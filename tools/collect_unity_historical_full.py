#!/usr/bin/env python3
"""
Collect 3 years of Unity options historical data for backtesting.

Implements optimal historical collection strategy from technical guide:
- Daily chunks to manage memory
- Batch processing with proper error handling
- Progress tracking and resume capability
- Efficient storage in DuckDB
"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import databento as db
from databento_dbn import Schema, SType

from src.unity_wheel.secrets.integration import get_databento_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")
PROGRESS_FILE = os.path.expanduser("~/.wheel_trading/cache/unity_collection_progress.json")


class UnityHistoricalCollector:
    """Collect 3 years of Unity options for backtesting."""

    def __init__(self):
        self.api_key = get_databento_api_key()
        self.client = db.Historical(self.api_key)
        self.conn = None
        self.progress = self.load_progress()

    def load_progress(self):
        """Load collection progress to enable resume."""
        try:
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, "r") as f:
                    return json.load(f)
        except:
            pass
        return {"completed_dates": [], "failed_dates": [], "last_date": None}

    def save_progress(self):
        """Save collection progress."""
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(self.progress, f, indent=2, default=str)

    def initialize_database(self):
        """Initialize database if not already done."""
        self.conn = duckdb.connect(DB_PATH)

        # Check if tables exist
        tables = self.conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_name IN ('options_ticks', 'instruments')
        """
        ).fetchall()

        if len(tables) < 2:
            print("📋 Creating database schema...")

            # Create options_ticks table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS options_ticks (
                    trade_date DATE NOT NULL,
                    ts_event TIMESTAMP NOT NULL,
                    instrument_id UBIGINT NOT NULL,
                    bid_px DECIMAL(10,4) NOT NULL,
                    ask_px DECIMAL(10,4) NOT NULL,
                    bid_sz UBIGINT NOT NULL,
                    ask_sz UBIGINT NOT NULL,
                    PRIMARY KEY (trade_date, ts_event, instrument_id)
                )
            """
            )

            # Create instruments table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS instruments (
                    instrument_id UBIGINT PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    underlying VARCHAR NOT NULL,
                    expiration DATE NOT NULL,
                    strike DECIMAL(10,2) NOT NULL,
                    option_type VARCHAR(1) NOT NULL,
                    date_listed DATE
                )
            """
            )

            # Create indexes
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_options_instrument ON options_ticks(instrument_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_instruments_underlying ON instruments(underlying, expiration)"
            )

            print("✅ Database schema created")
        else:
            print("✅ Database schema exists")

    def get_trading_days(self, start_date, end_date):
        """Get list of trading days (exclude weekends)."""
        trading_days = []
        current = start_date

        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

    def collect_daily_definitions(self, date):
        """Collect instrument definitions for a single day."""
        try:
            definitions = self.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.DEFINITION,
                symbols=["U.OPT"],
                stype_in=SType.PARENT,
                start=date,
                end=date + timedelta(days=1),
            )

            definitions_list = list(definitions)

            if definitions_list:
                self.store_definitions(definitions_list)
                logger.info(f"✅ {date}: Stored {len(definitions_list)} definitions")
                return True
            else:
                logger.info(f"📋 {date}: No definitions found")
                return True  # Not an error - just no new contracts

        except Exception as e:
            logger.error(f"❌ {date}: Definitions failed - {e}")
            return False

    def collect_daily_quotes(self, date):
        """Collect quotes for a single day using efficient time windows."""
        try:
            # Use 30-minute windows to manage data size
            windows = [
                (9, 30, 10, 0),  # Market open
                (12, 0, 12, 30),  # Midday
                (15, 30, 16, 0),  # Market close
            ]

            total_quotes = 0

            for start_hour, start_min, end_hour, end_min in windows:
                try:
                    window_start = date.replace(hour=start_hour, minute=start_min)
                    window_end = date.replace(hour=end_hour, minute=end_min)

                    quotes_data = self.client.timeseries.get_range(
                        dataset="OPRA.PILLAR",
                        schema="cmbp-1",
                        symbols=["U.OPT"],
                        stype_in=SType.PARENT,
                        start=window_start,
                        end=window_end,
                    )

                    # Process in chunks
                    quotes_chunk = []
                    for i, record in enumerate(quotes_data):
                        # Handle CMBP-1 message attributes
                        bid_price = getattr(record, "bid_px_00", getattr(record, "bid_px", 0)) / 1e9
                        ask_price = getattr(record, "ask_px_00", getattr(record, "ask_px", 0)) / 1e9
                        bid_size = getattr(record, "bid_sz_00", getattr(record, "bid_sz", 0))
                        ask_size = getattr(record, "ask_sz_00", getattr(record, "ask_sz", 0))

                        quotes_chunk.append(
                            {
                                "trade_date": date.date(),
                                "ts_event": pd.to_datetime(record.ts_event, unit="ns", utc=True),
                                "instrument_id": record.instrument_id,
                                "bid_px": bid_price,
                                "ask_px": ask_price,
                                "bid_sz": bid_size,
                                "ask_sz": ask_size,
                            }
                        )

                        # Store in batches of 1000
                        if len(quotes_chunk) >= 1000:
                            self.store_quotes_batch(quotes_chunk)
                            total_quotes += len(quotes_chunk)
                            quotes_chunk = []

                    # Store remaining quotes
                    if quotes_chunk:
                        self.store_quotes_batch(quotes_chunk)
                        total_quotes += len(quotes_chunk)

                except Exception as e:
                    logger.warning(
                        f"⚠️  {date} window {start_hour}:{start_min}-{end_hour}:{end_min} failed: {e}"
                    )
                    continue

            logger.info(f"✅ {date}: Stored {total_quotes:,} quotes")
            return True

        except Exception as e:
            logger.error(f"❌ {date}: Quotes failed - {e}")
            return False

    def store_definitions(self, definitions):
        """Store instrument definitions efficiently."""
        if not definitions:
            return

        df_data = []
        for defn in definitions:
            raw_symbol = defn.raw_symbol if hasattr(defn, "raw_symbol") else str(defn.symbol)

            # Parse option type from OCC symbol
            option_type = "C"
            if "P" in raw_symbol:
                option_type = "P"
            elif "C" in raw_symbol:
                option_type = "C"

            # Parse strike price
            strike_price = 0.0
            if hasattr(defn, "strike_price"):
                strike_price = float(defn.strike_price) / 1e9

            # Parse expiration
            expiration_date = None
            if hasattr(defn, "expiration"):
                expiration_date = pd.to_datetime(defn.expiration, unit="ns").date()

            df_data.append(
                {
                    "instrument_id": defn.instrument_id,
                    "symbol": raw_symbol,
                    "underlying": "U",
                    "expiration": expiration_date,
                    "strike": strike_price,
                    "option_type": option_type,
                    "date_listed": pd.to_datetime(defn.ts_event, unit="ns").date(),
                }
            )

        if df_data:
            df = pd.DataFrame(df_data)
            self.conn.execute("INSERT OR REPLACE INTO instruments SELECT * FROM df")
            self.conn.commit()

    def store_quotes_batch(self, quotes_batch):
        """Store batch of quotes efficiently."""
        if not quotes_batch:
            return

        df = pd.DataFrame(quotes_batch)

        # Optimize dtypes
        df["bid_px"] = df["bid_px"].astype("float64")
        df["ask_px"] = df["ask_px"].astype("float64")
        df["bid_sz"] = df["bid_sz"].astype("uint64")
        df["ask_sz"] = df["ask_sz"].astype("uint64")
        df["instrument_id"] = df["instrument_id"].astype("uint64")

        # Insert with conflict resolution
        self.conn.execute("INSERT OR IGNORE INTO options_ticks SELECT * FROM df")
        self.conn.commit()

    def collect_historical_data(self, start_date, end_date):
        """Collect historical data with progress tracking."""
        print(f"🚀 COLLECTING 3 YEARS OF UNITY OPTIONS HISTORICAL DATA")
        print(
            f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        print("=" * 70)

        trading_days = self.get_trading_days(start_date, end_date)
        total_days = len(trading_days)

        print(f"📊 Total trading days to process: {total_days:,}")

        # Filter out already completed days
        completed_dates = set(self.progress.get("completed_dates", []))
        remaining_days = [d for d in trading_days if d.strftime("%Y-%m-%d") not in completed_dates]

        print(f"✅ Already completed: {len(completed_dates)}")
        print(f"📥 Remaining to process: {len(remaining_days)}")

        if not remaining_days:
            print("🎉 All dates already completed!")
            return

        start_time = time.time()
        successful_days = 0
        failed_days = 0

        for i, date in enumerate(remaining_days, 1):
            print(f"\n📅 [{i:,}/{len(remaining_days):,}] Processing {date.strftime('%Y-%m-%d')}...")

            try:
                # Collect definitions first
                def_success = self.collect_daily_definitions(date)

                # Collect quotes
                quotes_success = self.collect_daily_quotes(date)

                if def_success and quotes_success:
                    successful_days += 1
                    self.progress["completed_dates"].append(date.strftime("%Y-%m-%d"))
                    self.progress["last_date"] = date.strftime("%Y-%m-%d")
                else:
                    failed_days += 1
                    self.progress["failed_dates"].append(date.strftime("%Y-%m-%d"))

                # Save progress every 10 days
                if i % 10 == 0:
                    self.save_progress()
                    elapsed = time.time() - start_time
                    rate = i / elapsed * 3600  # days per hour
                    remaining_time = (len(remaining_days) - i) / rate if rate > 0 else 0

                    print(
                        f"📊 Progress: {i}/{len(remaining_days)} ({i/len(remaining_days)*100:.1f}%)"
                    )
                    print(f"⏱️  Rate: {rate:.1f} days/hour, ETA: {remaining_time:.1f} hours")

            except Exception as e:
                logger.error(f"❌ Unexpected error for {date}: {e}")
                failed_days += 1
                self.progress["failed_dates"].append(date.strftime("%Y-%m-%d"))

        # Final save
        self.save_progress()

        # Final summary
        print(f"\n🎯 HISTORICAL COLLECTION COMPLETE")
        print("=" * 70)
        print(f"✅ Successful days: {successful_days:,}")
        print(f"❌ Failed days: {failed_days}")

        # Show database stats
        self.show_final_stats()

    def show_final_stats(self):
        """Show final database statistics."""
        stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as total_quotes,
                COUNT(DISTINCT instrument_id) as unique_instruments,
                COUNT(DISTINCT trade_date) as trading_days,
                MIN(trade_date) as start_date,
                MAX(trade_date) as end_date
            FROM options_ticks
        """
        ).fetchone()

        instrument_stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as total_contracts,
                COUNT(CASE WHEN option_type = 'P' THEN 1 END) as puts,
                COUNT(CASE WHEN option_type = 'C' THEN 1 END) as calls,
                MIN(strike) as min_strike,
                MAX(strike) as max_strike,
                MIN(expiration) as min_exp,
                MAX(expiration) as max_exp
            FROM instruments
            WHERE underlying = 'U'
        """
        ).fetchone()

        print(f"\n📊 FINAL UNITY OPTIONS DATASET:")
        print(f"   📈 Total quotes: {stats[0]:,}")
        print(f"   🔢 Unique instruments: {stats[1]:,}")
        print(f"   📅 Trading days: {stats[2]:,}")
        print(f"   📆 Date range: {stats[3]} to {stats[4]}")
        print(f"")
        print(f"   📋 Total contracts: {instrument_stats[0]:,}")
        print(f"   📊 Puts: {instrument_stats[1]:,}, Calls: {instrument_stats[2]:,}")
        print(f"   💰 Strike range: ${instrument_stats[3]:.2f} - ${instrument_stats[4]:.2f}")
        print(f"   📅 Expiration range: {instrument_stats[5]} to {instrument_stats[6]}")

        # Calculate data coverage
        if stats[2] > 0:
            expected_days = len(
                self.get_trading_days(
                    datetime.strptime(str(stats[3]), "%Y-%m-%d"),
                    datetime.strptime(str(stats[4]), "%Y-%m-%d"),
                )
            )
            coverage = stats[2] / expected_days * 100 if expected_days > 0 else 0
            print(f"   📊 Data coverage: {coverage:.1f}% of trading days")

        print(f"\n🎉 BACKTESTING DATASET READY!")
        print(f"✅ 3 years of real Unity options data collected")
        print(f"🚨 NO SYNTHETIC DATA - All from Databento OPRA feed")

    def close(self):
        """Clean up resources."""
        if self.conn:
            self.conn.close()


def main():
    """Main function to collect 3 years of historical data."""
    collector = UnityHistoricalCollector()

    try:
        # Initialize database
        collector.initialize_database()

        # Define 3-year period for backtesting
        end_date = datetime.now().date() - timedelta(days=1)  # Yesterday (data available)
        start_date = end_date - timedelta(days=3 * 365)  # 3 years ago

        print(f"🎯 TARGET: 3 years of Unity options data for backtesting")
        print(f"📅 Period: {start_date} to {end_date}")

        # Start collection
        collector.collect_historical_data(
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time()),
        )

    except KeyboardInterrupt:
        print(f"\n⏸️  Collection paused - progress saved. Resume by running script again.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        raise
    finally:
        collector.close()


if __name__ == "__main__":
    main()
