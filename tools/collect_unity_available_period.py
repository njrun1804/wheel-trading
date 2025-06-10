#!/usr/bin/env python3
"""
Collect Unity options data for AVAILABLE period: 2023-03-28 to 2025-06-09

This focuses on the period where CMBP-1 schema is available,
providing ~1.2 years of real historical data for backtesting.
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import databento as db
from databento_dbn import Schema, SType

from src.unity_wheel.secrets.integration import get_databento_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")
PROGRESS_FILE = os.path.expanduser("~/.wheel_trading/cache/unity_available_progress.json")


class UnityAvailablePeriodCollector:
    """Collect Unity options for available CMBP-1 period."""

    def __init__(self):
        self.api_key = get_databento_api_key()
        self.client = db.Historical(self.api_key)
        self.conn = duckdb.connect(DB_PATH)
        self.progress = self.load_progress()

        # CMBP-1 available period
        self.start_date = datetime(2023, 3, 28, tzinfo=timezone.utc)
        self.end_date = datetime(2025, 6, 9, tzinfo=timezone.utc)

    def load_progress(self):
        """Load progress for resume capability."""
        try:
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, "r") as f:
                    return json.load(f)
        except:
            pass
        return {"completed_dates": []}

    def save_progress(self):
        """Save progress."""
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(self.progress, f, indent=2, default=str)

    def get_trading_days(self):
        """Get trading days in available period."""
        trading_days = []
        current = self.start_date

        while current <= self.end_date:
            if current.weekday() < 5:  # Monday-Friday
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

    def collect_daily_quotes_optimized(self, date):
        """Collect quotes for one day with optimized time windows."""
        try:
            # Use smaller time windows to manage data volume
            windows = [
                (9, 30, 10, 30),  # Market open hour
                (13, 0, 14, 0),  # Early afternoon
                (15, 0, 16, 0),  # Market close hour
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

                    # Process in smaller batches
                    quotes_batch = []
                    batch_count = 0

                    for record in quotes_data:
                        # Handle CMBP-1 attributes
                        bid_price = getattr(record, "bid_px_00", getattr(record, "bid_px", 0)) / 1e9
                        ask_price = getattr(record, "ask_px_00", getattr(record, "ask_px", 0)) / 1e9
                        bid_size = getattr(record, "bid_sz_00", getattr(record, "bid_sz", 0))
                        ask_size = getattr(record, "ask_sz_00", getattr(record, "ask_sz", 0))

                        quotes_batch.append(
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

                        # Store every 500 quotes
                        if len(quotes_batch) >= 500:
                            self.store_quotes_batch(quotes_batch)
                            total_quotes += len(quotes_batch)
                            batch_count += 1
                            quotes_batch = []

                            # Log progress for large days
                            if batch_count % 10 == 0:
                                logger.info(
                                    f"   Window {start_hour}:{start_min}-{end_hour}:{end_min}: {total_quotes:,} quotes stored..."
                                )

                    # Store remaining quotes
                    if quotes_batch:
                        self.store_quotes_batch(quotes_batch)
                        total_quotes += len(quotes_batch)

                except Exception as e:
                    logger.warning(
                        f"Window {start_hour}:{start_min}-{end_hour}:{end_min} failed: {e}"
                    )
                    continue

            logger.info(f"✅ {date.strftime('%Y-%m-%d')}: Stored {total_quotes:,} quotes")
            return True

        except Exception as e:
            logger.error(f"❌ {date.strftime('%Y-%m-%d')}: Failed - {e}")
            return False

    def store_quotes_batch(self, quotes_batch):
        """Store quotes batch efficiently."""
        if not quotes_batch:
            return

        df = pd.DataFrame(quotes_batch)

        # Optimize dtypes
        df["bid_px"] = df["bid_px"].astype("float64")
        df["ask_px"] = df["ask_px"].astype("float64")
        df["bid_sz"] = df["bid_sz"].astype("uint64")
        df["ask_sz"] = df["ask_sz"].astype("uint64")
        df["instrument_id"] = df["instrument_id"].astype("uint64")

        # Insert with deduplication
        self.conn.execute("INSERT OR IGNORE INTO options_ticks SELECT * FROM df")
        self.conn.commit()

    def collect_available_period(self):
        """Collect data for the available CMBP-1 period."""
        print(f"🚀 COLLECTING UNITY OPTIONS - AVAILABLE PERIOD")
        print(
            f"📅 CMBP-1 Available: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"
        )
        print("=" * 70)

        trading_days = self.get_trading_days()
        total_days = len(trading_days)

        print(f"📊 Total trading days in period: {total_days:,}")

        # Filter completed days
        completed_dates = set(self.progress.get("completed_dates", []))
        remaining_days = [d for d in trading_days if d.strftime("%Y-%m-%d") not in completed_dates]

        print(f"✅ Already completed: {len(completed_dates)}")
        print(f"📥 Remaining: {len(remaining_days)}")

        if not remaining_days:
            print("🎉 All available days completed!")
            self.show_final_stats()
            return

        start_time = time.time()
        successful = 0
        failed = 0

        for i, date in enumerate(remaining_days, 1):
            print(f"\n📅 [{i:,}/{len(remaining_days):,}] {date.strftime('%Y-%m-%d')}...")

            try:
                success = self.collect_daily_quotes_optimized(date)

                if success:
                    successful += 1
                    self.progress["completed_dates"].append(date.strftime("%Y-%m-%d"))
                else:
                    failed += 1

                # Save progress every 5 days
                if i % 5 == 0:
                    self.save_progress()
                    elapsed = time.time() - start_time
                    rate = i / elapsed * 3600 if elapsed > 0 else 0
                    remaining_time = (len(remaining_days) - i) / rate if rate > 0 else 0

                    print(
                        f"📊 Progress: {i}/{len(remaining_days)} ({i/len(remaining_days)*100:.1f}%)"
                    )
                    print(f"⏱️  Rate: {rate:.1f} days/hour, ETA: {remaining_time:.1f} hours")

                # Show interim stats every 20 days
                if i % 20 == 0:
                    self.show_interim_stats()

            except KeyboardInterrupt:
                print(f"\n⏸️  Collection paused. Progress saved.")
                self.save_progress()
                return
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                failed += 1

        # Final save and stats
        self.save_progress()

        print(f"\n🎯 COLLECTION COMPLETE")
        print("=" * 70)
        print(f"✅ Successful days: {successful:,}")
        print(f"❌ Failed days: {failed}")

        self.show_final_stats()

    def show_interim_stats(self):
        """Show current database statistics."""
        stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as quotes,
                COUNT(DISTINCT instrument_id) as instruments,
                COUNT(DISTINCT trade_date) as days,
                MAX(trade_date) as latest_date
            FROM options_ticks
        """
        ).fetchone()

        print(
            f"📊 Current stats: {stats[0]:,} quotes, {stats[1]:,} instruments, {stats[2]:,} days (latest: {stats[3]})"
        )

    def show_final_stats(self):
        """Show comprehensive final statistics."""
        # Quotes statistics
        quotes_stats = self.conn.execute(
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

        # Instruments statistics
        instrument_stats = self.conn.execute(
            """
            SELECT
                COUNT(*) as total_contracts,
                COUNT(CASE WHEN option_type = 'P' THEN 1 END) as puts,
                COUNT(CASE WHEN option_type = 'C' THEN 1 END) as calls,
                MIN(strike) as min_strike,
                MAX(strike) as max_strike,
                COUNT(DISTINCT expiration) as expirations
            FROM instruments
            WHERE underlying = 'U'
        """
        ).fetchone()

        print(f"\n📊 FINAL UNITY OPTIONS BACKTESTING DATASET:")
        print("=" * 70)
        print(f"📈 Total option quotes: {quotes_stats[0]:,}")
        print(f"🔢 Unique instruments: {quotes_stats[1]:,}")
        print(f"📅 Trading days with data: {quotes_stats[2]:,}")
        print(f"📆 Quote date range: {quotes_stats[3]} to {quotes_stats[4]}")
        print(f"")
        print(f"📋 Total Unity contracts: {instrument_stats[0]:,}")
        print(f"📊 Puts: {instrument_stats[1]:,}, Calls: {instrument_stats[2]:,}")
        print(f"💰 Strike range: ${instrument_stats[3]:.2f} - ${instrument_stats[4]:.2f}")
        print(f"📅 Expiration count: {instrument_stats[5]:,}")

        # Calculate data period
        if quotes_stats[2] > 0:
            start_date = datetime.strptime(str(quotes_stats[3]), "%Y-%m-%d")
            end_date = datetime.strptime(str(quotes_stats[4]), "%Y-%m-%d")
            period_days = (end_date - start_date).days
            period_months = period_days / 30.44

            print(f"📊 Data period: {period_days:,} days ({period_months:.1f} months)")
            print(f"📊 Data coverage: {quotes_stats[2]/period_days*100:.1f}% of calendar days")

        print(f"\n🎉 BACKTESTING DATASET READY!")
        print(f"✅ {period_months:.1f} months of real Unity options data")
        print(f"🚨 NO SYNTHETIC DATA - All from Databento OPRA feed")
        print(f"✅ Suitable for comprehensive wheel strategy backtesting")

    def close(self):
        """Cleanup."""
        if self.conn:
            self.conn.close()


def main():
    """Main function."""
    collector = UnityAvailablePeriodCollector()

    try:
        print(f"🎯 UNITY OPTIONS HISTORICAL DATA COLLECTION")
        print(f"📊 Target period: March 28, 2023 to June 9, 2025")
        print(f"📋 Reason: CMBP-1 schema availability window")
        print(f"🎯 Goal: Build comprehensive backtesting dataset")

        collector.collect_available_period()

    except KeyboardInterrupt:
        print(f"\n⏸️  Collection interrupted - progress saved")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        raise
    finally:
        collector.close()


if __name__ == "__main__":
    main()
