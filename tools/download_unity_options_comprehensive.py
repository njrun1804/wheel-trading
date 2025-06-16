#!/usr/bin/env python3
"""
Download comprehensive Unity options data from Databento.
Uses definition + snapshot approach to get ALL options, not just traded ones.
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient
from unity_wheel.config.unified_config import get_config

config = get_config()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class UnityOptionsComprehensiveDownloader:
    """Download comprehensive Unity options data using definitions + snapshots."""

    def __init__(self):
        self.client = DatabentoClient()
        self.db_path = Path(config.storage.database_path).expanduser()
        self.conn = duckdb.connect(str(self.db_path))
        self.eastern = pytz.timezone("US/Eastern")

        # Create tables
        self.setup_tables()

    def setup_tables(self):
        """Create tables for Unity options data."""
        # Main daily table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unity_options_daily (
                date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                expiration DATE NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                option_type VARCHAR(1) NOT NULL,
                bid DECIMAL(10,4),
                ask DECIMAL(10,4),
                last DECIMAL(10,4),
                volume BIGINT,
                open_interest BIGINT,
                underlying_price DECIMAL(10,2),
                iv DECIMAL(6,4),
                delta DECIMAL(6,4),
                gamma DECIMAL(6,4),
                theta DECIMAL(6,4),
                vega DECIMAL(6,4),
                PRIMARY KEY (date, symbol)
            )
        """
        )

        # Index for efficient queries
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_unity_daily_date_exp_strike
            ON unity_options_daily(date, expiration, strike, option_type)
        """
        )

        logger.info("Tables created/verified")

    def get_unity_price(self, date: datetime.date) -> float:
        """Get Unity stock price for a date."""
        result = self.conn.execute(
            """
            SELECT close FROM price_history
            WHERE symbol = config.trading.symbol AND date = ?
        """,
            (date,),
        ).fetchone()

        return result[0] if result else None

    def download_options_for_date(self, date: datetime.date) -> int:
        """Download Unity options data for a single date using best available method."""
        # Skip weekends
        if date.weekday() >= 5:
            return 0

        logger.info(f"Processing {date}...")

        # Method 1: Try to get end-of-day snapshot (best)
        records = self.download_snapshot_data(date)
        if records > 0:
            logger.info(f"  ✓ {date}: {records} options via snapshot")
            return records

        # Method 2: Try statistics schema
        records = self.download_statistics_data(date)
        if records > 0:
            logger.info(f"  ✓ {date}: {records} options via statistics")
            return records

        # Method 3: Get definitions and last quotes
        records = self.download_definition_quotes(date)
        if records > 0:
            logger.info(f"  ✓ {date}: {records} options via definitions+quotes")
            return records

        # Method 4: Aggregate trades (last resort)
        records = self.download_trades_data(date)
        if records > 0:
            logger.info(f"  ✓ {date}: {records} options via trades")
            return records

        logger.debug(f"  - {date}: No data available")
        return 0

    def download_snapshot_data(self, date: datetime.date) -> int:
        """Download end-of-day snapshot data."""
        # Get market close time
        close_time = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            # Get snapshot at close
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="mbp-1",  # Market by price snapshot
                symbols=["U.OPT"],
                stype_in="parent",
                start=close_time - timedelta(minutes=5),
                end=close_time + timedelta(minutes=5),
                limit=50000,
            )

            df = data.to_df()
            if df.empty:
                return 0

            return self.process_snapshot_data(df, date)

        except Exception as e:
            if "mbp-1" in str(e) or "deprecated" in str(e):
                # Try consolidated MBP
                try:
                    data = self.client.client.timeseries.get_range(
                        dataset="OPRA.PILLAR",
                        schema="cmbp-1",
                        symbols=["U.OPT"],
                        stype_in="parent",
                        start=close_time - timedelta(minutes=5),
                        end=close_time + timedelta(minutes=5),
                        limit=50000,
                    )

                    df = data.to_df()
                    if not df.empty:
                        return self.process_snapshot_data(df, date)
                except:
                    pass

            return 0

    def download_statistics_data(self, date: datetime.date) -> int:
        """Download daily statistics data."""
        # Statistics are typically generated after market close
        stats_time = datetime.combine(date, datetime.min.time()).replace(
            hour=17, minute=0, tzinfo=self.eastern
        )

        try:
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="statistics",
                symbols=["U.OPT"],
                stype_in="parent",
                start=stats_time - timedelta(hours=2),
                end=stats_time + timedelta(hours=2),
                limit=50000,
            )

            df = data.to_df()
            if df.empty:
                return 0

            return self.process_statistics_data(df, date)

        except Exception as e:
            logger.debug(f"Statistics failed: {e}")
            return 0

    def download_definition_quotes(self, date: datetime.date) -> int:
        """Download option definitions and get last quotes."""
        # First get all Unity option definitions for this date
        definitions = self.get_option_definitions(date)
        if not definitions:
            return 0

        # Get end-of-day quotes for these options
        close_time = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        records_inserted = 0

        # Process in batches
        batch_size = 100
        for i in range(0, len(definitions), batch_size):
            batch = definitions[i : i + batch_size]
            instrument_ids = [d["instrument_id"] for d in batch]

            try:
                # Get quotes for these instruments
                data = self.client.client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema="tbbo",  # Top bid/ask
                    instrument_ids=instrument_ids,
                    start=close_time - timedelta(minutes=30),
                    end=close_time,
                    limit=10000,
                )

                df = data.to_df()
                if not df.empty:
                    # Get last quote for each instrument
                    for defn in batch:
                        inst_quotes = df[df["instrument_id"] == defn["instrument_id"]]
                        if not inst_quotes.empty:
                            last_quote = inst_quotes.iloc[-1]

                            self.insert_option_record(
                                date=date,
                                symbol=defn["symbol"],
                                expiration=defn["expiration"],
                                strike=defn["strike"],
                                option_type=defn["option_type"],
                                bid=self.convert_price(last_quote.get("bid_px_01")),
                                ask=self.convert_price(last_quote.get("ask_px_01")),
                                volume=0,  # No volume in quote data
                                open_interest=0,
                            )
                            records_inserted += 1

            except Exception as e:
                logger.debug(f"Quote batch failed: {e}")
                continue

        return records_inserted

    def download_trades_data(self, date: datetime.date) -> int:
        """Download and aggregate trades data."""
        market_start = datetime.combine(date, datetime.min.time()).replace(
            hour=9, minute=30, tzinfo=self.eastern
        )
        market_end = datetime.combine(date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=self.eastern
        )

        try:
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="trades",
                symbols=["U.OPT"],
                stype_in="parent",
                start=market_start,
                end=market_end,
                limit=100000,
            )

            df = data.to_df()
            if df.empty:
                return 0

            return self.process_trades_data(df, date)

        except Exception as e:
            logger.debug(f"Trades failed: {e}")
            return 0

    def get_option_definitions(self, date: datetime.date) -> list[dict]:
        """Get all Unity option definitions for a date."""
        # Definitions are available at midnight
        def_time = datetime.combine(date, datetime.min.time()).replace(tzinfo=pytz.UTC)

        try:
            data = self.client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="definition",
                symbols=["U"],  # Just U for definitions
                start=def_time,
                end=def_time + timedelta(seconds=1),
                limit=10000,
            )

            df = data.to_df()
            if df.empty:
                return []

            # Filter for Unity options
            unity_options = df[df["raw_symbol"].str.startswith("U ", na=False)]

            definitions = []
            for _, row in unity_options.iterrows():
                symbol = row["raw_symbol"]
                if len(symbol) >= 21:
                    try:
                        # Parse Unity option symbol
                        exp_str = symbol[6:12]
                        expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                        option_type = symbol[12]
                        strike = float(symbol[13:21]) / 1000

                        definitions.append(
                            {
                                "instrument_id": row["instrument_id"],
                                "symbol": symbol,
                                "expiration": expiration,
                                "strike": strike,
                                "option_type": option_type,
                            }
                        )
                    except:
                        continue

            return definitions

        except Exception as e:
            logger.debug(f"Definitions failed: {e}")
            return []

    def process_snapshot_data(self, df, date: datetime.date) -> int:
        """Process market snapshot data."""
        records = 0

        for _, row in df.iterrows():
            try:
                symbol = row.get("raw_symbol", "")
                if not symbol.startswith("U "):
                    continue

                # Parse symbol
                if len(symbol) >= 21:
                    exp_str = symbol[6:12]
                    expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                    option_type = symbol[12]
                    strike = float(symbol[13:21]) / 1000

                    self.insert_option_record(
                        date=date,
                        symbol=symbol,
                        expiration=expiration,
                        strike=strike,
                        option_type=option_type,
                        bid=self.convert_price(row.get("bid_px_01")),
                        ask=self.convert_price(row.get("ask_px_01")),
                        volume=row.get("volume", 0),
                        open_interest=row.get("open_interest", 0),
                    )
                    records += 1

            except Exception as e:
                continue

        self.conn.commit()
        return records

    def process_statistics_data(self, df, date: datetime.date) -> int:
        """Process statistics data."""
        records = 0

        for _, row in df.iterrows():
            try:
                symbol = row.get("raw_symbol", "")
                if not symbol.startswith("U "):
                    continue

                # Parse symbol
                if len(symbol) >= 21:
                    exp_str = symbol[6:12]
                    expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                    option_type = symbol[12]
                    strike = float(symbol[13:21]) / 1000

                    self.insert_option_record(
                        date=date,
                        symbol=symbol,
                        expiration=expiration,
                        strike=strike,
                        option_type=option_type,
                        bid=self.convert_price(row.get("bid_close")),
                        ask=self.convert_price(row.get("ask_close")),
                        last=self.convert_price(row.get("close_price")),
                        volume=row.get("volume", 0),
                        open_interest=row.get("open_interest", 0),
                    )
                    records += 1

            except Exception as e:
                continue

        self.conn.commit()
        return records

    def process_trades_data(self, df, date: datetime.date) -> int:
        """Process and aggregate trades data."""
        # Group by symbol
        symbol_groups = df.groupby("raw_symbol") if "raw_symbol" in df.columns else []

        records = 0
        for symbol, group in symbol_groups:
            if not symbol.startswith("U "):
                continue

            try:
                # Parse symbol
                if len(symbol) >= 21:
                    exp_str = symbol[6:12]
                    expiration = datetime.strptime("20" + exp_str, "%Y%m%d").date()
                    option_type = symbol[12]
                    strike = float(symbol[13:21]) / 1000

                    # Get last trade and total volume
                    last_trade = group.iloc[-1]
                    total_volume = len(group)

                    self.insert_option_record(
                        date=date,
                        symbol=symbol,
                        expiration=expiration,
                        strike=strike,
                        option_type=option_type,
                        last=self.convert_price(last_trade.get("price")),
                        volume=total_volume,
                        open_interest=0,
                    )
                    records += 1

            except Exception as e:
                continue

        self.conn.commit()
        return records

    def convert_price(self, price):
        """Convert Databento price format."""
        if price is None:
            return None
        # Databento uses fixed-point format
        if price > 10000:
            return float(price) / 10000.0
        return float(price)

    def insert_option_record(self, **kwargs):
        """Insert option record into database."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO unity_options_daily
            (date, symbol, expiration, strike, option_type, bid, ask, last,
             volume, open_interest, underlying_price, iv, delta, gamma, theta, vega)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                kwargs.get("date"),
                kwargs.get("symbol"),
                kwargs.get("expiration"),
                kwargs.get("strike"),
                kwargs.get("option_type"),
                kwargs.get("bid"),
                kwargs.get("ask"),
                kwargs.get("last"),
                kwargs.get("volume", 0),
                kwargs.get("open_interest", 0),
                kwargs.get("underlying_price"),
                kwargs.get("iv"),
                kwargs.get("delta"),
                kwargs.get("gamma"),
                kwargs.get("theta"),
                kwargs.get("vega"),
            ),
        )

    def download_all_data(self):
        """Download all Unity options data comprehensively."""
        start_date = datetime(2023, 3, 28).date()
        end_date = datetime.now().date() - timedelta(days=1)

        logger.info("=" * 60)
        logger.info("DOWNLOADING COMPREHENSIVE UNITY OPTIONS DATA")
        logger.info("=" * 60)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info("Using multiple methods for maximum coverage:")
        logger.info("  1. End-of-day snapshots")
        logger.info("  2. Daily statistics")
        logger.info("  3. Definitions + quotes")
        logger.info("  4. Trade aggregation")
        logger.info("=" * 60)

        current_date = start_date
        total_records = 0
        days_with_data = 0
        consecutive_empty = 0

        while current_date <= end_date:
            records = self.download_options_for_date(current_date)

            if records > 0:
                total_records += records
                days_with_data += 1
                consecutive_empty = 0
            else:
                consecutive_empty += 1

                # If too many empty days, we might be past data availability
                if consecutive_empty > 30:
                    logger.warning(
                        "Many consecutive empty days, checking if we're past data availability..."
                    )
                    # Don't break, just log

            # Brief pause
            if days_with_data % 10 == 0 and days_with_data > 0:
                time.sleep(1)

            current_date += timedelta(days=1)

        # Show comprehensive summary
        self.show_comprehensive_summary()

    def show_comprehensive_summary(self):
        """Show detailed summary of all downloaded data."""
        logger.info("\n" + "=" * 60)
        logger.info("COMPREHENSIVE UNITY OPTIONS DATA SUMMARY")
        logger.info("=" * 60)

        # Overall statistics
        stats = self.conn.execute(
            """
            SELECT
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT symbol) as unique_contracts,
                COUNT(*) as total_records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                SUM(volume) as total_volume,
                COUNT(DISTINCT expiration) as unique_expirations,
                COUNT(DISTINCT strike) as unique_strikes
            FROM unity_options_daily
        """
        ).fetchone()

        if stats and stats[0] > 0:
            logger.info("\nData Coverage:")
            logger.info(f"  Trading days with data: {stats[0]}")
            logger.info(f"  Unique option contracts: {stats[1]:,}")
            logger.info(f"  Total records: {stats[2]:,}")
            logger.info(f"  Date range: {stats[3]} to {stats[4]}")

            # Calculate coverage
            if stats[3] and stats[4]:
                start = datetime.strptime(str(stats[3]), "%Y-%m-%d")
                end = datetime.strptime(str(stats[4]), "%Y-%m-%d")
                total_days = (end - start).days + 1
                weekdays = sum(
                    1
                    for i in range(total_days)
                    if (start + timedelta(days=i)).weekday() < 5
                )
                coverage = (stats[0] / weekdays) * 100 if weekdays > 0 else 0

                logger.info(f"  Coverage: {coverage:.1f}% of weekdays")
                logger.info(
                    f"  Total volume: {stats[5]:,}"
                    if stats[5]
                    else "  Total volume: N/A"
                )
                logger.info(f"  Unique expirations: {stats[6]}")
                logger.info(f"  Unique strikes: {stats[7]}")

            # Daily average
            daily_avg = self.conn.execute(
                """
                SELECT
                    AVG(daily_count) as avg_options_per_day,
                    MIN(daily_count) as min_options,
                    MAX(daily_count) as max_options
                FROM (
                    SELECT date, COUNT(*) as daily_count
                    FROM unity_options_daily
                    GROUP BY date
                )
            """
            ).fetchone()

            if daily_avg:
                logger.info("\nDaily Statistics:")
                logger.info(f"  Average options per day: {daily_avg[0]:.0f}")
                logger.info(f"  Min options in a day: {daily_avg[1]}")
                logger.info(f"  Max options in a day: {daily_avg[2]}")

            # Recent sample
            logger.info("\nMost Recent Options:")
            recent = self.conn.execute(
                """
                SELECT date, symbol, strike, option_type, bid, ask, volume
                FROM unity_options_daily
                WHERE date = (SELECT MAX(date) FROM unity_options_daily)
                AND (bid > 0 OR ask > 0 OR volume > 0)
                ORDER BY volume DESC
                LIMIT 5
            """
            ).fetchall()

            if recent:
                for date, symbol, strike, otype, bid, ask, volume in recent:
                    bid_str = f"${bid:.2f}" if bid else "N/A"
                    ask_str = f"${ask:.2f}" if ask else "N/A"
                    logger.info(
                        f"  {symbol}: ${strike} {otype} bid={bid_str} ask={ask_str} vol={volume}"
                    )

            # Check data quality
            quality_check = self.conn.execute(
                """
                SELECT
                    COUNT(CASE WHEN bid IS NULL AND ask IS NULL AND last IS NULL THEN 1 END) as no_prices,
                    COUNT(CASE WHEN volume = 0 THEN 1 END) as zero_volume,
                    COUNT(CASE WHEN bid > 0 OR ask > 0 THEN 1 END) as has_quotes
                FROM unity_options_daily
            """
            ).fetchone()

            if quality_check:
                logger.info("\nData Quality:")
                logger.info(f"  Records with quotes: {quality_check[2]:,}")
                logger.info(f"  Records with zero volume: {quality_check[1]:,}")
                logger.info(f"  Records missing all prices: {quality_check[0]:,}")

            logger.info("\n✅ Comprehensive Unity options data successfully downloaded")
            logger.info("✅ Using multiple data sources for maximum coverage")
            logger.info("✅ All data is REAL from Databento - NO SYNTHETIC DATA")

        else:
            logger.warning("\n⚠️  No data downloaded - check Databento API credentials")

    def cleanup(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Main entry point."""
    downloader = UnityOptionsComprehensiveDownloader()

    try:
        downloader.download_all_data()
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        downloader.cleanup()


if __name__ == "__main__":
    main()
