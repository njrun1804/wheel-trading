#!/usr/bin/env python3
"""
Intraday Data Collection for Unity Wheel Trading
Runs during market hours for near-real-time snapshots
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import databento as db
import duckdb

from src.unity_wheel.secrets.manager import SecretManager

# Production logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "intraday_collection.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class IntradayCollector:
    """Collects intraday snapshots during market hours"""

    def __init__(self):
        self.db_path = (
            Path(__file__).parent.parent / "data" / "wheel_trading_optimized.duckdb"
        )
        self.metrics = {"start_time": datetime.now(), "snapshots": 0, "errors": []}

        # Load credentials
        self._load_credentials()

        # Initialize clients
        self.databento_client = db.Historical(self.databento_key)
        self.conn = duckdb.connect(str(self.db_path))

    def _load_credentials(self):
        """Load API keys from SecretManager"""
        try:
            secret_mgr = SecretManager()
            self.databento_key = secret_mgr.get_secret("databento_api_key")

            if not self.databento_key:
                raise ValueError("Missing Databento API key")

            logger.info("✅ Credentials loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            raise

    def is_market_hours(self):
        """Check if current time is during market hours"""
        now = datetime.now(UTC)

        # Market hours: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
        market_open = now.replace(hour=14, minute=30, second=0)
        market_close = now.replace(hour=21, minute=0, second=0)

        # Check if weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        return market_open <= now <= market_close

    async def collect_snapshot(self):
        """Collect a single market snapshot"""
        logger.info("📸 Collecting intraday snapshot...")

        try:
            # Get current time (use a few minutes ago to ensure data availability)
            snapshot_time = datetime.now(UTC) - timedelta(minutes=5)

            # Collect Unity stock snapshot
            await self._collect_stock_snapshot(snapshot_time)

            # Collect option snapshot (limited to near-the-money)
            await self._collect_option_snapshot(snapshot_time)

            self.metrics["snapshots"] += 1
            logger.info(
                f"✅ Snapshot {self.metrics['snapshots']} collected successfully"
            )

        except Exception as e:
            error_msg = f"Snapshot collection failed: {e}"
            logger.error(error_msg)
            self.metrics["errors"].append(error_msg)

    async def _collect_stock_snapshot(self, snapshot_time):
        """Collect Unity stock price snapshot"""
        try:
            # Get trades around snapshot time
            start_time = snapshot_time - timedelta(minutes=1)
            end_time = snapshot_time

            trades = self.databento_client.timeseries.get_range(
                dataset="EQUS.MINI",
                symbols=["U"],
                stype_in="raw_symbol",
                schema="trades",
                start=start_time,
                end=end_time,
            )

            df = trades.to_df()

            if not df.empty:
                # Get latest trade
                latest = df.iloc[-1]
                price = float(latest.get("price", 0)) / 1e9

                # Store snapshot
                self.conn.execute(
                    """
                    INSERT INTO market.intraday_snapshots
                    (symbol, timestamp, price, volume, snapshot_type)
                    VALUES (?, ?, ?, ?, 'stock')
                """,
                    ["U", snapshot_time, price, int(latest.get("size", 0)), "stock"],
                )

                logger.info(f"   Stock: ${price:.2f}")
                return price
            else:
                logger.warning("   No stock trades in snapshot window")
                return None

        except Exception as e:
            logger.error(f"Stock snapshot failed: {e}")
            return None

    async def _collect_option_snapshot(self, snapshot_time):
        """Collect option chain snapshot (ATM +/- 5 strikes)"""
        try:
            # Get current stock price
            stock_price = self._get_current_stock_price()
            if not stock_price:
                logger.warning("Could not get stock price for option snapshot")
                return

            # For intraday, we can try more recent data
            snapshot_date = snapshot_time.date()

            # Get near-term expirations (next 2 Fridays)
            friday = snapshot_date + timedelta((4 - snapshot_date.weekday()) % 7)
            if friday <= snapshot_date:
                friday += timedelta(days=7)

            expirations = [friday, friday + timedelta(days=7)]

            option_count = 0

            for expiry in expirations:
                # Get ATM strikes
                atm_strike = round(stock_price / 5) * 5  # Round to nearest $5
                strikes = [atm_strike + (i * 5) for i in range(-5, 6)]  # +/- 5 strikes

                # Build symbols for specific strikes
                symbols = []
                for strike in strikes:
                    # Unity options format: U YYMMDD P/C STRIKE
                    exp_str = expiry.strftime("%y%m%d")
                    for opt_type in ["P", "C"]:
                        symbol = f"U  {exp_str} {opt_type} {strike:05d}000"
                        symbols.append(symbol)

                if symbols:
                    try:
                        # Get quotes
                        quotes = self.databento_client.timeseries.get_range(
                            dataset="OPRA.PILLAR",
                            symbols=symbols[:20],  # Limit
                            stype_in="raw_symbol",
                            schema="mbp-1",
                            start=snapshot_time - timedelta(minutes=1),
                            end=snapshot_time,
                        )

                        quote_df = quotes.to_df()
                        option_count += len(quote_df)

                    except Exception as e:
                        logger.debug(f"Could not get quotes for {expiry}: {e}")

            logger.info(f"   Options: {option_count} quotes collected")

        except Exception as e:
            logger.error(f"Option snapshot failed: {e}")

    def _get_current_stock_price(self):
        """Get most recent Unity stock price"""
        try:
            result = self.conn.execute(
                """
                SELECT price FROM market.intraday_snapshots
                WHERE symbol = 'U' AND snapshot_type = 'stock'
                ORDER BY timestamp DESC
                LIMIT 1
            """
            ).fetchone()

            if result:
                return float(result[0])

            # Fallback to daily data
            result = self.conn.execute(
                """
                SELECT close FROM market.price_data
                WHERE symbol = 'U'
                ORDER BY date DESC
                LIMIT 1
            """
            ).fetchone()

            return float(result[0]) if result else None

        except Exception:
            return None

    def ensure_tables(self):
        """Ensure intraday snapshot tables exist"""
        try:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market.intraday_snapshots (
                    symbol VARCHAR,
                    timestamp TIMESTAMP,
                    price DECIMAL(10,2),
                    volume BIGINT,
                    snapshot_type VARCHAR,
                    PRIMARY KEY (symbol, timestamp, snapshot_type)
                )
            """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS options.intraday_quotes (
                    symbol VARCHAR,
                    expiration DATE,
                    strike DECIMAL(10,2),
                    option_type VARCHAR(4),
                    timestamp TIMESTAMP,
                    bid DECIMAL(10,2),
                    ask DECIMAL(10,2),
                    bid_size INTEGER,
                    ask_size INTEGER,
                    PRIMARY KEY (symbol, expiration, strike, option_type, timestamp)
                )
            """
            )

        except Exception as e:
            logger.warning(f"Could not create tables: {e}")

    async def run_once(self):
        """Run a single collection"""
        self.ensure_tables()

        if not self.is_market_hours():
            logger.info("Outside market hours - skipping collection")
            return 1

        await self.collect_snapshot()

        # Log summary
        duration = (datetime.now() - self.metrics["start_time"]).total_seconds()
        logger.info(
            f"Collection complete in {duration:.1f}s - Snapshots: {self.metrics['snapshots']}, Errors: {len(self.metrics['errors'])}"
        )

        return 0 if len(self.metrics["errors"]) == 0 else 1

    async def run_continuous(self, interval_minutes=15):
        """Run continuous collection during market hours"""
        self.ensure_tables()

        logger.info(
            f"Starting continuous intraday collection (every {interval_minutes} minutes)"
        )

        while True:
            if self.is_market_hours():
                await self.collect_snapshot()
            else:
                logger.info("Outside market hours - waiting...")

            # Wait for next interval
            await asyncio.sleep(interval_minutes * 60)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Intraday data collection")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument(
        "--interval", type=int, default=15, help="Collection interval in minutes"
    )

    args = parser.parse_args()

    collector = IntradayCollector()

    if args.once:
        exit_code = await collector.run_once()
        sys.exit(exit_code)
    else:
        await collector.run_continuous(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
