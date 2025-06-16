#!/usr/bin/env python3
"""
Simple data migration focusing on actual available data
"""

import logging
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleMigrator:
    def __init__(self, target_db: str):
        self.target_db = target_db

    def migrate_from_existing_dbs(self):
        """Migrate data from existing DuckDB databases"""
        logger.info("🚀 Starting migration from existing databases...")

        # Market data from master database
        self._migrate_market_data()

        # Options data from all sources
        self._migrate_options_data()

        # Create materialized view
        self._create_materialized_views()

        logger.info("✅ Migration complete!")

    def _migrate_market_data(self):
        """Migrate market data from data/wheel_trading_optimized.duckdb"""
        source_db = "data/wheel_trading_optimized.duckdb"
        if not Path(source_db).exists():
            logger.warning(f"Source database not found: {source_db}")
            return

        logger.info("📊 Migrating market data...")

        try:
            # Connect to source
            source_conn = duckdb.connect(source_db, read_only=True)

            # Check what tables exist
            tables = source_conn.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """
            ).fetchall()

            logger.info(f"  Found tables: {[t[0] for t in tables]}")

            # Try to find market data
            if "market_data" in [t[0] for t in tables]:
                count = source_conn.execute(
                    "SELECT COUNT(*) FROM market_data"
                ).fetchone()[0]
                logger.info(f"  Found {count} rows in market_data table")

                if count > 0:
                    # Read data
                    df = source_conn.execute("SELECT * FROM market_data").df()

                    # Ensure correct column order and types
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values(["symbol", "date"])

                    # Calculate daily returns
                    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

                    # Calculate rolling volatility
                    df["volatility_20d"] = df.groupby("symbol")[
                        "daily_return"
                    ].transform(
                        lambda x: x.rolling(window=20, min_periods=1).std()
                        * (252**0.5)
                    )
                    df["volatility_60d"] = df.groupby("symbol")[
                        "daily_return"
                    ].transform(
                        lambda x: x.rolling(window=60, min_periods=1).std()
                        * (252**0.5)
                    )

                    # Add year_month for partitioning
                    df["year_month"] = df["date"].dt.year * 100 + df["date"].dt.month

                    # Convert float to decimal for database
                    numeric_cols = ["open", "high", "low", "close"]
                    for col in numeric_cols:
                        df[col] = df[col].round(2)

                    # Select columns in correct order
                    columns = [
                        "symbol",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "daily_return",
                        "volatility_20d",
                        "volatility_60d",
                        "year_month",
                    ]
                    df = df[columns]

                    # Insert into target
                    target_conn = duckdb.connect(self.target_db)
                    target_conn.execute(
                        "INSERT INTO market.price_data SELECT * FROM df"
                    )
                    target_conn.close()

                    logger.info(f"  ✓ Migrated {len(df)} market data rows")

            source_conn.close()

        except Exception as e:
            logger.error(f"  Error migrating market data: {e}")

    def _migrate_options_data(self):
        """Migrate options data from all sources"""
        logger.info("📈 Migrating options data...")

        sources = [
            ("data/wheel_trading_optimized.duckdb", ["options_data", "active_options"]),
            (
                "data/cache/data/wheel_trading_optimized.duckdb",
                ["options_data", "option_chains"],
            ),
            (
                str(Path.home() / "data/wheel_trading_optimized.duckdb"),
                [
                    "unity_options_daily",
                    "unity_options_comprehensive",
                    "unity_options_highperf",
                ],
            ),
        ]

        total_rows = 0

        for source_db, table_names in sources:
            if not Path(source_db).exists():
                logger.info(f"  Skipping {source_db} (not found)")
                continue

            logger.info(f"  Processing {Path(source_db).name}...")

            try:
                source_conn = duckdb.connect(source_db, read_only=True)

                # Get available tables
                tables = source_conn.execute(
                    """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'main'
                """
                ).fetchall()

                available_tables = [t[0] for t in tables]

                for table in table_names:
                    if table in available_tables:
                        count = source_conn.execute(
                            f"SELECT COUNT(*) FROM {table}"
                        ).fetchone()[0]
                        if count > 0:
                            logger.info(f"    Reading {count} rows from {table}...")

                            # Read in chunks to handle large tables
                            chunk_size = 50000
                            for offset in range(0, count, chunk_size):
                                df = source_conn.execute(
                                    f"""
                                    SELECT * FROM {table} 
                                    LIMIT {chunk_size} OFFSET {offset}
                                """
                                ).df()

                                # Process the data
                                df = self._process_options_data(df)

                                if len(df) > 0:
                                    # Insert into target
                                    target_conn = duckdb.connect(self.target_db)
                                    target_conn.execute(
                                        "INSERT INTO options.contracts SELECT * FROM df"
                                    )
                                    target_conn.close()
                                    total_rows += len(df)

                                logger.info(
                                    f"      Processed {min(offset + chunk_size, count)}/{count} rows"
                                )

                source_conn.close()

            except Exception as e:
                logger.error(f"  Error processing {source_db}: {e}")

        logger.info(f"  ✓ Migrated {total_rows} total options rows")

    def _process_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process options data to match target schema"""
        df = df.copy()

        # Ensure required columns exist
        required_cols = {
            "symbol": None,
            "expiration": None,
            "strike": None,
            "option_type": "PUT",  # Default if missing
            "timestamp": datetime.now(),
            "bid": None,
            "ask": None,
            "mid": None,
            "volume": 0,
            "open_interest": 0,
            "implied_volatility": None,
            "delta": None,
            "gamma": None,
            "theta": None,
            "vega": None,
            "rho": None,
            "moneyness": 1.0,
            "days_to_expiry": None,
            "year_month": None,
        }

        # Add missing columns with defaults
        for col, default in required_cols.items():
            if col not in df.columns:
                df[col] = default

        # Calculate derived fields
        if "bid" in df.columns and "ask" in df.columns and "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2

        if "expiration" in df.columns and "timestamp" in df.columns:
            df["expiration"] = pd.to_datetime(df["expiration"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["days_to_expiry"] = (df["expiration"] - df["timestamp"]).dt.days

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["year_month"] = df["timestamp"].dt.year * 100 + df["timestamp"].dt.month

        # Filter to valid data
        df = df[(df["strike"] > 0) & (df["bid"] >= 0) & (df["ask"] >= 0)]

        # Select only required columns in correct order
        return df[list(required_cols.keys())]

    def _create_materialized_views(self):
        """Create materialized views"""
        logger.info("👁️  Creating materialized views...")

        conn = duckdb.connect(self.target_db)

        # Drop existing materialized view if exists
        conn.execute("DROP TABLE IF EXISTS analytics.wheel_opportunities_mv")

        # Create as table (DuckDB's equivalent of materialized view)
        conn.execute(
            """
            CREATE TABLE analytics.wheel_opportunities_mv AS
            WITH latest_options AS (
                SELECT 
                    symbol, expiration, strike, option_type,
                    MAX(timestamp) as latest_timestamp
                FROM options.contracts
                WHERE option_type = 'PUT'
                GROUP BY symbol, expiration, strike, option_type
            )
            SELECT 
                o.symbol,
                o.expiration,
                o.strike,
                o.bid,
                o.ask,
                o.delta,
                o.implied_volatility,
                o.volume,
                o.days_to_expiry,
                o.moneyness,
                CASE 
                    WHEN o.strike > 0 THEN (o.bid * 100) / (o.strike * 100)
                    ELSE 0
                END as premium_yield,
                CASE 
                    WHEN o.ask > 0 THEN (o.ask - o.bid) / o.ask
                    ELSE 1.0
                END as spread_pct,
                o.timestamp
            FROM options.contracts o
            JOIN latest_options l ON 
                o.symbol = l.symbol AND
                o.expiration = l.expiration AND
                o.strike = l.strike AND
                o.option_type = l.option_type AND
                o.timestamp = l.latest_timestamp
            WHERE 
                o.option_type = 'PUT'
                AND o.delta BETWEEN -0.35 AND -0.25
                AND o.days_to_expiry BETWEEN 20 AND 45
                AND o.bid > 0 
                AND o.ask > 0
                AND (o.ask - o.bid) / o.ask < 0.10
        """
        )

        # Create index
        conn.execute(
            """
            CREATE INDEX idx_wheel_mv_symbol ON analytics.wheel_opportunities_mv(symbol)
        """
        )

        count = conn.execute(
            "SELECT COUNT(*) FROM analytics.wheel_opportunities_mv"
        ).fetchone()[0]
        logger.info(f"  ✓ Created wheel_opportunities_mv with {count} rows")

        conn.close()

    def show_summary(self):
        """Show migration summary"""
        logger.info("\n📊 Migration Summary:")

        conn = duckdb.connect(self.target_db)

        tables = [
            ("market.price_data", "Market Data"),
            ("options.contracts", "Options Contracts"),
            ("analytics.wheel_opportunities_mv", "Wheel Opportunities"),
        ]

        for table, name in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                logger.info(f"  {name}: {count:,} rows")

                if count > 0:
                    # Get date range
                    if (
                        "timestamp"
                        in conn.execute(f"SELECT * FROM {table} LIMIT 0").description
                    ):
                        date_col = "timestamp"
                    elif (
                        "date"
                        in conn.execute(f"SELECT * FROM {table} LIMIT 0").description
                    ):
                        date_col = "date"
                    else:
                        continue

                    min_date, max_date = conn.execute(
                        f"""
                        SELECT MIN({date_col}), MAX({date_col}) FROM {table}
                    """
                    ).fetchone()

                    logger.info(f"    Date range: {min_date} to {max_date}")

            except Exception as e:
                logger.error(f"  Error reading {table}: {e}")

        conn.close()


def main():
    """Run migration"""
    target_db = "data/wheel_trading_optimized.duckdb"

    if not Path(target_db).exists():
        logger.error(f"Target database not found: {target_db}")
        logger.info("Please run create_optimized_schema_simple.py first")
        return

    migrator = SimpleMigrator(target_db)
    migrator.migrate_from_existing_dbs()
    migrator.show_summary()


if __name__ == "__main__":
    main()
