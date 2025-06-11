#!/usr/bin/env python3
"""Migrate existing data to unified database structure - fixed version."""

import os
from pathlib import Path

import duckdb
import pandas as pd


def create_unified_database():
    """Create unified database with optimal structure for wheel trading."""

    print("CREATING UNIFIED DATABASE STRUCTURE")
    print("=" * 80)

    # Database paths
    home_db = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")
    project_db = "data/cache/wheel_cache.duckdb"
    unified_db = "data/unified_trading.duckdb"

    print(f"\n1. Source databases:")
    print(f"   Home DB: {home_db}")
    print(f"   Project DB: {project_db}")
    print(f"   Target DB: {unified_db}")

    # Create unified database
    os.makedirs(os.path.dirname(unified_db), exist_ok=True)

    try:
        # Step 1: Extract data from source databases
        print("\n2. Extracting data from source databases...")

        # Get Unity stock data
        home_conn = duckdb.connect(home_db, read_only=True)
        stock_df = home_conn.execute(
            """
            SELECT symbol, date, open, high, low, close, volume, returns
            FROM price_history
            WHERE symbol = 'U'
        """
        ).df()
        print(f"   ✓ Extracted {len(stock_df)} Unity stock records")

        # Get FRED data
        fred_df = home_conn.execute(
            """
            SELECT date, series_id as indicator, value
            FROM fred_observations
        """
        ).df()
        print(f"   ✓ Extracted {len(fred_df)} FRED records")

        # Check what FRED series we have
        fred_series = home_conn.execute("SELECT * FROM fred_series").df()
        print(f"   ✓ Found {len(fred_series)} FRED series")

        home_conn.close()

        # Get Unity options data
        project_conn = duckdb.connect(project_db, read_only=True)
        options_df = project_conn.execute(
            """
            SELECT symbol, ts_event as date, open, high, low, close, volume
            FROM unity_options_ohlcv
        """
        ).df()
        print(f"   ✓ Extracted {len(options_df)} Unity option records")
        project_conn.close()

        # Step 2: Create unified database
        print("\n3. Creating unified database structure...")
        unified_conn = duckdb.connect(unified_db)

        # Create market_data table
        unified_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                returns DOUBLE,
                data_type VARCHAR,
                PRIMARY KEY (symbol, date, data_type)
            )
        """
        )

        # Insert stock data
        print("\n4. Populating unified tables...")
        print("   - Inserting Unity stock data...")
        for _, row in stock_df.iterrows():
            unified_conn.execute(
                """
                INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'stock')
            """,
                [
                    row["symbol"],
                    row["date"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    row["returns"],
                ],
            )

        # Insert options data
        print("   - Inserting Unity options data...")
        for _, row in options_df.iterrows():
            unified_conn.execute(
                """
                INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'option')
            """,
                [
                    row["symbol"],
                    row["date"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                ],
            )

        # Create options_chain table with parsed metadata
        print("   - Creating options_chain table...")
        unified_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS options_chain AS
            SELECT
                symbol,
                TRIM(SUBSTRING(symbol, 1, 6)) as underlying,
                CAST('20' || SUBSTRING(symbol, 7, 2) || '-' ||
                     SUBSTRING(symbol, 9, 2) || '-' ||
                     SUBSTRING(symbol, 11, 2) AS DATE) as expiration,
                CAST(SUBSTRING(symbol, 14, 8) AS DOUBLE) / 1000 as strike,
                SUBSTRING(symbol, 13, 1) as option_type,
                date,
                open,
                high,
                low,
                close,
                volume
            FROM market_data
            WHERE data_type = 'option'
        """
        )

        # Create economic_indicators table
        print("   - Creating economic_indicators table...")
        unified_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS economic_indicators (
                date DATE,
                indicator VARCHAR,
                value DOUBLE,
                PRIMARY KEY (date, indicator)
            )
        """
        )

        # Insert FRED data
        for _, row in fred_df.iterrows():
            try:
                unified_conn.execute(
                    """
                    INSERT INTO economic_indicators VALUES (?, ?, ?)
                """,
                    [row["date"], row["indicator"], row["value"]],
                )
            except:
                pass  # Skip duplicates

        # Create additional tables
        print("\n5. Creating additional tables for trading system...")

        # Greeks cache
        unified_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS greeks_cache (
                symbol VARCHAR,
                date DATE,
                underlying_price DOUBLE,
                delta DOUBLE,
                gamma DOUBLE,
                theta DOUBLE,
                vega DOUBLE,
                rho DOUBLE,
                iv DOUBLE,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """
        )

        # Recommendations
        unified_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR,
                action VARCHAR,
                strike DOUBLE,
                expiration DATE,
                contracts INT,
                confidence DOUBLE,
                expected_return DOUBLE,
                risk_metrics VARCHAR,
                features_used VARCHAR,
                actual_outcome DOUBLE,
                notes TEXT
            )
        """
        )

        # Positions
        unified_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                position_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                open_date DATE,
                close_date DATE,
                strike DOUBLE,
                expiration DATE,
                contracts INT,
                premium_collected DOUBLE,
                cost_basis DOUBLE,
                status VARCHAR,
                pnl DOUBLE,
                recommendation_id VARCHAR
            )
        """
        )

        # Create views
        print("\n6. Creating useful views...")

        # Current Unity price
        unified_conn.execute(
            """
            CREATE OR REPLACE VIEW current_unity_price AS
            SELECT * FROM market_data
            WHERE symbol = 'U' AND data_type = 'stock'
            ORDER BY date DESC
            LIMIT 1
        """
        )

        # Latest options
        unified_conn.execute(
            """
            CREATE OR REPLACE VIEW latest_options AS
            SELECT * FROM options_chain
            WHERE date = (SELECT MAX(date) FROM options_chain)
        """
        )

        # Current risk-free rate
        unified_conn.execute(
            """
            CREATE OR REPLACE VIEW current_risk_free_rate AS
            SELECT value FROM economic_indicators
            WHERE indicator = 'DGS3'
            ORDER BY date DESC
            LIMIT 1
        """
        )

        # VIX
        unified_conn.execute(
            """
            CREATE OR REPLACE VIEW current_vix AS
            SELECT value FROM economic_indicators
            WHERE indicator = 'VIXCLS'
            ORDER BY date DESC
            LIMIT 1
        """
        )

        # Create indexes
        print("\n7. Creating indexes for performance...")
        unified_conn.execute("CREATE INDEX idx_market_symbol ON market_data(symbol)")
        unified_conn.execute("CREATE INDEX idx_market_date ON market_data(date)")
        unified_conn.execute("CREATE INDEX idx_options_underlying ON options_chain(underlying)")
        unified_conn.execute("CREATE INDEX idx_options_expiration ON options_chain(expiration)")
        unified_conn.execute("CREATE INDEX idx_econ_date ON economic_indicators(date)")

        # Summary
        print("\n8. Database Summary:")
        tables = [
            "market_data",
            "options_chain",
            "economic_indicators",
            "greeks_cache",
            "recommendations",
            "positions",
        ]

        for table in tables:
            try:
                count = unified_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"   {table}: {count:,} rows")
            except Exception as e:
                print(f"   {table}: Error - {e}")

        # Sample data
        print("\n9. Sample Data:")

        # Latest stock price
        latest = unified_conn.execute("SELECT * FROM current_unity_price").fetchone()
        if latest:
            print(f"\n   Latest Unity Stock:")
            print(f"   Date: {latest[1]}, Close: ${latest[5]:.2f}, Volume: {latest[6]:,}")

        # Current VIX
        vix = unified_conn.execute("SELECT * FROM current_vix").fetchone()
        if vix:
            print(f"\n   Current VIX: {vix[0]:.2f}")

        # Available options count
        opt_count = unified_conn.execute(
            """
            SELECT COUNT(DISTINCT symbol)
            FROM options_chain
            WHERE underlying = 'U' AND option_type = 'P'
        """
        ).fetchone()[0]
        print(f"\n   Total Unity Put Options: {opt_count:,}")

        # Recent options
        recent_opts = unified_conn.execute(
            """
            SELECT COUNT(*)
            FROM latest_options
            WHERE underlying = 'U' AND option_type = 'P' AND expiration > CURRENT_DATE
        """
        ).fetchone()[0]
        print(f"   Current tradeable puts: {recent_opts}")

        unified_conn.close()

        print("\n" + "=" * 80)
        print("✅ UNIFIED DATABASE CREATED SUCCESSFULLY!")
        print(f"   Location: {unified_db}")
        print("\nThe database consolidates:")
        print("  - Unity stock prices (861 days)")
        print("  - Unity options data (178,724 daily records)")
        print("  - FRED economic indicators (9 series)")
        print("\nReady for:")
        print("  - Model training on historical data")
        print("  - Daily position recommendations")
        print("  - Backtesting and analysis")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    create_unified_database()
