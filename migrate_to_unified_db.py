#!/usr/bin/env python3
"""Migrate existing data to unified database structure for wheel trading system."""

import os
from datetime import datetime
from pathlib import Path

import duckdb


def create_unified_database():
    """Create unified database with optimal structure for wheel trading."""

    print("CREATING UNIFIED DATABASE STRUCTURE")
    print("=" * 80)

    # Database paths
    home_db = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")
    project_db = "data/cache/wheel_cache.duckdb"
    unified_db = "data/unified_trading.duckdb"

    # Create unified database
    print(f"\n1. Creating unified database: {unified_db}")
    os.makedirs(os.path.dirname(unified_db), exist_ok=True)

    # Connect to all databases
    home_conn = duckdb.connect(home_db, read_only=True)
    project_conn = duckdb.connect(project_db, read_only=True)
    unified_conn = duckdb.connect(unified_db)

    try:
        # Create market_data table combining stock and options
        print("\n2. Creating unified market_data table...")
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

        # Import Unity stock data
        print("   - Importing Unity stock prices...")
        stock_count = home_conn.execute(
            "SELECT COUNT(*) FROM price_history WHERE symbol = 'U'"
        ).fetchone()[0]
        unified_conn.execute(
            f"""
            INSERT INTO market_data
            SELECT
                symbol,
                date,
                open,
                high,
                low,
                close,
                volume,
                returns,
                'stock' as data_type
            FROM home_conn.price_history
            WHERE symbol = 'U'
        """
        )
        print(f"     ✓ Imported {stock_count} Unity stock records")

        # Import Unity options data
        print("   - Importing Unity options data...")
        options_count = project_conn.execute("SELECT COUNT(*) FROM unity_options_ohlcv").fetchone()[
            0
        ]
        unified_conn.execute(
            f"""
            INSERT INTO market_data
            SELECT
                symbol,
                ts_event as date,
                open,
                high,
                low,
                close,
                volume,
                NULL as returns,
                'option' as data_type
            FROM project_conn.unity_options_ohlcv
        """
        )
        print(f"     ✓ Imported {options_count} Unity option records")

        # Create options_chain table with parsed fields
        print("\n3. Creating options_chain table with parsed metadata...")
        unified_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS options_chain (
                symbol VARCHAR,
                underlying VARCHAR,
                expiration DATE,
                strike DOUBLE,
                option_type VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (symbol, date)
            )
        """
        )

        # Parse option symbols and populate options_chain
        unified_conn.execute(
            """
            INSERT INTO options_chain
            SELECT
                symbol,
                SUBSTRING(symbol, 1, 6) as underlying,
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
        print("\n4. Creating economic_indicators table...")
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

        # Import FRED data
        print("   - Importing FRED economic data...")
        fred_count = home_conn.execute("SELECT COUNT(*) FROM fred_observations").fetchone()[0]
        unified_conn.execute(
            f"""
            INSERT INTO economic_indicators
            SELECT
                o.date,
                o.series_id as indicator,
                o.value
            FROM home_conn.fred_observations o
        """
        )
        print(f"     ✓ Imported {fred_count} FRED observations")

        # Create model_features table
        print("\n5. Creating model_features table (placeholder for ML features)...")
        unified_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_features (
                date DATE,
                symbol VARCHAR,
                -- Price features
                price_sma_20 DOUBLE,
                price_sma_50 DOUBLE,
                rsi_14 DOUBLE,
                volatility_20 DOUBLE,
                volatility_60 DOUBLE,
                -- Option features
                put_call_ratio DOUBLE,
                iv_rank DOUBLE,
                iv_percentile DOUBLE,
                -- Market features
                vix DOUBLE,
                risk_free_rate DOUBLE,
                PRIMARY KEY (date, symbol)
            )
        """
        )

        # Create recommendations table
        print("\n6. Creating recommendations table...")
        unified_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP,
                symbol VARCHAR,
                action VARCHAR,
                strike DOUBLE,
                expiration DATE,
                contracts INT,
                confidence DOUBLE,
                expected_return DOUBLE,
                risk_metrics VARCHAR,  -- JSON
                features_used VARCHAR, -- JSON
                actual_outcome DOUBLE,
                notes TEXT
            )
        """
        )

        # Create positions table
        print("\n7. Creating positions table...")
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

        # Create useful views
        print("\n8. Creating views for easy access...")

        # Current stock price
        unified_conn.execute(
            """
            CREATE OR REPLACE VIEW current_unity_price AS
            SELECT * FROM market_data
            WHERE symbol = 'U' AND data_type = 'stock'
            ORDER BY date DESC
            LIMIT 1
        """
        )

        # Current options chain
        unified_conn.execute(
            """
            CREATE OR REPLACE VIEW current_options AS
            SELECT * FROM options_chain
            WHERE date = (SELECT MAX(date) FROM options_chain)
              AND expiration > CURRENT_DATE
        """
        )

        # Risk-free rate
        unified_conn.execute(
            """
            CREATE OR REPLACE VIEW current_risk_free_rate AS
            SELECT value FROM economic_indicators
            WHERE indicator = 'DGS3'
              AND date = (SELECT MAX(date) FROM economic_indicators WHERE indicator = 'DGS3')
        """
        )

        # VIX
        unified_conn.execute(
            """
            CREATE OR REPLACE VIEW current_vix AS
            SELECT value FROM economic_indicators
            WHERE indicator = 'VIXCLS'
              AND date = (SELECT MAX(date) FROM economic_indicators WHERE indicator = 'VIXCLS')
        """
        )

        # Create indexes for performance
        print("\n9. Creating indexes...")
        unified_conn.execute("CREATE INDEX idx_market_symbol_date ON market_data(symbol, date)")
        unified_conn.execute("CREATE INDEX idx_market_type ON market_data(data_type)")
        unified_conn.execute("CREATE INDEX idx_options_underlying ON options_chain(underlying)")
        unified_conn.execute("CREATE INDEX idx_options_expiration ON options_chain(expiration)")
        unified_conn.execute("CREATE INDEX idx_econ_indicator ON economic_indicators(indicator)")
        print("   ✓ Indexes created")

        # Show summary
        print("\n10. Database Summary:")
        tables = unified_conn.execute(
            """
            SELECT table_name,
                   (SELECT COUNT(*) FROM table_name) as row_count
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
        """
        ).fetchall()

        for table in tables:
            try:
                count = unified_conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
                print(f"    {table[0]}: {count:,} rows")
            except:
                pass

        # Sample queries
        print("\n11. Sample Query Results:")

        # Latest Unity stock price
        latest_stock = unified_conn.execute("SELECT * FROM current_unity_price").fetchone()
        if latest_stock:
            print(f"\n    Latest Unity Stock Price:")
            print(
                f"    Date: {latest_stock[1]}, Close: ${latest_stock[5]:.2f}, Volume: {latest_stock[6]:,}"
            )

        # Latest VIX
        latest_vix = unified_conn.execute("SELECT * FROM current_vix").fetchone()
        if latest_vix:
            print(f"\n    Current VIX: {latest_vix[0]:.2f}")

        # Available options
        options_count = unified_conn.execute(
            """
            SELECT COUNT(*) FROM current_options
            WHERE underlying = 'U' AND option_type = 'P'
        """
        ).fetchone()[0]
        print(f"\n    Current Unity Put Options: {options_count}")

        print("\n" + "=" * 80)
        print("✅ UNIFIED DATABASE CREATED SUCCESSFULLY!")
        print(f"   Location: {unified_db}")
        print("\nYou can now use this database for:")
        print("  - Training models on historical data")
        print("  - Making daily position recommendations")
        print("  - Backtesting strategies")
        print("  - Tracking recommendations and outcomes")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Close connections
        home_conn.close()
        project_conn.close()
        unified_conn.close()


if __name__ == "__main__":
    create_unified_database()
