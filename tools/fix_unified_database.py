#!/usr/bin/env python3
"""
Fix data quality issues in unified_wheel_trading.duckdb.
Ensures all price data is properly populated and adds missing indexes.
"""

import sys
from pathlib import Path

import duckdb


def fix_data_quality():
    """Fix NULL/zero values and data quality issues in unified database."""

    db_path = Path("data/unified_wheel_trading.duckdb")
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return False

    print(f"Connecting to {db_path}")
    conn = duckdb.connect(str(db_path))

    try:
        # 1. Check current data quality
        print("\n=== Current Data Quality ===")
        result = conn.execute(
            """
            SELECT
                COUNT(*) as total_rows,
                COUNT(CASE WHEN close IS NULL OR close = 0 THEN 1 END) as null_close,
                COUNT(CASE WHEN premium IS NULL OR premium = 0 THEN 1 END) as null_premium,
                COUNT(CASE WHEN open IS NULL THEN 1 END) as null_open
            FROM market_data
        """
        ).fetchone()

        total, null_close, null_premium, null_open = result
        print(f"Total rows: {total:,}")
        print(f"NULL/zero close: {null_close:,} ({null_close/total*100:.1f}%)")
        print(f"NULL/zero premium: {null_premium:,} ({null_premium/total*100:.1f}%)")
        print(f"NULL open: {null_open:,}")

        # 2. Fix stock prices where close is NULL but OHLC data exists
        print("\n=== Fixing Stock Prices ===")
        fixed = conn.execute(
            """
            UPDATE market_data
            SET close = (open + high + low) / 3.0
            WHERE close IS NULL
            AND open IS NOT NULL
            AND high IS NOT NULL
            AND low IS NOT NULL
            AND symbol = 'U'
        """
        ).rowcount
        print(f"Fixed {fixed} stock price records using OHLC average")

        # 3. Calculate returns for stock data
        print("\n=== Calculating Returns ===")
        conn.execute(
            """
            -- Add returns column if it doesn't exist
            ALTER TABLE market_data ADD COLUMN IF NOT EXISTS returns DOUBLE;

            -- Calculate daily returns for stock data
            UPDATE market_data m1
            SET returns = (
                SELECT (m1.close - m2.close) / m2.close
                FROM market_data m2
                WHERE m2.symbol = m1.symbol
                AND m2.date < m1.date
                AND m1.symbol = 'U'
                ORDER BY m2.date DESC
                LIMIT 1
            )
            WHERE m1.symbol = 'U' AND m1.close IS NOT NULL;
        """
        )
        print("Calculated returns for stock data")

        # 4. Fix option premium data
        print("\n=== Fixing Option Premiums ===")
        # For options, use mid-point of bid-ask if available
        fixed_options = conn.execute(
            """
            UPDATE market_data
            SET premium = CASE
                WHEN close IS NOT NULL AND close > 0 THEN close
                WHEN bid IS NOT NULL AND ask IS NOT NULL THEN (bid + ask) / 2.0
                ELSE premium
            END
            WHERE symbol LIKE 'U %'  -- Options format
            AND (premium IS NULL OR premium = 0)
        """
        ).rowcount
        print(f"Fixed {fixed_options} option premium records")

        # 5. Add volatility column
        print("\n=== Adding Volatility Calculations ===")
        conn.execute(
            """
            -- Add volatility columns
            ALTER TABLE market_data ADD COLUMN IF NOT EXISTS volatility_20d DOUBLE;
            ALTER TABLE market_data ADD COLUMN IF NOT EXISTS volatility_250d DOUBLE;
        """
        )

        # 6. Create optimized indexes
        print("\n=== Creating Indexes ===")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_symbol_date ON market_data(symbol, date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_market_date ON market_data(date)",
            "CREATE INDEX IF NOT EXISTS idx_options_metadata ON options_metadata(symbol, expiration, strike)",
            "CREATE INDEX IF NOT EXISTS idx_econ_indicator_date ON economic_indicators(indicator, date DESC)",
        ]

        for idx_sql in indexes:
            conn.execute(idx_sql)
            print(f"Created: {idx_sql.split('idx_')[1].split(' ')[0]}")

        # 7. Create Greeks cache table if missing
        print("\n=== Ensuring Greeks Cache Table ===")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS greeks_cache (
                symbol VARCHAR,
                date DATE,
                expiration DATE,
                strike DOUBLE,
                option_type CHAR(1),
                spot_price DOUBLE,
                risk_free_rate DOUBLE,
                delta DOUBLE,
                gamma DOUBLE,
                theta DOUBLE,
                vega DOUBLE,
                rho DOUBLE,
                implied_volatility DOUBLE,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, expiration, strike, option_type)
            )
        """
        )

        # 8. Update views to handle NULLs better
        print("\n=== Updating Views ===")
        conn.execute(
            """
            CREATE OR REPLACE VIEW unity_volatility AS
            SELECT
                date,
                symbol,
                close,
                returns,
                STDDEV(returns) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) * SQRT(252) as volatility_20d,
                STDDEV(returns) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                ) * SQRT(252) as volatility_250d
            FROM market_data
            WHERE symbol = 'U'
            AND close IS NOT NULL
            AND returns IS NOT NULL
        """
        )

        # 9. Final data quality check
        print("\n=== Final Data Quality Check ===")
        result = conn.execute(
            """
            SELECT
                COUNT(*) as total_rows,
                COUNT(CASE WHEN close IS NULL OR close = 0 THEN 1 END) as null_close,
                COUNT(CASE WHEN returns IS NULL THEN 1 END) as null_returns
            FROM market_data
            WHERE symbol = 'U'
        """
        ).fetchone()

        total, null_close, null_returns = result
        print(f"Unity stock records: {total:,}")
        print(
            f"Records with valid close: {total - null_close:,} ({(total-null_close)/total*100:.1f}%)"
        )
        print(f"Records with returns: {total - null_returns:,}")

        # 10. Vacuum to optimize
        print("\n=== Optimizing Database ===")
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
        print("Database optimized")

        conn.close()
        return True

    except Exception as e:
        print(f"\nError: {e}")
        conn.close()
        return False


if __name__ == "__main__":
    success = fix_data_quality()
    sys.exit(0 if success else 1)
