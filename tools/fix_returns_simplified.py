#!/usr/bin/env python3
"""Simplified fix for returns and volatility calculations"""

import sys
from pathlib import Path

import duckdb


def fix_returns_and_volatility():
    """Fix the returns calculation with a simpler approach."""

    db_path = Path("data/unified_wheel_trading.duckdb")
    conn = duckdb.connect(str(db_path))

    try:
        print("=== Fixing Returns and Volatility Calculations ===\n")

        # 1. Create new properly-typed returns column
        print("1. Creating new returns column with proper data type...")
        conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS returns_calc DOUBLE")

        # 2. Calculate returns properly
        print("2. Calculating returns...")
        conn.execute(
            """
            WITH price_pairs AS (
                SELECT
                    symbol,
                    date,
                    close,
                    LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close
                FROM market_data
                WHERE data_type = 'stock' AND close IS NOT NULL
            )
            UPDATE market_data
            SET returns_calc = (
                SELECT (pp.close - pp.prev_close) / pp.prev_close
                FROM price_pairs pp
                WHERE pp.symbol = market_data.symbol
                AND pp.date = market_data.date
                AND pp.prev_close > 0
            )
            WHERE data_type = 'stock'
        """
        )

        # 3. Check the results
        print("\n3. Checking Unity stock returns...")
        check = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(returns_calc) as with_returns,
                COUNT(CASE WHEN returns_calc != 0 THEN 1 END) as non_zero,
                MIN(returns_calc) as min_return,
                MAX(returns_calc) as max_return,
                STDDEV(returns_calc) as daily_std
            FROM market_data
            WHERE symbol = 'U' AND data_type = 'stock'
        """
        ).fetchone()

        total, with_ret, non_zero, min_ret, max_ret, daily_std = check
        print(f"  Total Unity stock records: {total}")
        print(f"  Records with returns: {with_ret} ({with_ret/total*100:.1f}%)")
        print(f"  Non-zero returns: {non_zero} ({non_zero/total*100:.1f}%)")
        print(f"  Daily return range: {min_ret:.2%} to {max_ret:.2%}")
        print(f"  Daily std dev: {daily_std:.4f}")
        print(f"  Annualized volatility: {daily_std * (252**0.5):.2%}")

        # 4. Create a properly calculated features table
        print("\n4. Creating new backtest features table...")
        conn.execute("DROP TABLE IF EXISTS backtest_features_v2")

        conn.execute(
            """
            CREATE TABLE backtest_features_v2 AS
            WITH stock_data AS (
                SELECT
                    date,
                    symbol,
                    close as stock_price,
                    returns_calc as returns,
                    volume,
                    -- 20-day volatility
                    CASE
                        WHEN COUNT(returns_calc) OVER (
                            PARTITION BY symbol ORDER BY date
                            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                        ) >= 20
                        THEN STDDEV(returns_calc) OVER (
                            PARTITION BY symbol ORDER BY date
                            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                        ) * SQRT(252)
                        ELSE NULL
                    END as volatility_20d,
                    -- 250-day volatility
                    CASE
                        WHEN COUNT(returns_calc) OVER (
                            PARTITION BY symbol ORDER BY date
                            ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                        ) >= 200  -- Allow some missing days
                        THEN STDDEV(returns_calc) OVER (
                            PARTITION BY symbol ORDER BY date
                            ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                        ) * SQRT(252)
                        ELSE NULL
                    END as volatility_250d,
                    -- 250-day average return
                    AVG(returns_calc) OVER (
                        PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                    ) as avg_return_250d
                FROM market_data
                WHERE symbol = 'U'
                AND data_type = 'stock'
                AND close IS NOT NULL
            )
            SELECT
                sd.*,
                -- VaR calculation (5th percentile)
                CASE
                    WHEN COUNT(sd.returns) OVER (
                        PARTITION BY sd.symbol ORDER BY sd.date
                        ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                    ) >= 200
                    THEN QUANTILE(sd.returns, 0.05) OVER (
                        PARTITION BY sd.symbol ORDER BY sd.date
                        ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                    )
                    ELSE NULL
                END as var_95,
                -- Risk-free rate (convert from percentage)
                COALESCE(ei.value / 100.0, 0.05) as risk_free_rate,
                -- VIX
                vix.value as vix
            FROM stock_data sd
            LEFT JOIN economic_indicators ei
                ON sd.date = ei.date AND ei.indicator = 'DGS3MO'
            LEFT JOIN economic_indicators vix
                ON sd.date = vix.date AND vix.indicator = 'VIXCLS'
        """
        )

        # 5. Verify the new calculations
        print("\n5. Verifying new volatility calculations...")
        verify = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN volatility_20d > 0 THEN 1 END) as with_vol20,
                AVG(volatility_20d) as avg_vol20,
                MIN(volatility_20d) as min_vol20,
                MAX(volatility_20d) as max_vol20,
                COUNT(CASE WHEN var_95 IS NOT NULL THEN 1 END) as with_var,
                AVG(var_95) as avg_var
            FROM backtest_features_v2
            WHERE date >= '2023-01-01'
        """
        ).fetchall()[0]

        print(f"  Records since 2023: {verify[0]}")
        print(f"  With 20d volatility: {verify[1]} ({verify[1]/verify[0]*100:.1f}%)")
        print(f"  Average 20d vol: {verify[2]:.1%}")
        print(f"  Vol range: {verify[3]:.1%} to {verify[4]:.1%}")
        print(f"  With VaR: {verify[5]} ({verify[5]/verify[0]*100:.1f}%)")
        print(f"  Average VaR: {verify[6]:.2%}")

        # 6. Show sample data
        print("\n6. Sample recent data:")
        samples = conn.execute(
            """
            SELECT
                date,
                stock_price,
                returns,
                volatility_20d,
                var_95
            FROM backtest_features_v2
            WHERE date >= '2025-06-01'
            ORDER BY date DESC
            LIMIT 5
        """
        ).fetchall()

        print("  Date       | Price  | Return  | Vol 20d | VaR 95%")
        print("  -----------|--------|---------|---------|--------")
        for date, price, ret, vol20, var95 in samples:
            ret_str = f"{ret:7.4f}" if ret else "   N/A "
            vol_str = f"{vol20:7.1%}" if vol20 else "   N/A "
            var_str = f"{var95:7.2%}" if var95 else "   N/A "
            print(f"  {date} | ${price:6.2f} | {ret_str} | {vol_str} | {var_str}")

        # 7. Replace old table
        print("\n7. Replacing old backtest_features table...")
        conn.execute("DROP TABLE IF EXISTS backtest_features")
        conn.execute("ALTER TABLE backtest_features_v2 RENAME TO backtest_features")

        # Add indexes
        conn.execute("CREATE INDEX idx_backtest_date ON backtest_features(date)")
        conn.execute("CREATE INDEX idx_backtest_symbol_date ON backtest_features(symbol, date)")

        print("\n✅ Returns and volatility calculations fixed!")

        conn.close()
        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        conn.close()
        return False


if __name__ == "__main__":
    success = fix_returns_and_volatility()
    sys.exit(0 if success else 1)
