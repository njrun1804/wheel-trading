#!/usr/bin/env python3
"""Fix returns calculation and volatility by updating data types"""

import sys
from pathlib import Path

import duckdb


def fix_returns_and_volatility():
    """Fix the returns calculation by updating data types."""

    db_path = Path("data/unified_wheel_trading.duckdb")
    conn = duckdb.connect(str(db_path))

    try:
        print("=== Fixing Returns and Volatility Calculations ===\n")

        # 1. Fix the data type for returns column
        print("1. Updating returns column data type...")
        # Create a new column with proper type
        conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS returns_new DOUBLE")

        # Copy data from old column (though it's all zeros)
        conn.execute("UPDATE market_data SET returns_new = returns")

        # Drop old column if we can, otherwise just use the new one
        try:
            conn.execute("ALTER TABLE market_data DROP COLUMN returns")
            conn.execute("ALTER TABLE market_data RENAME COLUMN returns_new TO returns")
        except:
            # If we can't drop, just use returns_new
            print("  Using returns_new column due to index dependencies")

        # Check which column we're using
        returns_col = "returns"
        try:
            conn.execute("SELECT returns FROM market_data LIMIT 1")
        except:
            returns_col = "returns_new"

        # 2. Recalculate returns properly
        print(f"2. Recalculating returns (using column: {returns_col})...")
        conn.execute(
            f"""
            UPDATE market_data m1
            SET {returns_col} = (
                SELECT
                    CASE
                        WHEN prev_close IS NOT NULL AND prev_close > 0
                        THEN (m1.close - prev_close) / prev_close
                        ELSE NULL
                    END
                FROM (
                    SELECT
                        symbol,
                        date,
                        close,
                        LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close
                    FROM market_data
                    WHERE close IS NOT NULL AND close > 0
                ) prev
                WHERE prev.symbol = m1.symbol AND prev.date = m1.date
            )
            WHERE m1.close IS NOT NULL AND m1.close > 0
        """
        )

        # Check the results
        updated = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(returns) as with_returns,
                COUNT(CASE WHEN returns != 0 THEN 1 END) as non_zero,
                MIN(returns) as min_return,
                MAX(returns) as max_return,
                AVG(returns) as avg_return,
                STDDEV(returns) as std_return
            FROM market_data
            WHERE symbol = 'U' AND data_type = 'stock'
        """
        ).fetchone()

        total, with_ret, non_zero, min_ret, max_ret, avg_ret, std_ret = updated
        print("\n  Returns updated:")
        print(f"  Total rows: {total}")
        print(f"  Rows with returns: {with_ret} ({with_ret/total*100:.1f}%)")
        print(f"  Non-zero returns: {non_zero} ({non_zero/total*100:.1f}%)")
        print(f"  Return range: {min_ret:.4f} to {max_ret:.4f}")
        print(f"  Daily std dev: {std_ret:.4f}")
        print(f"  Annualized volatility: {std_ret * (252**0.5):.2%}")

        # 3. Recreate the stock_price_history view
        print("\n3. Recreating stock_price_history view...")
        conn.execute(
            """
            CREATE OR REPLACE VIEW stock_price_history AS
            SELECT
                symbol,
                date,
                open,
                high,
                low,
                close,
                volume,
                returns,
                -- Rolling calculations for risk metrics
                AVG(returns) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                ) as avg_return_250d,
                STDDEV(returns) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                ) * SQRT(252) as volatility_250d,
                STDDEV(returns) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) * SQRT(252) as volatility_20d
            FROM market_data
            WHERE data_type = 'stock'
            AND close IS NOT NULL
            ORDER BY symbol, date
        """
        )

        # 4. Recreate backtest_features with proper calculations
        print("4. Recreating backtest_features table...")
        conn.execute("DROP TABLE IF EXISTS backtest_features")

        conn.execute(
            """
            CREATE TABLE backtest_features AS
            SELECT
                s.date,
                s.symbol,
                s.close as stock_price,
                s.returns,
                s.volume,
                s.volatility_20d,
                s.volatility_250d,
                -- VaR calculation (5th percentile of returns)
                QUANTILE(s.returns, 0.05) OVER (
                    PARTITION BY s.symbol
                    ORDER BY s.date
                    ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                ) as var_95,
                -- Risk-free rate from FRED data
                COALESCE(ei.value / 100.0, 0.05) as risk_free_rate,  -- Convert percentage to decimal
                -- VIX if available
                vix.value as vix
            FROM stock_price_history s
            LEFT JOIN economic_indicators ei
                ON s.date = ei.date
                AND ei.indicator = 'DGS3MO'
            LEFT JOIN economic_indicators vix
                ON s.date = vix.date
                AND vix.indicator = 'VIXCLS'
            WHERE s.symbol = 'U'  -- Focus on Unity for now
        """
        )

        # 5. Verify the fix
        print("\n5. Verifying volatility calculations...")
        verify = conn.execute(
            """
            SELECT
                COUNT(*) as total_rows,
                COUNT(CASE WHEN volatility_20d > 0 THEN 1 END) as rows_with_vol20,
                MIN(volatility_20d) as min_vol20,
                MAX(volatility_20d) as max_vol20,
                AVG(volatility_20d) as avg_vol20,
                MIN(volatility_250d) as min_vol250,
                MAX(volatility_250d) as max_vol250,
                AVG(volatility_250d) as avg_vol250
            FROM backtest_features
            WHERE date >= '2023-01-01'  -- Recent data only
        """
        ).fetchone()

        total, with_vol, min20, max20, avg20, min250, max250, avg250 = verify
        print(f"  Total rows: {total}")
        print(f"  Rows with 20d volatility > 0: {with_vol} ({with_vol/total*100:.1f}%)")
        print(f"  20d volatility range: {min20:.1%} to {max20:.1%}")
        print(f"  Average 20d volatility: {avg20:.1%}")
        print(f"  250d volatility range: {min250:.1%} to {max250:.1%}")
        print(f"  Average 250d volatility: {avg250:.1%}")

        # 6. Check VaR calculations
        print("\n6. Verifying VaR calculations...")
        var_check = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(var_95) as with_var,
                MIN(var_95) as worst_var,
                MAX(var_95) as best_var,
                AVG(var_95) as avg_var
            FROM backtest_features
            WHERE date >= '2023-01-01'
        """
        ).fetchone()

        total, with_var, worst, best, avg_var = var_check
        print(f"  Rows with VaR: {with_var} ({with_var/total*100:.1f}%)")
        print(f"  VaR range: {worst:.2%} to {best:.2%}")
        print(f"  Average VaR (95%): {avg_var:.2%}")

        # 7. Sample data to verify
        print("\n7. Sample data verification:")
        samples = conn.execute(
            """
            SELECT
                date,
                stock_price,
                returns,
                volatility_20d,
                volatility_250d,
                var_95,
                risk_free_rate
            FROM backtest_features
            WHERE date >= '2025-06-01'
            ORDER BY date DESC
            LIMIT 5
        """
        ).fetchall()

        print("  Date       | Price  | Return  | Vol20 | Vol250 | VaR95  | RF Rate")
        print("  -----------|--------|---------|-------|--------|--------|--------")
        for date, price, ret, vol20, vol250, var95, rf in samples:
            ret_str = f"{ret:7.4f}" if ret else "   N/A "
            vol20_str = f"{vol20:6.1%}" if vol20 else "  N/A "
            vol250_str = f"{vol250:6.1%}" if vol250 else "  N/A  "
            var_str = f"{var95:6.2%}" if var95 else "  N/A "
            print(
                f"  {date} | ${price:6.2f} | {ret_str} | {vol20_str} | {vol250_str} | {var_str} | {rf:6.2%}"
            )

        # 8. Update indexes
        print("\n8. Updating indexes...")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backtest_features_date ON backtest_features(date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backtest_features_symbol_date ON backtest_features(symbol, date)"
        )

        # 9. Vacuum and analyze
        print("\n9. Optimizing database...")
        conn.execute("VACUUM")
        conn.execute("ANALYZE")

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
