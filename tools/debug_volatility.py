#!/usr/bin/env python3
"""Debug why volatility calculations are returning 0.0%"""

from pathlib import Path

import duckdb

from unity_wheel.config.unified_config import get_config
config = get_config()


db_path = Path("data/unified_wheel_trading.duckdb")
conn = duckdb.connect(str(db_path))

print("=== Debugging Volatility Calculations ===\n")

# 1. Check if returns are calculated
print("1. Checking returns data:")
returns_check = conn.execute(
    """
    SELECT
        COUNT(*) as total_rows,
        COUNT(returns) as rows_with_returns,
        COUNT(CASE WHEN returns != 0 THEN 1 END) as non_zero_returns,
        MIN(returns) as min_return,
        MAX(returns) as max_return,
        AVG(returns) as avg_return,
        STDDEV(returns) as std_return
    FROM market_data
    WHERE symbol = config.trading.symbol AND data_type = 'stock'
"""
).fetchone()

total, with_returns, non_zero, min_ret, max_ret, avg_ret, std_ret = returns_check
print(f"  Total rows: {total}")
print(f"  Rows with returns: {with_returns} ({with_returns/total*100:.1f}%)")
print(f"  Non-zero returns: {non_zero} ({non_zero/total*100:.1f}%)")
print(f"  Return range: {min_ret:.4f} to {max_ret:.4f}")
print(f"  Average return: {avg_ret:.4f}")
print(f"  Std dev of returns: {std_ret:.4f}")

# 2. Sample some actual returns
print("\n2. Sample returns data:")
sample_returns = conn.execute(
    """
    SELECT date, close, returns,
           LAG(close) OVER (ORDER BY date) as prev_close,
           (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) as calc_return
    FROM market_data
    WHERE symbol = config.trading.symbol AND data_type = 'stock'
    AND returns IS NOT NULL
    ORDER BY date DESC
    LIMIT 10
"""
).fetchall()

print("  Date        | Close  | Stored Return | Prev Close | Calc Return")
print("  ------------|--------|---------------|------------|------------")
for date, close, ret, prev_close, calc_ret in sample_returns:
    print(f"  {date} | ${close:>6.2f} | {ret:>13.4f} | ${prev_close:>9.2f} | {calc_ret:>11.4f}")

# 3. Check the backtest_features table
print("\n3. Checking backtest_features volatility:")
vol_check = conn.execute(
    """
    SELECT
        COUNT(*) as total,
        COUNT(volatility_20d) as with_vol20,
        COUNT(CASE WHEN volatility_20d > 0 THEN 1 END) as positive_vol20,
        MIN(volatility_20d) as min_vol,
        MAX(volatility_20d) as max_vol,
        AVG(volatility_20d) as avg_vol
    FROM backtest_features
    WHERE symbol = config.trading.symbol
"""
).fetchone()

total, with_vol, positive_vol, min_vol, max_vol, avg_vol = vol_check
print(f"  Total rows: {total}")
print(f"  Rows with volatility_20d: {with_vol}")
print(f"  Positive volatility: {positive_vol}")
print(f"  Volatility range: {min_vol:.4f} to {max_vol:.4f}")
print(f"  Average volatility: {avg_vol:.4f}")

# 4. Manual volatility calculation
print("\n4. Manual volatility calculation test:")
manual_vol = conn.execute(
    """
    WITH returns_data AS (
        SELECT
            date,
            returns,
            COUNT(returns) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as count_returns,
            STDDEV(returns) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as rolling_std
        FROM market_data
        WHERE symbol = config.trading.symbol
        AND data_type = 'stock'
        AND returns IS NOT NULL
        ORDER BY date DESC
        LIMIT 30
    )
    SELECT
        date,
        returns,
        count_returns,
        rolling_std,
        rolling_std * SQRT(252) as annualized_vol
    FROM returns_data
    WHERE count_returns >= 20
    LIMIT 5
"""
).fetchall()

print("  Date        | Return  | Count | Rolling Std | Annual Vol")
print("  ------------|---------|-------|-------------|------------")
for date, ret, count, std, ann_vol in manual_vol:
    print(f"  {date} | {ret:>7.4f} | {count:>5} | {std:>11.4f} | {ann_vol:>10.2%}")

# 5. Check if it's a data type issue
print("\n5. Checking data types in tables:")
schema_check = conn.execute(
    """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'market_data'
    AND column_name IN ('returns', 'close', 'volatility_20d', 'volatility_250d')
"""
).fetchall()

for col, dtype in schema_check:
    print(f"  {col:<15} {dtype}")

# 6. Direct calculation in backtest_features
print("\n6. Recreating volatility calculation:")
# Drop and recreate the table with proper calculation
conn.execute(
    """
    CREATE OR REPLACE TABLE backtest_features_fixed AS
    WITH price_data AS (
        SELECT
            date,
            symbol,
            close as stock_price,
            returns,
            volume,
            -- Calculate rolling statistics only when we have enough data
            CASE
                WHEN COUNT(returns) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) >= 20
                THEN STDDEV(returns) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * SQRT(252)
                ELSE NULL
            END as volatility_20d,
            CASE
                WHEN COUNT(returns) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) >= 250
                THEN STDDEV(returns) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) * SQRT(252)
                ELSE NULL
            END as volatility_250d
        FROM market_data
        WHERE symbol = config.trading.symbol AND data_type = 'stock'
        AND close IS NOT NULL
    )
    SELECT
        p.*,
        -- VaR calculation
        CASE
            WHEN COUNT(p.returns) OVER (PARTITION BY p.symbol ORDER BY p.date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) >= 250
            THEN QUANTILE(p.returns, 0.05) OVER (PARTITION BY p.symbol ORDER BY p.date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW)
            ELSE NULL
        END as var_95,
        COALESCE(ei.value, 0.05) as risk_free_rate,
        vix.value as vix
    FROM price_data p
    LEFT JOIN economic_indicators ei
        ON p.date = ei.date AND ei.indicator = 'DGS3MO'
    LEFT JOIN economic_indicators vix
        ON p.date = vix.date AND vix.indicator = 'VIXCLS'
"""
)

# Check the fixed table
fixed_check = conn.execute(
    """
    SELECT
        COUNT(*) as total,
        COUNT(CASE WHEN volatility_20d > 0 THEN 1 END) as positive_vol20,
        MIN(volatility_20d) as min_vol,
        MAX(volatility_20d) as max_vol,
        AVG(volatility_20d) as avg_vol
    FROM backtest_features_fixed
    WHERE volatility_20d IS NOT NULL
"""
).fetchone()

total, positive_vol, min_vol, max_vol, avg_vol = fixed_check
print("\n  Fixed table results:")
print(f"  Total rows with volatility: {total}")
print(f"  Positive volatility: {positive_vol}")
print(f"  Volatility range: {min_vol:.2%} to {max_vol:.2%}")
print(f"  Average volatility: {avg_vol:.2%}")

conn.close()
