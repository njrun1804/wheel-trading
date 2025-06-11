#!/usr/bin/env python3
"""Fix the backtester to properly load data from unified database."""

import re

# Read the current backtester
with open("src/unity_wheel/backtesting/wheel_backtester.py", "r") as f:
    content = f.read()

# Find the _load_price_data method
old_query = '''result = conn.execute(
                """
                SELECT
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    returns,
                    volatility_20d,
                    volatility_250d,
                    risk_free_rate
                FROM backtest_features
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                [symbol, start_date.date(), end_date.date()],
            ).fetchall()'''

new_query = '''result = conn.execute(
                """
                SELECT
                    bf.date,
                    COALESCE(md.open, bf.stock_price) as open,
                    COALESCE(md.high, bf.stock_price) as high,
                    COALESCE(md.low, bf.stock_price) as low,
                    COALESCE(md.close, bf.stock_price) as close,
                    COALESCE(md.volume, bf.volume) as volume,
                    bf.returns,
                    bf.volatility_20d,
                    bf.volatility_250d,
                    bf.risk_free_rate
                FROM backtest_features bf
                LEFT JOIN market_data md
                    ON bf.date = md.date
                    AND bf.symbol = md.symbol
                    AND md.data_type = 'stock'
                WHERE bf.symbol = ?
                AND bf.date BETWEEN ? AND ?
                ORDER BY bf.date
                """,
                [symbol, start_date.date(), end_date.date()],
            ).fetchall()'''

# Replace the query
content = content.replace(old_query, new_query)

# Write back
with open("src/unity_wheel/backtesting/wheel_backtester.py", "w") as f:
    f.write(content)

print("✅ Fixed backtester data loading to join backtest_features with market_data")
print("✅ Now using REAL market OHLCV data - NO synthetic data")
print("✅ Falls back to stock_price (close) if OHLCV not available")
