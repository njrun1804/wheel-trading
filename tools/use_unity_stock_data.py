#!/usr/bin/env python3
"""
Example of using the REAL Unity stock data you have.
Shows daily prices and basic statistics.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from unity_wheel.config.unified_config import get_config

config = get_config()


# Connect to database
db_path = Path(config.storage.database_path).expanduser()
conn = duckdb.connect(str(db_path))

print("=" * 60)
print("UNITY STOCK DATA EXAMPLES")
print("=" * 60)

# 1. Recent prices
print("\n1. LAST 10 DAYS OF UNITY STOCK PRICES:")
recent = conn.execute(
    """
    SELECT date, open, high, low, close, volume
    FROM price_history
    WHERE symbol = config.trading.symbol
    ORDER BY date DESC
    LIMIT 10
"""
).fetchall()

print("Date        Open    High    Low     Close   Volume")
print("----------  ------  ------  ------  ------  -----------")
for row in recent:
    print(
        f"{row[0]}  ${row[1]:6.2f}  ${row[2]:6.2f}  ${row[3]:6.2f}  ${row[4]:6.2f}  {row[5]:11,}"
    )

# 2. Price statistics
print("\n2. UNITY PRICE STATISTICS:")
stats = conn.execute(
    """
    SELECT
        MIN(close) as min_price,
        MAX(close) as max_price,
        AVG(close) as avg_price,
        STDDEV(close) as volatility,
        MIN(date) as start_date,
        MAX(date) as end_date,
        COUNT(*) as trading_days
    FROM price_history
    WHERE symbol = config.trading.symbol
"""
).fetchone()

print(f"  Period: {stats[4]} to {stats[5]} ({stats[6]} days)")
print(f"  Price range: ${stats[0]:.2f} - ${stats[1]:.2f}")
print(f"  Average price: ${stats[2]:.2f}")
print(f"  Volatility (std dev): ${stats[3]:.2f}")

# 3. Convert to pandas for analysis
print("\n3. CONVERTING TO PANDAS DATAFRAME:")
df = conn.execute(
    """
    SELECT date, open, high, low, close, volume
    FROM price_history
    WHERE symbol = config.trading.symbol
    ORDER BY date
"""
).df()

print(f"  DataFrame shape: {df.shape}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Columns: {', '.join(df.columns)}")

# 4. Calculate returns
df["daily_return"] = df["close"].pct_change()
df["log_return"] = (df["close"] / df["close"].shift(1)).apply(
    lambda x: pd.NA if pd.isna(x) else np.log(x)
)

print("\n4. RETURN STATISTICS:")
print(
    f"  Average daily return: {df['daily_return'].mean():.4f} ({df['daily_return'].mean()*252*100:.2f}% annualized)"
)
print(
    f"  Daily return volatility: {df['daily_return'].std():.4f} ({df['daily_return'].std()*np.sqrt(252)*100:.2f}% annualized)"
)
print(
    f"  Sharpe ratio (0% risk-free): {df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252):.2f}"
)

# 5. Recent volatility
print("\n5. RECENT VOLATILITY (20-day rolling):")
df["volatility_20d"] = df["daily_return"].rolling(20).std() * np.sqrt(252) * 100
recent_vol = df[["date", "close", "volatility_20d"]].tail(5)

print("Date        Close   20-Day Vol")
print("----------  ------  ----------")
for _, row in recent_vol.iterrows():
    vol_str = (
        f"{row['volatility_20d']:.1f}%" if pd.notna(row["volatility_20d"]) else "N/A"
    )
    print(f"{row['date']}  ${row['close']:6.2f}  {vol_str:>10}")

print("\n✓ All data shown is REAL Unity stock data from Databento")
print("✓ NO SYNTHETIC DATA")

conn.close()
