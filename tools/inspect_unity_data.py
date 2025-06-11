#!/usr/bin/env python3
"""
Inspect the structure of Unity options data from Databento.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient

# Initialize client
client = DatabentoClient()

# Get a small sample of data
print("Getting sample Unity options data...")
data = client.client.timeseries.get_range(
    dataset="OPRA.PILLAR",
    symbols=["U.OPT"],
    stype_in="parent",
    schema="ohlcv-1d",
    start="2025-06-01",
    end="2025-06-06",
    path=None,
)

# Convert to DataFrame
df = data.to_df()
print(f"\nReceived {len(df)} records")
print(f"\nDataFrame columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check the structure
print("\n\nDetailed look at first record:")
if len(df) > 0:
    first_row = df.iloc[0]
    print(f"Index (timestamp): {df.index[0]}")
    for col in df.columns:
        print(f"{col}: {first_row[col]} (type: {type(first_row[col])})")

# Check if symbol is in a different column
print("\n\nChecking for symbol data:")
for col in df.columns:
    if "symbol" in col.lower() or col in ["raw_symbol", "ts_symbol"]:
        print(f"Found symbol column '{col}':")
        print(df[col].head())
        break
