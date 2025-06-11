#!/usr/bin/env python3
"""Analyze existing options data for liquidity patterns."""

import numpy as np
import pandas as pd

# Load our options data
df = pd.read_parquet("data/unity-options/processed/unity_ohlcv_3y.parquet")

# Parse option symbols to extract strikes and expirations
df["expiry_str"] = df["symbol"].str[6:12]  # YYMMDD format
df["strike_str"] = df["symbol"].str[13:21]  # Strike price
df["strike"] = df["strike_str"].astype(float) / 1000

# Get unique expirations
print("=== OPTION CHAIN COVERAGE ===")
print(f"Total option records: {len(df):,}")
print(f"Unique symbols: {df.symbol.nunique():,}")
print(f"Date range: {df.ts_event.min()} to {df.ts_event.max()}")

# Analyze volume as liquidity proxy
print("\n=== VOLUME ANALYSIS (Liquidity Proxy) ===")
volume_stats = df.groupby("symbol")["volume"].agg(["mean", "median", "max", "sum"])
liquid_options = volume_stats[volume_stats["sum"] > 1000].index
print(f"Options with >1000 total volume: {len(liquid_options):,} / {len(volume_stats):,}")

# Find most liquid strikes for current expirations
print("\n=== MOST LIQUID STRIKES (by volume) ===")
# Convert ts_event to datetime if needed
if df["ts_event"].dtype == "object":
    df["ts_event"] = pd.to_datetime(df["ts_event"])
recent_options = df[df.ts_event > pd.to_datetime("2025-05-01")]
liquid_strikes = (
    recent_options.groupby(["strike", "expiry_str"])
    .agg({"volume": "sum", "symbol": "first"})
    .sort_values("volume", ascending=False)
    .head(20)
)

for (strike, expiry), row in liquid_strikes.iterrows():
    print(f"  Strike ${strike:.0f}, Exp {expiry}, Volume: {row.volume:,}")

# Estimate spreads from high/low
df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
df["hl_spread_pct"] = df["hl_spread"] * 100

print("\n=== SPREAD ESTIMATES (from High-Low) ===")
print(f"Median H-L spread: {df.hl_spread_pct.median():.1f}%")
print(
    f"Options with <10% H-L spread: {(df.hl_spread_pct < 10).sum():,} ({(df.hl_spread_pct < 10).mean():.1%})"
)

# Find optimal blending candidates
print("\n=== BLENDING CANDIDATES ===")
# Current Unity price around $25.68
spot = 25.68
target_strikes = [22, 23, 24, 25, 26, 27]
target_expiries = ["250718", "250815", "250919"]  # July, Aug, Sept 2025

print("\nLooking for liquid options at target strikes:")
for strike in target_strikes:
    for expiry in target_expiries:
        # Find matching options
        matches = df[(df["strike"] == strike) & (df["expiry_str"] == expiry)]
        if len(matches) > 0:
            total_vol = matches["volume"].sum()
            avg_spread = matches["hl_spread_pct"].mean()
            print(f"  ${strike} @ {expiry}: Volume={total_vol:,}, H-L Spread={avg_spread:.1f}%")

# Analyze current portfolio from my_positions.yaml
print("\n=== CURRENT POSITION ANALYSIS ===")
print("From my_positions.yaml:")
print("  Strike: $23, Expiration: 2025-07-18")
print("  Delta: -0.40, Contracts: 5")
print("  This is a SINGLE strike approach")
print("\nWith greek blending, we could instead have:")
print("  2 contracts @ $22 (21 DTE) - capture theta")
print("  2 contracts @ $24 (35 DTE) - core position")
print("  1 contract @ $26 (60 DTE) - vega play")
print("  Same total capital, better risk distribution!")
