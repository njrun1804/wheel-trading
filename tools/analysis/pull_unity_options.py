#!/usr/bin/env python3
"""
Pull real Unity options from Databento using the correct OPRA.PILLAR dataset.
Unity DOES have options - we were just using the wrong query format!
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unity_wheel.utils.databento_unity import (
    cost_estimate,
    get_equity_bars,
    get_wheel_candidates,
    store_options_in_duckdb,
)


async def main():
    """Pull Unity options and equity data."""

    print("🎯 Unity Options Pull - Using Correct Databento Format")
    print("=" * 60)

    # First, check cost estimate
    now = datetime.now()
    start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (now - timedelta(days=1)).strftime("%Y-%m-%d")  # T-1 for historical

    print("\n💰 Cost Estimate:")
    est = cost_estimate(start, end)
    print(f"   Data size: {est['readable_bytes']}")
    print(f"   Estimated cost: ${est['cost_usd']:.2f}")

    # Pull equity bars first
    print("\n📈 Pulling Unity daily bars...")
    bars = get_equity_bars(days=250)

    if not bars.empty:
        print(f"✅ Retrieved {len(bars)} days of Unity price data")
        print(f"   Latest close: ${bars['close'].iloc[-1]:.2f}")
        print(f"   Volatility: {bars['returns'].std() * (252**0.5) * 100:.1f}%")

        # Store in DuckDB
        import duckdb

        db_path = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")
        conn = duckdb.connect(db_path)

        # Update price history
        conn.execute("DELETE FROM price_history WHERE symbol = 'U'")

        for _, row in bars.iterrows():
            conn.execute(
                """
                INSERT INTO price_history
                (symbol, date, open, high, low, close, volume, returns, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                [
                    "U",
                    row["ts_event"].date() if hasattr(row["ts_event"], "date") else row.name.date(),
                    float(row.get("open", row["close"])),
                    float(row.get("high", row["close"])),
                    float(row.get("low", row["close"])),
                    float(row["close"]),
                    int(row.get("volume", 0)),
                    float(row.get("returns", 0)),
                ],
            )

        conn.commit()
        conn.close()
        print("✅ Stored Unity price history in DuckDB")

    # Now pull options
    print("\n📊 Pulling Unity options for wheel strategy...")
    print("   Target: 30-delta puts, 30-60 DTE")

    # Get wheel candidates
    candidates = get_wheel_candidates(
        target_delta=0.30, dte_range=(30, 60), moneyness_range=0.15  # ±15% from ATM
    )

    if candidates.empty:
        print("❌ No Unity options found matching criteria")
        print("\nPossible reasons:")
        print("1. Market is closed (data is T-1)")
        print("2. No options in the 30-60 DTE range")
        print("3. Need to check different date ranges")
        return

    print(f"\n✅ Found {len(candidates)} Unity put options!")

    # Display top candidates
    print("\n🎯 Top Wheel Candidates (sorted by closest to 30-delta):")
    print("-" * 80)
    print(
        f"{'Strike':>7} | {'DTE':>4} | {'Bid':>6} | {'Ask':>6} | {'Mid':>6} | {'Delta':>7} | {'Moneyness':>10}"
    )
    print("-" * 80)

    for _, opt in candidates.head(10).iterrows():
        print(
            f"${opt['strike']:6.2f} | {opt['dte']:3d} | "
            f"${opt['bid']:5.2f} | ${opt['ask']:5.2f} | ${opt['mid']:5.2f} | "
            f"{opt['approx_delta']:6.3f} | {opt['moneyness']*100:9.1f}%"
        )

    # Store in database
    stored = store_options_in_duckdb(candidates)
    print(f"\n✅ Stored {stored} Unity options in database")

    # Show data quality
    print("\n📊 Data Quality Check:")
    print(f"   Options with quotes: {candidates['bid'].notna().sum()}")
    print(f"   Average bid-ask spread: ${(candidates['ask'] - candidates['bid']).mean():.2f}")
    print(f"   Tightest spread: ${(candidates['ask'] - candidates['bid']).min():.2f}")

    # Test recommendation
    print("\n💡 Recommended Trade:")
    best = candidates.iloc[0]
    print(f"   Sell {best['raw_symbol']}")
    print(f"   Strike: ${best['strike']:.2f}")
    print(f"   Expiration: {best['expiration'].date()}")
    print(f"   Premium: ${best['mid']:.2f}")
    print(f"   Approximate delta: {best['approx_delta']:.3f}")

    if best["moneyness"] < 0:
        print(f"   ⚠️  This put is ITM by {abs(best['moneyness'])*100:.1f}%")
    else:
        print(f"   ✅ This put is OTM by {best['moneyness']*100:.1f}%")

    print("\n✅ Unity options successfully pulled from Databento!")
    print("🎉 Unity DOES have options - we just needed the right query!")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("DATABENTO_API_KEY"):
        print("❌ Error: DATABENTO_API_KEY environment variable not set")
        print("Set it with: export DATABENTO_API_KEY='your-key-here'")
        sys.exit(1)

    asyncio.run(main())
