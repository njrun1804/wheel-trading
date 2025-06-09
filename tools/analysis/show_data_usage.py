#!/usr/bin/env python3
"""Show exactly how the wheel strategy uses data."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def show_data_usage():
    """Display how each component uses data."""
    
    print("🎯 Wheel Strategy Data Usage Map")
    print("=" * 70)
    
    print("\n1️⃣ WHEEL STRATEGY DECISIONS (src/unity_wheel/strategy/wheel.py)")
    print("   Data needed: Current option chain only")
    print("   Historical: NONE")
    print("   Example:")
    print("   - Current SPY price: $450")
    print("   - Find 30-delta put with 30-45 DTE")
    print("   - Calculate expected return")
    print("   → Recommend: Sell SPY 440P expiring in 35 days")
    
    print("\n2️⃣ RISK CALCULATIONS (src/unity_wheel/risk/analytics.py)")
    print("   Data needed: 250 days of daily returns")
    print("   Historical: Stock prices only (not options)")
    print("   Example:")
    print("   - Calculate 95% VaR from returns distribution")
    print("   - Estimate worst-case daily loss")
    print("   - Size position using Kelly criterion")
    print("   → Risk limit: Max 20% of portfolio in single position")
    
    print("\n3️⃣ GREEKS CALCULATIONS (src/unity_wheel/math/options.py)")
    print("   Data needed: Current spot, strike, time, volatility")
    print("   Historical: NONE")
    print("   Example:")
    print("   - Spot: $30, Strike: $27, DTE: 35, IV: 45%")
    print("   - Black-Scholes → Delta: -0.30")
    print("   → Real-time calculation, no history needed")
    
    print("\n4️⃣ POSITION MONITORING (src/unity_wheel/monitoring/)")
    print("   Data needed: Current positions + current prices")
    print("   Historical: Position entry data only")
    print("   Example:")
    print("   - Sold U 27P for $0.75 on Jan 1")
    print("   - Current price: $0.25")
    print("   - Profit: $0.50 (67%)")
    print("   → Track P&L, no historical market data needed")
    
    print("\n" + "=" * 70)
    print("📊 DATABENTO API CALLS")
    print("=" * 70)
    
    print("\nONE-TIME SETUP (per symbol):")
    print("┌─────────────────────────────────────────┐")
    print("│ GET /timeseries.get_range               │")
    print("│ Dataset: XNAS.BASIC                     │")
    print("│ Schema: OHLC_1D (daily bars)            │")
    print("│ Period: 250 days                        │")
    print("│ Purpose: Load returns for risk calcs    │")
    print("└─────────────────────────────────────────┘")
    
    print("\nREAL-TIME OPERATIONS (when user requests):")
    print("┌─────────────────────────────────────────┐")
    print("│ GET /timeseries.get_range               │")
    print("│ Dataset: OPRA.PILLAR                    │")
    print("│ Schema: DEFINITION + MBP_1              │")
    print("│ Period: Current snapshot                │")
    print("│ Purpose: Get option chain               │")
    print("│ Cache: 15 minutes                       │")
    print("└─────────────────────────────────────────┘")
    
    print("\nDAILY MAINTENANCE (cron job):")
    print("┌─────────────────────────────────────────┐")
    print("│ GET /timeseries.get_range               │")
    print("│ Dataset: XNAS.BASIC                     │")
    print("│ Schema: OHLC_1D                         │")
    print("│ Period: Yesterday only                  │")
    print("│ Purpose: Append latest price            │")
    print("└─────────────────────────────────────────┘")
    
    print("\n" + "=" * 70)
    print("💾 STORAGE BREAKDOWN")
    print("=" * 70)
    
    print("\nDuckDB Tables:")
    print("1. price_history      - 250 days × 10 symbols = ~20 KB")
    print("2. option_chains      - Current only (15-min TTL) = ~500 KB")  
    print("3. wheel_candidates   - Pre-filtered results = ~10 KB")
    print("4. position_snapshots - User positions = ~5 KB")
    
    print("\nTotal Storage: < 1 MB (vs 5 GB limit)")
    
    print("\n" + "=" * 70)
    print("🚀 KEY INSIGHTS")
    print("=" * 70)
    
    print("\n✅ This is NOT a backtesting system")
    print("✅ Minimal historical data required (just daily prices)")
    print("✅ No historical options data needed")
    print("✅ Real-time recommendations based on current market")
    print("✅ Pull-when-asked pattern (no streaming)")
    
    print("\n💡 Why this approach?")
    print("- Wheel strategy is mechanical (sell 30-delta puts)")
    print("- Decisions based on current conditions, not patterns")
    print("- Risk management needs returns history, not option history")
    print("- Keeps costs minimal (<$1/month)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    show_data_usage()