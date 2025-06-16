#!/usr/bin/env python3
"""
What data granularity do we actually need for wheel strategy?
Daily? Minute? Tick? Let's find out.
"""

import math

from unity_wheel.config.unified_config import get_config

config = get_config()


def analyze_wheel_strategy_needs():
    """What does the wheel strategy actually use?"""
    print("🎯 WHEEL STRATEGY DATA NEEDS")
    print("=" * 60)

    print("\n1. OPTION SELECTION (Real-time)")
    print("   Need: Current bid/ask spreads")
    print("   Granularity: Snapshot when requested")
    print("   Historical: NONE")
    print("   Why: We only sell options at current market prices")

    print("\n2. RISK CALCULATIONS (VaR, volatility)")
    print("   Need: Returns for statistical analysis")
    print("   Granularity: DAILY is sufficient")
    print("   Historical: 750-1000 days")
    print("   Why: Multi-day holding periods, not day trading")

    print("\n3. KELLY SIZING")
    print("   Need: Win/loss rates from monthly cycles")
    print("   Granularity: MONTHLY outcomes")
    print("   Historical: 36-50 monthly results")
    print("   Why: Each option cycle is ~30-45 days")


def compare_data_requirements():
    """Compare storage and cost for different granularities."""
    print("\n\n💾 DATA REQUIREMENTS BY GRANULARITY")
    print("=" * 60)

    # For 750 days of history
    days = 750
    hours_per_day = 6.5  # Market hours
    minutes_per_day = hours_per_day * 60
    ticks_per_minute = 100  # Rough estimate for Unity

    # Data points needed
    daily_points = days
    minute_points = days * minutes_per_day
    tick_points = days * minutes_per_day * ticks_per_minute

    # Storage (assuming 50 bytes per record)
    bytes_per_record = 50

    print(f"\nFor {days} days of Unity data:")
    print("\nDAILY BARS:")
    print(f"  Data points: {daily_points:,}")
    print(f"  Storage: {daily_points * bytes_per_record / 1024:.1f} KB")
    print("  API calls: 3-4 (daily bar batches)")

    print("\n15-MINUTE BARS:")
    minute_15_points = days * hours_per_day * 4
    print(f"  Data points: {minute_15_points:,}")
    print(f"  Storage: {minute_15_points * bytes_per_record / 1024 / 1024:.1f} MB")
    print("  API calls: ~100 (limited batch size)")

    print("\nMINUTE BARS:")
    print(f"  Data points: {minute_points:,}")
    print(f"  Storage: {minute_points * bytes_per_record / 1024 / 1024:.1f} MB")
    print("  API calls: ~500+")

    print("\nTICK DATA:")
    print(f"  Data points: {tick_points:,}")
    print(f"  Storage: {tick_points * bytes_per_record / 1024 / 1024 / 1024:.1f} GB")
    print("  API calls: Thousands")
    print("  Cost: $$$$ (tick data is expensive!)")


def calculate_risk_metrics_accuracy():
    """Does higher granularity improve risk calculations?"""
    print("\n\n📊 RISK CALCULATION ACCURACY BY GRANULARITY")
    print("=" * 60)

    # For wheel strategy with monthly holding periods
    holding_period_days = 30

    print(f"\nHolding period: {holding_period_days} days")
    print("\nVaR calculation accuracy:")

    # Daily VaR
    print("\nDAILY DATA:")
    print("  Measures: Daily returns")
    print("  Accuracy: ✅ Perfect for multi-day holdings")
    print("  Use case: Standard risk management")

    # Intraday VaR
    print("\nINTRADAY DATA:")
    print("  Measures: Intraday volatility")
    print("  Accuracy: ⚠️ Overstates risk for monthly holdings")
    print("  Use case: Day trading (not wheel strategy)")

    # The key insight
    print("\n💡 KEY INSIGHT:")
    print("Wheel positions are held 30-45 days")
    print("Intraday noise averages out over month")
    print("Daily data captures the risk we actually face")


def show_actual_vol_calculations():
    """Compare volatility estimates from different granularities."""
    print("\n\n📈 VOLATILITY ESTIMATION COMPARISON")
    print("=" * 60)

    # Unity's characteristics
    annual_vol_daily = 0.65  # From daily data

    # Intraday vol is typically higher due to microstructure
    intraday_multiplier = 1.3  # Typical for volatile stocks
    annual_vol_intraday = annual_vol_daily * intraday_multiplier

    print("\nUnity volatility estimates:")
    print(f"From daily closes: {annual_vol_daily:.0%} annual")
    print(f"From intraday data: {annual_vol_intraday:.0%} annual")

    # Impact on position sizing
    portfolio = config.trading.portfolio_value_000
    target_risk = 0.02  # 2% daily risk

    position_size_daily = (target_risk * portfolio) / (
        annual_vol_daily / math.sqrt(252)
    )
    position_size_intraday = (target_risk * portfolio) / (
        annual_vol_intraday / math.sqrt(252)
    )

    print("\nPosition sizing impact:")
    print(f"Using daily vol: ${position_size_daily:,.0f}")
    print(f"Using intraday vol: ${position_size_intraday:,.0f}")
    print(
        f"Difference: ${position_size_daily - position_size_intraday:,.0f} smaller position"
    )

    print("\n⚠️ Using intraday data would make us too conservative!")


def final_recommendation():
    """What do we actually need?"""
    print("\n\n" + "=" * 60)
    print("🎯 FINAL ANSWER: DAILY DATA ONLY")
    print("=" * 60)

    print("\nFor Unity wheel strategy, we need:")
    print("\n1. HISTORICAL (one-time load):")
    print("   - Type: Daily OHLC bars")
    print("   - Period: 750-1000 days")
    print("   - Dataset: XASE.BASIC")
    print("   - Storage: ~40KB")
    print("   - Purpose: Risk calculations")

    print("\n2. REAL-TIME (when requested):")
    print("   - Type: Option chain snapshot")
    print("   - Period: Current only")
    print("   - Dataset: OPRA.PILLAR")
    print("   - Storage: ~50KB (15-min cache)")
    print("   - Purpose: Find options to sell")

    print("\n3. ONGOING (daily update):")
    print("   - Type: Previous day close")
    print("   - Period: 1 day")
    print("   - Dataset: XASE.BASIC")
    print("   - Storage: 50 bytes")
    print("   - Purpose: Update risk metrics")

    print("\n❌ NOT NEEDED:")
    print("   - Tick data (expensive, overkill)")
    print("   - Minute bars (unnecessary granularity)")
    print("   - Intraday data (wrong risk profile)")
    print("   - Historical options (decisions use current only)")

    print("\n💰 TOTAL COST:")
    print("   - Initial load: ~$0.10")
    print("   - Monthly: ~$0.50")
    print("   - Storage: <1MB")


if __name__ == "__main__":
    analyze_wheel_strategy_needs()
    compare_data_requirements()
    calculate_risk_metrics_accuracy()
    show_actual_vol_calculations()
    final_recommendation()
