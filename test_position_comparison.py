#!/usr/bin/env python3
"""Test position comparison with your actual positions."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime

from src.unity_wheel.models.position import Position
from src.unity_wheel.strategy.position_evaluator import PositionEvaluator


def main():
    """Test position evaluation with your positions."""

    print("🔍 Testing Position Comparison System")
    print("=" * 50)

    # Your current position: Short 75 calls at $25 strike, July 18 2025
    current_position = Position(symbol="U250718C00025000", quantity=-75)

    print(f"\n📊 Current Position:")
    print(f"   {current_position}")
    print(f"   Strike: $25.00")
    print(f"   Expiry: July 18, 2025")
    print(f"   Contracts: 75 (short)")

    # Market data (from your paste)
    current_price = 24.84
    current_option_bid = 2.07  # From your data: $2.08 - spread
    current_option_ask = 2.09  # From your data: $2.08 + spread

    # Calculate days to expiry
    expiry_date = datetime(2025, 7, 18)
    today = datetime.now()
    days_to_expiry = (expiry_date - today).days

    print(f"\n📈 Market Data:")
    print(f"   Unity Price: ${current_price}")
    print(f"   Option Bid/Ask: ${current_option_bid}/{current_option_ask}")
    print(f"   Days to Expiry: {days_to_expiry}")

    # Initialize evaluator
    evaluator = PositionEvaluator()

    # Evaluate current position
    print(f"\n⚖️ Evaluating Current Position...")
    current_value = evaluator.evaluate_position(
        position=current_position,
        current_price=current_price,
        risk_free_rate=0.05,
        volatility=0.45,  # Typical Unity volatility
        days_to_expiry=days_to_expiry,
        bid=current_option_bid,
        ask=current_option_ask,
        contracts=75,
    )

    print(f"\n📊 Current Position Analysis:")
    print(f"   Current Value: ${current_value.current_value * 100 * 75:,.2f}")
    print(f"   Intrinsic Value: ${current_value.intrinsic_value * 100 * 75:,.2f}")
    print(f"   Time Value: ${current_value.time_value * 100 * 75:,.2f}")
    print(f"   Probability ITM: {current_value.probability_itm:.1%}")
    print(f"   Expected Daily Return: ${current_value.daily_expected_return:.2f}")
    print(f"   Annualized Return: {current_value.annualized_return:.1%}")

    # Test some alternative positions
    print(f"\n🔄 Analyzing Switch Opportunities...")

    # Alternative strikes to consider
    alternatives = [
        # (strike, expiry_days, bid, ask, description)
        (23.00, 45, 2.80, 2.90, "Lower strike, shorter term"),
        (26.00, 45, 1.50, 1.60, "Higher strike, shorter term"),
        (25.00, 30, 1.80, 1.90, "Same strike, shorter term"),
        (27.00, 60, 1.20, 1.30, "Higher strike, longer term"),
    ]

    best_switch = None

    for strike, dte, bid, ask, desc in alternatives:
        print(f"\n   Analyzing: ${strike} strike, {dte} DTE ({desc})")

        analysis = evaluator.analyze_switch(
            current_position=current_position,
            current_bid=current_option_bid,
            current_ask=current_option_ask,
            current_dte=days_to_expiry,
            new_strike=strike,
            new_expiry_days=dte,
            new_bid=bid,
            new_ask=ask,
            underlying_price=current_price,
            volatility=0.45,
            risk_free_rate=0.05,
            contracts=75,
            min_benefit_threshold=1000.0,  # $1,000 minimum benefit
        )

        if analysis.should_switch:
            print(f"   ✅ Beneficial: {analysis.rationale}")
            print(f"      Switch Benefit: ${analysis.switch_benefit:,.2f}")
            print(f"      Daily Return Improvement: ${analysis.daily_return_improvement:.2f}")
            print(f"      Switch Cost: ${analysis.total_switch_cost:,.2f}")

            if not best_switch or analysis.switch_benefit > best_switch.switch_benefit:
                best_switch = analysis
        else:
            print(f"   ❌ Not Beneficial: {analysis.rationale}")

    print(f"\n📋 Recommendation:")
    if best_switch:
        print(
            f"   🔄 SWITCH to ${best_switch.new_position.strike} strike, {best_switch.new_position.expiry_days} DTE"
        )
        print(f"   Expected Benefit: ${best_switch.switch_benefit:,.2f}")
        print(f"   Improved Daily Return: ${best_switch.daily_return_improvement:.2f}/day")
        print(f"   Total Switch Cost: ${best_switch.total_switch_cost:,.2f}")
    else:
        print(f"   ✋ HOLD current position - no better alternatives found")
        print(f"   Your $25 calls expiring July 18 are well-positioned")
        print(
            f"   Continue collecting time decay of ${current_value.time_value * 100 * 75 / days_to_expiry:.2f}/day"
        )

    print(f"\n💡 Note: With Unity at ${current_price}, your $25 calls are slightly OTM")
    print(
        f"   This is ideal for theta collection with {current_value.probability_itm:.0%} assignment risk"
    )


if __name__ == "__main__":
    main()
