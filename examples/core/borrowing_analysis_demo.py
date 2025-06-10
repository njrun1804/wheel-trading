"""Demonstration of borrowing cost analysis for capital allocation decisions."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.unity_wheel.risk.borrowing_cost_analyzer import BorrowingCostAnalyzer, BorrowingSource


def demo_borrowing_analysis():
    """Demonstrate borrowing cost analysis for Unity wheel positions."""

    analyzer = BorrowingCostAnalyzer()

    print("Borrowing Cost Analysis for Unity Wheel Trading")
    print("=" * 60)

    # Show current borrowing status
    summary = analyzer.get_current_borrowing_summary()
    print("\nCurrent Debt Summary:")
    print("-" * 40)
    for source_name, details in summary.items():
        if source_name == "totals":
            continue
        print(f"\n{source_name.replace('_', ' ').title()}:")
        print(f"  Balance: ${details['balance']:,.0f}")
        print(f"  Rate: {details['annual_rate']}")
        print(f"  Daily Cost: ${details['daily_cost']:.2f}")
        print(f"  Monthly Cost: ${details['monthly_cost']:.2f}")
        print(f"  Annual Cost: ${details['annual_cost']:.0f}")

    totals = summary["totals"]
    print(f"\nTotal Debt: ${totals['total_debt']:,.0f}")
    print(f"Blended Rate: {totals['blended_rate']}")
    print(f"Total Daily Cost: ${totals['daily_cost']:.2f}")
    print(f"Total Monthly Cost: ${totals['monthly_cost']:.2f}")

    # Calculate hurdle rates
    print("\n" + "=" * 60)
    print("Hurdle Rates (Minimum Returns Needed):")
    print("-" * 40)

    for source_name in ["amex_loan", "schwab_margin"]:
        hurdle = analyzer.calculate_hurdle_rate(source_name)

        print(f"\n{source_name.replace('_', ' ').title()}:")
        print(f"  Pure Hurdle Rate: {hurdle:.1%}")
        print(f"  Note: Tax-free environment, no safety factors")

    # Example Unity wheel scenarios
    print("\n" + "=" * 60)
    print("Unity Wheel Position Scenarios:")
    print("-" * 40)

    scenarios = [
        {
            "name": "Conservative Put Sale",
            "position_size": 10000,
            "expected_return": 0.15,  # 15% annualized
            "confidence": 0.70,
            "holding_days": 45,
        },
        {
            "name": "Aggressive Put Sale",
            "position_size": 25000,
            "expected_return": 0.25,  # 25% annualized
            "confidence": 0.60,
            "holding_days": 30,
        },
        {
            "name": "Premium Harvest",
            "position_size": 15000,
            "expected_return": 0.12,  # 12% annualized
            "confidence": 0.85,
            "holding_days": 60,
        },
        {
            "name": "High Conviction Trade",
            "position_size": 50000,
            "expected_return": 0.35,  # 35% annualized
            "confidence": 0.90,
            "holding_days": 45,
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Position Size: ${scenario['position_size']:,.0f}")
        print(f"  Expected Return: {scenario['expected_return']:.1%}")
        print(f"  Confidence: {scenario['confidence']:.0%}")
        print(f"  Holding Period: {scenario['holding_days']} days")

        # Analyze with no cash
        result_no_cash = analyzer.analyze_position_allocation(
            position_size=scenario["position_size"],
            expected_annual_return=scenario["expected_return"],
            confidence=scenario["confidence"],
            holding_period_days=scenario["holding_days"],
            available_cash=0,
        )

        print(f"\n  With $0 cash available:")
        print(f"    Decision: {result_no_cash.action.upper()}")
        print(f"    Reasoning: {result_no_cash.reasoning}")

        if result_no_cash.action == "invest":
            print(f"    Borrow From: {result_no_cash.source_to_use}")
            print(f"    Borrowing Cost: ${result_no_cash.borrowing_cost:.0f}")
            print(f"    Expected Profit: ${result_no_cash.details['expected_profit']:.0f}")
            print(f"    Net Benefit: ${result_no_cash.net_benefit:.0f}")

        # Analyze with some cash
        cash_available = scenario["position_size"] * 0.4  # 40% cash
        result_with_cash = analyzer.analyze_position_allocation(
            position_size=scenario["position_size"],
            expected_annual_return=scenario["expected_return"],
            confidence=scenario["confidence"],
            holding_period_days=scenario["holding_days"],
            available_cash=cash_available,
        )

        print(f"\n  With ${cash_available:,.0f} cash available:")
        print(f"    Decision: {result_with_cash.action.upper()}")
        if result_with_cash.action == "invest" and result_with_cash.source_to_use:
            print(f"    Need to Borrow: ${result_with_cash.details['need_to_borrow']:,.0f}")
            print(f"    Borrowing Cost: ${result_with_cash.borrowing_cost:.0f}")
            print(f"    Net Benefit: ${result_with_cash.net_benefit:.0f}")

    # Paydown benefit analysis
    print("\n" + "=" * 60)
    print("Debt Paydown Analysis:")
    print("-" * 40)

    paydown_amounts = [5000, 10000, 20000]

    for amount in paydown_amounts:
        print(f"\nPaying down ${amount:,} of Amex loan:")
        benefits = analyzer.calculate_paydown_benefit(
            paydown_amount=amount, source_name="amex_loan", time_horizon_days=365
        )

        print(f"  Interest Saved (1 year): ${benefits['interest_saved']:.0f}")
        print(f"  Effective Return: {benefits['effective_return']:.1%}")
        print(f"  Daily Savings: ${benefits['daily_savings']:.2f}")
        print(f"  Monthly Savings: ${benefits['monthly_savings']:.2f}")
        print(f"  After-Tax Benefit: ${benefits['after_tax_benefit']:.0f}")

    # Key insights
    print("\n" + "=" * 60)
    print("Key Insights for Unity Wheel Trading:")
    print("-" * 40)
    print("1. Amex loan at 7% requires ~14% return after tax/risk adjustment")
    print("2. Schwab margin at 10% requires ~20% return after adjustments")
    print("3. Daily borrowing cost on $45k Amex: $8.63")
    print("4. Unity wheel typically returns 15-30% annualized")
    print("5. High confidence trades (>80%) more likely to justify borrowing")
    print("6. Paying down debt provides guaranteed 7-10% 'return'")
    print("\nRule of Thumb: Only borrow for positions with >20% expected return")


if __name__ == "__main__":
    demo_borrowing_analysis()
