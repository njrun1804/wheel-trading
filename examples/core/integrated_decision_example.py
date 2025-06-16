"""Example of integrated decision making with borrowing cost analysis."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.unity_wheel.risk.borrowing_cost_analyzer import BorrowingCostAnalyzer


def make_integrated_decision():
    """Example of making a wheel decision considering borrowing costs."""

    print("Integrated Wheel + Borrowing Decision")
    print("=" * 50)
    print()

    # Initialize analyzer
    analyzer = BorrowingCostAnalyzer()

    # Scenario: You found a good Unity wheel opportunity
    wheel_opportunity = {
        "strike": 35.0,
        "premium": 1.50,  # $1.50 per share
        "dte": 45,
        "contracts": 10,  # 10 contracts = 1000 shares
        "delta": 0.30,
        "iv_rank": 65,  # High IV percentile
    }

    # Calculate position details
    position_size = wheel_opportunity["contracts"] * 100 * wheel_opportunity["strike"]
    premium_total = wheel_opportunity["contracts"] * 100 * wheel_opportunity["premium"]

    # Calculate expected return (simplified)
    # Assume 80% chance of keeping premium, 20% chance of assignment
    win_rate = 0.80
    expected_pnl = (win_rate * premium_total) - ((1 - win_rate) * premium_total * 0.5)

    # Annualize return
    days_per_year = 365
    expected_annual_return = (expected_pnl / position_size) * (
        days_per_year / wheel_opportunity["dte"]
    )

    # Confidence based on IV rank and delta
    confidence = 0.70 if wheel_opportunity["iv_rank"] > 50 else 0.60
    confidence *= 1.1 if wheel_opportunity["delta"] <= 0.30 else 1.0
    confidence = min(confidence, 0.90)  # Cap at 90%

    print("Wheel Opportunity Analysis:")
    print(f"  Strike: ${wheel_opportunity['strike']}")
    print(f"  Premium: ${wheel_opportunity['premium']}/share")
    print(f"  Contracts: {wheel_opportunity['contracts']}")
    print(f"  DTE: {wheel_opportunity['dte']} days")
    print(f"  Delta: {wheel_opportunity['delta']}")
    print(f"  IV Rank: {wheel_opportunity['iv_rank']}")
    print()

    print("Position Economics:")
    print(f"  Position Size: ${position_size:,.0f}")
    print(f"  Premium Collected: ${premium_total:,.0f}")
    print(f"  Expected P&L: ${expected_pnl:,.0f}")
    print(f"  Expected Annual Return: {expected_annual_return:.1%}")
    print(f"  Confidence: {confidence:.0%}")
    print()

    # Analyze borrowing decision for different cash scenarios
    cash_scenarios = [0, 10000, 20000, position_size]

    print("Borrowing Analysis by Available Cash:")
    print("-" * 50)

    for cash in cash_scenarios:
        result = analyzer.analyze_position_allocation(
            position_size=position_size,
            expected_annual_return=expected_annual_return,
            holding_period_days=wheel_opportunity["dte"],
            available_cash=cash,
            confidence=confidence,
        )

        print(f"\nWith ${cash:,} cash available:")
        print(f"  Action: {result.action.upper()}")

        if result.action == "invest":
            need_to_borrow = max(0, position_size - cash)
            if need_to_borrow > 0:
                print(f"  Borrow: ${need_to_borrow:,} from {result.source_to_use}")
                print(
                    f"  Cost: ${result.borrowing_cost:.0f} for {wheel_opportunity['dte']} days"
                )
            print(f"  Expected Profit: ${result.details['expected_profit']:.0f}")
            print(f"  Net After Borrowing: ${result.net_benefit:.0f}")
            print("  → GO: Take the position")
        else:
            print(f"  Hurdle Rate: {result.hurdle_rate:.1%}")
            print(f"  Your Return: {result.expected_return:.1%}")
            print("  → NO GO: Pay down debt instead")

    # Final recommendation
    print("\n" + "=" * 50)
    print("RECOMMENDATION:")

    if expected_annual_return * confidence >= 0.14:  # 14% hurdle
        print("✅ This trade meets the borrowing hurdle rate")
        print("   Consider taking the position if you have conviction")
    else:
        print("❌ This trade doesn't justify borrowing")
        print("   Better to pay down debt or wait for better setup")

    # Additional considerations
    print("\nAdditional Considerations:")
    print(
        "• Unity earnings in {} days - avoid if <7 days to earnings".format(
            "XX"  # Would check actual calendar
        )
    )
    print("• Current Unity volatility regime: Check if >80% (stop trading)")
    print("• Portfolio concentration: Check existing Unity exposure")
    print("• Tax implications: Short-term gains taxed at ordinary rates")


if __name__ == "__main__":
    make_integrated_decision()
