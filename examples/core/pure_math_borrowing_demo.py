"""Demonstration of pure mathematical borrowing analysis."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.unity_wheel.risk.pure_borrowing_analyzer import (
    PureBorrowingAnalyzer,
    analyze_pure_borrowing,
)


def demo_pure_math_analysis():
    """Demonstrate pure mathematical borrowing analysis."""

    print("Pure Mathematical Borrowing Analysis")
    print("=" * 60)
    print("No safety factors, no tax adjustments - just math\n")

    analyzer = PureBorrowingAnalyzer()

    # Show effective rates with daily compounding
    print("Loan Terms with Daily Compounding:")
    print("-" * 40)
    for name, loan in analyzer.loans.items():
        ear = loan.effective_annual_rate()
        print(f"{name.upper()}:")
        print(f"  Stated APR: {loan.annual_rate:.2%}")
        print(f"  Effective Annual Rate: {ear:.3%}")
        print(f"  Daily Rate: {loan.daily_rate:.5%}")
        print()

    # Example Unity wheel position
    print("Unity Wheel Position Analysis:")
    print("-" * 40)

    position = {
        "amount": 35000,  # $35k position
        "expected_return": 0.03,  # 3% total return (not annualized)
        "holding_days": 45,
        "contracts": 10,
        "strike": 35,
        "premium": 1.50,
    }

    print(f"Position Size: ${position['amount']:,}")
    print(
        f"Expected Return: {position['expected_return']:.1%} over {position['holding_days']} days"
    )
    print(f"Annualized Return: {position['expected_return'] * 365 / position['holding_days']:.1%}")
    print()

    # Analyze with different cash scenarios
    for cash in [0, 10000, 35000]:
        analysis = analyzer.analyze_investment(
            investment_amount=position["amount"],
            expected_return=position["expected_return"],
            holding_days=position["holding_days"],
            available_cash=cash,
            loan_source="schwab",  # 10% margin
        )

        print(f"\nWith ${cash:,} cash available:")
        print(f"  NPV: ${analysis.npv:,.2f}")
        print(f"  IRR: {analysis.irr:.3%}" if analysis.irr else "  IRR: N/A")
        print(f"  Decision: {analysis.action.upper()}")

        if cash < position["amount"]:
            borrow = position["amount"] - cash
            print(f"  Borrow: ${borrow:,}")
            print(f"  Borrowing Cost: ${analysis.borrowing_cost:.2f}")
            print(f"  Break-even Return: {analysis.break_even_return:.3%}")
            print(f"  Return Multiple: {analysis.return_multiple:.2f}x")

            if analysis.days_to_break_even:
                print(f"  Days to Break Even: {analysis.days_to_break_even}")

    # Break-even analysis
    print("\n" + "=" * 60)
    print("Break-Even Analysis (borrowing 100% from Schwab):")
    print("-" * 40)

    # What return do we need to exactly break even?
    schwab_loan = analyzer.loans["schwab"]

    for days in [30, 45, 60, 90]:
        # Calculate exact borrowing cost
        borrowing_cost = schwab_loan.compound_interest(days, position["amount"])
        break_even_return = borrowing_cost / position["amount"]
        annualized_break_even = break_even_return * 365 / days

        print(f"{days} days:")
        print(f"  Borrowing Cost: ${borrowing_cost:.2f}")
        print(f"  Break-even Return: {break_even_return:.3%}")
        print(f"  Annualized: {annualized_break_even:.2%}")

    # Sensitivity analysis
    print("\n" + "=" * 60)
    print("Sensitivity Analysis (3% return, 45 days, 100% borrowed):")
    print("-" * 40)

    base_analysis = analyzer.analyze_investment(
        investment_amount=position["amount"],
        expected_return=position["expected_return"],
        holding_days=position["holding_days"],
        available_cash=0,
    )

    print(f"Base Case NPV: ${base_analysis.npv:.2f}")
    print("\nSensitivity to changes:")
    for key, impact in base_analysis.sensitivity.items():
        print(f"  {key}: ${impact:+.2f} NPV impact")

    # Compare multiple opportunities
    print("\n" + "=" * 60)
    print("Comparing Multiple Opportunities:")
    print("-" * 40)

    opportunities = [
        {"amount": 20000, "return": 0.025, "days": 30},  # 2.5% in 30 days
        {"amount": 35000, "return": 0.030, "days": 45},  # 3.0% in 45 days
        {"amount": 50000, "return": 0.040, "days": 60},  # 4.0% in 60 days
    ]

    comparisons = analyzer.compare_opportunities(opportunities, available_capital=30000)

    for opp_name, analysis in comparisons.items():
        idx = int(opp_name.split("_")[1])
        opp = opportunities[idx]
        print(
            f"\nOpportunity {idx + 1}: ${opp['amount']:,} for {opp['return']:.1%} in {opp['days']} days"
        )
        print(f"  NPV: ${analysis.npv:,.2f}")
        print(f"  Action: {analysis.action}")
        print(f"  Annualized Return: {opp['return'] * 365 / opp['days']:.1%}")

    # Mathematical insights
    print("\n" + "=" * 60)
    print("Pure Mathematical Insights:")
    print("-" * 40)
    print("• Schwab margin at 10% APR = 10.52% EAR with daily compounding")
    print("• Break-even for 45-day position: 1.27% total return needed")
    print("• Each 1% return = $350 profit on $35k position")
    print("• Each 1% borrowing rate increase = ~$43 cost for 45 days")
    print("• NPV > 0 means mathematically profitable (no safety margin)")
    print("• Tax-free environment means gross = net returns")


if __name__ == "__main__":
    demo_pure_math_analysis()
