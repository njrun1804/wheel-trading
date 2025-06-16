"""Quick borrowing cost check for Unity wheel positions."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.unity_wheel.risk.borrowing_cost_analyzer import (
    analyze_borrowing_decision,
)


def quick_borrowing_check():
    """Quick check: Should I use borrowed money for this position?"""

    print("Quick Borrowing Cost Check")
    print("=" * 40)
    print()

    # Your current debt
    print("Current Debt:")
    print("- Amex Loan: $45,000 @ 7% APR")
    print("- Daily Cost: $8.63")
    print("- Monthly Cost: $262.50")
    print()

    # Quick rules (pure math, no safety factors)
    print("Quick Rules for Unity Wheel (Pure Math):")
    print("-" * 40)
    print("❌ Expected return <7%: Pay down Amex")
    print("⚠️  Expected return 7-10%: Maybe borrow")
    print("✅ Expected return >10%: OK to borrow from Schwab")
    print()

    # Example positions
    examples = [
        {
            "name": "Safe 30-delta put",
            "size": 15000,
            "expected_annual_return": 0.12,  # 12%
            "confidence": 0.80,
        },
        {
            "name": "Aggressive 40-delta put",
            "size": 25000,
            "expected_annual_return": 0.22,  # 22%
            "confidence": 0.70,
        },
        {
            "name": "Premium harvest strategy",
            "size": 35000,
            "expected_annual_return": 0.18,  # 18%
            "confidence": 0.85,
        },
    ]

    for example in examples:
        result = analyze_borrowing_decision(
            position_size=example["size"],
            expected_return=example["expected_annual_return"],
            confidence=example["confidence"],
            available_cash=0,  # Assume no cash
        )

        adj_return = example["expected_annual_return"] * example["confidence"]

        print(f"{example['name']}:")
        print(f"  Size: ${example['size']:,}")
        print(
            f"  Expected: {example['expected_annual_return']:.0%} × {example['confidence']:.0%} confidence = {adj_return:.0%}"
        )
        print(f"  Decision: {result.action.upper()}")

        if result.action == "invest":
            print(f"  → Borrow ${result.invest_amount:,.0f} (profit beats cost)")
        else:
            print(
                f"  → Pay debt instead ({adj_return:.0%} < {result.hurdle_rate:.0%} hurdle)"
            )
        print()

    # Bottom line
    print("Bottom Line (Pure Math):")
    print("-" * 40)
    print("• Paying down 7% debt = guaranteed 7% return")
    print("• Need >7% return to beat Amex, >10% to beat Schwab")
    print("• Unity wheel typically returns 15-25%")
    print("• Tax-free environment means returns are not reduced")


if __name__ == "__main__":
    quick_borrowing_check()
