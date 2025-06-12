"""Demonstration of Unity margin calculations with account type differentiation."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unity_wheel.risk.unity_margin import UnityMarginCalculator


def demo_unity_margin():
    """Demonstrate Unity margin calculations for different account types."""

    calculator = UnityMarginCalculator()

    # Common parameters
    contracts = 10
    strike = 35.0
    current_price = 36.0
    premium_received = 150.0  # $1.50 per share

    print("Unity Margin Calculator Demonstration")
    print("=" * 50)
    print(f"Contracts: {contracts}")
    print(f"Strike: ${strike}")
    print(f"Current Price: ${current_price}")
    print(f"Premium: ${premium_received/100:.2f} per share")
    print()

    # 1. IRA Account - Full cash securing
    ira_result = calculator.calculate_unity_margin(
        contracts=contracts,
        strike=strike,
        current_price=current_price,
        premium_received=premium_received,
        account_type="ira",
        option_type="put",
    )

    print("1. IRA Account")
    print(f"   Margin Required: ${ira_result.margin_required:,.2f}")
    print(f"   Margin Type: {ira_result.margin_type}")
    print(f"   Note: 100% cash secured (no margin allowed)")
    print()

    # 2. Cash Account - Also full cash for puts
    cash_result = calculator.calculate_unity_margin(
        contracts=contracts,
        strike=strike,
        current_price=current_price,
        premium_received=premium_received,
        account_type="cash",
        option_type="put",
    )

    print("2. Cash Account")
    print(f"   Margin Required: ${cash_result.margin_required:,.2f}")
    print(f"   Margin Type: {cash_result.margin_type}")
    print(f"   Note: Cash secured puts (no margin for puts)")
    print()

    # 3. Margin Account - Standard margin with Unity adjustment
    margin_result = calculator.calculate_unity_margin(
        contracts=contracts,
        strike=strike,
        current_price=current_price,
        premium_received=premium_received,
        account_type="margin",
        option_type="put",
    )

    print("3. Margin Account")
    print(f"   Standard Margin: ${margin_result.details['standard_margin']:,.2f}")
    print(f"   Unity Multiplier: {margin_result.details['unity_multiplier']}x")
    print(f"   Unity Adjusted: ${margin_result.margin_required:,.2f}")
    print(f"   Calculation Method: {margin_result.calculation_method}")
    print(f"   Note: 1.5x standard margin due to Unity volatility")
    print()

    # 4. Portfolio Margin - Risk-based with Unity adjustment
    portfolio_result = calculator.calculate_portfolio_margin(
        contracts=contracts,
        strike=strike,
        current_price=current_price,
        premium_received=premium_received,
        implied_volatility=0.60,  # 60% IV typical for Unity
        account_type="portfolio",
    )

    print("4. Portfolio Margin Account")
    print(f"   Stress Move: {portfolio_result.details['stress_move_percent']:.1%}")
    print(f"   Stressed Price: ${portfolio_result.details['stressed_price']:.2f}")
    print(
        f"   Standard Portfolio Margin: ${portfolio_result.details['standard_portfolio_margin']:,.2f}"
    )
    print(f"   Unity Adjusted: ${portfolio_result.margin_required:,.2f}")
    print(f"   Note: Even portfolio margin gets Unity adjustment")
    print()

    # 5. Comparison Summary
    print("Margin Comparison Summary")
    print("-" * 50)
    print(f"IRA/Cash Account:      ${ira_result.margin_required:>10,.2f} (100% secured)")
    print(
        f"Margin Account:        ${margin_result.margin_required:>10,.2f} ({margin_result.margin_required/ira_result.margin_required:.1%} of cash)"
    )
    print(
        f"Portfolio Account:     ${portfolio_result.margin_required:>10,.2f} ({portfolio_result.margin_required/ira_result.margin_required:.1%} of cash)"
    )
    print()

    # 6. Impact of premium on margin
    print("Premium Impact on Margin (Margin Account)")
    print("-" * 50)

    for premium_per_share in [0.50, 1.00, 2.00, 3.00, 5.00]:
        premium_dollars = premium_per_share * 100
        test_result = calculator.calculate_unity_margin(
            contracts=1,
            strike=strike,
            current_price=current_price,
            premium_received=premium_dollars,
            account_type="margin",
            option_type="put",
        )
        print(
            f"Premium ${premium_per_share:.2f}/share: Margin ${test_result.margin_required:>8,.2f} per contract"
        )

    print()
    print("Key Takeaways:")
    print("- IRA accounts always require 100% cash securing")
    print("- Unity positions require 1.5x standard margin due to volatility")
    print("- Higher premiums reduce margin requirements")
    print("- Portfolio margin still gets Unity adjustment for safety")


if __name__ == "__main__":
    demo_unity_margin()
