#!/usr/bin/env python3
"""Example usage of risk metrics and analytics functions."""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.unity_wheel.risk.analytics import (
    RiskAnalyzer,
    calculate_cvar,
    calculate_sharpe_ratio,
    calculate_var,
)


def main():
    print("=== Risk Metrics Demo ===\n")

    # Initialize risk analyzer
    analyzer = RiskAnalyzer()

    # 1. Value at Risk (VaR) and Conditional VaR (CVaR)
    print("1. VaR and CVaR Example:")

    # Create sample returns data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year

    var_95 = calculate_var(returns, confidence_level=0.95)
    cvar_95 = calculate_cvar(returns, var_95)

    print(f"   95% VaR: {var_95:.2%}")
    print(f"   95% CVaR: {cvar_95:.2%}")
    print(f"   CVaR/VaR ratio: {cvar_95/var_95:.2f}\n")

    # 2. Kelly Sizing for Options
    print("2. Kelly Sizing Example:")

    # Option selling scenario
    win_rate = 0.70  # 70% of puts expire worthless
    avg_win = 1.0  # Keep full premium
    avg_loss = 3.0  # Average loss if assigned (3x premium)

    kelly_fraction, confidence = analyzer.calculate_kelly_criterion(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        apply_half_kelly=True,  # Conservative sizing
    )

    print(f"   Win rate: {win_rate:.0%}")
    print(f"   Avg win: {avg_win:.1f}x premium")
    print(f"   Avg loss: {avg_loss:.1f}x premium")
    print(f"   Recommended allocation: {kelly_fraction:.1%} of portfolio")
    print(f"   Confidence: {confidence:.0%}\n")

    # 3. Position Risk Assessment
    print("3. Position Risk Assessment:")

    portfolio_value = 100000
    position_value = 10000
    position_delta = 0.30

    risk_score, risk_factors = analyzer.assess_position_risk(
        position_value=position_value,
        portfolio_value=portfolio_value,
        position_delta=position_delta,
        underlying_volatility=0.65,  # Unity typical IV
    )

    print(f"   Portfolio: ${portfolio_value:,.0f}")
    print(f"   Position: ${position_value:,.0f} ({position_value/portfolio_value:.0%})")
    print(f"   Delta: {position_delta:.2f}")
    print(f"   Risk score: {risk_score:.2f}")
    print(f"   Risk factors: {', '.join(risk_factors) if risk_factors else 'None'}\n")

    # 4. Sharpe Ratio
    print("4. Performance Metrics:")

    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.05)
    annual_return = np.mean(returns) * 252
    annual_vol = np.std(returns) * np.sqrt(252)

    print(f"   Annual return: {annual_return:.1%}")
    print(f"   Annual volatility: {annual_vol:.1%}")
    print(f"   Sharpe ratio: {sharpe:.2f}")

    # 5. Risk Limits Check
    print("\n5. Risk Limits Check:")

    # Sample risk metrics
    current_var = 0.15
    current_cvar = 0.22
    current_margin_util = 0.45

    limits_ok, violations = analyzer.check_risk_limits(
        var_95=current_var,
        cvar_95=current_cvar,
        total_delta_exposure=0.30,
        margin_utilization=current_margin_util,
    )

    print(f"   VaR: {current_var:.1%}")
    print(f"   CVaR: {current_cvar:.1%}")
    print(f"   Margin utilization: {current_margin_util:.0%}")
    print(f"   Risk limits OK: {limits_ok}")

    if violations:
        print(f"   Violations: {', '.join(violations)}")


if __name__ == "__main__":
    main()
