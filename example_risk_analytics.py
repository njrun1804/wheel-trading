#!/usr/bin/env python3
"""Example usage of risk metrics and analytics functions."""

import numpy as np
from src.utils.math import calculate_var, calculate_cvar, half_kelly_size, margin_requirement
from src.utils.analytics import calculate_edge, expected_value, sharpe_ratio, maximum_drawdown

def main():
    print("=== Risk Metrics Demo ===\n")
    
    # 1. Value at Risk (VaR) and Conditional VaR (CVaR)
    print("1. VaR and CVaR Example:")
    volatility = 0.20  # 20% annual volatility
    var_95 = calculate_var(volatility, confidence_level=0.95)
    cvar_95 = calculate_cvar(volatility, confidence_level=0.95)
    print(f"   95% VaR: {var_95:.2%}")
    print(f"   95% CVaR: {cvar_95:.2%}")
    print(f"   CVaR/VaR ratio: {cvar_95/var_95:.2f}\n")
    
    # 2. Kelly Sizing for Options
    print("2. Kelly Sizing Example:")
    # Selling a put: $5 premium, $45 max loss if assigned
    premium = 5.0
    max_loss = 45.0
    edge = 0.05  # 5% edge
    odds = premium / max_loss
    bankroll = 100000
    
    position_size = half_kelly_size(edge, odds, bankroll)
    print(f"   Edge: {edge:.1%}")
    print(f"   Odds: {odds:.3f}")
    print(f"   Recommended position size: ${position_size:,.2f}")
    print(f"   As % of bankroll: {position_size/bankroll:.1%}\n")
    
    # 3. Margin Requirements
    print("3. Margin Requirement Example:")
    underlying = 450  # SPY at $450
    strikes = np.array([440, 430, 420])
    premiums = np.array([5.0, 3.0, 1.5])
    
    margins = margin_requirement(strikes, underlying, premiums)
    for strike, premium, margin in zip(strikes, premiums, margins):
        print(f"   Strike ${strike}, Premium ${premium:.2f}: Margin ${margin:,.0f}")
    print()
    
    # 4. Edge Calculation
    print("4. Edge Calculation Example:")
    theoretical_values = np.array([10.50, 5.25, 2.10])
    market_prices = np.array([10.00, 5.00, 2.00])
    
    edges = calculate_edge(theoretical_values, market_prices)
    for theo, market, edge in zip(theoretical_values, market_prices, edges):
        print(f"   Theoretical: ${theo:.2f}, Market: ${market:.2f}, Edge: {edge:.1%}")
    print()
    
    # 5. Expected Value
    print("5. Expected Value Example:")
    # Option selling scenarios
    outcomes = [500, -1000, -3000]  # Keep premium, small loss, large loss
    probabilities = [0.7, 0.25, 0.05]
    
    ev = expected_value(outcomes, probabilities)
    print(f"   Outcomes: {outcomes}")
    print(f"   Probabilities: {probabilities}")
    print(f"   Expected Value: ${ev:.2f}\n")
    
    # 6. Performance Metrics
    print("6. Performance Metrics Example:")
    # Simulated returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    cumulative = 10000 * np.cumprod(1 + returns)
    
    sharpe = sharpe_ratio(returns)
    max_dd, peak_idx, trough_idx = maximum_drawdown(cumulative)
    
    print(f"   Annual return: {np.mean(returns) * 252:.1%}")
    print(f"   Annual volatility: {np.std(returns) * np.sqrt(252):.1%}")
    print(f"   Sharpe ratio: {sharpe:.2f}")
    print(f"   Maximum drawdown: {max_dd:.1%}")
    print(f"   Drawdown period: {peak_idx} to {trough_idx}")


if __name__ == "__main__":
    main()