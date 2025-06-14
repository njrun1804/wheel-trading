#!/usr/bin/env python3
"""
Test dynamic optimization system for Unity wheel strategy.
Demonstrates autonomous operation with continuous parameter adjustment.
"""

import os
import sys
from datetime import datetime

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unity_wheel.analytics.dynamic_optimizer import (
    DynamicOptimizer,
    MarketState,
    OptimizationResult,
)
from src.config.loader import get_config

config = get_config()
DB_PATH = os.path.expanduser(config.storage.database_path)


def calculate_market_state(returns: np.ndarray, prices: list) -> MarketState:
    """Calculate current market state from historical data."""

    # 20-day realized volatility
    recent_returns = returns[-20:] if len(returns) >= 20 else returns
    realized_vol = np.std(recent_returns) * np.sqrt(252)

    # Volatility percentile (where we are in historical distribution)
    all_vols = []
    for i in range(20, len(returns)):
        vol = np.std(returns[i - 20 : i]) * np.sqrt(252)
        all_vols.append(vol)

    if all_vols:
        vol_percentile = sum(v < realized_vol for v in all_vols) / len(all_vols)
    else:
        vol_percentile = 0.5

    # 20-day momentum
    if len(prices) >= 20:
        momentum = float(prices[-1][4] - prices[-20][4]) / float(prices[-20][4])  # close prices
    else:
        momentum = 0.0

    # Volume ratio (would need volume data)
    volume_ratio = 1.0  # Placeholder

    return MarketState(
        realized_volatility=realized_vol,
        volatility_percentile=vol_percentile,
        price_momentum=momentum,
        volume_ratio=volume_ratio,
        iv_rank=None,  # Would come from options data
        days_to_earnings=None,  # Would come from calendar
    )


def main():
    """Test dynamic optimization with real Unity data."""

    print("🚀 Unity Dynamic Parameter Optimization")
    print("=" * 60)
    print("Objective: Maximize CAGR - 0.20 × |CVaR₉₅| with ½-Kelly sizing")
    print("=" * 60)

    # Load Unity data
    conn = duckdb.connect(DB_PATH)
    data = conn.execute(
        """
        SELECT date, open, high, low, close, volume, returns
        FROM price_history
        WHERE symbol = config.trading.symbol AND returns IS NOT NULL
        ORDER BY date
    """
    ).fetchall()

    returns = np.array([float(row[6]) for row in data])

    print(f"\n📊 Data: {len(returns)} days of Unity returns")

    # Calculate current market state
    market_state = calculate_market_state(returns, data)

    print("\n📈 Current Market State:")
    print(f"   Realized Vol: {market_state.realized_volatility:.1%} annualized")
    print(f"   Vol Percentile: {market_state.volatility_percentile:.1%} (vs history)")
    print(f"   20-day Momentum: {market_state.price_momentum:+.1%}")

    # Initialize optimizer
    optimizer = DynamicOptimizer(symbol = config.trading.symbol)

    # Run optimization
    print("\n🔧 Running Dynamic Optimization...")
    result = optimizer.optimize_parameters(market_state, returns)

    # Display results
    print("\n✅ Optimization Results:")
    print("\n   📊 Optimal Parameters:")
    print(f"   Delta Target: {result.delta_target:.3f}")
    print(f"   DTE Target: {result.dte_target} days")
    print(f"   Kelly Fraction: {result.kelly_fraction:.3f} ({result.kelly_fraction*100:.1f}%)")

    print("\n   📈 Expected Outcomes:")
    print(f"   Expected CAGR: {result.expected_cagr:.1%}")
    print(f"   Expected CVaR₉₅: {result.expected_cvar:.1%}")
    print(f"   Objective Value: {result.objective_value:.4f}")

    print(f"\n   🎯 Confidence: {result.confidence_score:.1%}")

    # Show diagnostics
    print("\n   🔍 Diagnostics:")
    for key, value in result.diagnostics.items():
        print(f"   {key}: {value:+.3f}")

    # Validate results
    print("\n🔒 Autonomous Validation:")
    validation = optimizer.validate_optimization(result)
    all_passed = all(validation.values())

    for check, passed in validation.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")

    if all_passed:
        print("\n✅ All validation checks passed!")
    else:
        print("\n⚠️  Some validation checks failed - review parameters")

    # Compare with static approach
    print("\n📊 Dynamic vs Static Comparison:")

    # Static parameters
    static_delta = 0.30
    static_dte = 45
    static_kelly = 0.50

    print("\n   Static Approach:")
    print(f"   - Delta: {static_delta:.2f} (fixed)")
    print(f"   - DTE: {static_dte} days (fixed)")
    print(f"   - Kelly: {static_kelly:.2f} (fixed)")

    print("\n   Dynamic Approach:")
    print(
        f"   - Delta: {result.delta_target:.3f} (adjusted for vol percentile: {market_state.volatility_percentile:.0%})"
    )
    print(f"   - DTE: {result.dte_target} days (shorter due to high vol)")
    print(f"   - Kelly: {result.kelly_fraction:.3f} (reduced for risk management)")

    # Show continuous adjustment
    print("\n📉 Continuous Adjustment Example:")
    print("   As volatility percentile changes, parameters adjust smoothly:")

    vol_percentiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    for vp in vol_percentiles:
        test_state = MarketState(
            realized_volatility=0.40 + 0.60 * vp,  # 40% to 100% vol
            volatility_percentile=vp,
            price_momentum=0.0,
            volume_ratio=1.0,
        )
        test_result = optimizer.optimize_parameters(test_state, returns)
        print(f"\n   Vol Percentile {vp:.0%}:")
        print(
            f"     Delta: {test_result.delta_target:.3f}, DTE: {test_result.dte_target}, Kelly: {test_result.kelly_fraction:.3f}"
        )

    # Show how it optimizes objective function
    print("\n🎯 Objective Function Optimization:")
    print("   Formula: CAGR - 0.20 × |CVaR₉₅|")
    print("\n   Current optimization:")
    print(
        f"   {result.expected_cagr:.3f} - 0.20 × |{result.expected_cvar:.3f}| = {result.objective_value:.4f}"
    )

    # Integration with system
    print("\n🔗 System Integration:")
    print(
        """
    async def get_wheel_recommendation():
        # 1. Load current market data
        market_state = calculate_market_state(...)

        # 2. Run dynamic optimization
        optimal = optimizer.optimize_parameters(market_state, returns)

        # 3. Validate autonomously
        if not optimizer.validate_optimization(optimal):
            return fallback_parameters()

        # 4. Find options matching parameters
        options = await find_options(
            delta_target=optimal.delta_target,
            dte_target=optimal.dte_target
        )

        # 5. Size position
        position_size = portfolio_value * optimal.kelly_fraction

        return WheelRecommendation(
            action="SELL_PUT",
            strike=options.best_strike,
            size=position_size,
            confidence=optimal.confidence_score
        )
    """
    )

    conn.close()


if __name__ == "__main__":
    main()
