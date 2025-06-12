#!/usr/bin/env python3
"""
Test market calibration with Unity's historical data.
Shows how to use historical data to set optimal parameters.
"""

import asyncio
import os
import sys
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unity_wheel.analytics.market_calibrator import MarketCalibrator
from unity_wheel.risk.regime_detector import RegimeDetector

DB_PATH = os.path.expanduser(config.storage.database_path)


async def test_calibration():
    """Test market calibration with real Unity data."""

    print("🎯 Unity Wheel Strategy Calibration")
    print("=" * 60)

    # Connect to database
    conn = duckdb.connect(DB_PATH)

    # Load Unity data
    data = conn.execute(
        """
        SELECT date, open, high, low, close, volume, returns
        FROM price_history
        WHERE symbol = config.trading.symbol
        ORDER BY date
    """
    ).fetchall()

    # Convert to arrays
    dates = [row[0] for row in data]
    prices_df = pd.DataFrame(
        data, columns=["date", "open", "high", "low", "close", "volume", "returns"]
    )
    prices_df.set_index("date", inplace=True)
    returns = np.array([float(row[6]) for row in data if row[6] is not None])

    print(f"\n📊 Data loaded: {len(returns)} days")
    print(f"   Date range: {dates[0]} to {dates[-1]}")

    # Initialize calibrator
    calibrator = MarketCalibrator(symbol = config.trading.symbol)

    # Calibrate parameters
    print("\n🔧 Calibrating optimal parameters...")
    optimal_params = await calibrator.calibrate_from_history(
        returns=returns, prices=prices_df, iv_history=None  # Would include IV data in production
    )

    # Display results
    print(f"\n📈 Current Market Regime: {optimal_params.regime}")
    print(f"   Confidence: {optimal_params.confidence_score:.1%}")

    print("\n🎯 Optimal Parameters:")
    print("\n   Strike Selection:")
    print(f"   - Put Delta Target: {optimal_params.put_delta_target:.2f}")
    print(
        f"   - Delta Range: {optimal_params.delta_range[0]:.2f} to {optimal_params.delta_range[1]:.2f}"
    )
    print(f"   - Call Delta (if assigned): {optimal_params.call_delta_target:.2f}")

    print("\n   Expiration Selection:")
    print(f"   - Target DTE: {optimal_params.dte_target} days")
    print(f"   - DTE Range: {optimal_params.dte_range[0]} to {optimal_params.dte_range[1]} days")

    print("\n   Position Sizing:")
    print(
        f"   - Kelly Fraction: {optimal_params.kelly_fraction:.2f} ({optimal_params.kelly_fraction*100:.0f}%)"
    )
    print(f"   - Max Position: {optimal_params.max_position_pct*100:.0f}% of portfolio")

    print("\n   Risk Management:")
    print(f"   - Max Daily VaR: {optimal_params.max_var_95*100:.1f}%")
    print(f"   - Profit Target: {optimal_params.profit_target*100:.0f}% of max profit")
    print(f"   - Stop Loss: {optimal_params.stop_loss*100:.0f}% of position")

    print("\n   Rolling Rules:")
    print(f"   - Roll at {optimal_params.roll_at_dte} DTE")
    print(f"   - Roll at {optimal_params.roll_at_profit_pct*100:.0f}% profit")
    print(f"   - Roll if delta reaches {optimal_params.roll_at_delta:.2f}")

    # Generate trading rules
    rules = calibrator.generate_trading_rules(optimal_params)
    print("\n" + "\n".join(rules))

    # Compare with static parameters
    print("\n📊 Comparison with Static Parameters:")
    print("   Static approach: Always 0.30 delta, 45 DTE, 50% Kelly")
    print(
        f"   Dynamic approach: {optimal_params.put_delta_target:.2f} delta, {optimal_params.dte_target} DTE, {optimal_params.kelly_fraction*100:.0f}% Kelly"
    )

    # Show impact
    if optimal_params.regime == "Medium Vol" or optimal_params.regime == "High Vol":
        print("\n   ⚠️  Regime-based adjustments active:")
        print("   - Reduced delta target (lower probability of assignment)")
        print("   - Shorter DTE (less time risk)")
        print("   - Reduced Kelly fraction (smaller positions)")
        print("   - Earlier profit taking")

    # Example trade sizing
    portfolio_value = config.trading.portfolio_value
    print(f"\n💰 Example Position Sizing (${portfolio_value:,} portfolio):")

    position_size = (
        portfolio_value * optimal_params.kelly_fraction * optimal_params.max_position_pct
    )
    print(f"   Max position size: ${position_size:,.0f}")
    print(f"   Contracts (assuming $100 shares): {position_size/10000:.0f} contracts")

    # Risk metrics
    daily_var_dollars = position_size * optimal_params.max_var_95
    print(f"\n   Daily VaR: ${daily_var_dollars:,.0f}")
    print(f"   Stop loss: ${position_size * optimal_params.stop_loss:,.0f}")

    # Close connection
    conn.close()

    return optimal_params


async def main():
    """Run calibration test."""

    # Run calibration
    params = await test_calibration()

    # Additional analysis
    print("\n" + "=" * 60)
    print("🔍 Additional Insights:")

    # Volatility impact
    print("\n📊 Unity's Extreme Volatility Impact:")
    print("   - Standard wheel strategy would use 0.30 delta")
    print(f"   - Calibrated strategy uses {params.put_delta_target:.2f} delta")
    print(
        f"   - This reduces assignment probability from ~30% to ~{params.put_delta_target*100:.0f}%"
    )

    # Time decay optimization
    print("\n⏱️  Time Decay Optimization:")
    theta_days = params.dte_target
    print(f"   - Targeting {theta_days} DTE balances:")
    print("     • Enough premium collection")
    print("     • Manageable gamma risk")
    print("     • Multiple rolls per quarter")

    # What's missing
    print("\n🔨 To Fully Optimize, We Still Need:")
    print("   1. Current IV rank (need options data)")
    print("   2. Earnings calendar integration")
    print("   3. Real-time option chain analysis")
    print("   4. Correlation with QQQ/tech sector")
    print("   5. Historical assignment frequency by strike")


if __name__ == "__main__":
    asyncio.run(main())
