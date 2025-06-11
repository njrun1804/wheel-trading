#!/usr/bin/env python3
"""Daily parameter optimization example for Unity wheel strategy.

This script combines the dynamic optimizer, grid-search backtester,
and Monte Carlo modeling to refine strike and sizing parameters.
It is designed to run once per day on a local machine.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.unity_wheel.analytics.dynamic_optimizer import DynamicOptimizer, MarketState
from src.unity_wheel.backtesting import WheelBacktester
from src.unity_wheel.risk.advanced_financial_modeling import AdvancedFinancialModeling
from src.unity_wheel.storage import Storage
from src.unity_wheel.data_providers.databento import DatabentoClient, PriceHistoryLoader


async def run_daily_optimization(symbol: str = "U") -> None:
    """Run dynamic optimization, grid search, and Monte Carlo analysis."""

    storage = Storage()
    await storage.initialize()

    # Ensure we have at least a year of price history
    client = DatabentoClient()
    loader = PriceHistoryLoader(client, storage)
    await loader.load_price_history(symbol, days=250)

    # Load historical data from cache
    async with storage.cache.connection() as conn:
        df = conn.execute(
            """
            SELECT date, close, volume, returns
            FROM price_history
            WHERE symbol = ?
            ORDER BY date
            """,
            [symbol],
        ).df()

    if df.empty:
        raise RuntimeError("Price history not available")

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Build market state from recent data
    realized_vol = df["returns"].iloc[-20:].std() * np.sqrt(252)
    all_vols = df["returns"].rolling(20).std().dropna()
    vol_percentile = float((all_vols < realized_vol).mean()) if len(all_vols) else 0.5
    momentum = (df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20]
    volume_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]

    market_state = MarketState(
        realized_volatility=realized_vol,
        volatility_percentile=vol_percentile,
        price_momentum=float(momentum),
        volume_ratio=float(volume_ratio),
    )

    returns = df["returns"].values

    # Dynamic optimization
    optimizer = DynamicOptimizer(symbol)
    dyn_result = optimizer.optimize_parameters(market_state, returns)

    # Grid search around dynamic parameters
    delta_range = (
        max(0.10, dyn_result.delta_target - 0.05),
        min(0.40, dyn_result.delta_target + 0.05),
    )
    dte_range = (
        max(21, dyn_result.dte_target - 7),
        min(49, dyn_result.dte_target + 7),
    )

    backtester = WheelBacktester(storage=storage)
    opt_result = await backtester.optimize_parameters(
        symbol=symbol,
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        delta_range=delta_range,
        dte_range=dte_range,
        optimization_metric="sharpe",
    )

    best_delta = opt_result["optimal_delta"] or dyn_result.delta_target
    best_dte = opt_result["optimal_dte"] or dyn_result.dte_target

    # Monte Carlo analysis of best parameters
    mc_model = AdvancedFinancialModeling()
    mc = mc_model.monte_carlo_simulation(
        expected_return=opt_result["best_results"].annualized_return,
        volatility=realized_vol,
        time_horizon=best_dte,
        position_size=10000,
        n_simulations=1000,
    )

    print("=== Daily Parameter Optimization ===")
    print(f"Dynamic Delta Target : {dyn_result.delta_target:.3f}")
    print(f"Dynamic DTE Target   : {dyn_result.dte_target} days")
    print(f"Kelly Fraction       : {dyn_result.kelly_fraction:.3f}")
    print()
    print("Grid Search Refinement")
    print(f"  Optimal Delta      : {best_delta:.3f}")
    print(f"  Optimal DTE        : {best_dte} days")
    print(f"  Best Sharpe Ratio  : {opt_result['best_metric']:.2f}")
    print()
    print("Monte Carlo Analysis (1000 paths)")
    print(f"  Probability Profit : {mc.probability_profit:.1%}")
    print(f"  Expected Shortfall : {mc.expected_shortfall:.2%}")
    print(f"  Max Drawdown       : {mc.max_drawdown:.2%}")


if __name__ == "__main__":
    asyncio.run(run_daily_optimization())
