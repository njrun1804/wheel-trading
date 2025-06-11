#!/usr/bin/env python3
"""
Part 2: Regime-aware backtesting with optimal parameter selection.
"""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from run_3year_statistical_backtest import run_statistical_3year_analysis
from src.unity_wheel.backtesting import WheelBacktester
from src.unity_wheel.storage import Storage
from src.unity_wheel.strategy import WheelParameters


def calculate_kelly_criterion(returns: np.ndarray, confidence: float = 0.5) -> float:
    """
    Calculate Kelly criterion with safety factor.
    Uses empirical distribution rather than assuming normality.
    """
    if len(returns) < 30:
        return 0.1  # Conservative default

    # Calculate win rate and avg win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.1

    p = len(wins) / len(returns)  # Win probability
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))

    # Kelly formula: f = p/a - q/b
    # where p = win prob, q = loss prob, a = loss size, b = win size
    kelly = (p * avg_win - (1 - p) * avg_loss) / (avg_win * avg_loss)

    # Apply confidence factor for safety
    return max(0.05, min(kelly * confidence, 0.5))


def optimize_parameters_for_regime(returns: np.ndarray, volatility: float, skewness: float) -> dict:
    """
    Optimize wheel parameters based on regime characteristics.
    Uses statistical properties to determine optimal settings.
    """
    # Base parameters
    base_delta = 0.30
    base_dte = 45
    base_position = 0.20

    # Adjust delta based on volatility and skewness
    # Higher vol = lower delta, negative skew = lower delta
    vol_adjustment = np.clip((0.60 - volatility) / 0.40, -0.15, 0.10)
    skew_adjustment = np.clip(skewness * 0.05, -0.05, 0.05)

    optimal_delta = np.clip(base_delta + vol_adjustment + skew_adjustment, 0.15, 0.40)

    # Adjust DTE based on volatility
    # Higher vol = shorter DTE for gamma risk management
    if volatility > 0.80:
        optimal_dte = 30
    elif volatility > 0.60:
        optimal_dte = 45
    else:
        optimal_dte = 60

    # Position sizing using modified Kelly
    kelly = calculate_kelly_criterion(returns, confidence=0.5)

    # Further reduce in high vol regimes
    if volatility > 1.0:
        kelly *= 0.5
    elif volatility > 0.80:
        kelly *= 0.75

    optimal_position = np.clip(kelly, 0.05, 0.25)

    return {
        "delta": optimal_delta,
        "dte": optimal_dte,
        "position_size": optimal_position,
        "kelly_fraction": kelly,
    }


async def backtest_with_regime_parameters():
    """Run backtests using regime-specific optimized parameters."""

    # First run statistical analysis
    print("Running statistical analysis first...")
    full_data, regime_results, regime_names, regime_order = await run_statistical_3year_analysis()

    print("\n\n=== REGIME-AWARE BACKTESTING ===")
    print("-" * 60)

    # Initialize backtester
    storage = Storage()
    await storage.initialize()
    backtester = WheelBacktester(storage)

    # 1. Optimize parameters for each regime
    print("\n1. REGIME-SPECIFIC PARAMETER OPTIMIZATION")
    print("-" * 60)

    regime_params = {}
    for regime in regime_order:
        stats = regime_results["statistics"][regime]

        # Get returns for this regime
        regime_mask = regime_results["labels"] == regime
        regime_indices = regime_results["features"].index[regime_mask]
        regime_returns = full_data.loc[regime_indices, "returns"].values

        # Optimize parameters
        params = optimize_parameters_for_regime(
            regime_returns, stats["volatility"], stats["skewness"]
        )

        regime_params[regime] = params

        print(f"\n  {regime_names[regime]} Volatility Regime:")
        print(f"    Volatility: {stats['volatility']:.1%}")
        print(f"    Skewness: {stats['skewness']:.2f}")
        print(f"    Optimal Delta: {params['delta']:.2f}")
        print(f"    Optimal DTE: {params['dte']} days")
        print(
            f"    Position Size: {params['position_size']:.1%} (Kelly: {params['kelly_fraction']:.1%})"
        )

    # 2. Backtest each regime period
    print("\n\n2. REGIME-SPECIFIC BACKTESTING")
    print("-" * 60)

    # Group consecutive days in same regime
    regime_periods = []
    current_regime = regime_results["labels"][0]
    start_idx = 0

    for i in range(1, len(regime_results["labels"])):
        if regime_results["labels"][i] != current_regime:
            # End of regime period
            if i - start_idx >= 20:  # Minimum 20 days
                regime_periods.append(
                    {
                        "regime": current_regime,
                        "start": regime_results["features"].index[start_idx],
                        "end": regime_results["features"].index[i - 1],
                        "days": i - start_idx,
                    }
                )
            current_regime = regime_results["labels"][i]
            start_idx = i

    # Add final period
    if len(regime_results["labels"]) - start_idx >= 20:
        regime_periods.append(
            {
                "regime": current_regime,
                "start": regime_results["features"].index[start_idx],
                "end": regime_results["features"].index[-1],
                "days": len(regime_results["labels"]) - start_idx,
            }
        )

    # Backtest significant regime periods
    regime_performance = {r: [] for r in regime_order}

    print(f"\n  Testing {len(regime_periods)} regime periods...")

    for i, period in enumerate(regime_periods[:20]):  # Test up to 20 periods
        regime = period["regime"]
        params = regime_params[regime]

        try:
            wheel_params = WheelParameters(
                target_delta=params["delta"],
                target_dte=params["dte"],
                max_position_size=params["position_size"],
            )

            result = await backtester.backtest_strategy(
                symbol="U",
                start_date=period["start"],
                end_date=period["end"],
                initial_capital=100000,
                parameters=wheel_params,
            )

            regime_performance[regime].append(
                {
                    "return": result.total_return,
                    "sharpe": result.sharpe_ratio,
                    "trades": result.total_trades,
                    "win_rate": result.win_rate,
                    "days": period["days"],
                }
            )

            if i % 5 == 0:
                print(f"    Completed {i+1}/{min(20, len(regime_periods))} periods...")

        except Exception as e:
            print(f"    Error in period {i}: {str(e)[:50]}...")

    # 3. Aggregate regime performance
    print("\n\n3. REGIME PERFORMANCE SUMMARY")
    print("-" * 60)

    print("\n  Regime | Periods | Avg Return | Avg Sharpe | Win Rate | Total Days")
    print("  -------|---------|------------|------------|----------|------------")

    for regime in regime_order:
        perfs = regime_performance[regime]
        if perfs:
            avg_return = np.mean([p["return"] for p in perfs])
            avg_sharpe = np.mean([p["sharpe"] for p in perfs if p["sharpe"] != 0])
            avg_win_rate = np.mean([p["win_rate"] for p in perfs])
            total_days = sum([p["days"] for p in perfs])

            print(
                f"  {regime_names[regime]:<6} | {len(perfs):>7} | {avg_return:>10.1%} | "
                f"{avg_sharpe:>10.2f} | {avg_win_rate:>8.1%} | {total_days:>10}"
            )

    # 4. Full period backtest with dynamic regime switching
    print("\n\n4. FULL 3-YEAR BACKTEST WITH REGIME AWARENESS")
    print("-" * 60)

    start_date = full_data.index.min()
    end_date = full_data.index.max()

    # Use weighted average parameters based on regime distribution
    weighted_delta = sum(
        regime_params[r]["delta"] * regime_results["statistics"][r]["percentage"]
        for r in regime_order
    )
    weighted_position = sum(
        regime_params[r]["position_size"] * regime_results["statistics"][r]["percentage"]
        for r in regime_order
    )

    print(f"\n  Weighted Parameters (based on regime distribution):")
    print(f"    Delta: {weighted_delta:.2f}")
    print(f"    Position Size: {weighted_position:.1%}")

    full_params = WheelParameters(
        target_delta=weighted_delta, target_dte=45, max_position_size=weighted_position
    )

    print(f"\n  Running full {(end_date - start_date).days / 365.25:.1f}-year backtest...")

    full_result = await backtester.backtest_strategy(
        symbol="U",
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        parameters=full_params,
    )

    print(f"\n  Full Period Results:")
    print(f"    Total Return: {full_result.total_return:.1%}")
    print(f"    Annual Return: {full_result.annualized_return:.1%}")
    print(f"    Sharpe Ratio: {full_result.sharpe_ratio:.2f}")
    print(f"    Max Drawdown: {full_result.max_drawdown:.1%}")
    print(f"    Win Rate: {full_result.win_rate:.1%}")
    print(f"    Total Trades: {full_result.total_trades}")

    # Calculate risk-adjusted metrics
    sortino = calculate_sortino_ratio(full_result.daily_returns.values)
    calmar = (
        abs(full_result.annualized_return / full_result.max_drawdown)
        if full_result.max_drawdown != 0
        else 0
    )

    print(f"\n  Advanced Metrics:")
    print(f"    Sortino Ratio: {sortino:.2f}")
    print(f"    Calmar Ratio: {calmar:.2f}")
    print(f"    Avg Win/Loss: {calculate_win_loss_ratio(full_result.positions):.2f}")

    return full_result, regime_results, regime_params


def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0) -> float:
    """Calculate Sortino ratio using downside deviation."""
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return 0

    downside_dev = np.std(downside_returns) * np.sqrt(252)
    return (np.mean(returns) * 252) / downside_dev if downside_dev > 0 else 0


def calculate_win_loss_ratio(positions: list) -> float:
    """Calculate average win to average loss ratio."""
    wins = [p.realized_pnl for p in positions if p.realized_pnl > 0]
    losses = [abs(p.realized_pnl) for p in positions if p.realized_pnl < 0]

    if not wins or not losses:
        return 0

    return np.mean(wins) / np.mean(losses)


if __name__ == "__main__":
    asyncio.run(backtest_with_regime_parameters())
