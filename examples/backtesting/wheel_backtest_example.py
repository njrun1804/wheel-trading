"""Example usage of wheel strategy backtester.

DATA REQUIREMENTS FOR UNITY WHEEL STRATEGY VALIDATION:

1. HISTORICAL STOCK DATA:
   - Minimum: 1 year of daily OHLCV data
   - Recommended: 2-3 years for robust testing
   - Required fields: Date, Open, High, Low, Close, Volume
   - Purpose: Track underlying price movements, gap risk, assignment scenarios

2. HISTORICAL OPTIONS DATA (OPTIONAL BUT RECOMMENDED):
   - Minimum: 6 months of end-of-day option chains
   - Recommended: 1 year of data
   - Required fields: Strike, Expiration, Bid, Ask, Volume, Open Interest, IV
   - Purpose: Realistic premium calculation, liquidity validation, strike selection

3. SPECIFIC DATA POINTS NEEDED:
   - Unity price history: At least 250 trading days (1 year)
   - Volatility data: Historical or implied volatility
   - Earnings dates: To avoid trading around earnings
   - Dividend dates: If Unity pays dividends

4. WHY THIS AMOUNT:
   - 1 year captures multiple market regimes
   - Includes at least 4 earnings cycles
   - Sufficient for statistical significance (12+ trades)
   - Captures seasonal volatility patterns
   - Allows testing through different volatility environments
"""

import asyncio
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

from src.config.loader import get_config
from src.unity_wheel.backtesting import WheelBacktester
from src.unity_wheel.data_providers.databento import DatabentoClient, PriceHistoryLoader
from src.unity_wheel.storage import Storage
from src.unity_wheel.strategy.wheel import WheelParameters


async def run_basic_backtest():
    """Run a basic wheel strategy backtest on Unity."""

    # Initialize components
    config = get_config()
    storage = Storage()

    # For actual backtesting, you need historical data loaded
    # This example assumes data is already in the database
    backtester = WheelBacktester(storage=storage)

    # Define backtest period
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)

    print(f"Running backtest for Unity from {start_date} to {end_date}")

    # Run backtest with default parameters
    results = await backtester.backtest_strategy(
        symbol="U",
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        contracts_per_trade=1,  # 100 shares
    )

    # Display results
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annualized Return: {results.annualized_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"\nTotal Trades: {results.total_trades}")
    print(f"Assignments: {results.assignments}")
    print(f"Average Trade P&L: ${results.average_trade_pnl:.2f}")
    print(f"\nRisk Metrics:")
    print(f"VaR (95%): {results.var_95:.2%}")
    print(f"CVaR (95%): {results.cvar_95:.2%}")
    print(f"Max Gap Loss: ${results.max_gap_loss:.2f}")
    print(f"\nUnity-Specific:")
    print(f"Gap Events: {results.gap_events}")
    print(f"Earnings Avoided: {results.earnings_avoided}")

    return results


async def optimize_parameters():
    """Find optimal delta and DTE parameters for Unity."""

    storage = Storage()
    backtester = WheelBacktester(storage=storage)

    # Define optimization period
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)

    print(f"Optimizing parameters from {start_date} to {end_date}")

    # Run optimization
    optimization_results = await backtester.optimize_parameters(
        symbol="U",
        start_date=start_date,
        end_date=end_date,
        delta_range=(0.20, 0.40),  # Test 20-40 delta
        dte_range=(30, 60),  # Test 30-60 DTE
        optimization_metric="sharpe",  # Optimize for risk-adjusted returns
    )

    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Optimal Delta: {optimization_results['optimal_delta']:.2f}")
    print(f"Optimal DTE: {optimization_results['optimal_dte']} days")
    print(f"Best Sharpe Ratio: {optimization_results['best_metric']:.2f}")

    # Show all tested combinations
    print("\nAll Parameter Combinations:")
    print(optimization_results["all_results"].to_string())

    return optimization_results


async def backtest_different_deltas():
    """Compare performance of different delta targets."""

    storage = Storage()
    backtester = WheelBacktester(storage=storage)

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)

    deltas = [0.20, 0.25, 0.30, 0.35, 0.40]
    results = []

    for delta in deltas:
        print(f"\nTesting delta {delta}...")

        params = WheelParameters(target_delta=delta)

        result = await backtester.backtest_strategy(
            symbol="U",
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            parameters=params,
        )

        results.append(
            {
                "delta": delta,
                "return": result.annualized_return,
                "sharpe": result.sharpe_ratio,
                "win_rate": result.win_rate,
                "assignments": result.assignments,
                "max_dd": result.max_drawdown,
            }
        )

    # Create comparison table
    df = pd.DataFrame(results)
    print("\n=== DELTA COMPARISON ===")
    print(df.to_string(index=False))

    return df


async def walk_forward_windows():
    """Run walk-forward backtests over yearly windows."""

    storage = Storage()
    backtester = WheelBacktester(storage=storage)

    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 1, 1)

    results_df = await backtester.backtest_walk_forward(
        symbol="U",
        start_date=start_date,
        end_date=end_date,
        window_size=timedelta(days=365),
        step_size=timedelta(days=365),
        delta_range=(0.20, 0.40),
        dte_range=(30, 60),
    )

    print("\n=== WALK-FORWARD RESULTS ===")
    print(results_df.to_string(index=False))

    return results_df


async def load_required_data():
    """Load the minimum required data for backtesting."""

    config = get_config()
    storage = Storage()

    # Initialize Databento client
    client = DatabentoClient()
    price_loader = PriceHistoryLoader(client, storage)

    # Load 1 year of Unity price data
    symbol = config.unity.ticker
    success = await price_loader.load_price_history(
        symbol=symbol, days=250  # 1 year of trading days
    )

    if success:
        # Check what we have
        info = await price_loader.check_data_availability(symbol)
        print(f"\nData loaded for {symbol}:")
        print(f"Days available: {info['days_available']}")
        print(f"Date range: {info['date_range']}")
        print(f"Sufficient for risk: {info['sufficient_for_risk']}")
        print(f"Annualized volatility: {info['annualized_volatility']:.2%}")
    else:
        print("Failed to load price data")

    return success


def plot_backtest_results(results):
    """Visualize backtest results."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Equity curve
    ax1 = axes[0, 0]
    results.equity_curve.plot(ax=ax1)
    ax1.set_title("Portfolio Value Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value ($)")

    # Daily returns distribution
    ax2 = axes[0, 1]
    results.daily_returns.hist(bins=50, ax=ax2)
    ax2.set_title("Daily Returns Distribution")
    ax2.set_xlabel("Daily Return")
    ax2.set_ylabel("Frequency")

    # Drawdown chart
    ax3 = axes[1, 0]
    rolling_max = results.equity_curve.expanding().max()
    drawdown = (results.equity_curve - rolling_max) / rolling_max
    drawdown.plot(ax=ax3)
    ax3.set_title("Drawdown Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Drawdown %")

    # Trade outcomes
    ax4 = axes[1, 1]
    trade_pnls = [p.realized_pnl for p in results.positions]
    if trade_pnls:
        pd.Series(trade_pnls).plot(kind="bar", ax=ax4)
        ax4.set_title("Individual Trade P&L")
        ax4.set_xlabel("Trade Number")
        ax4.set_ylabel("P&L ($)")

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    print(
        """
    WHEEL STRATEGY BACKTEST DATA REQUIREMENTS
    ========================================

    Minimum Data Needed:
    - 1 year (250 days) of Unity daily OHLCV data
    - Covers multiple market conditions
    - Includes 4 earnings cycles
    - Provides ~12-24 trades for statistical significance

    Recommended Data:
    - 2-3 years for more robust results
    - Historical option chains for accurate premiums
    - Implied volatility history
    - Earnings calendar dates

    Data Loading Options:
    1. Use Databento API (requires subscription)
    2. Import from CSV files
    3. Use free data sources (Yahoo Finance for stocks)
    """
    )

    # Run examples
    asyncio.run(run_basic_backtest())
    asyncio.run(walk_forward_windows())
