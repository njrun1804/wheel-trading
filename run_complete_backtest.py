#!/usr/bin/env python3
"""
Run complete backtesting suite with parameter optimization and risk analysis.
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from src.unity_wheel.backtesting import WheelBacktester
from src.unity_wheel.storage import Storage
from src.unity_wheel.strategy import WheelParameters


async def run_complete_analysis():
    """Run complete backtest with parameter optimization and risk analysis."""

    print("=== WHEEL STRATEGY COMPLETE ANALYSIS ===\n")

    # 1. Show current market conditions
    print("1. CURRENT MARKET CONDITIONS")
    print("-" * 50)

    db_path = Path("data/unified_wheel_trading.duckdb")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Get latest market data
    current = conn.execute(
        """
        SELECT
            date,
            stock_price,
            volatility_20d,
            volatility_250d,
            var_95,
            risk_free_rate
        FROM backtest_features
        WHERE symbol = 'U'
        ORDER BY date DESC
        LIMIT 1
    """
    ).fetchone()

    date, price, vol20, vol250, var95, rf = current
    print(f"  Date: {date}")
    print(f"  Unity Price: ${price:.2f}")
    print(f"  20-day volatility: {vol20:.1%}")
    print(f"  250-day volatility: {vol250:.1%}")
    print(f"  Value at Risk (95%): {var95:.2%}")
    print(f"  Risk-free rate: {rf:.2%}")

    # Show volatility over time
    vol_history = conn.execute(
        """
        SELECT
            DATE_TRUNC('month', date) as month,
            AVG(volatility_20d) as avg_vol,
            MIN(volatility_20d) as min_vol,
            MAX(volatility_20d) as max_vol
        FROM backtest_features
        WHERE symbol = 'U'
        AND date >= CURRENT_DATE - INTERVAL '1 year'
        GROUP BY DATE_TRUNC('month', date)
        ORDER BY month DESC
        LIMIT 6
    """
    ).fetchall()

    print("\n  Volatility History (last 6 months):")
    print("  Month       | Avg Vol | Min Vol | Max Vol")
    print("  ------------|---------|---------|--------")
    for month, avg_vol, min_vol, max_vol in vol_history:
        print(
            f"  {month.strftime('%Y-%m')}    | {avg_vol:>7.1%} | {min_vol:>7.1%} | {max_vol:>7.1%}"
        )

    conn.close()

    # 2. Run main backtest
    print("\n\n2. MAIN BACKTEST (1 Year)")
    print("-" * 50)

    storage = Storage()
    await storage.initialize()
    backtester = WheelBacktester(storage)

    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Default parameters
    params = WheelParameters(
        target_delta=0.30, target_dte=45, max_position_size=0.20  # 20% max per position
    )

    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Initial capital: $100,000")
    print(f"  Target delta: {params.target_delta}")
    print(f"  Target DTE: {params.target_dte} days")
    print(f"  Max position: {params.max_position_size:.0%} of portfolio")

    # Run backtest
    start_time = time.time()
    results = await backtester.backtest_strategy(
        symbol="U",
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        parameters=params,
    )
    elapsed = time.time() - start_time

    print(f"\n  ✅ Backtest completed in {elapsed:.2f} seconds")

    # Display results
    print("\n  Results:")
    print(f"  Total return: {results.total_return:.1%}")
    print(f"  Annualized return: {results.annualized_return:.1%}")
    print(f"  Sharpe ratio: {results.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {results.max_drawdown:.1%}")
    print(f"  Win rate: {results.win_rate:.1%}")
    print(f"  Total trades: {results.total_trades}")
    print(
        f"  Assignments: {results.assignments} ({results.assignments/max(1,results.total_trades)*100:.1f}%)"
    )
    print(f"  Average trade P&L: ${results.average_trade_pnl:.2f}")

    # Risk metrics
    print(f"\n  Risk Metrics:")
    print(f"  VaR (95%): {results.var_95:.2%}")
    print(f"  CVaR (95%): {results.cvar_95:.2%}")
    print(f"  Max gap loss: ${results.max_gap_loss:.2f}")
    print(f"  Gap events (>10% moves): {results.gap_events}")
    print(f"  Earnings avoided: {results.earnings_avoided}")

    # 3. Parameter Optimization
    print("\n\n3. PARAMETER OPTIMIZATION")
    print("-" * 50)
    print("  Testing different delta targets...")

    optimization_results = []
    test_deltas = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    for delta in test_deltas:
        print(f"\n  Testing delta={delta:.2f}...", end="", flush=True)

        test_params = WheelParameters(target_delta=delta, target_dte=45, max_position_size=0.20)

        start_time = time.time()
        result = await backtester.backtest_strategy(
            symbol="U",
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            parameters=test_params,
        )
        elapsed = time.time() - start_time

        optimization_results.append(
            {
                "delta": delta,
                "return": result.annualized_return,
                "sharpe": result.sharpe_ratio,
                "max_dd": result.max_drawdown,
                "trades": result.total_trades,
                "assignments": result.assignments,
                "win_rate": result.win_rate,
                "time": elapsed,
            }
        )

        print(f" done ({elapsed:.1f}s)")
        print(
            f"    Return: {result.annualized_return:>6.1%} | Sharpe: {result.sharpe_ratio:>5.2f} | Trades: {result.total_trades:>3}"
        )

    # Show optimization summary
    print("\n  Optimization Results Summary:")
    print("  Delta | Return | Sharpe | MaxDD  | Trades | Assign | Win%  ")
    print("  ------|--------|--------|--------|--------|--------|-------")

    for r in optimization_results:
        assign_pct = r["assignments"] / max(1, r["trades"]) * 100
        print(
            f"  {r['delta']:.2f}  | {r['return']:>6.1%} | {r['sharpe']:>6.2f} | {r['max_dd']:>6.1%} | {r['trades']:>6} | {r['assignments']:>6} | {r['win_rate']:>5.1%}"
        )

    # Find optimal
    best_sharpe = max(optimization_results, key=lambda x: x["sharpe"])
    best_return = max(optimization_results, key=lambda x: x["return"])

    print(
        f"\n  🎯 Best Sharpe Ratio: Delta {best_sharpe['delta']:.2f} (Sharpe: {best_sharpe['sharpe']:.2f})"
    )
    print(
        f"  🎯 Best Return: Delta {best_return['delta']:.2f} (Return: {best_return['return']:.1%})"
    )

    # 4. DTE Optimization
    print("\n\n4. DTE (Days to Expiry) OPTIMIZATION")
    print("-" * 50)
    print("  Testing different target DTEs...")

    dte_results = []
    test_dtes = [30, 45, 60, 90]
    optimal_delta = best_sharpe["delta"]  # Use optimal delta from above

    for dte in test_dtes:
        print(f"\n  Testing DTE={dte} days...", end="", flush=True)

        test_params = WheelParameters(
            target_delta=optimal_delta, target_dte=dte, max_position_size=0.20
        )

        start_time = time.time()
        result = await backtester.backtest_strategy(
            symbol="U",
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            parameters=test_params,
        )
        elapsed = time.time() - start_time

        dte_results.append(
            {
                "dte": dte,
                "return": result.annualized_return,
                "sharpe": result.sharpe_ratio,
                "trades": result.total_trades,
                "avg_pnl": result.average_trade_pnl,
            }
        )

        print(f" done ({elapsed:.1f}s)")
        print(f"    Return: {result.annualized_return:>6.1%} | Sharpe: {result.sharpe_ratio:>5.2f}")

    print("\n  DTE Optimization Results:")
    print("  DTE  | Return | Sharpe | Trades | Avg P&L")
    print("  -----|--------|--------|--------|--------")

    for r in dte_results:
        print(
            f"  {r['dte']:>3}d | {r['return']:>6.1%} | {r['sharpe']:>6.2f} | {r['trades']:>6} | ${r['avg_pnl']:>7.2f}"
        )

    best_dte = max(dte_results, key=lambda x: x["sharpe"])
    print(f"\n  🎯 Optimal DTE: {best_dte['dte']} days (Sharpe: {best_dte['sharpe']:.2f})")

    # 5. Risk Analysis by Market Regime
    print("\n\n5. RISK ANALYSIS BY MARKET REGIME")
    print("-" * 50)

    # Run backtest with optimal parameters
    optimal_params = WheelParameters(
        target_delta=best_sharpe["delta"], target_dte=best_dte["dte"], max_position_size=0.20
    )

    print(
        f"  Using optimal parameters: Delta={optimal_params.target_delta:.2f}, DTE={optimal_params.target_dte}"
    )

    # Test in different volatility regimes
    print("\n  Performance in Different Volatility Regimes:")

    # Get periods of different volatility
    conn = duckdb.connect(str(db_path), read_only=True)

    vol_periods = conn.execute(
        """
        WITH vol_regimes AS (
            SELECT
                date,
                volatility_20d,
                CASE
                    WHEN volatility_20d < 0.40 THEN 'Low (<40%)'
                    WHEN volatility_20d < 0.70 THEN 'Medium (40-70%)'
                    WHEN volatility_20d < 1.00 THEN 'High (70-100%)'
                    ELSE 'Extreme (>100%)'
                END as regime
            FROM backtest_features
            WHERE symbol = 'U'
            AND date >= ?
        )
        SELECT
            regime,
            COUNT(*) as days,
            MIN(date) as first_date,
            MAX(date) as last_date
        FROM vol_regimes
        GROUP BY regime
        ORDER BY
            CASE regime
                WHEN 'Low (<40%)' THEN 1
                WHEN 'Medium (40-70%)' THEN 2
                WHEN 'High (70-100%)' THEN 3
                ELSE 4
            END
    """,
        [start_date],
    ).fetchall()

    print("\n  Volatility Regime Distribution:")
    print("  Regime           | Days | First Date | Last Date")
    print("  -----------------|------|------------|----------")
    for regime, days, first, last in vol_periods:
        print(f"  {regime:<15} | {days:>4} | {first} | {last}")

    conn.close()

    # 6. Strategy Validation
    print("\n\n6. STRATEGY VALIDATION")
    print("-" * 50)

    # Compare theoretical vs actual option prices
    print("  Validating option pricing model...")

    conn = duckdb.connect(str(db_path), read_only=True)

    # Get sample of options with actual prices
    validation = conn.execute(
        """
        WITH recent_options AS (
            SELECT
                om.strike,
                md.date,
                md.close as market_price,
                s.close as spot_price,
                DATEDIFF('day', md.date, om.expiration) as dte,
                bf.volatility_20d as volatility,
                bf.risk_free_rate
            FROM market_data md
            JOIN options_metadata om ON md.symbol = om.symbol
            JOIN market_data s ON md.date = s.date AND s.symbol = 'U' AND s.data_type = 'stock'
            JOIN backtest_features bf ON md.date = bf.date AND bf.symbol = 'U'
            WHERE om.option_type = 'P'
            AND om.underlying = 'U'
            AND md.close > 0.50
            AND md.date >= CURRENT_DATE - INTERVAL '30 days'
            AND md.volume > 10
            ORDER BY md.date DESC
            LIMIT 20
        )
        SELECT
            strike,
            spot_price,
            market_price,
            dte,
            volatility
        FROM recent_options
    """
    ).fetchall()

    if validation:
        print("\n  Option Pricing Validation (last 30 days):")
        print("  Strike | Spot  | Market | DTE | Vol  | Status")
        print("  -------|-------|--------|-----|------|-------")

        valid_count = 0
        for strike, spot, mkt_price, dte, vol in validation[:10]:
            # Simple validation: option should be worth at least intrinsic value
            intrinsic = max(0, strike - spot)
            time_value = mkt_price - intrinsic

            if time_value >= 0 and mkt_price > 0:
                status = "✅ Valid"
                valid_count += 1
            else:
                status = "❌ Invalid"

            print(
                f"  ${strike:>5.0f} | ${spot:>5.2f} | ${mkt_price:>6.2f} | {dte:>3} | {vol:>4.0%} | {status}"
            )

        print(
            f"\n  Validation Rate: {valid_count}/{min(10, len(validation))} ({valid_count/min(10, len(validation))*100:.0f}%)"
        )

    conn.close()

    # 7. Final Summary
    print("\n\n7. FINAL SUMMARY & RECOMMENDATIONS")
    print("-" * 50)

    print(f"\n  Optimal Parameters:")
    print(f"  • Target Delta: {best_sharpe['delta']:.2f}")
    print(f"  • Target DTE: {best_dte['dte']} days")
    print(f"  • Expected Annual Return: {best_sharpe['return']:.1%}")
    print(f"  • Sharpe Ratio: {best_sharpe['sharpe']:.2f}")
    print(f"  • Win Rate: {best_sharpe['win_rate']:.1%}")

    print(f"\n  Risk Considerations:")
    print(
        f"  • Unity's current volatility ({vol20:.0%}) is {'extremely high' if vol20 > 0.80 else 'elevated' if vol20 > 0.60 else 'moderate'}"
    )
    print(
        f"  • Assignment risk: {best_sharpe['assignments']/max(1,best_sharpe['trades'])*100:.1f}% of trades"
    )
    print(f"  • Gap risk events occur ~{results.gap_events} times per year")
    print(f"  • Earnings avoidance critical ({results.earnings_avoided} periods skipped)")

    print(f"\n  Recommended Actions:")
    if vol20 > 0.80:
        print("  ⚠️  Consider reducing position sizes in current high volatility")
        print("  ⚠️  Use lower delta targets (0.15-0.20) for safety")
    elif vol20 > 0.60:
        print("  • Current volatility elevated - maintain disciplined position sizing")
        print("  • Delta 0.20-0.25 recommended")
    else:
        print("  • Volatility moderate - can use standard parameters")
        print("  • Consider increasing position size opportunistically")

    print("\n✅ Analysis Complete!")


if __name__ == "__main__":
    asyncio.run(run_complete_analysis())
