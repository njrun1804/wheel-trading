#!/usr/bin/env python3
"""
Demo: Using the unified database for backtesting and analysis.
Shows the performance benefits of pre-calculated features.
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from src.unity_wheel.backtesting import WheelBacktester
from src.unity_wheel.storage import Storage


async def demo_backtesting_performance():
    """Demonstrate backtesting with the optimized unified database."""

    print("=== Wheel Strategy Backtesting Demo ===\n")

    # 1. Show available data
    print("1. Checking available data in unified database:")
    unified_db = Path("data/unified_wheel_trading.duckdb")

    if not unified_db.exists():
        print("❌ Unified database not found. Run tools/etl_unified_database.py first!")
        return

    conn = duckdb.connect(str(unified_db), read_only=True)

    # Check data availability
    stats = conn.execute(
        """
        SELECT
            'Stock Data' as type,
            COUNT(*) as records,
            MIN(date) as start_date,
            MAX(date) as end_date
        FROM market_data
        WHERE symbol = 'U' AND data_type = 'stock'

        UNION ALL

        SELECT
            'Backtest Features',
            COUNT(*),
            MIN(date),
            MAX(date)
        FROM backtest_features
        WHERE symbol = 'U'
    """
    ).fetchall()

    for data_type, records, start, end in stats:
        print(f"  {data_type:<20} {records:>6,} records ({start} to {end})")

    # 2. Quick data quality check
    print("\n2. Data quality check:")
    quality = conn.execute(
        """
        SELECT
            COUNT(*) as total_days,
            COUNT(CASE WHEN volatility_20d IS NOT NULL THEN 1 END) as days_with_vol,
            AVG(volatility_20d) as avg_volatility,
            MIN(stock_price) as min_price,
            MAX(stock_price) as max_price
        FROM backtest_features
        WHERE symbol = 'U'
    """
    ).fetchone()

    total, with_vol, avg_vol, min_price, max_price = quality
    print(f"  Total days: {total:,}")
    print(f"  Days with volatility: {with_vol:,} ({with_vol/total*100:.1f}%)")
    print(f"  Average volatility: {avg_vol:.1%}")
    print(f"  Price range: ${min_price:.2f} - ${max_price:.2f}")

    # 3. Run backtest
    print("\n3. Running backtest:")
    print("  Period: Last 1 year")
    print("  Initial capital: $100,000")
    print("  Strategy: Wheel (selling puts)")

    # Initialize backtester
    storage = Storage()
    await storage.initialize()
    backtester = WheelBacktester(storage)

    # Run backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    start_time = time.time()
    results = await backtester.backtest_strategy(
        symbol="U",
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        contracts_per_trade=10,  # Fixed size for demo
    )
    elapsed = time.time() - start_time

    print(f"\n  ✅ Backtest completed in {elapsed:.2f} seconds")

    # 4. Display results
    print("\n4. Backtest Results:")
    print(f"  Total return: {results.total_return:.1%}")
    print(f"  Annualized return: {results.annualized_return:.1%}")
    print(f"  Sharpe ratio: {results.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {results.max_drawdown:.1%}")
    print(f"  Win rate: {results.win_rate:.1%}")
    print(f"  Total trades: {results.total_trades}")
    print(f"  Assignments: {results.assignments}")
    print(f"  Unity gap events: {results.gap_events}")

    # 5. Parameter optimization demo
    print("\n5. Parameter Optimization Demo:")
    print("  Testing different delta targets...")

    # Test different parameters
    param_results = []
    for delta in [0.20, 0.25, 0.30, 0.35, 0.40]:
        print(f"  Testing delta={delta:.2f}...", end="", flush=True)

        from src.unity_wheel.strategy import WheelParameters

        params = WheelParameters(target_delta=delta)

        start_time = time.time()
        result = await backtester.backtest_strategy(
            symbol="U",
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            parameters=params,
        )
        elapsed = time.time() - start_time

        param_results.append(
            {
                "delta": delta,
                "return": result.annualized_return,
                "sharpe": result.sharpe_ratio,
                "trades": result.total_trades,
                "time": elapsed,
            }
        )
        print(f" done ({elapsed:.1f}s)")

    # Show optimization results
    print("\n  Parameter Optimization Results:")
    print("  Delta | Return | Sharpe | Trades | Time")
    print("  ------|--------|--------|--------|------")
    for r in param_results:
        print(
            f"  {r['delta']:.2f}  | {r['return']:>6.1%} | {r['sharpe']:>6.2f} | {r['trades']:>6} | {r['time']:.1f}s"
        )

    best = max(param_results, key=lambda x: x["sharpe"])
    print(f"\n  🎯 Optimal delta: {best['delta']:.2f} (Sharpe: {best['sharpe']:.2f})")

    # 6. Query performance comparison
    print("\n6. Query Performance Comparison:")

    # Test query on unified DB with pre-calculated features
    print("  Unified DB (with features):", end="", flush=True)
    start_time = time.time()
    df_unified = conn.execute(
        """
        SELECT * FROM backtest_features
        WHERE symbol = 'U'
        AND date >= CURRENT_DATE - INTERVAL '1 year'
    """
    ).df()
    unified_time = time.time() - start_time
    print(f" {len(df_unified)} rows in {unified_time:.3f}s")

    # Show what would happen without pre-calculated features
    print("  Without pre-calc features:", end="", flush=True)
    start_time = time.time()
    df_raw = conn.execute(
        """
        SELECT
            date,
            symbol,
            close,
            -- Calculate volatility on the fly (slow!)
            STDDEV(returns) OVER (
                PARTITION BY symbol
                ORDER BY date
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) * SQRT(252) as volatility_20d
        FROM market_data
        WHERE symbol = 'U'
        AND date >= CURRENT_DATE - INTERVAL '1 year'
        AND data_type = 'stock'
    """
    ).df()
    raw_time = time.time() - start_time
    print(f" {len(df_raw)} rows in {raw_time:.3f}s")

    print(f"\n  ⚡ Speedup: {raw_time/unified_time:.1f}x faster with pre-calculated features!")

    # 7. Sample analysis queries
    print("\n7. Sample Analysis Queries:")

    # High volatility periods
    high_vol = conn.execute(
        """
        SELECT
            date,
            stock_price,
            volatility_20d,
            var_95
        FROM backtest_features
        WHERE symbol = 'U'
        AND volatility_20d > 0.60  -- 60% annualized vol
        ORDER BY date DESC
        LIMIT 5
    """
    ).df()

    print("\n  Recent high volatility periods:")
    print(high_vol.to_string(index=False))

    # Monthly performance
    monthly = conn.execute(
        """
        SELECT
            DATE_TRUNC('month', date) as month,
            LAST(stock_price) - FIRST(stock_price) as price_change,
            AVG(volatility_20d) as avg_volatility,
            MIN(stock_price) as min_price,
            MAX(stock_price) as max_price
        FROM backtest_features
        WHERE symbol = 'U'
        AND date >= CURRENT_DATE - INTERVAL '6 months'
        GROUP BY DATE_TRUNC('month', date)
        ORDER BY month
    """
    ).df()

    print("\n  Monthly summary (last 6 months):")
    print(monthly.to_string(index=False))

    conn.close()

    print("\n✅ Demo completed!")
    print("\nKey Benefits of Unified Database:")
    print("- Pre-calculated features for fast backtesting")
    print("- Optimized indexes for time-series queries")
    print("- Consistent data quality across all analyses")
    print("- Easy parameter optimization and research")


async def demo_live_vs_backtest():
    """Show how the two databases work together."""

    print("\n\n=== Live vs Backtest Database Demo ===\n")

    # Check both databases
    operational_db = Path.home() / ".wheel_trading/cache/wheel_cache.duckdb"
    analytical_db = Path("data/unified_wheel_trading.duckdb")

    print("Database Status:")
    print(f"  Operational DB: {'✅ Found' if operational_db.exists() else '❌ Not found'}")
    print(f"  Analytical DB:  {'✅ Found' if analytical_db.exists() else '❌ Not found'}")

    if analytical_db.exists():
        conn = duckdb.connect(str(analytical_db), read_only=True)

        # Show latest data points
        latest = conn.execute(
            """
            SELECT
                'Stock' as type,
                MAX(date) as latest_date
            FROM market_data
            WHERE symbol = 'U' AND data_type = 'stock'

            UNION ALL

            SELECT
                'Options',
                MAX(date)
            FROM market_data
            WHERE symbol LIKE 'U %' AND data_type = 'option'

            UNION ALL

            SELECT
                'Features',
                MAX(date)
            FROM backtest_features
            WHERE symbol = 'U'
        """
        ).fetchall()

        print("\nData Freshness:")
        for data_type, latest_date in latest:
            days_old = (datetime.now().date() - latest_date).days
            status = "✅ Current" if days_old <= 1 else f"⚠️  {days_old} days old"
            print(f"  {data_type:<10} {latest_date}  {status}")

        conn.close()

    print("\nRecommended Workflow:")
    print("1. Daily: Run ETL to sync operational → analytical")
    print("2. Real-time: Use operational DB for live recommendations")
    print("3. Research: Use analytical DB for backtesting & optimization")
    print("4. Weekly: Full database optimization and cleanup")


if __name__ == "__main__":
    # Run the demos
    asyncio.run(demo_backtesting_performance())
    asyncio.run(demo_live_vs_backtest())
