#!/usr/bin/env python3
"""Verify we have complete market data coverage for the full 3-year period."""

from pathlib import Path

import duckdb

from unity_wheel.config.unified_config import get_config
config = get_config()



def verify_data_coverage():
    """Check data coverage and gaps in our 3-year dataset."""

    db_path = Path("data/unified_wheel_trading.duckdb")
    conn = duckdb.connect(str(db_path), read_only=True)

    print("=== VERIFYING 3-YEAR DATA COVERAGE ===\n")

    # 1. Check stock data coverage
    print("1. Stock Data Coverage:")
    stock_coverage = conn.execute(
        """
        SELECT
            MIN(date) as first_date,
            MAX(date) as last_date,
            COUNT(DISTINCT date) as trading_days,
            DATEDIFF('day', MIN(date), MAX(date)) as calendar_days,
            COUNT(*) as total_records
        FROM backtest_features
        WHERE symbol = config.trading.symbol
    """
    ).fetchone()

    first, last, trading_days, calendar_days, total = stock_coverage
    print(f"  Date range: {first} to {last}")
    print(f"  Calendar days: {calendar_days}")
    print(f"  Trading days with data: {trading_days}")
    print(f"  Expected trading days (~252/year): {int(calendar_days * 252/365)}")
    print(f"  Coverage: {trading_days / (calendar_days * 252/365) * 100:.1f}%")

    # 2. Check for gaps in stock data
    print("\n2. Checking for Data Gaps:")
    gaps = conn.execute(
        """
        WITH daily_dates AS (
            SELECT
                date,
                LAG(date) OVER (ORDER BY date) as prev_date,
                DATEDIFF('day', LAG(date) OVER (ORDER BY date), date) as gap_days
            FROM backtest_features
            WHERE symbol = config.trading.symbol
            ORDER BY date
        )
        SELECT
            prev_date,
            date,
            gap_days
        FROM daily_dates
        WHERE gap_days > 5  -- More than a week gap (accounting for weekends)
        ORDER BY gap_days DESC
    """
    ).fetchall()

    if gaps:
        print("  Found significant gaps:")
        for prev, curr, gap in gaps[:5]:
            print(f"    {prev} to {curr}: {gap} days")
    else:
        print("  ✅ No significant gaps found")

    # 3. Check data quality by year
    print("\n3. Data Quality by Year:")
    yearly = conn.execute(
        """
        SELECT
            EXTRACT(YEAR FROM date) as year,
            COUNT(*) as days,
            AVG(CASE WHEN returns IS NOT NULL THEN 1 ELSE 0 END) as pct_with_returns,
            AVG(CASE WHEN volatility_20d IS NOT NULL THEN 1 ELSE 0 END) as pct_with_vol,
            AVG(volatility_20d) as avg_volatility
        FROM backtest_features
        WHERE symbol = config.trading.symbol
        GROUP BY EXTRACT(YEAR FROM date)
        ORDER BY year
    """
    ).fetchall()

    print("  Year | Days | Returns | Vol Data | Avg Vol")
    print("  -----|------|---------|----------|--------")
    for year, days, ret_pct, vol_pct, avg_vol in yearly:
        print(f"  {int(year)} | {days:>4} | {ret_pct:>6.1%} | {vol_pct:>7.1%} | {avg_vol:>6.1%}")

    # 4. Check option data coverage
    print("\n4. Option Data Coverage:")
    option_coverage = conn.execute(
        """
        SELECT
            MIN(date) as first_date,
            MAX(date) as last_date,
            COUNT(DISTINCT date) as days_with_options,
            COUNT(DISTINCT om.expiration) as unique_expirations,
            AVG(strikes_per_day) as avg_strikes_per_day
        FROM (
            SELECT
                md.date,
                COUNT(DISTINCT om.strike) as strikes_per_day
            FROM market_data md
            JOIN options_metadata om ON md.symbol = om.symbol
            WHERE om.underlying = 'U'
            AND om.option_type = 'P'
            AND md.close > 0
            GROUP BY md.date
        ) daily_strikes,
        market_data md
        JOIN options_metadata om ON md.symbol = om.symbol
        WHERE om.underlying = 'U'
        AND om.option_type = 'P'
    """
    ).fetchone()

    opt_first, opt_last, opt_days, opt_exp, avg_strikes = option_coverage
    print(f"  Date range: {opt_first} to {opt_last}")
    print(f"  Days with option data: {opt_days}")
    print(f"  Unique expirations: {opt_exp}")
    print(f"  Average strikes per day: {avg_strikes:.1f}")

    # 5. Check regime distribution over full period
    print("\n5. Volatility Regime Distribution (Full 3 Years):")
    regimes = conn.execute(
        """
        SELECT
            CASE
                WHEN volatility_20d < 0.40 THEN 'Low (<40%)'
                WHEN volatility_20d < 0.70 THEN 'Medium (40-70%)'
                WHEN volatility_20d < 1.00 THEN 'High (70-100%)'
                ELSE 'Extreme (>100%)'
            END as regime,
            COUNT(*) as days,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
        FROM backtest_features
        WHERE symbol = config.trading.symbol
        AND volatility_20d IS NOT NULL
        GROUP BY regime
        ORDER BY
            CASE regime
                WHEN 'Low (<40%)' THEN 1
                WHEN 'Medium (40-70%)' THEN 2
                WHEN 'High (70-100%)' THEN 3
                ELSE 4
            END
    """
    ).fetchall()

    print("  Regime          | Days | Percentage")
    print("  ----------------|------|------------")
    for regime, days, pct in regimes:
        print(f"  {regime:<15} | {days:>4} | {pct:>5.1f}%")

    # 6. Summary
    print("\n6. SUMMARY:")
    years_of_data = calendar_days / 365.25
    print(f"  ✅ Total data period: {years_of_data:.1f} years")
    print(f"  ✅ Stock data: {trading_days} trading days")
    print(f"  ✅ Option data: Available from {opt_first}")

    if trading_days / (calendar_days * 252 / 365) > 0.95:
        print("  ✅ Excellent data coverage (>95%)")
    else:
        print(
            f"  ⚠️  Some gaps in data ({100 - trading_days / (calendar_days * 252/365) * 100:.1f}% missing)"
        )

    conn.close()


if __name__ == "__main__":
    verify_data_coverage()
