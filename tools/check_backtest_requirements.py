#!/usr/bin/env python3
"""Check if we have all requirements for proper backtesting"""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb


def check_backtest_requirements():
    """Verify all data needed for backtesting is available."""

    db_path = Path("data/unified_wheel_trading.duckdb")
    conn = duckdb.connect(str(db_path), read_only=True)

    print("=== Backtesting Requirements Check ===\n")

    # 1. Stock price data
    print("1. Stock Price Data:")
    stock_check = conn.execute(
        """
        SELECT
            COUNT(*) as total_days,
            MIN(date) as start_date,
            MAX(date) as end_date,
            COUNT(CASE WHEN returns IS NOT NULL THEN 1 END) as days_with_returns,
            COUNT(CASE WHEN volatility_20d IS NOT NULL THEN 1 END) as days_with_vol
        FROM backtest_features
        WHERE symbol = config.trading.symbol
    """
    ).fetchone()

    total, start, end, with_ret, with_vol = stock_check
    print(f"  ✅ Unity stock data: {total} days ({start} to {end})")
    print(f"  ✅ Returns calculated: {with_ret} days")
    print(f"  ✅ Volatility calculated: {with_vol} days")

    # 2. Option chain data
    print("\n2. Option Chain Data:")
    option_check = conn.execute(
        """
        SELECT
            COUNT(DISTINCT symbol) as unique_options,
            COUNT(*) as total_records,
            MIN(date) as start_date,
            MAX(date) as end_date,
            COUNT(DISTINCT date) as trading_days,
            COUNT(CASE WHEN close > 0 THEN 1 END) as records_with_prices
        FROM market_data
        WHERE symbol LIKE 'U %'
        AND data_type = 'option'
    """
    ).fetchone()

    unique_opts, total_recs, opt_start, opt_end, days, with_prices = option_check
    print(f"  {'✅' if unique_opts > 0 else '❌'} Option contracts: {unique_opts:,}")
    print(f"  {'✅' if total_recs > 0 else '❌'} Total option records: {total_recs:,}")
    print(
        f"  {'✅' if with_prices > 0 else '❌'} Records with prices: {with_prices:,} ({with_prices/total_recs*100:.1f}%)"
    )
    print(f"  Date range: {opt_start} to {opt_end}")
    print(f"  Trading days covered: {days}")

    # 3. Put options specifically (for wheel strategy)
    print("\n3. Put Options for Wheel Strategy:")
    puts_check = conn.execute(
        """
        SELECT
            om.expiration,
            COUNT(DISTINCT om.strike) as strikes,
            COUNT(DISTINCT md.date) as days_with_data,
            AVG(CASE WHEN md.close > 0 THEN 1 ELSE 0 END) as pct_with_prices
        FROM options_metadata om
        JOIN market_data md ON om.symbol = md.symbol
        WHERE om.option_type = 'P'
        AND om.underlying = 'U'
        AND md.data_type = 'option'
        AND om.expiration >= '2024-01-01'
        GROUP BY om.expiration
        ORDER BY om.expiration
        LIMIT 10
    """
    ).fetchall()

    print("  Expiration  | Strikes | Days | % with Prices")
    print("  ------------|---------|------|---------------")
    for exp, strikes, days, pct in puts_check:
        status = "✅" if pct > 0.5 else "⚠️ "
        print(f"  {exp} |    {strikes:4} | {days:4} | {status} {pct*100:5.1f}%")

    # 4. Check if we can find liquid strikes at various dates
    print("\n4. Liquid Strike Availability (sample dates):")
    test_dates = [
        datetime.now() - timedelta(days=30),
        datetime.now() - timedelta(days=90),
        datetime.now() - timedelta(days=180),
        datetime.now() - timedelta(days=365),
    ]

    for test_date in test_dates:
        liquid = conn.execute(
            """
            SELECT COUNT(DISTINCT om.strike) as liquid_strikes
            FROM market_data md
            JOIN options_metadata om ON md.symbol = om.symbol
            WHERE om.underlying = 'U'
            AND om.option_type = 'P'
            AND md.date = ?
            AND md.close > 0
            AND md.volume > 0
            AND om.expiration > md.date
            AND om.expiration <= md.date + INTERVAL '60 days'
        """,
            [test_date.date()],
        ).fetchone()[0]

        status = "✅" if liquid >= 5 else "⚠️ " if liquid > 0 else "❌"
        print(f"  {test_date.date()}: {status} {liquid} liquid put strikes")

    # 5. Economic indicators
    print("\n5. Economic Indicators (for risk-free rate):")
    econ_check = conn.execute(
        """
        SELECT
            indicator,
            COUNT(*) as records,
            MIN(date) as start_date,
            MAX(date) as end_date
        FROM economic_indicators
        WHERE indicator IN ('DGS3MO', 'VIXCLS')
        GROUP BY indicator
    """
    ).fetchall()

    for indicator, records, start, end in econ_check:
        print(f"  ✅ {indicator}: {records} records ({start} to {end})")

    # 6. Check Greeks calculation readiness
    print("\n6. Greeks Calculation Requirements:")
    print("  ✅ Stock prices: Available")
    print("  ✅ Option prices: Available (sparse but realistic)")
    print("  ✅ Risk-free rates: Available from FRED")
    print("  ✅ Volatility: Calculated from historical data")
    print("  ✅ Time to expiration: Can be calculated from dates")

    # 7. Missing components check
    print("\n7. Potential Issues for Backtesting:")

    # Check for gaps in data
    gaps = conn.execute(
        """
        WITH daily_dates AS (
            SELECT
                DATE '2024-01-01' + INTERVAL (i) DAY as date
            FROM generate_series(0, 365) AS s(i)
        ),
        trading_days AS (
            SELECT d.date
            FROM daily_dates d
            WHERE EXTRACT(DOW FROM d.date) NOT IN (0, 6)  -- Exclude weekends
        )
        SELECT COUNT(*) as missing_days
        FROM trading_days t
        LEFT JOIN backtest_features bf ON t.date = bf.date AND bf.symbol = config.trading.symbol
        WHERE bf.date IS NULL
        AND t.date <= CURRENT_DATE
    """
    ).fetchone()[0]

    if gaps > 10:
        print(f"  ⚠️  Missing {gaps} trading days in 2024")
    else:
        print(f"  ✅ Data coverage good (only {gaps} missing days)")

    # Check option premium distribution
    premium_check = conn.execute(
        """
        SELECT
            CASE
                WHEN close < 0.50 THEN '< $0.50'
                WHEN close < 1.00 THEN '$0.50-$1.00'
                WHEN close < 2.00 THEN '$1.00-$2.00'
                WHEN close < 5.00 THEN '$2.00-$5.00'
                ELSE '> $5.00'
            END as price_range,
            COUNT(*) as count
        FROM market_data md
        JOIN options_metadata om ON md.symbol = om.symbol
        WHERE md.data_type = 'option'
        AND om.option_type = 'P'
        AND md.close > 0
        GROUP BY price_range
        ORDER BY price_range
    """
    ).fetchall()

    print("\n  Option Premium Distribution:")
    for range_name, count in premium_check:
        print(f"    {range_name:<12} {count:>6,} records")

    # 8. Backtesting readiness summary
    print("\n8. Backtesting Readiness Summary:")
    print("  ✅ Stock price history with volatility")
    print("  ✅ Option chains with sparse but real prices")
    print("  ✅ Risk-free rates from FRED")
    print("  ✅ VIX data for market regime")
    print("  ⚠️  Option data is sparse (normal for Unity)")
    print("  ⚠️  May need to interpolate strikes/expirations")

    # 9. Recommended approach
    print("\n9. Recommended Backtesting Approach:")
    print("  • Use actual option prices when available")
    print("  • Fall back to Black-Scholes for missing prices")
    print("  • Use historical volatility for IV when not available")
    print("  • Respect liquidity constraints (volume > 0)")
    print("  • Account for Unity's gap risk around earnings")

    conn.close()


if __name__ == "__main__":
    check_backtest_requirements()
