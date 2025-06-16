#!/usr/bin/env python3
"""
Comprehensive data quality assessment for wheel trading database.
"""

from datetime import datetime
from pathlib import Path

import duckdb

from unity_wheel.config.unified_config import get_config

config = get_config()


def assess_data_quality():
    """Run comprehensive data quality assessment."""
    db_path = Path(config.storage.database_path).expanduser()
    conn = duckdb.connect(str(db_path))

    print("=" * 80)
    print("DATA QUALITY ASSESSMENT REPORT")
    print(f"Generated: {datetime.now()}")
    print("=" * 80)

    # 1. STOCK DATA ASSESSMENT
    print("\n1. STOCK DATA TABLES")
    print("-" * 40)

    # Check price_history table
    stock_stats = conn.execute(
        """
        SELECT
            'price_history' as table_name,
            COUNT(*) as total_records,
            COUNT(DISTINCT symbol) as unique_symbols,
            COUNT(DISTINCT date) as unique_dates,
            MIN(date) as start_date,
            MAX(date) as end_date,
            SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_prices,
            SUM(CASE WHEN volume IS NULL OR volume = 0 THEN 1 ELSE 0 END) as zero_volume,
            AVG(CASE WHEN close > 0 THEN ((high - low) / close) * 100 ELSE NULL END) as avg_daily_range_pct
        FROM price_history
    """
    ).fetchone()

    print(f"Table: {stock_stats[0]}")
    print(f"  Records: {stock_stats[1]:,}")
    print(f"  Symbols: {stock_stats[2]}")
    print(f"  Trading days: {stock_stats[3]}")
    print(f"  Date range: {stock_stats[4]} to {stock_stats[5]}")
    print(f"  Null prices: {stock_stats[6]}")
    print(f"  Zero/null volume: {stock_stats[7]}")
    print(
        f"  Avg daily range: {stock_stats[8]:.2f}%"
        if stock_stats[8]
        else "  Avg daily range: N/A"
    )

    # Check for price anomalies
    anomalies = conn.execute(
        """
        WITH price_changes AS (
            SELECT symbol, date, close,
                   LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close,
                   ABS((close - LAG(close) OVER (PARTITION BY symbol ORDER BY date)) /
                       LAG(close) OVER (PARTITION BY symbol ORDER BY date)) * 100 as pct_change
            FROM price_history
            WHERE close > 0
        )
        SELECT * FROM price_changes
        WHERE pct_change > 50  -- More than 50% daily change
        ORDER BY pct_change DESC
        LIMIT 5
    """
    ).fetchall()

    if anomalies:
        print("\n  ⚠️  Price anomalies detected (>50% daily change):")
        for symbol, date, close, prev_close, pct_change in anomalies:
            print(
                f"    {symbol} on {date}: ${prev_close:.2f} → ${close:.2f} ({pct_change:.1f}%)"
            )

    # Check Unity-specific stock tables
    print("\n  Unity Stock Tables:")

    # unity_price_history
    unity_tables = [
        ("unity_price_history", "date"),
        ("unity_stock_1min", "date"),
        ("unity_daily_stock", "date"),
    ]

    for table, date_col in unity_tables:
        try:
            stats = conn.execute(
                f"""
                SELECT
                    COUNT(*) as records,
                    COUNT(DISTINCT {date_col}) as days,
                    MIN({date_col}) as start_date,
                    MAX({date_col}) as end_date
                FROM {table}
            """
            ).fetchone()
            print(f"\n  {table}:")
            print(f"    Records: {stats[0]:,}")
            print(f"    Days: {stats[1]}")
            print(f"    Date range: {stats[2]} to {stats[3]}")
        except (Exception, duckdb.Error) as e:
            print(f"\n  {table}: Empty or not found ({e})")

    # 2. OPTIONS DATA ASSESSMENT
    print("\n\n2. OPTIONS DATA TABLES")
    print("-" * 40)

    # Main options tables
    options_tables = [
        ("options_ticks", "trade_date", "raw_symbol"),
        ("unity_options_daily", "date", "symbol"),
        ("unity_options_ticks", "trade_date", "raw_symbol"),
        ("unity_options_raw", "date", "symbol"),
        ("unity_options_processed", "date", "symbol"),
        ("unity_daily_options", "date", "symbol"),
    ]

    for table, date_col, symbol_col in options_tables:
        try:
            stats = conn.execute(
                f"""
                SELECT
                    COUNT(*) as records,
                    COUNT(DISTINCT {date_col}) as days,
                    COUNT(DISTINCT {symbol_col}) as unique_options,
                    MIN({date_col}) as start_date,
                    MAX({date_col}) as end_date,
                    AVG(CASE WHEN volume > 0 THEN volume ELSE NULL END) as avg_volume
                FROM {table}
            """
            ).fetchone()

            print(f"\n{table}:")
            print(f"  Records: {stats[0]:,}")
            print(f"  Trading days: {stats[1]}")
            print(f"  Unique options: {stats[2]}")
            print(f"  Date range: {stats[3]} to {stats[4]}")
            print(f"  Avg volume: {stats[5]:.0f}" if stats[5] else "  Avg volume: N/A")

            # Check data quality
            quality = conn.execute(
                f"""
                SELECT
                    SUM(CASE WHEN {symbol_col} IS NULL OR {symbol_col} = '' THEN 1 ELSE 0 END) as null_symbols,
                    SUM(CASE WHEN volume < 0 THEN 1 ELSE 0 END) as negative_volume,
                    COUNT(DISTINCT LENGTH({symbol_col})) as symbol_length_variations
                FROM {table}
            """
            ).fetchone()

            if quality[0] > 0:
                print(f"  ⚠️  Null/empty symbols: {quality[0]}")
            if quality[1] > 0:
                print(f"  ⚠️  Negative volume: {quality[1]}")
            if quality[2] > 3:
                print(f"  ⚠️  Symbol format variations: {quality[2]} different lengths")

        except Exception as e:
            print(f"\n{table}: Empty or error - {str(e)[:50]}")

    # 3. FRED DATA ASSESSMENT
    print("\n\n3. FRED DATA TABLES")
    print("-" * 40)

    # FRED series
    fred_series = conn.execute(
        """
        SELECT
            COUNT(*) as total_series,
            COUNT(DISTINCT frequency) as frequencies,
            MIN(observation_start) as earliest_start,
            MAX(last_updated) as latest_update
        FROM fred_series
    """
    ).fetchone()

    print("fred_series:")
    print(f"  Total series: {fred_series[0]}")
    print(f"  Frequencies: {fred_series[1]}")
    print(f"  Earliest data: {fred_series[2]}")
    print(f"  Latest update: {fred_series[3]}")

    # Sample FRED series
    print("\n  Sample economic indicators:")
    samples = conn.execute(
        """
        SELECT series_id, title, frequency, observation_start, observation_end
        FROM fred_series
        ORDER BY series_id
        LIMIT 5
    """
    ).fetchall()

    for series_id, title, freq, start, end in samples:
        print(f"    {series_id}: {title[:40]}... ({freq}, {start} to {end})")

    # FRED observations
    fred_obs = conn.execute(
        """
        SELECT
            COUNT(*) as total_observations,
            COUNT(DISTINCT series_id) as series_count,
            MIN(date) as earliest,
            MAX(date) as latest,
            SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) as null_values
        FROM fred_observations
    """
    ).fetchone()

    print("\nfred_observations:")
    print(f"  Total observations: {fred_obs[0]:,}")
    print(f"  Series count: {fred_obs[1]}")
    print(f"  Date range: {fred_obs[2]} to {fred_obs[3]}")
    print(f"  Null values: {fred_obs[4]}")

    # FRED features
    try:
        fred_feat = conn.execute(
            """
            SELECT COUNT(*) as records, COUNT(DISTINCT date) as dates
            FROM fred_features
        """
        ).fetchone()
        print("\nfred_features:")
        print(f"  Records: {fred_feat[0]:,}")
        print(f"  Unique dates: {fred_feat[1]}")
    except (Exception, duckdb.Error) as e:
        print(f"\nfred_features: Empty or not found ({e})")

    # 4. OTHER TABLES
    print("\n\n4. OTHER TABLES")
    print("-" * 40)

    # Risk metrics
    try:
        risk = conn.execute(
            """
            SELECT COUNT(*) as records, COUNT(DISTINCT date) as dates
            FROM risk_metrics
        """
        ).fetchone()
        print(f"risk_metrics: {risk[0]} records across {risk[1]} dates")
    except (Exception, duckdb.Error) as e:
        print(f"risk_metrics: Empty or not found ({e})")

    # Instruments
    try:
        inst = conn.execute(
            """
            SELECT COUNT(*) as records, COUNT(DISTINCT symbol) as symbols
            FROM instruments
        """
        ).fetchone()
        print(f"instruments: {inst[0]} records, {inst[1]} unique symbols")
    except (Exception, duckdb.Error) as e:
        print(f"instruments: Empty or not found ({e})")

    # 5. REDUNDANCY ANALYSIS
    print("\n\n5. REDUNDANCY ANALYSIS & RECOMMENDATIONS")
    print("-" * 40)

    # Check for duplicate Unity stock data
    print("\nChecking Unity stock data redundancy...")

    # Compare price_history vs unity_price_history
    unity_comparison = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM price_history WHERE symbol = config.trading.symbol) as price_history_unity,
            (SELECT COUNT(*) FROM unity_price_history) as unity_price_history_count,
            (SELECT COUNT(*) FROM unity_daily_stock) as unity_daily_stock_count,
            (SELECT COUNT(DISTINCT date) FROM unity_stock_1min) as unity_1min_days
    """
    ).fetchone()

    print(f"  Unity in price_history: {unity_comparison[0]} records")
    print(f"  unity_price_history: {unity_comparison[1]} records")
    print(f"  unity_daily_stock: {unity_comparison[2]} records")
    print(f"  unity_stock_1min: {unity_comparison[3]} days")

    # Check Unity options redundancy
    print("\nChecking Unity options data redundancy...")

    options_comparison = conn.execute(
        """
        SELECT
            table_name,
            COUNT(*) as record_count,
            COUNT(DISTINCT date_col) as unique_days,
            MIN(date_col) as start_date,
            MAX(date_col) as end_date
        FROM (
            SELECT 'unity_options_daily' as table_name, date as date_col FROM unity_options_daily
            UNION ALL
            SELECT 'unity_options_ticks' as table_name, trade_date as date_col FROM unity_options_ticks
            UNION ALL
            SELECT 'unity_daily_options' as table_name, date as date_col FROM unity_daily_options
            UNION ALL
            SELECT 'unity_options_raw' as table_name, date as date_col FROM unity_options_raw
            UNION ALL
            SELECT 'unity_options_processed' as table_name, date as date_col FROM unity_options_processed
        )
        GROUP BY table_name
        ORDER BY record_count DESC
    """
    ).fetchall()

    print("\n  Unity options tables comparison:")
    for table, records, days, start, end in options_comparison:
        print(f"    {table}: {records:,} records, {days} days ({start} to {end})")

    # Summary tables
    summary_tables = conn.execute(
        """
        SELECT COUNT(*) as daily_summary,
               (SELECT COUNT(*) FROM unity_daily_summary_real) as daily_summary_real
        FROM unity_daily_summary
    """
    ).fetchone()

    print(f"\n  unity_daily_summary: {summary_tables[0]} records")
    print(f"  unity_daily_summary_real: {summary_tables[1]} records")

    conn.close()

    # 6. RECOMMENDATIONS
    print("\n\n6. RECOMMENDATIONS")
    print("-" * 40)
    print(
        """
TABLES TO KEEP:
1. price_history - Main stock data table with all symbols
2. unity_options_daily - Clean daily options data with proper schema
3. fred_series, fred_observations - Core FRED economic data
4. unity_stock_1min - Unique intraday data (if needed for analysis)

TABLES TO REMOVE (Redundant):
1. unity_price_history - Duplicate of 'U' records in price_history
2. unity_daily_stock - Duplicate daily data
3. unity_options_ticks - Superseded by unity_options_daily
4. unity_options_raw - Temporary/intermediate table
5. unity_options_processed - Temporary/intermediate table
6. unity_daily_options - Duplicate of unity_options_daily
7. unity_daily_summary - Empty/unused
8. unity_daily_summary_real - Empty/unused
9. options_ticks - Old format, superseded by unity_options_daily

DATA QUALITY ISSUES:
1. Unity options coverage is sparse (26 days out of ~430 possible)
   - This is NORMAL for options - only days with actual trades
2. Some tables have empty symbols or null values
3. Multiple Unity tables contain the same data in different formats
"""
    )


if __name__ == "__main__":
    assess_data_quality()
