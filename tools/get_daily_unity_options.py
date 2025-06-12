#!/usr/bin/env python3
"""
Get actual daily Unity options data using different Databento schemas.
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unity_wheel.data_providers.databento import DatabentoClient


def test_daily_options_schemas():
    """Test different schemas to get daily Unity options data."""
    client = DatabentoClient()
    db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
    conn = duckdb.connect(str(db_path))

    # Test with a week of recent data
    test_dates = [
        datetime(2025, 6, 2).date(),  # Monday
        datetime(2025, 6, 3).date(),  # Tuesday
        datetime(2025, 6, 4).date(),  # Wednesday
        datetime(2025, 6, 5).date(),  # Thursday
        datetime(2025, 6, 6).date(),  # Friday
    ]

    print("Testing different Databento schemas for daily Unity options data...")
    print("=" * 70)

    schemas_to_test = [
        ("cmbp-1", "Consolidated Market by Price - End of day snapshots"),
        ("mbp-1", "Market by Price - Best bid/ask throughout day"),
        ("trades", "All trades - can aggregate to daily"),
        ("statistics", "Daily statistics"),
        ("ohlcv-1s", "Second-level OHLCV"),
        ("ohlcv-1m", "Minute-level OHLCV"),
        ("ohlcv-1h", "Hourly OHLCV"),
    ]

    for schema, description in schemas_to_test:
        print(f"\n{schema.upper()} Schema: {description}")
        print("-" * 50)

        daily_results = []

        for test_date in test_dates:
            # Convert to market hours
            if schema in ["mbp-1", "trades"]:
                # Use full trading day for tick data
                start = datetime.combine(test_date, datetime.min.time()).replace(
                    hour=9, minute=30, tzinfo=pytz.timezone("US/Eastern")
                )
                end = datetime.combine(test_date, datetime.min.time()).replace(
                    hour=16, minute=0, tzinfo=pytz.timezone("US/Eastern")
                )
            else:
                # Use UTC midnight for bar data
                start = datetime.combine(test_date, datetime.min.time()).replace(tzinfo=pytz.UTC)
                end = start + timedelta(days=1)

            try:
                data = client.client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema=schema,
                    symbols=["U.OPT"],
                    stype_in="parent",
                    start=start,
                    end=end,
                    limit=1000,  # Limit to avoid huge downloads
                )

                df = data.to_df()

                if not df.empty:
                    # Count Unity options
                    if "symbol" in df.columns:
                        unity_count = len(df[df["symbol"].str.startswith("U")])
                    elif "raw_symbol" in df.columns:
                        unity_count = len(df[df["raw_symbol"].str.startswith("U")])
                    else:
                        unity_count = len(df)

                    daily_results.append((test_date, unity_count))
                    print(f"  {test_date}: {unity_count} Unity options")
                else:
                    print(f"  {test_date}: No data")

            except Exception as e:
                error_msg = str(e)
                if "deprecated" in error_msg.lower():
                    print(f"  {test_date}: Schema deprecated")
                elif "422" in error_msg:
                    print(f"  {test_date}: No data available")
                elif "not_fully_available" in error_msg:
                    print(f"  {test_date}: Schema not available for this date")
                else:
                    print(f"  {test_date}: Error - {error_msg[:50]}")

            # Brief pause between requests
            time.sleep(0.2)

        # Summary for this schema
        if daily_results:
            total_days = len(daily_results)
            avg_options = sum(count for _, count in daily_results) / total_days
            print(
                f"\n  📊 Summary: {total_days}/5 days with data, avg {avg_options:.0f} options per day"
            )

            if total_days >= 4:  # Good coverage
                print(f"  ✅ This schema provides good daily coverage!")

                # Test a longer date range with this schema
                print(f"\n  🔍 Testing {schema} with longer date range...")
                test_longer_range(client, schema)
        else:
            print(f"  ❌ No data available with this schema")

    conn.close()


def test_longer_range(client, schema):
    """Test a schema with a longer date range."""
    # Test last 2 weeks
    end_date = datetime(2025, 6, 6).date()
    start_date = end_date - timedelta(days=14)

    # Convert to appropriate time range
    if schema in ["mbp-1", "trades"]:
        start = datetime.combine(start_date, datetime.min.time()).replace(
            hour=9, minute=30, tzinfo=pytz.timezone("US/Eastern")
        )
        end = datetime.combine(end_date, datetime.min.time()).replace(
            hour=16, minute=0, tzinfo=pytz.timezone("US/Eastern")
        )
    else:
        start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=pytz.UTC)
        end = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).replace(
            tzinfo=pytz.UTC
        )

    try:
        data = client.client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema=schema,
            symbols=["U.OPT"],
            stype_in="parent",
            start=start,
            end=end,
            limit=10000,
        )

        df = data.to_df()

        if not df.empty:
            # Analyze by date
            if "ts_event" in df.columns:
                df["date"] = df["ts_event"].dt.date
            else:
                df["date"] = start_date

            daily_counts = df["date"].value_counts().sort_index()
            trading_days = len(daily_counts)

            print(f"    📅 {trading_days} trading days with data over 2 weeks")
            print(f"    📊 Total records: {len(df):,}")
            print(f"    💹 Avg records per day: {len(df)/trading_days:.0f}")

            if trading_days >= 8:  # Should have ~10 trading days in 2 weeks
                print(f"    ✅ EXCELLENT! This schema gives us daily options data!")
                return True
        else:
            print(f"    ❌ No data in longer range")

    except Exception as e:
        print(f"    ❌ Error in longer range: {str(e)[:50]}")

    return False


if __name__ == "__main__":
    test_daily_options_schemas()
