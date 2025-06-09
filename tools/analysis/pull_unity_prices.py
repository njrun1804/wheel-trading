#!/usr/bin/env python3
"""
Simple script to pull Unity historical prices directly from Databento.
No complex abstractions - just get the data we need.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta

import duckdb
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.unity_wheel.databento import DatabentoClient
from src.unity_wheel.secrets import SecretManager
from src.config.loader import get_config

# Constants
config = get_config()
TICKER = config.unity.ticker
REQUIRED_DAYS = 750
DB_PATH = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")


async def main():
    """Fetch Unity historical prices and store in DuckDB."""

    print("📊 Unity Historical Price Fetcher")
    print("=" * 60)

    # Get API key
    try:
        secrets = SecretManager()
        api_key = secrets.get_secret("databento_api_key")
        if not api_key:
            print("❌ No Databento API key found")
            return
    except Exception as e:
        print(f"❌ Error getting API key: {e}")
        return

    print(f"✅ API key found: {api_key[:8]}...")

    # Create database directory
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Connect to DuckDB
    print(f"\n🔄 Connecting to DuckDB at {DB_PATH}")
    conn = duckdb.connect(DB_PATH)

    # Create price history table
    print("📋 Creating price_history table...")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10,2),
            high DECIMAL(10,2),
            low DECIMAL(10,2),
            close DECIMAL(10,2),
            volume BIGINT,
            returns DECIMAL(8,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
        )
    """
    )

    # Check existing data
    existing = conn.execute(
        """
        SELECT
            COUNT(*) as days,
            MIN(date) as start_date,
            MAX(date) as end_date
        FROM price_history
        WHERE symbol = ?
    """,
        [TICKER],
    ).fetchone()

    days, start_date, end_date = existing
    print(f"\n📊 Current data status:")
    print(f"   Symbol: {TICKER}")
    print(f"   Days available: {days}")
    if days > 0:
        print(f"   Date range: {start_date} to {end_date}")

    if days >= REQUIRED_DAYS:
        print(f"\n✅ Already have {days} days of data (need {REQUIRED_DAYS})")

        # Show some statistics
        stats = conn.execute(
            """
            SELECT
                AVG(returns) * 252 as annual_return,
                STDDEV(returns) * SQRT(252) as annual_vol,
                MIN(returns) as worst_day,
                MAX(returns) as best_day
            FROM price_history
            WHERE symbol = ?
        """,
            [TICKER],
        ).fetchone()

        annual_return, annual_vol, worst_day, best_day = stats
        print(f"\n📈 Unity Statistics:")
        print(f"   Annual return: {annual_return*100:.1f}%")
        print(f"   Annual volatility: {annual_vol*100:.1f}%")
        print(f"   Worst day: {worst_day*100:.1f}%")
        print(f"   Best day: {best_day*100:.1f}%")

        conn.close()
        return

    # Initialize Databento client
    print(f"\n🔄 Initializing Databento client...")
    client = DatabentoClient(api_key=api_key)

    # Calculate date range
    # Use Friday if it's a weekend
    end_date = datetime.now()
    if end_date.weekday() >= 5:  # Saturday=5, Sunday=6
        # Go back to Friday
        days_back = end_date.weekday() - 4
        end_date = end_date - timedelta(days=days_back)

    # Add buffer for weekends/holidays
    start_date = end_date - timedelta(days=int(REQUIRED_DAYS * 1.5))

    print(f"\n📈 Fetching data from {start_date.date()} to {end_date.date()}...")
    print("This may take a few minutes due to API rate limits...")

    try:
        # Import databento directly to use their API
        import databento as db

        # Create databento client
        db_client = db.Historical(api_key)

        # Import Schema enum from databento
        from databento import Schema

        # Fetch daily bars
        # Only use valid datasets based on error messages
        datasets_to_try = [
            ("XNYS.PILLAR", Schema.OHLCV_1D),  # NYSE Pillar (valid)
            ("DBEQ.BASIC", Schema.OHLCV_1D),  # Databento Equities (valid)
        ]

        data = None
        for dataset, schema in datasets_to_try:
            try:
                print(f"\n🔄 Trying dataset: {dataset} with schema: {schema}...")

                # Adjust dates based on dataset requirements
                adjusted_end = end_date
                adjusted_start = start_date

                # Special handling for different datasets
                if dataset.startswith("XNYS"):
                    # NYSE datasets have delayed data, try 7 days earlier
                    adjusted_end = end_date - timedelta(days=7)
                    print(f"   Adjusting end date to {adjusted_end.date()} for NYSE dataset")

                if dataset == "DBEQ.BASIC":
                    # DBEQ has delayed data and limited history
                    adjusted_start = max(start_date, end_date - timedelta(days=365))
                    adjusted_end = end_date - timedelta(days=7)
                    print(
                        f"   Adjusting to 1 year of data ending {adjusted_end.date()} for DBEQ.BASIC"
                    )

                # Try different symbol formats
                symbols_to_try = [TICKER, f"{TICKER}.XNYS", f"{TICKER}.XNAS"]

                for symbol in symbols_to_try:
                    try:
                        # Use databento's timeseries API
                        data = db_client.timeseries.get_range(
                            dataset=dataset,
                            symbols=[symbol],
                            schema=schema,
                            start=adjusted_start,
                            end=adjusted_end,
                        )

                        # Check if we got any data
                        first_record = None
                        for record in data:
                            first_record = record
                            break

                        if first_record:
                            print(f"   ✅ Found data with symbol: {symbol}")
                            # Reset iterator to beginning
                            data = db_client.timeseries.get_range(
                                dataset=dataset,
                                symbols=[symbol],
                                schema=schema,
                                start=adjusted_start,
                                end=adjusted_end,
                            )
                            break
                        else:
                            print(f"   ❌ No data for symbol: {symbol}")
                            data = None

                    except Exception as symbol_error:
                        print(f"   ❌ Symbol {symbol} failed: {str(symbol_error)[:50]}...")
                        data = None
                        continue

                # If we got data, break out of dataset loop
                if data is not None:
                    print(f"✅ Success with dataset: {dataset}")
                    break

            except Exception as e:
                error_msg = str(e)
                if "data_end_after_available_end" in error_msg:
                    print(f"❌ {dataset} failed: Data not available up to {end_date.date()}")
                elif "data_start_before_available_start" in error_msg:
                    print(f"❌ {dataset} failed: Data not available from {start_date.date()}")
                elif "dataset_unavailable" in error_msg:
                    print(f"❌ {dataset} failed: Dataset not available with current subscription")
                else:
                    print(f"❌ {dataset} failed: {error_msg[:100]}...")
                continue

        if data is None:
            print("\n❌ Failed to fetch data from any dataset")
            print("\nPossible issues:")
            print("1. Unity (U) might not be available in these datasets")
            print("2. The date range might be too recent")
            print("3. The API key might not have access to these datasets")
            return

        # Convert to records
        records = []
        prev_close = None
        count = 0

        for record in data:
            # Extract fields
            date = pd.to_datetime(record.ts_event, unit="ns").date()
            close = record.close / 1e9  # Databento uses fixed-point

            # Calculate returns
            returns = 0
            if prev_close and prev_close > 0:
                returns = (close - prev_close) / prev_close

            records.append(
                (
                    TICKER,
                    date,
                    record.open / 1e9,
                    record.high / 1e9,
                    record.low / 1e9,
                    close,
                    record.volume,
                    returns,
                )
            )

            prev_close = close
            count += 1

            # Show progress every 100 records
            if count % 100 == 0:
                print(f"   Processed {count} days...")

        print(f"\n✅ Retrieved {len(records)} days of data")

        if records:
            # Insert into database
            print("💾 Storing in database...")
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_history
                (symbol, date, open, high, low, close, volume, returns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                records,
            )

            # Verify final count
            final_days = conn.execute(
                """
                SELECT COUNT(*) FROM price_history WHERE symbol = ?
            """,
                [TICKER],
            ).fetchone()[0]

            print(f"\n✅ Successfully stored {final_days} days of data")

            # Show final statistics
            stats = conn.execute(
                """
                SELECT
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    AVG(returns) * 252 as annual_return,
                    STDDEV(returns) * SQRT(252) as annual_vol
                FROM price_history
                WHERE symbol = ?
            """,
                [TICKER],
            ).fetchone()

            start_date, end_date, annual_return, annual_vol = stats

            print(f"\n📊 Final Data Summary:")
            print(f"   Date range: {start_date} to {end_date}")
            print(f"   Annual return: {annual_return*100:.1f}%")
            print(f"   Annual volatility: {annual_vol*100:.1f}%")

            if final_days >= REQUIRED_DAYS:
                print(f"\n✅ Sufficient data for all risk calculations!")
            elif final_days >= 500:
                print(f"\n⚠️  Have {final_days} days - adequate for most calculations")
            else:
                print(f"\n❌ Only {final_days} days - may need more data")

        else:
            print("❌ No data returned from API")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        conn.close()
        await client.close()

    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
