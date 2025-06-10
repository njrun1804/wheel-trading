#!/usr/bin/env python3
"""Check if Unity has any options at all in Databento."""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import databento as db

from src.unity_wheel.secrets.integration import SecretInjector, get_databento_api_key


async def check_unity_options():
    """Check if Unity has options using different methods."""

    print("🔍 Checking Unity Options Availability")
    print("=" * 50)

    # Get API key
    api_key = get_databento_api_key()
    client = db.Historical(api_key)

    # Test 1: Check metadata
    print("\n1️⃣ Checking dataset metadata...")
    try:
        # Get available date range for OPRA
        metadata = client.metadata.get_dataset_range(dataset="OPRA.PILLAR")
        print(f"   OPRA.PILLAR available from {metadata.start_date} to {metadata.end_date}")
    except Exception as e:
        print(f"   Metadata error: {e}")

    # Test 2: Try to get ANY Unity options in the last month
    print("\n2️⃣ Searching for Unity options in last 30 days...")

    end_date = datetime(2025, 6, 9, tzinfo=timezone.utc)
    start_date = end_date - timedelta(days=30)

    try:
        # Try getting trades to see if ANY Unity options traded
        response = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="trades",
            start=start_date,
            end=end_date,
            symbols=["U.OPT"],
            stype_in=db.SType.PARENT,
            limit=10,  # Just get first 10
        )

        count = 0
        for record in response:
            count += 1
            print(f"   Found trade: {record}")

        if count == 0:
            print("   ❌ No Unity option trades found in last 30 days")
        else:
            print(f"   ✅ Found {count} Unity option trades")

    except Exception as e:
        print(f"   Trade search error: {e}")

    # Test 3: Check for popular tech stocks that should have options
    print("\n3️⃣ Testing known optionable stocks for comparison...")
    test_symbols = ["AAPL", "MSFT", "NVDA", "U"]

    for symbol in test_symbols:
        try:
            response = client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="definition",
                start=end_date - timedelta(days=1),
                end=end_date,
                symbols=[f"{symbol}.OPT"],
                stype_in=db.SType.PARENT,
                limit=1,
            )

            found = False
            for _ in response:
                found = True
                break

            print(f"   {symbol}: {'✅ Has options' if found else '❌ No options found'}")

        except Exception as e:
            print(f"   {symbol}: Error - {e}")

    # Test 4: Try searching by underlying
    print("\n4️⃣ Searching by underlying symbol...")
    try:
        # Some datasets support searching by underlying
        response = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="definition",
            start=end_date - timedelta(days=5),
            end=end_date,
            symbols=["U"],
            stype_in=db.SType.UNDERLYING,
            limit=10,
        )

        count = 0
        for record in response:
            count += 1

        print(f"   Found {count} option definitions via underlying search")

    except Exception as e:
        if "stype_in" in str(e):
            print("   Underlying search not supported in this way")
        else:
            print(f"   Underlying search error: {e}")


if __name__ == "__main__":
    with SecretInjector(service="databento"):
        asyncio.run(check_unity_options())
