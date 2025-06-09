#!/usr/bin/env python3
"""Test Databento with direct option symbols."""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import databento as db
from databento_dbn import Schema, SType

from src.unity_wheel.databento import DatentoClient
from src.unity_wheel.utils import setup_structured_logging


async def test_direct_options():
    """Try different approaches to get options data."""
    setup_structured_logging()

    print("🔍 Testing different approaches for options data...\n")

    client = DatentoClient()

    try:
        # Get a valid date range
        end = datetime(2025, 6, 5, 0, 0, 0, tzinfo=timezone.utc)  # Thursday
        start = datetime(2025, 6, 4, 0, 0, 0, tzinfo=timezone.utc)  # Wednesday

        print(f"Using date range: {start.date()} to {end.date()}")

        # Test 1: Try using raw symbol wildcard without .OPT
        print("\n1. Testing raw symbol queries...")
        test_symbols = [
            "U     *",  # Unity with wildcard
            "U*",  # Simple wildcard
            "SPY   *",  # SPY with wildcard
            "SPY*",  # SPY simple wildcard
        ]

        for symbol in test_symbols:
            print(f"\n   Trying: '{symbol}'")
            try:
                response = client.client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema=Schema.DEFINITION,
                    start=start,
                    end=end,
                    symbols=[symbol],
                    limit=5,
                )

                count = 0
                for record in response:
                    count += 1
                    print(f"   ✅ Found: {record.raw_symbol}")

                if count == 0:
                    print(f"   ❌ No results")
                else:
                    print(f"   Total found: {count}")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        # Test 2: Try without stype_in parameter
        print("\n2. Testing without stype_in parameter...")
        try:
            response = client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.DEFINITION,
                start=start,
                end=end,
                symbols=["U"],
                limit=10,
            )

            count = 0
            for record in response:
                if hasattr(record, "raw_symbol") and "U" in record.raw_symbol:
                    count += 1
                    print(f"   Found: {record.raw_symbol}")
                    if count >= 5:
                        break

            print(f"   Total Unity options found: {count}")

        except Exception as e:
            print(f"   Error: {e}")

        # Test 3: Try to get ALL definitions and filter
        print("\n3. Getting all definitions and filtering...")
        try:
            # This might be expensive, so limit it
            response = client.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.DEFINITION,
                start=start,
                end=start + timedelta(minutes=1),  # Just 1 minute of data
                limit=1000,
            )

            unity_options = []
            spy_options = []

            for record in response:
                if hasattr(record, "raw_symbol"):
                    if record.raw_symbol.startswith("U "):
                        unity_options.append(record.raw_symbol)
                    elif record.raw_symbol.startswith("SPY"):
                        spy_options.append(record.raw_symbol)

            print(f"   Unity options found: {len(unity_options)}")
            if unity_options:
                print(f"   Examples: {unity_options[:3]}")

            print(f"   SPY options found: {len(spy_options)}")
            if spy_options:
                print(f"   Examples: {spy_options[:3]}")

        except Exception as e:
            print(f"   Error: {e}")

        # Test 4: Check available datasets
        print("\n4. Testing data availability...")
        datasets = ["OPRA.PILLAR", "XNAS.BASIC", "GLBX.MDP3"]

        for dataset in datasets:
            try:
                # Try to get minimal data to check if dataset is accessible
                response = client.client.timeseries.get_range(
                    dataset=dataset,
                    schema=Schema.TRADES if "BASIC" in dataset else Schema.DEFINITION,
                    start=start,
                    end=start + timedelta(seconds=1),
                    limit=1,
                )

                # Try to consume one record
                for _ in response:
                    print(f"   ✅ {dataset} is accessible")
                    break
                else:
                    print(f"   ⚠️  {dataset} returned no data")

            except Exception as e:
                print(f"   ❌ {dataset}: {str(e)[:100]}")

    finally:
        await client.close()

    print("\n✅ Test complete")
    print("\nConclusions:")
    print("- Check if OPRA.PILLAR subscription is active")
    print("- Unity options should have symbols like 'U     YYMMDDCXXXXX'")
    print("- May need to contact Databento support for correct query format")


if __name__ == "__main__":
    asyncio.run(test_direct_options())
