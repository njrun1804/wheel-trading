#!/usr/bin/env python3
"""Debug Databento connection and data retrieval."""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.unity_wheel.databento import DatabentoClient
from src.unity_wheel.utils import setup_structured_logging, get_logger

logger = get_logger(__name__)


async def debug_databento():
    """Debug why we're getting 0 options."""
    setup_structured_logging()

    print("🔍 Debugging Databento data retrieval...\n")

    client = DatabentoClient()

    try:
        # Test 1: Check if we can get any data
        print("1. Testing basic connectivity...")
        from databento_dbn import Schema

        # Get last trading day (skip weekends)
        today = datetime.now(timezone.utc)

        # Find last Thursday (OPRA data usually available through Thursday)
        # Go back more days to ensure we get valid trading data
        if today.weekday() == 6:  # Sunday
            days_back = 3  # Back to Thursday
        elif today.weekday() == 5:  # Saturday
            days_back = 2  # Back to Thursday
        elif today.weekday() == 0:  # Monday
            days_back = 4  # Back to previous Thursday
        else:
            days_back = 1  # Previous day

        last_trading_day = today - timedelta(days=days_back)

        # Use last trading day for data
        end = last_trading_day.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=1)

        print(f"   Date range: {start.date()} to {end.date()}")

        # Test different symbol formats
        symbols_to_test = [
            "U.OPT",  # Unity options parent symbol
            "U",  # Just Unity
            "UNIT.OPT",  # Alternative ticker
            "SPY.OPT",  # Test with known liquid ETF
        ]

        for symbol in symbols_to_test:
            print(f"\n2. Testing symbol: {symbol}")
            try:
                response = client.client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema=Schema.DEFINITION,
                    start=start,
                    end=end,
                    symbols=[symbol],
                    stype_in="parent",
                    limit=10,  # Just get first 10
                )

                count = 0
                for record in response:
                    count += 1
                    print(
                        f"   Found definition: {record.raw_symbol} strike={record.strike_price} exp={record.expiration}"
                    )
                    if count >= 5:  # Just show first 5
                        break

                if count == 0:
                    print(f"   ❌ No definitions found for {symbol}")
                else:
                    print(f"   ✅ Found {count} definitions for {symbol}")

            except Exception as e:
                print(f"   ❌ Error with {symbol}: {e}")

        # Test 3: Try getting trade data for Unity
        print("\n3. Testing Unity stock trades...")
        try:
            response = client.client.timeseries.get_range(
                dataset="XNAS.BASIC",
                schema=Schema.TRADES,
                start=start,
                end=end,
                symbols=["U"],
                limit=5,
            )

            count = 0
            for trade in response:
                count += 1
                print(f"   Trade: price={trade.price} size={trade.size} time={trade.ts_event}")
                if count >= 3:
                    break

            print(f"   {'✅' if count > 0 else '❌'} Found {count} trades")

        except Exception as e:
            print(f"   ❌ Error getting trades: {e}")

        # Test 4: List available datasets
        print("\n4. Checking available data...")
        try:
            # This would require admin API access
            print("   Note: Full dataset listing requires admin access")
        except Exception as exc:
            logger.warning("Failed to list datasets", exc_info=exc)

    finally:
        await client.close()

    print("\n✅ Debug complete")
    print("\nRecommendations:")
    print("- If U.OPT returns no data, Unity may not have listed options")
    print("- Try using SPY or QQQ for testing as they have very liquid options")
    print("- Check if the date range includes trading days")
    print("- Verify the Databento subscription includes OPRA data")


if __name__ == "__main__":
    asyncio.run(debug_databento())
