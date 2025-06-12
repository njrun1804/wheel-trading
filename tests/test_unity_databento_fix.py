import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#!/usr/bin/env python3
"""Test Unity options data access with the updated Databento client."""

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unity_wheel.data_providers.databento.client import DatabentoClient
from unity_wheel.utils.logging import setup_structured_logging


async def test_unity_options():
    """Test Unity options data retrieval with the updated client."""
    setup_structured_logging()

    print("🧪 Testing Unity options data access...")
    print("=" * 60)

    # Initialize client
    client = DatabentoClient()

    try:
        # Test 1: Get Unity spot price
        print("\n1. Testing Unity spot price retrieval...")
        try:
            spot_price = await client._get_underlying_price("U")
            print(f"   ✅ Unity spot price: ${spot_price.last_price:.2f}")
            print(f"   Bid: ${spot_price.bid:.2f}, Ask: ${spot_price.ask:.2f}")
            print(f"   Timestamp: {spot_price.timestamp}")
        except Exception as e:
            print(f"   ❌ Failed to get Unity spot price: {e}")

        # Test 2: Get Unity option definitions
        print("\n2. Testing Unity option definitions...")

        # Calculate next monthly expiration (3rd Friday)
        today = datetime.now(UTC)

        # Find next month
        if today.day > 15:  # Past mid-month, go to next month
            if today.month == 12:
                next_month = today.replace(year=today.year + 1, month=1, day=1)
            else:
                next_month = today.replace(month=today.month + 1, day=1)
        else:
            next_month = today.replace(day=1)

        # Find first Friday of the month
        days_until_friday = (4 - next_month.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        first_friday = next_month + timedelta(days=days_until_friday)

        # Third Friday
        expiration = first_friday + timedelta(weeks=2)

        print(f"   Looking for options expiring: {expiration.date()}")

        try:
            definitions = await client._get_definitions("U", expiration)

            if definitions:
                print(f"   ✅ Found {len(definitions)} Unity option definitions")

                # Show a few examples
                puts = [d for d in definitions if d.option_type.value == "P"]
                calls = [d for d in definitions if d.option_type.value == "C"]

                print(f"   Puts: {len(puts)}, Calls: {len(calls)}")

                if puts:
                    print("\n   Sample puts:")
                    for put in sorted(puts, key=lambda x: x.strike_price)[:3]:
                        print(f"     Strike: ${put.strike_price}, Symbol: {put.raw_symbol}")

            else:
                print("   ⚠️  No Unity option definitions found")
                print("   This might mean:")
                print("   - Unity doesn't have listed options")
                print("   - The expiration date has no options")
                print("   - Databento subscription doesn't include Unity options")

        except Exception as e:
            print(f"   ❌ Failed to get Unity options: {e}")

        # Test 3: Full option chain retrieval
        print("\n3. Testing full option chain retrieval...")
        try:
            chain = await client.get_option_chain("U", expiration)

            print("   ✅ Option chain retrieved")
            print(f"   Spot price: ${chain.spot_price:.2f}")
            print(f"   Calls: {len(chain.calls)}, Puts: {len(chain.puts)}")

            if chain.puts:
                # Find ~30 delta put
                target_strike = float(chain.spot_price) * 0.95  # Rough approximation
                closest_put = min(
                    chain.puts, key=lambda x: abs(float(x.strike_price) - target_strike)
                )

                print("\n   Example put near 30 delta:")
                print(f"   Strike: ${closest_put.strike_price}")
                print(f"   Bid: ${closest_put.bid_price}, Ask: ${closest_put.ask_price}")
                print(f"   Spread: {closest_put.spread_pct:.1f}%")

        except Exception as e:
            print(f"   ❌ Failed to get option chain: {e}")

        # Test 4: Check symbol format cache
        print("\n4. Checking symbol format cache...")
        if hasattr(client, "_symbol_format_cache") and client._symbol_format_cache:
            print("   ✅ Symbol format cache:")
            for symbol, (formats, stype) in client._symbol_format_cache.items():
                print(f"     {symbol}: {formats} (stype={stype})")
        else:
            print("   ℹ️  No cached symbol formats yet")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await client.close()

    print("\n" + "=" * 60)
    print("✅ Test complete")

    # Provide troubleshooting tips
    print("\n💡 Troubleshooting tips:")
    print("1. If no options found, try setting: export DATABENTO_SKIP_VALIDATION=true")
    print("2. Check if your Databento subscription includes OPRA data")
    print("3. Unity trades on NYSE American (XAMER), not NASDAQ")
    print("4. Use the debug script for more detailed diagnostics:")
    print("   python tools/debug/debug_databento.py")


if __name__ == "__main__":
    asyncio.run(test_unity_options())
