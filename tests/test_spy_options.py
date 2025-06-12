#!/usr/bin/env python3
"""Test Databento with SPY options (very liquid, always has data)."""

import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unity_wheel.data_providers.base.storage import DataStorage
from unity_wheel.data_providers.databento import DatabentoClient
from unity_wheel.data_providers.databento.integration import DatabentoIntegration
from unity_wheel.utils import setup_structured_logging


async def test_spy_options():
    """Test with SPY to verify Databento integration works."""
    setup_structured_logging()

    print("🔍 Testing Databento with SPY options...\n")

    client = DatabentoClient()
    storage = DataStorage(local_dir="data/databento")
    integration = DatabentoIntegration(client, storage)

    try:
        # First, test basic SPY trades to confirm connectivity
        print("1. Testing SPY stock data...")
        spy_price = await client._get_underlying_price("SPY")
        print(f"   ✅ SPY last trade: ${float(spy_price.last_price):.2f}")
        print(f"   Timestamp: {spy_price.timestamp}")

        # Now test options for a specific expiration
        print("\n2. Looking for SPY options expirations...")

        # Get next monthly expiration (3rd Friday)
        today = datetime.now()
        target_month = today.month + 1 if today.day > 15 else today.month
        target_year = today.year
        if target_month > 12:
            target_month = 1
            target_year += 1

        # Find 3rd Friday of target month
        first_day = datetime(target_year, target_month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)

        print(f"   Testing expiration: {third_friday.date()}")

        # Get definitions for this expiration
        definitions = await client._get_definitions("SPY", third_friday)
        print(f"   Found {len(definitions)} option definitions")

        if definitions:
            # Show a few examples
            print("\n   Sample options:")
            for i, defn in enumerate(definitions[:5]):
                print(
                    f"   - {defn.raw_symbol}: Strike ${defn.strike_price}, {defn.option_type.value}"
                )

        # Test wheel candidates
        print("\n3. Finding wheel candidates for SPY...")
        candidates = await integration.get_wheel_candidates(
            underlying="SPY",
            target_delta=0.30,
            dte_range=(20, 50),
            min_premium_pct=0.5,  # Lower for SPY since it's less volatile
        )

        print(f"\n   Found {len(candidates)} wheel candidates")

        if candidates:
            print("\n   Top 3 candidates:")
            for i, cand in enumerate(candidates[:3]):
                print(f"\n   {i+1}. Strike ${cand['strike']:.2f}, {cand['dte']} DTE")
                print(f"      Premium: ${cand['mid']:.2f} ({cand['premium_pct']:.1f}%)")
                print(f"      Delta: {cand['delta']:.3f}")
                print(f"      Expected return: {cand['expected_return']:.1f}%")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await client.close()

    print("\n✅ SPY test complete")
    print("\nIf SPY works but Unity doesn't, Unity may not have listed options.")


if __name__ == "__main__":
    asyncio.run(test_spy_options())
