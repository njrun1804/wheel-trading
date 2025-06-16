#!/usr/bin/env python3
"""Debug Unity options - find the correct symbol format."""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import databento as db

from src.unity_wheel.secrets.integration import get_databento_api_key


def debug_unity():
    """Debug Unity options access."""

    print("🔍 Debugging Unity Options")
    print("=" * 50)

    api_key = get_databento_api_key()
    client = db.Historical(api_key)

    # Use a date we KNOW had options - let's try early 2025
    # Options expire on 3rd Friday of each month
    test_date = datetime(2025, 1, 17)  # January 2025 expiration

    print(f"\n1️⃣ Testing with known expiration date: {test_date.date()}")

    # Method 1: Try getting MBP data for a specific Unity option
    # Unity Jan 17 2025 $25 Put would be something like: U     250117P00025000
    print("\n   Trying specific option symbol format...")

    # The OCC symbol format is: ROOT + YYMMDD + C/P + 8-digit strike
    # Unity with 5 spaces, then date, then type, then strike * 1000
    test_symbols = [
        "U     250117P00025000",  # $25 Put
        "U     250117C00025000",  # $25 Call
        "U 250117P00025000",  # Try with 1 space
        "U250117P00025000",  # No spaces
    ]

    for symbol in test_symbols:
        try:
            response = client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="mbp-1",  # Top of book
                start=datetime(2025, 1, 10),
                end=datetime(2025, 1, 11),
                symbols=[symbol],
            )

            count = 0
            for record in response:
                count += 1
                if count == 1:
                    print(f"   ✅ Symbol '{symbol}' works!")
                    print(f"      Bid: ${record.levels[0].bid_px / 1e9:.2f}")
                    print(f"      Ask: ${record.levels[0].ask_px / 1e9:.2f}")
                break

            if count == 0:
                print(f"   ❌ Symbol '{symbol}' - no data")

        except Exception as e:
            print(f"   ❌ Symbol '{symbol}' - error: {str(e)[:50]}")

    # Method 2: Get all definitions for a specific date and search for Unity
    print("\n2️⃣ Searching all option definitions for Unity...")

    try:
        # Get definitions for options expiring on Jan 17, 2025
        response = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="definition",
            start=datetime(2025, 1, 10),
            end=datetime(2025, 1, 11),
            limit=100000,  # Get many records
        )

        unity_options = []
        checked = 0

        for record in response:
            checked += 1
            # Check if this is a Unity option
            if hasattr(record, "raw_symbol") and record.raw_symbol.startswith("U"):
                # Also check underlying field
                if hasattr(record, "underlying") and record.underlying.strip() == "U":
                    unity_options.append(record)

            if checked >= 10000 and not unity_options:  # Stop early if no Unity found
                break

        print(f"   Checked {checked} definitions")
        print(f"   Found {len(unity_options)} Unity options")

        if unity_options:
            # Show first one
            opt = unity_options[0]
            print("\n   Sample Unity option:")
            print(f"   Raw symbol: {opt.raw_symbol}")
            print(f"   Strike: ${opt.strike_price / 1e9:.2f}")
            print(
                f"   Expiration: {datetime.fromtimestamp(opt.expiration / 1e9).date()}"
            )
            print(f"   Type: {'Call' if opt.instrument_class == 'C' else 'Put'}")

    except Exception as e:
        print(f"   Definition search error: {e}")

    # Method 3: Check what Unity stock trades as
    print("\n3️⃣ Verifying Unity stock symbol...")

    try:
        response = client.timeseries.get_range(
            dataset="XNYS.PILLAR",
            schema="trades",
            start=datetime(2025, 1, 10),
            end=datetime(2025, 1, 11),
            symbols=["U"],
            limit=1,
        )

        for record in response:
            print("   ✅ Unity trades as 'U' on NYSE")
            print(f"   Price: ${record.price / 1e9:.2f}")
            break

    except Exception as e:
        print(f"   Stock check error: {e}")


if __name__ == "__main__":
    debug_unity()
