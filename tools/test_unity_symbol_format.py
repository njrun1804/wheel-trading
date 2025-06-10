#!/usr/bin/env python3
"""
Test different Unity symbol formats to find what works with Databento.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient


def test_symbol_formats():
    """Test various Unity symbol formats."""
    client = DatabentoClient()
    eastern = pytz.timezone("US/Eastern")

    # Test date - recent trading day
    test_date = datetime(2025, 6, 6)  # Friday June 6, 2025
    market_close = test_date.replace(hour=15, minute=55, tzinfo=eastern)

    # Different symbol formats to try based on the guide
    symbol_tests = [
        # Format 1: Standard parent symbol
        {
            "symbols": ["U.OPT"],
            "stype_in": "parent",
            "description": "Standard parent symbol (U.OPT)",
        },
        # Format 2: Unity with spaces as shown in instruments table
        {
            "symbols": ["U     *"],  # 5 spaces then wildcard
            "stype_in": "parent",
            "description": "Unity with 5 spaces (U     *)",
        },
        # Format 3: Just U as parent
        {"symbols": ["U"], "stype_in": "parent", "description": "Just U as parent"},
        # Format 4: Specific contract example
        {
            "symbols": ["U     250620C00025000"],  # June 2025 $25 call
            "stype_in": "raw_symbol",
            "description": "Specific contract (June 2025 $25 call)",
        },
        # Format 5: No symbol type specified
        {"symbols": ["U.OPT"], "stype_in": None, "description": "U.OPT with no stype_in"},
    ]

    print("Testing Unity symbol formats with Databento...")
    print("=" * 60)
    print(f"Test date: {test_date.date()}")
    print(f"Time window: {market_close} to {market_close + timedelta(minutes=5)}")
    print("=" * 60)

    for test in symbol_tests:
        print(f"\nTesting: {test['description']}")
        print(f"  Symbols: {test['symbols']}")
        print(f"  stype_in: {test['stype_in']}")

        try:
            # Make the request
            params = {
                "dataset": "OPRA.PILLAR",
                "schema": "cmbp-1",
                "symbols": test["symbols"],
                "start": market_close,
                "end": market_close + timedelta(minutes=5),
                "limit": 10,  # Just get a few records
            }

            if test["stype_in"] is not None:
                params["stype_in"] = test["stype_in"]

            data = client.client.timeseries.get_range(**params)

            # Convert to dataframe
            df = data.to_df()

            if df.empty:
                print("  Result: No data returned")
            else:
                print(f"  Result: SUCCESS! Got {len(df)} records")

                # Show unique symbols
                if "raw_symbol" in df.columns:
                    unique_symbols = df["raw_symbol"].unique()[:5]
                    print(f"  Sample symbols: {list(unique_symbols)}")

                # This format works!
                return test

        except Exception as e:
            error_msg = str(e)
            if "Could not resolve" in error_msg:
                print(f"  Result: Symbol not recognized")
            elif "422" in error_msg:
                print(f"  Result: Invalid request format")
            else:
                print(f"  Result: Error - {error_msg[:100]}")

    print("\n" + "=" * 60)
    print("No working format found. May need to check:")
    print("- Databento subscription includes Unity options")
    print("- Date range has available data")
    print("- API credentials are valid")

    return None


if __name__ == "__main__":
    working_format = test_symbol_formats()

    if working_format:
        print("\n" + "=" * 60)
        print("WORKING FORMAT FOUND!")
        print(f"Use: symbols={working_format['symbols']}, stype_in='{working_format['stype_in']}'")
        print("Update the download script with this format.")
