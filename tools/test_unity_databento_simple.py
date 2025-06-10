#!/usr/bin/env python3
"""
Simple test to get Unity options from Databento using correct conventions.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient


def test_unity_options():
    """Test getting Unity options the right way."""
    client = DatabentoClient()

    # Test date - a recent trading day
    test_date = datetime(2025, 6, 6)
    start = test_date.replace(hour=0, minute=0, second=0, tzinfo=pytz.UTC)
    end = start + timedelta(seconds=1)

    print("Testing Unity options download...")
    print("=" * 60)

    # Step 1: Get definitions to find Unity options
    print("\n1. Getting option definitions...")
    try:
        data = client.client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="definition",  # lowercase
            start=start,
            end=end,
            symbols=["U"],  # Just U, not U.OPT
            limit=1000,
        )

        df = data.to_df()
        print(f"   Got {len(df)} total records")

        if not df.empty and "raw_symbol" in df.columns:
            # Filter for Unity options (symbol starts with 'U ')
            unity_options = df[df["raw_symbol"].str.startswith("U ", na=False)]
            print(f"   Found {len(unity_options)} Unity option contracts")

            if not unity_options.empty:
                # Show sample
                print("\n   Sample Unity options:")
                for idx, row in unity_options.head(5).iterrows():
                    print(
                        f"     {row['raw_symbol']}: ID={row.get('instrument_id')}, "
                        f"Strike=${row.get('strike_price')}, Exp={row.get('expiration')}"
                    )

                # Get some instrument IDs for testing
                instrument_ids = unity_options["instrument_id"].head(10).tolist()

                # Step 2: Try to get trades for these instruments
                print(f"\n2. Getting trades for {len(instrument_ids)} instruments...")

                market_open = test_date.replace(hour=14, minute=30, tzinfo=pytz.UTC)  # 9:30 AM ET
                market_close = test_date.replace(hour=21, minute=0, tzinfo=pytz.UTC)  # 4:00 PM ET

                trades_data = client.client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema="trades",
                    start=market_open,
                    end=market_close,
                    instrument_ids=instrument_ids,
                    limit=100,
                )

                trades_df = trades_data.to_df()
                print(f"   Got {len(trades_df)} trade records")

                return True
        else:
            print("   No Unity options found in definitions")

    except Exception as e:
        print(f"   Error: {e}")

    return False


if __name__ == "__main__":
    success = test_unity_options()

    if success:
        print("\n✅ SUCCESS! Found Unity options data")
        print("Update the download scripts to use this approach")
    else:
        print("\n❌ Still unable to find Unity options")
        print("May need to check with Databento support")
