#!/usr/bin/env python3
"""
Test Databento download for a single day to verify connectivity.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytz

from src.unity_wheel.data_providers.databento import DatabentoClient


def test_single_day_download():
    """Test downloading Unity options for a single recent day."""

    print("Testing Databento connectivity...")

    # Initialize client
    client = DatabentoClient()
    print(f"✓ DatabentoClient initialized")

    # Test date - yesterday
    eastern = pytz.timezone("US/Eastern")
    test_date = datetime.now(eastern) - timedelta(days=1)
    test_date = test_date.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = test_date.replace(hour=16, minute=0)

    print(f"\nTesting download for: {test_date.date()}")
    print(f"Time range: {test_date} to {end_time}")

    try:
        # Test 1: Get Unity stock price
        print("\n1. Testing Unity stock price...")
        stock_data = client.client.timeseries.get_range(
            dataset="XNAS.ITCH",
            schema="ohlcv-1m",
            symbols=["U"],
            start=test_date,
            end=end_time,
        )

        # Convert to dataframe to see data
        stock_df = stock_data.to_df()
        if not stock_df.empty:
            print(f"✓ Got {len(stock_df)} stock price records")
            print(f"  Sample: Open={stock_df['open'].iloc[0]}, Close={stock_df['close'].iloc[-1]}")
        else:
            print("✗ No stock data returned")

    except Exception as e:
        print(f"✗ Stock data error: {e}")

    try:
        # Test 2: Get Unity options definitions
        print("\n2. Testing Unity options definitions...")
        definitions = client.client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="definition",
            symbols=["U.OPT"],
            stype_in="parent",
            start=test_date,
            end=end_time,
        )

        def_df = definitions.to_df()
        if not def_df.empty:
            print(f"✓ Got {len(def_df)} option definitions")
            print(
                f"  Unique strikes: {def_df['strike_price'].nunique() if 'strike_price' in def_df else 'N/A'}"
            )
        else:
            print("✗ No definitions returned")

    except Exception as e:
        print(f"✗ Definitions error: {e}")

    try:
        # Test 3: Get Unity options quotes
        print("\n3. Testing Unity options quotes...")
        quotes = client.client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="cmbp-1",
            symbols=["U.OPT"],
            stype_in="parent",
            start=test_date,
            end=end_time,
            limit=1000,  # Limit for testing
        )

        quotes_df = quotes.to_df()
        if not quotes_df.empty:
            print(f"✓ Got {len(quotes_df)} option quote records")
            if "bid_px_01" in quotes_df and "ask_px_01" in quotes_df:
                avg_spread = (quotes_df["ask_px_01"] - quotes_df["bid_px_01"]).mean() / 10000.0
                print(f"  Average spread: ${avg_spread:.3f}")
        else:
            print("✗ No quotes returned")

    except Exception as e:
        print(f"✗ Quotes error: {e}")

    print("\n" + "=" * 50)
    print("Test complete. Check results above.")


if __name__ == "__main__":
    test_single_day_download()
