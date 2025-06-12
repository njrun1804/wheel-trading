import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#!/usr/bin/env python3
"""Quick test of Databento connection for Unity data."""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import databento as db

from unity_wheel.data_providers.databento.client import DatabentoClient
from unity_wheel.secrets.integration import SecretInjector


async def test_connection():
    """Test basic Databento connection and data retrieval."""

    print("🔍 Testing Databento Connection for Unity")
    print("=" * 50)

    # Inject API key
    with SecretInjector(service="databento"):
        client = DatabentoClient()

        # Test 1: Get recent Unity stock data
        print("\n1️⃣ Testing stock data retrieval...")
        try:
            end_date = datetime(2025, 6, 9, tzinfo=timezone.utc)
            start_date = end_date - timedelta(days=5)

            response = client.client.timeseries.get_range(
                dataset="XNYS.PILLAR",
                schema="ohlcv-1d",
                start=start_date,
                end=end_date,
                symbols=["U"],
            )

            count = 0
            for record in response:
                count += 1
                print(
                    f"   Date: {datetime.fromtimestamp(record.ts_event / 1e9, tz=timezone.utc).date()}"
                )
                print(f"   Close: ${float(record.close) / 1e9:.2f}")
                print(f"   Volume: {record.volume:,}")
                if count >= 2:  # Just show first 2 records
                    break

            print(f"✅ Stock data working - found {count} records")

        except Exception as e:
            print(f"❌ Stock data failed: {e}")

        # Test 2: Get Unity option definitions
        print("\n2️⃣ Testing options definitions...")
        try:
            # Try multiple approaches
            # First, get raw definitions using wildcard
            print("   Trying U* symbol pattern...")

            end = datetime(2025, 6, 9, tzinfo=timezone.utc)
            start = end - timedelta(days=1)

            # Try different approaches
            # Approach 1: Use U.OPT format
            try:
                response = client.client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema="definition",
                    start=start,
                    end=end,
                    symbols=["U.OPT"],  # Correct parent format
                    stype_in=db.SType.PARENT,
                )

                definitions = []
                for record in response:
                    definitions.append(record)
                    if len(definitions) >= 10:
                        break

                print(f"   With U.OPT: Found {len(definitions)} definitions")

            except Exception as e1:
                print(f"   U.OPT failed: {e1}")

                # Approach 2: Try raw symbol without stype
                if not definitions:
                    print("   Trying raw symbols...")
                    response = client.client.timeseries.get_range(
                        dataset="OPRA.PILLAR",
                        schema="definition",
                        start=start,
                        end=end,
                        symbols=["U"],  # Just U
                    )

                    for record in response:
                        definitions.append(record)
                        if len(definitions) >= 10:
                            break

                    print(f"   With raw U: Found {len(definitions)} definitions")

            if definitions:
                # Show sample raw records
                print(f"   First record fields: {dir(definitions[0])}")
                print(f"   Raw symbol: {getattr(definitions[0], 'raw_symbol', 'N/A')}")
                print(f"   Instrument class: {getattr(definitions[0], 'instrument_class', 'N/A')}")

        except Exception as e:
            print(f"❌ Options definitions failed: {e}")

            # Check if it's a subscription issue
            if "subscription" in str(e).lower():
                print("\n⚠️  Your Databento subscription may not include Unity options.")
                print("   Options: Set DATABENTO_SKIP_VALIDATION=true to skip options collection")

        await client.close()


if __name__ == "__main__":
    asyncio.run(test_connection())
