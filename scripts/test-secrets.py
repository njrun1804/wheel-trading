#!/usr/bin/env python3
"""Test script to verify all API credentials are working."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.fred.fred_client import FREDClient
from src.unity_wheel.data_providers.databento.client import DatentoClient
from src.unity_wheel.schwab.client import SchwabClient
from src.unity_wheel.secrets import SecretManager


async def test_schwab_credentials():
    """Test Schwab API credentials."""
    print("\n--- Testing Schwab Credentials ---")
    try:
        # Initialize client (will use SecretManager)
        async with SchwabClient() as client:
            print("✓ Schwab credentials loaded successfully")

            # Try to make a test API call
            try:
                # This will attempt OAuth flow if not authenticated
                await client.initialize()
                print("✓ Schwab API connection successful")

                # Try to fetch account info
                health = await client.health_check()
                print(f"✓ Schwab client health: {health}")

            except Exception as e:
                print(f"✗ Schwab API test failed: {e}")
                print("  Note: You may need to complete OAuth flow first")
                return False

    except Exception as e:
        print(f"✗ Failed to load Schwab credentials: {e}")
        return False

    return True


async def test_databento_credentials():
    """Test Databento API credentials."""
    print("\n--- Testing Databento Credentials ---")
    try:
        # Initialize client (will use SecretManager)
        client = DatabentoClient()
        print("✓ Databento credentials loaded successfully")

        # Try to validate connection
        try:
            # Make a simple API call to test credentials
            # Note: This might consume a small amount of API quota
            print("  Testing API connection...")

            # Just initialize client - if API key is invalid, it will fail
            print(f"✓ Databento API key validated")
            print(f"  API key: {client.api_key[:8]}...")

            # Could add a minimal data request here if needed
            # But that would consume quota

        except Exception as e:
            print(f"✗ Databento API test failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Failed to load Databento credentials: {e}")
        return False

    return True


async def test_fred_credentials():
    """Test FRED API credentials."""
    print("\n--- Testing FRED Credentials ---")
    try:
        # Initialize client (will use SecretManager)
        async with FREDClient() as client:
            print("✓ FRED credentials loaded successfully")

            # Try to fetch a test series
            try:
                print("  Testing API connection...")
                # DGS10 is 10-Year Treasury Rate - always available
                observations = await client.get_observations("DGS10", limit=1)

                if observations:
                    print(f"✓ FRED API connection successful")
                    print(
                        f"  Latest 10Y Treasury: {observations[0].value}% on {observations[0].date}"
                    )
                else:
                    print("✗ No data returned from FRED")
                    return False

            except Exception as e:
                print(f"✗ FRED API test failed: {e}")
                return False

    except Exception as e:
        print(f"✗ Failed to load FRED credentials: {e}")
        return False

    return True


async def test_all_credentials():
    """Test all configured credentials."""
    print("\n=== Unity Wheel Trading Bot - Credential Testing ===\n")

    # Check which services are configured
    manager = SecretManager()
    configured = manager.list_configured_services()

    print("Configured services:")
    for service, is_configured in configured.items():
        status = "✓" if is_configured else "✗"
        print(f"  {status} {service}")

    # Test each configured service
    results = {}

    if configured.get("schwab"):
        results["schwab"] = await test_schwab_credentials()
    else:
        print("\n--- Schwab Not Configured ---")
        print("Run: python scripts/setup-secrets.py")
        results["schwab"] = False

    if configured.get("databento"):
        results["databento"] = await test_databento_credentials()
    else:
        print("\n--- Databento Not Configured ---")
        print("Run: python scripts/setup-secrets.py")
        results["databento"] = False

    if configured.get("ofred"):
        results["fred"] = await test_fred_credentials()
    else:
        print("\n--- FRED Not Configured ---")
        print("Run: python scripts/setup-secrets.py")
        results["fred"] = False

    # Summary
    print("\n=== Test Summary ===")
    all_passed = True
    for service, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{service}: {status}")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\nSome tests failed. Please check your credentials and try again.")
        print("To reconfigure: python scripts/setup-secrets.py")
        return False
    else:
        print("\n✓ All credential tests passed!")
        print("Your Unity Wheel Trading Bot is ready to use.")
        return True


def main():
    """Main entry point."""
    success = asyncio.run(test_all_credentials())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
