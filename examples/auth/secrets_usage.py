#!/usr/bin/env python3
"""Example usage of the new secret management system."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.unity_wheel.secrets import SecretManager, SecretProvider
from src.unity_wheel.secrets.integration import (
    SecretInjector,
    get_databento_api_key,
    get_ofred_api_key,
    migrate_env_to_secrets,
)


async def demo_basic_usage():
    """Demonstrate basic secret management usage."""
    print("\n=== Basic Secret Management Usage ===\n")

    # Initialize SecretManager (auto-detects provider)
    manager = SecretManager()
    print(f"Using provider: {manager.provider.value}")

    # Check which services are configured
    configured = manager.list_configured_services()
    print("\nConfigured services:")
    for service, is_configured in configured.items():
        status = "✓" if is_configured else "✗"
        print(f"  {status} {service}")

    # Get individual secret (will prompt if missing)
    try:
        print("\nRetrieving Databento API key...")
        databento_key = manager.get_secret("databento_api_key", prompt_if_missing=True)
        print(f"✓ Retrieved Databento API key: {databento_key[:8]}...")
    except Exception as e:
        print(f"✗ Failed to get Databento API key: {e}")

    # Get all credentials for a service
    try:
        print("\nRetrieving Schwab credentials...")
        schwab_creds = manager.get_credentials("schwab", prompt_if_missing=True)
        print(f"✓ Retrieved Schwab client_id: {schwab_creds['client_id'][:8]}...")
        print(f"✓ Retrieved Schwab client_secret: {'*' * 8}")
    except Exception as e:
        print(f"✗ Failed to get Schwab credentials: {e}")


async def demo_integration_helpers():
    """Demonstrate integration helper functions."""
    print("\n=== Integration Helper Functions ===\n")

    try:
        # Get Databento API key using helper
        databento_key = get_databento_api_key()
        print(f"✓ Got Databento API key via helper: {databento_key[:8]}...")

        # Get FRED API key using helper
        fred_key = get_ofred_api_key()
        print(f"✓ Got FRED API key via helper: {fred_key[:8]}...")

    except Exception as e:
        print(f"✗ Error using integration helpers: {e}")


async def demo_env_injection():
    """Demonstrate environment variable injection."""
    print("\n=== Environment Variable Injection ===\n")

    # Check initial state
    print("Before injection:")
    print(f"  WHEEL_AUTH__CLIENT_ID: {os.environ.get('WHEEL_AUTH__CLIENT_ID', 'Not set')}")
    print(f"  DATABENTO_API_KEY: {os.environ.get('DATABENTO_API_KEY', 'Not set')}")

    # Use SecretInjector for temporary env vars
    print("\nUsing SecretInjector context manager...")
    with SecretInjector(service="schwab"):
        print("Inside context:")
        print(
            f"  WHEEL_AUTH__CLIENT_ID: {os.environ.get('WHEEL_AUTH__CLIENT_ID', 'Not set')[:8]}..."
        )
        print(f"  WHEEL_AUTH__CLIENT_SECRET: {'*' * 8}")

    print("\nAfter context:")
    print(f"  WHEEL_AUTH__CLIENT_ID: {os.environ.get('WHEEL_AUTH__CLIENT_ID', 'Not set')}")
    print(f"  WHEEL_AUTH__CLIENT_SECRET: {os.environ.get('WHEEL_AUTH__CLIENT_SECRET', 'Not set')}")


async def demo_auth_client_integration():
    """Demonstrate AuthClient integration with SecretManager."""
    print("\n=== AuthClient Integration ===\n")

    try:
        # Import the enhanced AuthClient
        from src.unity_wheel.auth.client_v2 import AuthClient

        # Create client without providing credentials (uses SecretManager)
        print("Creating AuthClient without explicit credentials...")
        async with AuthClient(use_secret_manager=True) as client:
            # Perform health check
            health = await client.health_check()
            print(f"\nAuthClient health status: {health['status']}")
            print(f"Has credentials: {health['has_credentials']}")
            print(f"Has stored token: {health['has_stored_token']}")

            # Show SecretManager status
            if "secret_manager" in health:
                sm_status = health["secret_manager"]
                print(f"\nSecretManager provider: {sm_status.get('provider', 'unknown')}")
                print(f"Schwab configured: {sm_status.get('schwab_configured', False)}")

    except Exception as e:
        print(f"✗ Error with AuthClient integration: {e}")


async def demo_schwab_client_integration():
    """Demonstrate SchwabClient integration with SecretManager."""
    print("\n=== SchwabClient Integration ===\n")

    try:
        from src.unity_wheel.schwab import SchwabClient
        from src.unity_wheel.secrets.integration import SecretInjector

        # Use SecretInjector to provide credentials via environment
        print("Creating SchwabClient with SecretInjector...")
        with SecretInjector(service="schwab"):
            # Get credentials from environment (injected by SecretInjector)
            client_id = os.environ["WHEEL_AUTH__CLIENT_ID"]
            client_secret = os.environ["WHEEL_AUTH__CLIENT_SECRET"]

            async with SchwabClient(client_id, client_secret) as client:
                print("✓ SchwabClient initialized successfully")
                print(f"  Using client_id: {client_id[:8]}...")

    except Exception as e:
        print(f"✗ Error with SchwabClient integration: {e}")


async def demo_provider_specific():
    """Demonstrate provider-specific features."""
    print("\n=== Provider-Specific Features ===\n")

    # Local provider
    print("Local Provider:")
    local_manager = SecretManager(provider=SecretProvider.LOCAL)
    print(f"  Storage location: ~/.wheel_trading/secrets/")
    print(f"  Encryption: Machine-specific key (UID + hostname)")

    # GCP provider (only if configured)
    if os.environ.get("GCP_PROJECT_ID"):
        print("\nGCP Provider:")
        try:
            gcp_manager = SecretManager(provider=SecretProvider.GCP)
            print(f"  Project ID: {os.environ['GCP_PROJECT_ID']}")
            print(f"  Secret format: projects/{os.environ['GCP_PROJECT_ID']}/secrets/{{secret_id}}")
        except Exception as e:
            print(f"  ✗ GCP not available: {e}")

    # Environment provider (read-only)
    print("\nEnvironment Provider:")
    env_manager = SecretManager(provider=SecretProvider.ENVIRONMENT)
    print(f"  Prefix: WHEEL_")
    print(f"  Read-only: Cannot set or delete secrets")


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Unity Wheel Trading Bot - Secret Management Examples")
    print("=" * 60)

    # Check if we should migrate environment variables
    if any(
        var in os.environ
        for var in [
            "WHEEL_AUTH__CLIENT_ID",
            "WHEEL_AUTH__CLIENT_SECRET",
            "DATABENTO_API_KEY",
            "FRED_API_KEY",
            "OFRED_API_KEY",
        ]
    ):
        print("\n✓ Found environment variables that can be migrated to SecretManager")
        response = input("Migrate them now? (y/N): ").strip().lower()
        if response == "y":
            migrate_env_to_secrets()

    # Run demonstrations
    await demo_basic_usage()
    await demo_integration_helpers()
    await demo_env_injection()
    await demo_auth_client_integration()
    await demo_schwab_client_integration()
    await demo_provider_specific()

    print("\n" + "=" * 60)
    print("To set up all credentials interactively, run:")
    print("  python scripts/setup-secrets.py")
    print("\nTo set up Google Cloud Secret Manager, run:")
    print("  python scripts/setup-secrets.py --setup-gcp")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
