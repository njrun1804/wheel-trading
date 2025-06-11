#!/usr/bin/env python3
"""Migrate secrets from Google Cloud Secrets to macOS Keychain.

This script helps move Databento and FRED API keys from Google Cloud Secrets
to macOS Keychain for easier local access.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.unity_wheel.secrets.manager import SecretManager, SecretProvider


def migrate_to_keychain():
    """Migrate secrets from current provider to macOS Keychain."""
    print("=== Unity Wheel Trading - Migrate Secrets to macOS Keychain ===\n")

    # Check if we're on macOS
    import platform

    if platform.system() != "Darwin":
        print("❌ This script only works on macOS!")
        return 1

    # Check if keyring is installed
    try:
        import keyring
    except ImportError:
        print("❌ keyring library not installed.")
        print("Please run: pip install keyring")
        return 1

    # Get current secret manager (will use existing provider)
    print("Step 1: Reading secrets from current provider...")
    current_manager = SecretManager()
    print(f"Current provider: {current_manager.provider.value}")

    # Check which services are configured
    configured = current_manager.list_configured_services()
    print("\nConfigured services:")
    for service, is_configured in configured.items():
        status = "✓" if is_configured else "✗"
        print(f"  {status} {service}")

    if not any(configured.values()):
        print("\n❌ No secrets found to migrate!")
        return 1

    # Create Keychain manager
    print("\nStep 2: Initializing macOS Keychain...")
    keychain_manager = SecretManager(provider=SecretProvider.KEYCHAIN)

    # Migrate each configured service
    print("\nStep 3: Migrating secrets...")
    migrated = 0

    # Special handling for databento and fred
    if configured.get("databento", False):
        try:
            api_key = current_manager.get_secret("databento_api_key", prompt_if_missing=False)
            keychain_manager.backend.set_secret("databento_api_key", api_key)
            print("  ✓ Migrated Databento API key")
            migrated += 1
        except Exception as e:
            print(f"  ✗ Failed to migrate Databento: {e}")

    if configured.get("ofred", False):
        try:
            api_key = current_manager.get_secret("ofred_api_key", prompt_if_missing=False)
            keychain_manager.backend.set_secret("ofred_api_key", api_key)
            print("  ✓ Migrated FRED API key")
            migrated += 1
        except Exception as e:
            print(f"  ✗ Failed to migrate FRED: {e}")

    # Also migrate Schwab if present
    if configured.get("schwab", False):
        try:
            creds = current_manager.get_credentials("schwab", prompt_if_missing=False)
            keychain_manager.set_credentials("schwab", **creds)
            print("  ✓ Migrated Schwab credentials")
            migrated += 1
        except Exception as e:
            print(f"  ✗ Failed to migrate Schwab: {e}")

    print(f"\n✓ Migration complete! Migrated {migrated} service(s) to macOS Keychain.")

    # Verify migration
    print("\nStep 4: Verifying migration...")
    keychain_configured = keychain_manager.list_configured_services()
    for service, is_configured in keychain_configured.items():
        if is_configured:
            print(f"  ✓ {service} available in Keychain")

    print("\n✅ Success! Your secrets are now stored in macOS Keychain.")
    print("\nYou can verify using the Keychain Access app or by running:")
    print("  security find-generic-password -s DatabentoAPIKey -a $USER -w")
    print("  security find-generic-password -s FREDAPIKey -a $USER -w")

    # Optionally set environment variable to force keychain usage
    print("\nTo ensure the app uses Keychain, you can set:")
    print("  export WHEEL_SECRET_PROVIDER=keychain")

    return 0


if __name__ == "__main__":
    sys.exit(migrate_to_keychain())
