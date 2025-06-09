#!/usr/bin/env python3
"""Verify the stored secret matches what's in Schwab portal."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import getpass

from src.unity_wheel.secrets import SecretManager

print("\n=== Secret Verification ===\n")

# Get stored credentials
manager = SecretManager()
creds = manager.get_credentials("schwab")

print("STEP 1: In the Schwab Developer Portal")
print("Click on the Secret field to view/copy it")
print("\nPress Enter when you've copied it...")
input()

# Get the portal secret
portal_secret = getpass.getpass("\nSTEP 2: Paste the secret from portal here: ")

if not portal_secret:
    print("❌ No secret entered!")
    sys.exit(1)

# Compare
stored_secret = creds["client_secret"]

print(f"\nPortal secret length: {len(portal_secret)} characters")
print(f"Stored secret length: {len(stored_secret)} characters")

if portal_secret == stored_secret:
    print("\n✅ MATCH! The secrets are identical.")
    print("\nThis means the 'invalid_client' error is NOT due to wrong credentials.")
    print("\nPossible issues:")
    print("1. Schwab's API might be having issues")
    print("2. The app might need to be fully recreated")
    print("3. There might be account-level restrictions")
else:
    print("\n❌ MISMATCH! The secrets are different.")
    print("\nUpdating stored secret...")

    updated_creds = {"client_id": creds["client_id"], "client_secret": portal_secret}

    manager.set_credentials("schwab", updated_creds)
    print("✅ Secret updated!")
    print("\nNow run: python test_new_secret.py")
