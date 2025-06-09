#!/usr/bin/env python3
"""Quick script to update Schwab secret after regeneration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.secrets import SecretManager
import getpass

print("\n=== Schwab Secret Regeneration ===\n")

print("STEP 1: Go to Schwab Developer Portal")
print("   https://developer.schwab.com/")
print("   1. Log in")
print("   2. Go to 'My Apps'")
print("   3. Click on 'Github1'")
print("   4. Click 'Regenerate Secret'")
print("   5. Copy the NEW secret immediately!")
print("\nPress Enter when you've copied the new secret...")
input()

# Get current credentials
manager = SecretManager()
current_creds = manager.get_credentials("schwab")

print("\nSTEP 2: Enter the NEW secret")
print(f"Current App Key: {current_creds['client_id']}")
print("(This should NOT change, only the secret changes)")

new_secret = getpass.getpass("\nPaste the NEW secret here: ")

if not new_secret:
    print("❌ No secret entered!")
    sys.exit(1)

# Confirm before saving
print(f"\nNew secret length: {len(new_secret)} characters")
print("Does this look correct? (y/n): ", end="")
confirm = input().lower()

if confirm != 'y':
    print("Cancelled.")
    sys.exit(0)

# Update the secret
updated_creds = {
    "client_id": current_creds['client_id'],  # Keep same client ID
    "client_secret": new_secret
}

manager.set_credentials("schwab", updated_creds)
print("\n✅ Secret updated successfully!")

print("\nSTEP 3: Test immediately")
print("Run: python schwab_oauth_fixed.py")
print("\nNOTE: If you just regenerated, it might take a few minutes to activate.")