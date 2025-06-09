#!/usr/bin/env python3
"""Quick secret comparison."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.secrets import SecretManager
import getpass

# Get stored credentials
manager = SecretManager()
creds = manager.get_credentials("schwab")

print("\nYour stored App Key:", creds['client_id'])
print("Your stored Secret:", creds['client_secret'])
print("\nNow compare this with what's in the Schwab portal.")
print("Do they match EXACTLY? (every character)")
print("\nIf NOT, enter the correct secret below.")
print("If they DO match, just press Enter to skip.")

new_secret = getpass.getpass("\nPaste correct secret (or Enter to skip): ")

if new_secret:
    print(f"\nUpdating secret ({len(new_secret)} chars)...")
    updated_creds = {
        "client_id": creds['client_id'],
        "client_secret": new_secret
    }
    manager.set_credentials("schwab", updated_creds)
    print("✅ Secret updated!")
    print("\nNow run: python test_new_secret.py")
else:
    print("\nSecrets match but Schwab still doesn't recognize them.")
    print("\nYou need to either:")
    print("1. Create a new app in Schwab Developer Portal")
    print("2. Contact developer@schwab.com for help")