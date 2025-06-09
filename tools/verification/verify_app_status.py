#!/usr/bin/env python3
"""Verify the app credentials and status."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.secrets import SecretManager

print("\n=== Schwab App Verification ===\n")

# Get stored credentials
manager = SecretManager()
creds = manager.get_credentials("schwab")

print("WHAT TO CHECK IN SCHWAB DEVELOPER PORTAL:\n")

print("1. Verify these EXACTLY match:")
print(f"   App Key: {creds['client_id']}")
print(f"   Secret: {creds['client_secret']}")
print("   (Copy from portal and compare character by character)")

print("\n2. Check App Status:")
print("   - Is it still 'Ready For Use'?")
print("   - Not 'Inactive' or 'Suspended'?")

print("\n3. Check App Type:")
print("   - Is this a 'Personal' app?")
print("   - Is it linked to YOUR Schwab account?")

print("\n4. API Products:")
print("   - Is 'Accounts and Trading Production' enabled?")
print("   - Are there any warnings or messages?")

print("\n5. Try Regenerating Secret:")
print("   - In the portal, click 'Regenerate Secret'")
print("   - Copy the NEW secret")
print("   - Update it here with: python scripts/setup-secrets.py")

print("\n6. Account Issues:")
print("   - Is your Schwab brokerage account active?")
print("   - Any restrictions on your account?")
print("   - Try logging into schwab.com normally")

print("\n" + "=" * 50)
print("\nPOSSIBLE ISSUES:")
print("\n1. 🔑 Wrong Credentials")
print("   The most common issue - double-check every character")

print("\n2. 🚫 App Deactivated")
print("   Schwab might have deactivated it for inactivity")

print("\n3. 👤 Account Mismatch")
print("   App might be tied to a different account")

print("\n4. 🔄 Need Fresh Secret")
print("   Sometimes regenerating the secret fixes issues")

print("\n5. 🏦 Account Problem")
print("   Your brokerage account might have an issue")

print("\nNEXT STEPS:")
print("1. Verify credentials character-by-character")
print("2. Try regenerating the secret")
print("3. Contact developer@schwab.com if still failing")
