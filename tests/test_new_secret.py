#!/usr/bin/env python3
"""Quick test to see if new secret is recognized by Schwab."""

import asyncio
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.secrets import SecretManager


async def test_credentials():
    """Test if Schwab recognizes our credentials."""
    manager = SecretManager()
    creds = manager.get_credentials("schwab")

    print("\n=== Testing New Credentials ===\n")
    print(f"App Key: {creds['client_id']}")
    print(f"Secret: {'*' * len(creds['client_secret'])} ({len(creds['client_secret'])} chars)")

    # Test with a dummy code - we expect "invalid_grant" not "invalid_client"
    token_url = "https://api.schwabapi.com/v1/oauth/token"

    data = {
        "grant_type": "authorization_code",
        "code": "DUMMY_CODE_FOR_TEST",
        "redirect_uri": "https://127.0.0.1:8182/callback",
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(token_url, data=data) as response:
                text = await response.text()

                if "invalid_client" in text:
                    print("\n❌ Credentials NOT recognized by Schwab")
                    print("   Either wrong secret or not activated yet")
                elif "invalid_grant" in text or "invalid_request" in text:
                    print("\n✅ Credentials ARE recognized!")
                    print("   (The code is invalid, but credentials work)")
                else:
                    print(f"\n🤔 Unexpected response: {text[:100]}")

        except Exception as e:
            print(f"\n❌ Error: {e}")


asyncio.run(test_credentials())
