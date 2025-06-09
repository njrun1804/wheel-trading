#!/usr/bin/env python3
"""Test different OAuth methods to find what Schwab actually expects."""

import asyncio
import base64
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.secrets import SecretManager


async def test_oauth_methods():
    """Try different authentication methods."""
    print("\n=== Testing Different OAuth Methods ===\n")

    # Get credentials
    manager = SecretManager()
    creds = manager.get_credentials("schwab")
    client_id = creds["client_id"]
    client_secret = creds["client_secret"]

    # Use a definitely expired code to test auth methods
    test_code = "C0.TEST.CODE"
    token_url = "https://api.schwabapi.com/v1/oauth/token"
    redirect_uri = "https://127.0.0.1:8182/callback"

    print(f"Testing with:")
    print(f"  Client ID: {client_id}")
    print(f"  Secret: {'*' * len(client_secret)} ({len(client_secret)} chars)")
    print(f"  Redirect URI: {redirect_uri}\n")

    async with aiohttp.ClientSession() as session:
        # Method 1: Credentials in body (standard OAuth2)
        print("Method 1: Credentials in POST body")
        data = {
            "grant_type": "authorization_code",
            "code": test_code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        try:
            async with session.post(token_url, data=data) as response:
                text = await response.text()
                print(f"  Status: {response.status}")
                print(f"  Response: {text[:200]}\n")
        except Exception as e:
            print(f"  Error: {e}\n")

        # Method 2: Basic Auth header
        print("Method 2: Basic Auth header")
        auth_string = f"{client_id}:{client_secret}"
        auth_b64 = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data2 = {
            "grant_type": "authorization_code",
            "code": test_code,
            "redirect_uri": redirect_uri,
        }

        try:
            async with session.post(token_url, headers=headers, data=data2) as response:
                text = await response.text()
                print(f"  Status: {response.status}")
                print(f"  Response: {text[:200]}\n")
        except Exception as e:
            print(f"  Error: {e}\n")

        # Method 3: Client ID in body, secret in header
        print("Method 3: Mixed (ID in body, secret in header)")
        secret_b64 = base64.b64encode(f":{client_secret}".encode()).decode()

        headers3 = {
            "Authorization": f"Basic {secret_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data3 = {
            "grant_type": "authorization_code",
            "code": test_code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
        }

        try:
            async with session.post(token_url, headers=headers3, data=data3) as response:
                text = await response.text()
                print(f"  Status: {response.status}")
                print(f"  Response: {text[:200]}\n")
        except Exception as e:
            print(f"  Error: {e}\n")

    print("\nINTERPRETING RESULTS:")
    print("- 'invalid_client' = credentials not recognized")
    print("- 'invalid_grant' or 'invalid_request' = credentials OK, code is bad")
    print("- Look for which method gives 'invalid_grant' instead of 'invalid_client'")


asyncio.run(test_oauth_methods())
