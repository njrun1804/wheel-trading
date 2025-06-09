#!/usr/bin/env python3
"""Fixed Schwab OAuth with proper client authentication."""

import asyncio
import secrets
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import aiohttp

sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.auth.storage import SecureTokenStorage
from src.unity_wheel.secrets import SecretManager


async def schwab_oauth_fixed():
    """Complete OAuth flow with correct authentication."""
    print("\n=== Schwab OAuth Authorization (Fixed) ===\n")

    # Get credentials
    manager = SecretManager()
    creds = manager.get_credentials("schwab")
    client_id = creds["client_id"]
    client_secret = creds["client_secret"]

    # OAuth URLs
    auth_url = "https://api.schwabapi.com/v1/oauth/authorize"
    token_url = "https://api.schwabapi.com/v1/oauth/token"
    redirect_uri = "https://127.0.0.1:8182/callback"

    # Generate authorization URL
    state = secrets.token_urlsafe(32)
    auth_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "AccountsRead MarketDataRead",
        "state": state,
    }
    full_auth_url = f"{auth_url}?{urlencode(auth_params)}"

    print("Opening authorization URL in browser...")
    print(f"\nAuthorization URL:\n{full_auth_url}\n")

    # Open browser
    webbrowser.open(full_auth_url)

    print("Steps:")
    print("1. Log into your Schwab account")
    print("2. Complete MFA if required")
    print("3. Authorize the application")
    print("4. You'll be redirected to a page that may show an error")
    print("5. Copy the ENTIRE URL from your browser's address bar")
    print("6. Paste it here\n")

    # Get the redirect URL
    redirect_url = input("Paste the complete URL here: ").strip()

    try:
        # Parse the authorization code
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)

        if "code" not in params:
            print(f"\n❌ No authorization code found in URL")
            return False

        code = params["code"][0]
        print(f"\n✅ Authorization code extracted: {code[:20]}...")

        # IMPORTANT: Schwab expects client_id and client_secret in the POST body
        # NOT in Basic Auth header
        print("\nExchanging code for tokens...")

        async with aiohttp.ClientSession() as session:
            # Token exchange with credentials in body
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret,
            }

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            async with session.post(
                token_url, data=token_data, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_text = await response.text()

                if response.status == 200:
                    token_response = await response.json()

                    print("\n✅ Token exchange successful!")
                    print(f"   Access token: {token_response['access_token'][:30]}...")
                    print(f"   Expires in: {token_response['expires_in']} seconds")

                    # Save tokens
                    storage = SecureTokenStorage()
                    storage.save_tokens(
                        access_token=token_response["access_token"],
                        refresh_token=token_response["refresh_token"],
                        expires_in=token_response["expires_in"],
                        scope=token_response.get("scope", ""),
                        token_type=token_response.get("token_type", "Bearer"),
                    )

                    print("\n✅ Tokens saved successfully!")
                    print("\n🎉 OAuth setup complete!")
                    print("\nTest your connection:")
                    print("  python scripts/test-secrets.py")

                    return True
                else:
                    print(f"\n❌ Token exchange failed")
                    print(f"   Status: {response.status}")
                    print(f"   Response: {response_text}")
                    return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def main():
    """Run the fixed OAuth flow."""
    success = await schwab_oauth_fixed()

    if not success:
        print("\n💡 If you keep getting 'invalid_client', check:")
        print("1. Your app secret is correct in SecretManager")
        print("2. The app is fully approved (not pending)")
        print("3. Try regenerating your app secret in Schwab portal")


if __name__ == "__main__":
    asyncio.run(main())
