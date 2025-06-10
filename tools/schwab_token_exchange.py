#!/usr/bin/env python3
"""Exchange Schwab OAuth callback URL for access tokens."""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def exchange_callback_for_tokens(callback_url: str):
    """Exchange OAuth callback URL for access tokens."""
    print("\n=== Schwab Token Exchange ===\n")

    # Get credentials from SecretManager
    try:
        from src.unity_wheel.secrets.manager import SecretManager

        print("🔍 Retrieving credentials from secrets...")
        secret_manager = SecretManager()

        creds = secret_manager.get_credentials("schwab", prompt_if_missing=False)
        client_id = creds["client_id"]
        client_secret = creds["client_secret"]

        print(f"✅ Found credentials")

    except Exception as e:
        print(f"❌ Error retrieving credentials: {e}")
        return False

    try:
        print(f"\n🔗 Processing callback URL...")
        print(f"   URL: {callback_url[:100]}...")

        # Parse the authorization code
        parsed = urlparse(callback_url)
        params = parse_qs(parsed.query)

        if "error" in params:
            error = params["error"][0]
            error_desc = params.get("error_description", ["Unknown error"])[0]
            print(f"\n❌ OAuth Error: {error}")
            print(f"   Description: {error_desc}")
            return False

        if "code" not in params:
            print("\n❌ No authorization code found in URL")
            print("   Make sure you copied the entire URL including the ?code= part")
            print(f"   URL provided: {callback_url}")
            return False

        code = params["code"][0]
        print(f"\n✅ Authorization code found: {code[:20]}...")

        # OAuth configuration
        token_url = "https://api.schwabapi.com/v1/oauth/token"
        redirect_uri = "https://127.0.0.1:8182/callback"

        print(f"\n🔄 Exchanging code for tokens...")

        # Exchange code for tokens
        async with aiohttp.ClientSession() as session:
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

                if response.status == 200:
                    tokens = await response.json()

                    print("\n✅ Success! Tokens received:")
                    print(f"   Access Token: {tokens['access_token'][:30]}...")
                    print(f"   Refresh Token: {tokens['refresh_token'][:30]}...")
                    print(f"   Expires in: {tokens['expires_in']} seconds")
                    print(f"   Scope: {tokens.get('scope', 'N/A')}")

                    # Save tokens to file for Schwab client
                    token_dir = Path.home() / ".schwab_cache"
                    token_dir.mkdir(exist_ok=True)

                    token_data = {
                        "access_token": tokens["access_token"],
                        "refresh_token": tokens["refresh_token"],
                        "expires_at": (
                            datetime.now() + timedelta(seconds=tokens["expires_in"])
                        ).isoformat(),
                        "scope": tokens.get("scope", ""),
                        "token_type": tokens.get("token_type", "Bearer"),
                    }

                    token_file = token_dir / "token.json"
                    with open(token_file, "w") as f:
                        json.dump(token_data, f, indent=2)

                    print(f"\n💾 Tokens saved to: {token_file}")

                    print("\n" + "=" * 60)
                    print("🎉 OAUTH SETUP COMPLETE!")
                    print("=" * 60)
                    print("\nYou can now run:")
                    print("  python run.py --portfolio 100000")
                    print("  python scripts/test-secrets.py")

                    return True

                else:
                    error_text = await response.text()
                    print(f"\n❌ Token exchange failed!")
                    print(f"   Status: {response.status}")
                    print(f"   Error: {error_text}")

                    if response.status == 400 and "invalid_client" in error_text:
                        print("\n💡 Tips:")
                        print("   1. Verify your Client ID and Secret are correct")
                        print("   2. Make sure your app is fully approved in Schwab portal")
                        print("   3. Check that the redirect URI matches exactly")

                    return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python schwab_token_exchange.py 'CALLBACK_URL'")
        print("\nExample:")
        print(
            "  python schwab_token_exchange.py 'https://127.0.0.1:8182/callback?code=ABC123&state=XYZ'"
        )
        sys.exit(1)

    callback_url = sys.argv[1]

    success = asyncio.run(exchange_callback_for_tokens(callback_url))

    if not success:
        print("\n❓ Need help? Common issues:")
        print("   - Make sure you copied the entire URL including the code parameter")
        print("   - Your Schwab app must be approved and active")
        print("   - The redirect URI must be exactly: https://127.0.0.1:8182/callback")
        sys.exit(1)


if __name__ == "__main__":
    main()
