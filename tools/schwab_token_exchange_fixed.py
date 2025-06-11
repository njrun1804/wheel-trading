#!/usr/bin/env python3
"""Fixed Schwab token exchange using Basic Auth for client credentials."""

import asyncio
import base64
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def exchange_callback_for_tokens_fixed(callback_url: str):
    """Exchange OAuth callback URL for access tokens using correct Basic Auth."""
    print("\n=== Schwab Token Exchange (FIXED) ===\n")

    # Get credentials
    try:
        from src.unity_wheel.secrets.manager import SecretManager

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
            return False

        code = params["code"][0]  # Already URL-decoded by parse_qs
        print(f"\n✅ Authorization code found: {code[:30]}...")

        # OAuth configuration
        token_url = "https://api.schwabapi.com/v1/oauth/token"
        redirect_uri = "https://127.0.0.1:8182/callback"

        print(f"\n🔄 Exchanging code for tokens using Basic Auth...")

        # Create Basic Auth header
        auth_string = f"{client_id}:{client_secret}"
        auth_bytes = auth_string.encode("utf-8")
        auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")

        # Exchange code for tokens
        async with aiohttp.ClientSession() as session:
            # Official Schwab library uses Basic Auth for authorization_code grant
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            }

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {auth_b64}",
            }

            print(f"   Using Basic Auth (official method): {client_id}:***")

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
                    response_text = await response.text()
                    print(f"\n❌ Token exchange failed!")
                    print(f"   Status: {response.status}")
                    print(f"   Response: {response_text}")

                    if response.status == 400:
                        try:
                            error_data = json.loads(response_text)
                            if error_data.get("error") == "invalid_grant":
                                print(
                                    "\n💡 The authorization code has expired (codes expire in 5-10 minutes)"
                                )
                                print("   Get a fresh code: python tools/schwab_auth_url.py")
                        except:
                            pass

                    return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python schwab_token_exchange_fixed.py 'CALLBACK_URL'")
        sys.exit(1)

    callback_url = sys.argv[1]
    success = asyncio.run(exchange_callback_for_tokens_fixed(callback_url))

    if not success:
        print("\n🔄 To get a fresh authorization code:")
        print("   python tools/schwab_auth_url.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
