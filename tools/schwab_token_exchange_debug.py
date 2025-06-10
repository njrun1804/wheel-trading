#!/usr/bin/env python3
"""Debug version of Schwab token exchange with detailed logging."""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def exchange_callback_for_tokens_debug(callback_url: str):
    """Exchange OAuth callback URL for access tokens with debug info."""
    print("\n=== Schwab Token Exchange (DEBUG) ===\n")

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
        print(f"   Raw URL: {callback_url}")

        # Parse the authorization code
        parsed = urlparse(callback_url)
        params = parse_qs(parsed.query)

        print(f"\n📊 Parsed parameters:")
        for key, value in params.items():
            if key == "code":
                print(f"   {key}: {value[0][:30]}... (length: {len(value[0])})")
            else:
                print(f"   {key}: {value[0]}")

        if "error" in params:
            error = params["error"][0]
            error_desc = params.get("error_description", ["Unknown error"])[0]
            print(f"\n❌ OAuth Error: {error}")
            print(f"   Description: {error_desc}")
            return False

        if "code" not in params:
            print("\n❌ No authorization code found in URL")
            return False

        code = params["code"][0]  # parse_qs automatically URL-decodes
        print(f"\n✅ Authorization code (decoded): {code[:30]}...")
        print(f"   Code length: {len(code)} characters")
        print(f"   Code ends with: ...{code[-10:]}")

        # Check if code looks valid
        if not code.startswith("C0."):
            print("⚠️  WARNING: Code doesn't start with 'C0.' - might be invalid")

        # OAuth configuration
        token_url = "https://api.schwabapi.com/v1/oauth/token"
        redirect_uri = "https://127.0.0.1:8182/callback"

        print(f"\n🔄 Preparing token exchange...")
        print(f"   Token URL: {token_url}")
        print(f"   Redirect URI: {redirect_uri}")
        print(f"   Client ID: {client_id}")
        print(f"   Client Secret: {'*' * len(client_secret)}")

        # Exchange code for tokens
        async with aiohttp.ClientSession() as session:
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret,
            }

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }

            print(f"\n📤 Sending request...")
            print(f"   Data keys: {list(token_data.keys())}")

            async with session.post(
                token_url, data=token_data, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:

                response_text = await response.text()

                print(f"\n📥 Response received:")
                print(f"   Status: {response.status}")
                print(f"   Headers: {dict(response.headers)}")
                print(f"   Body: {response_text}")

                if response.status == 200:
                    tokens = await response.json()

                    print("\n✅ Success! Tokens received:")
                    print(f"   Access Token: {tokens['access_token'][:30]}...")
                    print(f"   Refresh Token: {tokens['refresh_token'][:30]}...")
                    print(f"   Expires in: {tokens['expires_in']} seconds")

                    # Save tokens
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
                    print("\n🎉 OAuth setup complete!")

                    return True

                else:
                    print(f"\n❌ Token exchange failed!")

                    if response.status == 400:
                        try:
                            error_data = await response.json()
                            if error_data.get("error") == "invalid_grant":
                                print("💡 DIAGNOSIS: Authorization code has expired!")
                                print("   Authorization codes expire in 5-10 minutes.")
                                print("   You need to get a fresh authorization code.")
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
        print("Usage: python schwab_token_exchange_debug.py 'CALLBACK_URL'")
        sys.exit(1)

    callback_url = sys.argv[1]
    success = asyncio.run(exchange_callback_for_tokens_debug(callback_url))

    if not success:
        print("\n🔄 To get a fresh authorization code:")
        print("   python tools/schwab_auth_url.py")
        print("   Then complete the OAuth flow again")
        sys.exit(1)


if __name__ == "__main__":
    main()
