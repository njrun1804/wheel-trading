#!/usr/bin/env python3
"""Simple Schwab OAuth setup script."""

import asyncio
import os
import secrets
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def setup_schwab_oauth():
    """Complete Schwab OAuth setup process."""
    print("\n=== Schwab OAuth Setup ===\n")
    
    # Get credentials from environment or prompt
    client_id = os.environ.get('SCHWAB_CLIENT_ID')
    client_secret = os.environ.get('SCHWAB_CLIENT_SECRET')
    
    if not client_id:
        print("No SCHWAB_CLIENT_ID found in environment.")
        client_id = input("Enter your Schwab Client ID: ").strip()
    else:
        print(f"Using CLIENT_ID from environment: {client_id[:8]}...")
    
    if not client_secret:
        print("\nNo SCHWAB_CLIENT_SECRET found in environment.")
        client_secret = input("Enter your Schwab Client Secret: ").strip()
    else:
        print("Using CLIENT_SECRET from environment: ***")
    
    # OAuth configuration
    auth_url = "https://api.schwabapi.com/v1/oauth/authorize"
    token_url = "https://api.schwabapi.com/v1/oauth/token"
    redirect_uri = "https://127.0.0.1:8182/callback"
    
    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    
    # Build authorization URL
    auth_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "AccountsRead MarketDataRead",
        "state": state,
    }
    
    full_auth_url = f"{auth_url}?{urlencode(auth_params)}"
    
    print("\n" + "="*60)
    print("STEP 1: AUTHORIZE")
    print("="*60)
    print("\n🔗 Authorization URL:")
    print(full_auth_url)
    print("\n" + "-"*60)
    
    # Try to open browser
    print("\n📌 Opening browser...")
    if not webbrowser.open(full_auth_url):
        print("❌ Could not open browser automatically.")
        print("👉 Please copy the URL above and paste it in your browser.")
    
    print("\n📋 Instructions:")
    print("1. Log into your Schwab account")
    print("2. Complete any multi-factor authentication")
    print("3. Click 'Allow' to authorize the application")
    print("4. You'll be redirected to a page that may show a connection error")
    print("5. ⚠️  IMPORTANT: Copy the ENTIRE URL from your browser's address bar")
    print("   (It should start with https://127.0.0.1:8182/callback?code=...)")
    
    print("\n" + "="*60)
    print("STEP 2: PASTE CALLBACK URL")
    print("="*60)
    
    # Get the callback URL
    callback_url = input("\n📍 Paste the complete callback URL here:\n> ").strip()
    
    try:
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
            return False
        
        code = params["code"][0]
        print(f"\n✅ Authorization code found: {code[:20]}...")
        
        print("\n" + "="*60)
        print("STEP 3: EXCHANGE CODE FOR TOKENS")
        print("="*60)
        
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
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            async with session.post(
                token_url, 
                data=token_data, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    tokens = await response.json()
                    
                    print("\n✅ Success! Tokens received:")
                    print(f"   Access Token: {tokens['access_token'][:30]}...")
                    print(f"   Refresh Token: {tokens['refresh_token'][:30]}...")
                    print(f"   Expires in: {tokens['expires_in']} seconds")
                    print(f"   Scope: {tokens.get('scope', 'N/A')}")
                    
                    # Save tokens to file
                    token_dir = Path.home() / ".schwab_cache"
                    token_dir.mkdir(exist_ok=True)
                    
                    import json
                    from datetime import datetime, timedelta
                    
                    token_data = {
                        "access_token": tokens["access_token"],
                        "refresh_token": tokens["refresh_token"],
                        "expires_at": (datetime.now() + timedelta(seconds=tokens["expires_in"])).isoformat(),
                        "scope": tokens.get("scope", ""),
                        "token_type": tokens.get("token_type", "Bearer"),
                    }
                    
                    token_file = token_dir / "token.json"
                    with open(token_file, "w") as f:
                        json.dump(token_data, f, indent=2)
                    
                    print(f"\n💾 Tokens saved to: {token_file}")
                    
                    # Save credentials if not in environment
                    if not os.environ.get('SCHWAB_CLIENT_ID'):
                        print("\n📝 To avoid entering credentials next time, set environment variables:")
                        print(f"   export SCHWAB_CLIENT_ID='{client_id}'")
                        print(f"   export SCHWAB_CLIENT_SECRET='{client_secret}'")
                    
                    print("\n" + "="*60)
                    print("🎉 OAUTH SETUP COMPLETE!")
                    print("="*60)
                    print("\nYou can now run:")
                    print("  python run.py --portfolio 100000")
                    
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


async def main():
    """Run the OAuth setup."""
    success = await setup_schwab_oauth()
    
    if not success:
        print("\n❓ Need help? Common issues:")
        print("   - Make sure you're using the production app credentials")
        print("   - The redirect URI must be exactly: https://127.0.0.1:8182/callback")
        print("   - Your Schwab app must be approved and active")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())