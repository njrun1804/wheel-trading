#!/usr/bin/env python3
"""Generate Schwab OAuth authorization URL."""

import secrets
import sys
from pathlib import Path
from urllib.parse import urlencode

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_auth_url():
    """Generate Schwab OAuth authorization URL."""
    print("\n=== Schwab OAuth URL Generator ===\n")
    
    # Get credentials from SecretManager
    try:
        from src.unity_wheel.secrets.manager import SecretManager
        
        print("🔍 Retrieving credentials from secrets...")
        secret_manager = SecretManager()
        print(f"   Using {secret_manager.provider.value} provider")
        
        creds = secret_manager.get_credentials("schwab", prompt_if_missing=False)
        client_id = creds["client_id"]
        
        print(f"✅ Found Client ID: {client_id[:8]}...")
        
    except Exception as e:
        print(f"❌ Error retrieving credentials: {e}")
        return None
    
    # OAuth configuration
    auth_url = "https://api.schwabapi.com/v1/oauth/authorize"
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
    
    print("\n" + "="*80)
    print("🔗 AUTHORIZATION URL")
    print("="*80)
    print(full_auth_url)
    print("="*80)
    
    print("\n📋 Instructions:")
    print("1. Copy the URL above and paste it in your browser")
    print("2. Log into your Schwab account")
    print("3. Complete multi-factor authentication if prompted")
    print("4. Click 'Allow' to authorize the application")
    print("5. You'll be redirected to a page that shows a connection error")
    print("6. Copy the ENTIRE URL from your browser's address bar")
    print("7. Use that URL with the token exchange script")
    
    print("\n💡 Next steps:")
    print("   python tools/schwab_token_exchange.py 'PASTE_CALLBACK_URL_HERE'")
    
    return full_auth_url


if __name__ == "__main__":
    generate_auth_url()