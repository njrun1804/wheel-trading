#!/usr/bin/env python3
"""Debug Schwab OAuth setup issues."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def debug_schwab_setup():
    """Debug common Schwab OAuth issues."""
    print("\n=== Schwab OAuth Debug ===\n")
    
    # Check credentials
    try:
        from src.unity_wheel.secrets.manager import SecretManager
        
        secret_manager = SecretManager()
        creds = secret_manager.get_credentials("schwab", prompt_if_missing=False)
        
        client_id = creds["client_id"]
        client_secret = creds["client_secret"]
        
        print("✅ Credentials found:")
        print(f"   Client ID: {client_id}")
        print(f"   Client ID length: {len(client_id)} characters")
        print(f"   Client Secret: {'*' * len(client_secret)} ({len(client_secret)} chars)")
        
        # Check format
        if not client_id.startswith('OB'):
            print("⚠️  WARNING: Client ID should start with 'OB' for Schwab")
        
        if len(client_id) != 32:
            print("⚠️  WARNING: Client ID should be 32 characters long")
            
        if len(client_secret) < 16:
            print("⚠️  WARNING: Client Secret seems too short")
            
    except Exception as e:
        print(f"❌ Error retrieving credentials: {e}")
        return
    
    print("\n📋 Common 'invalid_client' causes:")
    print("1. **App not approved**: Your Schwab app is still pending approval")
    print("2. **Wrong environment**: Using sandbox credentials for production API")
    print("3. **Incorrect secret**: The app secret was regenerated/changed")
    print("4. **Redirect URI mismatch**: The registered URI doesn't match exactly")
    
    print("\n🔧 Troubleshooting steps:")
    print("1. **Check Schwab Developer Portal**:")
    print("   - Go to: https://developer.schwab.com/dashboard")
    print("   - Verify your app status is 'Ready for use' (not 'Pending')")
    print("   - Check the redirect URI is exactly: https://127.0.0.1:8182/callback")
    
    print("\n2. **Verify credentials**:")
    print("   - Make sure you're using the Production app (not Sandbox)")
    print("   - If you regenerated your secret, update it in secrets")
    
    print("\n3. **Test with curl**:")
    print("   Try this command to test token exchange directly:")
    print(f"""
   curl -X POST https://api.schwabapi.com/v1/oauth/token \\
     -H "Content-Type: application/x-www-form-urlencoded" \\
     -d "grant_type=authorization_code" \\
     -d "code=YOUR_CODE_HERE" \\
     -d "redirect_uri=https://127.0.0.1:8182/callback" \\
     -d "client_id={client_id}" \\
     -d "client_secret=YOUR_SECRET"
   """)
    
    print("\n4. **Common fixes**:")
    print("   - Wait 24-48 hours after app approval")
    print("   - Try regenerating your app secret in Schwab portal")
    print("   - Ensure you're using the 'Individual' app type")
    
    print("\n📞 If still stuck:")
    print("   - Contact Schwab API Support: https://developer.schwab.com/contact")
    print("   - Check their status page: https://developer.schwab.com/products")


if __name__ == "__main__":
    debug_schwab_setup()