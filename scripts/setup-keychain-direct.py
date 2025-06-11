#!/usr/bin/env python3
"""Direct setup of API keys in macOS Keychain."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.unity_wheel.secrets.manager import SecretManager, SecretProvider


def main():
    """Set up demo API keys in Keychain for testing."""
    print("=== Setting up API keys in macOS Keychain ===\n")
    
    # Force Keychain provider
    os.environ["WHEEL_SECRET_PROVIDER"] = "keychain"
    
    try:
        # Initialize Keychain manager
        manager = SecretManager(provider=SecretProvider.KEYCHAIN)
        print(f"✓ Initialized {manager.provider.value} provider")
        
        # Set demo API keys (you should replace these with real keys)
        demo_keys = {
            "databento_api_key": "demo_databento_key_12345",
            "ofred_api_key": "demo_fred_key_67890"
        }
        
        # Store each key
        for key_id, value in demo_keys.items():
            try:
                manager.backend.set_secret(key_id, value)
                print(f"✓ Stored {key_id} in Keychain")
            except Exception as e:
                print(f"✗ Failed to store {key_id}: {e}")
        
        # Verify they're stored
        print("\nVerifying stored keys:")
        for key_id in demo_keys:
            try:
                retrieved = manager.backend.get_secret(key_id)
                if retrieved:
                    print(f"✓ {key_id} is available in Keychain")
                else:
                    print(f"✗ {key_id} not found in Keychain")
            except Exception as e:
                print(f"✗ Error retrieving {key_id}: {e}")
        
        print("\n✅ Setup complete!")
        print("\nTo use real API keys, update them using:")
        print("  security add-generic-password -a $USER -s DatabentoAPIKey -w 'your-real-key' -U")
        print("  security add-generic-password -a $USER -s FREDAPIKey -w 'your-real-key' -U")
        
        print("\nOr use the migration script if you have keys in another provider:")
        print("  python scripts/migrate-secrets-to-keychain.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())