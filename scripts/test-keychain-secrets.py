#!/usr/bin/env python3
"""Test macOS Keychain secret backend functionality."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.unity_wheel.secrets.manager import SecretManager, SecretProvider


def test_keychain_backend():
    """Test basic Keychain operations."""
    print("=== Testing macOS Keychain Secret Backend ===\n")
    
    # Force keychain provider
    os.environ["WHEEL_SECRET_PROVIDER"] = "keychain"
    
    try:
        # Initialize manager
        print("1. Initializing Keychain backend...")
        manager = SecretManager()
        
        if manager.provider != SecretProvider.KEYCHAIN:
            print(f"❌ Expected KEYCHAIN provider, got {manager.provider.value}")
            return 1
        print("✓ Keychain backend initialized")
        
        # Test setting a secret
        print("\n2. Testing secret storage...")
        test_key = "test_api_key"
        test_value = "test_value_12345"
        
        manager.backend.set_secret(test_key, test_value)
        print(f"✓ Stored test secret '{test_key}'")
        
        # Test retrieving the secret
        print("\n3. Testing secret retrieval...")
        retrieved = manager.backend.get_secret(test_key)
        
        if retrieved != test_value:
            print(f"❌ Retrieved value doesn't match: expected '{test_value}', got '{retrieved}'")
            return 1
        print("✓ Retrieved secret successfully")
        
        # Test listing secrets
        print("\n4. Testing secret listing...")
        secrets = manager.backend.list_secrets()
        print(f"Available secrets: {secrets}")
        
        # Test deleting the secret
        print("\n5. Testing secret deletion...")
        manager.backend.delete_secret(test_key)
        
        # Verify deletion
        retrieved_after_delete = manager.backend.get_secret(test_key)
        if retrieved_after_delete is not None:
            print("❌ Secret still exists after deletion")
            return 1
        print("✓ Secret deleted successfully")
        
        # Test with actual API keys (read-only)
        print("\n6. Checking for existing API keys...")
        for secret_id in ["databento_api_key", "ofred_api_key"]:
            value = manager.backend.get_secret(secret_id)
            if value:
                print(f"✓ Found {secret_id} in Keychain")
            else:
                print(f"ℹ️  {secret_id} not found in Keychain")
        
        print("\n✅ All tests passed!")
        
        # Show how to access via command line
        print("\nYou can also verify using the command line:")
        print("  security find-generic-password -s DatabentoAPIKey -a $USER")
        print("  security find-generic-password -s FREDAPIKey -a $USER")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up environment
        os.environ.pop("WHEEL_SECRET_PROVIDER", None)


if __name__ == "__main__":
    sys.exit(test_keychain_backend())