#!/usr/bin/env python3
"""
Debug API key retrieval issue.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# First check environment variable
env_key = os.environ.get("DATABENTO_API_KEY")
print(f"Environment DATABENTO_API_KEY: {env_key[:10]}..." if env_key else "Not set")
print(f"Environment key format valid: {env_key.startswith('db-') if env_key else 'N/A'}")

# Now check SecretManager
try:
    from src.unity_wheel.secrets.integration import get_databento_api_key
    
    secret_key = get_databento_api_key()
    print(f"\nSecretManager key: {secret_key[:10]}..." if secret_key else "Not found")
    print(f"SecretManager key format valid: {secret_key.startswith('db-') if secret_key else 'N/A'}")
    print(f"SecretManager key length: {len(secret_key) if secret_key else 0}")
    
    # Check for common issues
    if secret_key:
        if secret_key != secret_key.strip():
            print("WARNING: Key has extra whitespace!")
        if '\n' in secret_key or '\r' in secret_key:
            print("WARNING: Key contains newline characters!")
        if not secret_key.startswith('db-'):
            print(f"WARNING: Key doesn't start with 'db-', starts with: {repr(secret_key[:5])}")
            
except Exception as e:
    print(f"Error getting key from SecretManager: {e}")

# Try creating DatabentoClient with env var directly
print("\nTrying DatabentoClient with environment variable:")
try:
    from src.unity_wheel.data_providers.databento import DatabentoClient
    
    if env_key:
        client = DatabentoClient(api_key=env_key)
        print("✓ DatabentoClient created successfully with env var")
    else:
        print("✗ No environment variable to test with")
        
except Exception as e:
    print(f"✗ Failed to create DatabentoClient: {e}")

# Try without any key (uses SecretManager)
print("\nTrying DatabentoClient with SecretManager:")
try:
    client = DatabentoClient()
    print("✓ DatabentoClient created successfully with SecretManager")
except Exception as e:
    print(f"✗ Failed to create DatabentoClient: {e}")