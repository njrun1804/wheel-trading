# macOS Keychain Secrets Guide

This guide explains how to use macOS Keychain to store Databento and FRED API keys for the Unity Wheel Trading Bot.

## Overview

On macOS, the system now automatically detects and uses Keychain for secret storage. This provides:

- **AES encryption** at rest
- **Automatic unlocking** at login
- **User session scoping**
- **No environment variables** needed
- **No Google Cloud dependency** for local development

## Quick Start

### 1. Install Dependencies

```bash
# Install with keychain support
poetry install -E keychain

# Or just the keyring library
pip install keyring
```

### 2. One-Time Setup

#### Option A: Using the provided script
```bash
./scripts/setup-keychain-secrets.sh
```

#### Option B: Manual command line
```bash
# Databento
security add-generic-password \
    -a "$USER" \
    -s "DatabentoAPIKey" \
    -w "your-databento-api-key" \
    -T "" -U

# FRED
security add-generic-password \
    -a "$USER" \
    -s "FREDAPIKey" \
    -w "your-fred-api-key" \
    -T "" -U
```

#### Option C: Python setup
```bash
python scripts/setup-secrets.py
# The script will auto-detect macOS and use Keychain
```

## Migration from Google Cloud Secrets

If you have existing secrets in Google Cloud Secrets:

```bash
python scripts/migrate-secrets-to-keychain.py
```

This will:
1. Read secrets from your current provider
2. Store them in macOS Keychain
3. Verify the migration

## Usage

The application automatically uses Keychain on macOS. No code changes needed!

```python
# This automatically uses Keychain on macOS
from src.unity_wheel.secrets.integration import get_databento_api_key
api_key = get_databento_api_key()
```

## Verification

### Check stored secrets
```bash
# View Databento key (requires password)
security find-generic-password -s DatabentoAPIKey -a $USER -w

# View FRED key
security find-generic-password -s FREDAPIKey -a $USER -w

# List all wheel-trading secrets
security dump-keychain | grep -A5 "DatabentoAPIKey\|FREDAPIKey"
```

### Test the integration
```bash
python scripts/test-keychain-secrets.py
```

## Environment Variables

### Force Keychain usage (optional)
```bash
export WHEEL_SECRET_PROVIDER=keychain
```

### Fallback behavior
The system checks providers in this order:
1. `WHEEL_SECRET_PROVIDER` environment variable
2. macOS Keychain (if on Darwin and keyring installed)
3. Google Cloud Secrets (if configured)
4. Local encrypted storage

## Troubleshooting

### "keyring library not installed"
```bash
poetry install -E keychain
# or
pip install keyring
```

### "Failed to get secret from Keychain"
- Ensure you're logged in to macOS
- Check the secret exists: `security find-generic-password -s DatabentoAPIKey`
- Try re-adding the secret with the setup script

### Keychain prompts for access
- This is normal on first access
- Click "Always Allow" to avoid future prompts
- Or use `-T ""` flag when adding secrets to deny automatic access

## Security Notes

1. **Access Control**: Secrets are tied to your user account
2. **Code Signing**: New applications may prompt for Keychain access
3. **Backup**: Keychain is included in Time Machine backups
4. **iCloud Sync**: Login keychain doesn't sync to iCloud

## CI/CD Considerations

For GitHub Actions or remote CI:
- Store secrets as repository secrets
- The code falls back to environment variables automatically
- No Keychain access needed in CI environments
