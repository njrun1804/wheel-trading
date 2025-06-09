# Secret Management Setup Guide

This guide explains how to set up and use the Unity Wheel Trading Bot's secret management system, which supports both local encrypted storage and Google Cloud Secret Manager.

## Overview

The secret management system provides:
- **Secure credential storage** for Schwab, Databento, and FRED APIs
- **Local encrypted storage** using machine-specific encryption keys
- **Google Cloud Secret Manager** integration for cloud deployments
- **Automatic credential prompting** when secrets are missing
- **Environment variable migration** from legacy configurations

## Quick Start

### 1. Interactive Setup (Recommended)

Run the setup script to configure all credentials:

```bash
python scripts/setup-secrets.py
```

This will:
- Auto-detect your environment (local or GCP)
- Prompt for all required credentials
- Store them securely

### 2. Check Current Configuration

To see which services are configured:

```bash
python scripts/setup-secrets.py --check-only
```

## Storage Providers

### Local Storage (Default)

Credentials are stored encrypted at `~/.wheel_trading/secrets/`:
- Uses machine-specific encryption (UID + hostname)
- Files have restricted permissions (0600)
- Fully offline, no external dependencies

### Google Cloud Secret Manager

For production deployments:
- Centralized secret management
- Audit logging and access control
- Version history and rotation support

To set up GCP:

```bash
# Run the GCP setup wizard
python scripts/setup-secrets.py --setup-gcp

# Then configure credentials with GCP provider
python scripts/setup-secrets.py --provider gcp
```

### Environment Variables (Read-Only)

The system can read from environment variables as a fallback:
- Variables must use prefix: `WHEEL_`
- Example: `WHEEL_SCHWAB_CLIENT_ID`

## Required Credentials

### Schwab API
- **client_id**: OAuth application client ID
- **client_secret**: OAuth application client secret
- Obtain from: [Schwab Developer Portal](https://developer.schwab.com)

### Databento
- **api_key**: API key for market data access
- Obtain from: [Databento Dashboard](https://databento.com/dashboard)

### FRED (Federal Reserve Economic Data)
- **api_key**: API key for economic indicators
- Obtain from: [FRED API Keys](https://fred.stlouisfed.org/docs/api/api_key.html)

## Usage in Code

### Basic Usage

```python
from src.unity_wheel.secrets import SecretManager

# Initialize (auto-detects provider)
secrets = SecretManager()

# Get individual secret
databento_key = secrets.get_secret("databento_api_key")

# Get all credentials for a service
schwab_creds = secrets.get_credentials("schwab")
client_id = schwab_creds["client_id"]
client_secret = schwab_creds["client_secret"]
```

### Using Integration Helpers

```python
from src.unity_wheel.secrets.integration import (
    get_schwab_credentials,
    get_databento_api_key,
    get_ofred_api_key
)

# Direct credential access
schwab = get_schwab_credentials()
databento_key = get_databento_api_key()
fred_key = get_ofred_api_key()
```

### Environment Variable Injection

For compatibility with libraries expecting environment variables:

```python
from src.unity_wheel.secrets.integration import SecretInjector

# Temporarily inject credentials into environment
with SecretInjector(service="schwab"):
    # WHEEL_AUTH__CLIENT_ID and WHEEL_AUTH__CLIENT_SECRET
    # are now available as environment variables
    client = SchwabClient()  # Will read from env
```

### Enhanced AuthClient

The updated AuthClient automatically uses SecretManager:

```python
from src.unity_wheel.auth.client_v2 import AuthClient

# No need to provide credentials explicitly
async with AuthClient() as client:
    # Credentials loaded from SecretManager
    await client.initialize()
```

## Migration from Environment Variables

If you have existing environment variables, migrate them:

```bash
# Set environment variables (one-time)
export WHEEL_AUTH__CLIENT_ID="your-client-id"
export WHEEL_AUTH__CLIENT_SECRET="your-client-secret"
export DATABENTO_API_KEY="your-databento-key"

# Run migration
python -c "from src.unity_wheel.secrets.integration import migrate_env_to_secrets; migrate_env_to_secrets()"
```

## Google Cloud Setup Details

### Prerequisites

1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
2. Install Python client: `pip install google-cloud-secret-manager`

### Manual Setup Steps

```bash
# 1. Authenticate
gcloud auth login
gcloud auth application-default login

# 2. Create or select project
gcloud projects create wheel-trading-bot --name="Wheel Trading Bot"
gcloud config set project wheel-trading-bot

# 3. Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# 4. Set environment variable
export GCP_PROJECT_ID="wheel-trading-bot"
```

### Creating Secrets in GCP Console

1. Visit: https://console.cloud.google.com/security/secret-manager
2. Click "Create Secret"
3. Use these secret IDs:
   - `schwab_client_id`
   - `schwab_client_secret`
   - `databento_api_key`
   - `ofred_api_key`

## Security Best Practices

1. **Never commit credentials** to version control
2. **Use least privilege** access for GCP service accounts
3. **Rotate credentials** regularly
4. **Monitor access logs** in GCP
5. **Backup encryption keys** for local storage

## Troubleshooting

### "Secret not found" errors

1. Run setup: `python scripts/setup-secrets.py`
2. Check provider: Ensure correct provider is detected
3. Verify permissions: Check file/GCP permissions

### GCP authentication issues

```bash
# Re-authenticate
gcloud auth application-default login

# Verify project
gcloud config get-value project

# Check API enablement
gcloud services list --enabled | grep secretmanager
```

### Local storage issues

```bash
# Check storage location
ls -la ~/.wheel_trading/secrets/

# Verify file permissions (should be 0600)
stat ~/.wheel_trading/secrets/secrets.enc
```

## Examples

See `example_secrets_usage.py` for comprehensive examples:

```bash
python example_secrets_usage.py
```

This demonstrates:
- Basic secret operations
- Provider-specific features
- Integration with existing code
- Environment variable injection
- Migration from legacy configs
