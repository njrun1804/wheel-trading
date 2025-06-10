# Interactive Credential Setup Guide

This guide shows you how to set up and test your API credentials for the Unity Wheel Trading Bot.

## Quick Start

### 1. Run the Interactive Setup

```bash
python scripts/setup-secrets.py
```

This will:
1. Prompt you for each credential:
   - **Schwab OAuth Client ID**: Your Schwab app client ID
   - **Schwab OAuth Client Secret**: Your Schwab app secret (hidden input)
   - **Databento API Key**: Your Databento API key (hidden input)
   - **FRED API Key**: Your Federal Reserve data API key (hidden input)

2. Store them securely (encrypted locally by default)
3. Offer to test them immediately

### 2. What You'll See

```
=== Unity Wheel Trading Bot - Credential Setup ===

Using local provider for secret storage.

--- SCHWAB Credentials ---
Schwab OAuth Client ID: YOUR_CLIENT_ID_HERE
Schwab OAuth Client Secret: [hidden input]
✓ schwab credentials saved

--- DATABENTO Credentials ---
Databento API Key: [hidden input]
✓ databento credentials saved

--- OFRED Credentials ---
FRED (Federal Reserve Economic Data) API Key: [hidden input]
✓ ofred credentials saved

✓ All credentials configured successfully!

Credentials are stored in: local
Location: ~/.wheel_trading/secrets/

============================================================

Would you like to test the credentials now? (y/N): y
```

### 3. Credential Testing

If you choose to test, you'll see:

```
=== Unity Wheel Trading Bot - Credential Testing ===

Configured services:
  ✓ schwab
  ✓ databento
  ✓ ofred

--- Testing Schwab Credentials ---
✓ Schwab credentials loaded successfully
✓ Schwab API connection successful
✓ Schwab client health: {'status': 'healthy', ...}

--- Testing Databento Credentials ---
✓ Databento credentials loaded successfully
✓ Databento API key validated
  API key: db-12345...

--- Testing FRED Credentials ---
✓ FRED credentials loaded successfully
  Testing API connection...
✓ FRED API connection successful
  Latest 10Y Treasury: 4.25% on 2024-01-19

=== Test Summary ===
schwab: ✓ PASSED
databento: ✓ PASSED
fred: ✓ PASSED

✓ All credential tests passed!
Your Unity Wheel Trading Bot is ready to use.
```

## Manual Testing

You can test credentials anytime:

```bash
python scripts/test-secrets.py
```

## Updating Credentials

To update any credential:

```bash
# Re-run setup (it will detect existing credentials)
python scripts/setup-secrets.py

# You'll be asked if you want to update each service
```

## Check Current Configuration

```bash
python scripts/setup-secrets.py --check-only
```

Output:
```
=== Current Configuration ===
✓ schwab
✓ databento
✓ ofred
```

## Where Credentials Are Stored

- **Local**: `~/.wheel_trading/secrets/secrets.enc` (encrypted)
- **Google Cloud**: In Secret Manager (if using GCP)

## Security Notes

1. Credentials are encrypted at rest using machine-specific keys
2. Never stored in plain text
3. Never committed to version control
4. Can be migrated to Google Cloud Secret Manager for production

## Troubleshooting

### "Secret not found" Error
```bash
# Re-run setup
python scripts/setup-secrets.py
```

### Schwab OAuth Issues
- Ensure redirect URI is exactly: `http://localhost:8182/callback`
- Check client ID and secret are from an active Schwab app
- You may need to complete OAuth flow on first run

### Databento API Key Invalid
- Verify key at: https://databento.com/dashboard
- Check you have an active subscription

### FRED API Key Invalid
- Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
- Keys are usually activated immediately

## Using in Code

All clients now automatically use SecretManager:

```python
# Schwab - no credentials needed!
from src.unity_wheel.schwab import SchwabClient
async with SchwabClient() as client:
    # Credentials loaded automatically
    positions = await client.get_positions()

# Databento - no API key needed!
from src.unity_wheel.databento import DatentoClient
client = DatentoClient()  # API key loaded automatically

# FRED - no API key needed!
from src.unity_wheel.data import FREDClient
async with FREDClient() as client:
    # API key loaded automatically
    data = await client.get_series_observations("DGS10")
```

## Next Steps

1. Complete Schwab OAuth flow if needed
2. Run the main trading bot: `python run.py --portfolio 100000`
3. Monitor with: `./scripts/monitor.sh`
