# Authentication Setup Guide

This guide covers setting up authentication for the Unity Wheel Trading Bot with Schwab API integration.

## Overview

The authentication system provides:
- Zero manual intervention after initial setup
- Automatic token refresh before expiry
- Graceful degradation with cached responses
- Circuit breaker protection
- Rate limiting with exponential backoff

## Initial Setup

### 1. Register Your Application

1. Go to [Schwab Developer Portal](https://developer.schwab.com)
2. Create a new application
3. Set the redirect URI to: `http://localhost:8182/callback`
4. Note your Client ID and Client Secret

### 2. Configure Credentials

The Unity Wheel Trading Bot now uses a secure credential management system. Use the interactive setup:

```bash
# Run the secret management setup
python scripts/setup-secrets.py
```

This will prompt you for your Schwab credentials and store them securely.

#### Alternative Methods

If you prefer environment variables:

```bash
export WHEEL_AUTH__CLIENT_ID="your-client-id-here"
export WHEEL_AUTH__CLIENT_SECRET="your-client-secret-here"
```

Or use Google Cloud Secret Manager (for production):

```bash
# Set up GCP first
python scripts/setup-secrets.py --setup-gcp

# Then configure with GCP provider
python scripts/setup-secrets.py --provider gcp
```

See [SECRET_MANAGEMENT.md](SECRET_MANAGEMENT.md) for detailed information.

### 3. First-Time Authentication

Run the authentication setup:

```python
import asyncio
from src.unity_wheel.auth.client_v2 import AuthClient

async def setup_auth():
    # Create client - credentials loaded automatically from SecretManager
    auth_client = AuthClient()
    
    # Initialize and authenticate
    async with auth_client:
        # This will open browser for OAuth flow
        await auth_client.authenticate()
        
        # Check health
        health = await auth_client.health_check()
        print(f"Auth status: {health['status']}")

# Run setup
asyncio.run(setup_auth())
```

The browser will open automatically. Log in to your Schwab account and authorize the application.

## Using Authentication

### Basic Usage

```python
from src.unity_wheel.auth import AuthClient

async def get_account_data():
    async with AuthClient(
        client_id=os.getenv("WHEEL_AUTH__CLIENT_ID"),
        client_secret=os.getenv("WHEEL_AUTH__CLIENT_SECRET")
    ) as client:
        # Make authenticated requests
        accounts = await client.make_request(
            "GET",
            "https://api.schwabapi.com/v1/accounts"
        )
        return accounts
```

### Advanced Configuration

Configure auth settings in `config.yaml`:

```yaml
auth:
  # Token management
  auto_refresh: true
  token_refresh_buffer_minutes: 5
  
  # Cache settings
  enable_cache: true
  cache_ttl_seconds: 3600
  cache_max_size_mb: 100
  
  # Rate limiting
  rate_limit_rps: 10.0
  rate_limit_burst: 20
  enable_circuit_breaker: true
```

## Health Monitoring

Check authentication health:

```python
async def check_auth_health():
    async with AuthClient(...) as client:
        health = await client.health_check()
        
        print(f"Status: {health['status']}")
        print(f"Token valid: {health['token_valid']}")
        print(f"Rate limit: {health['rate_limiter']['tokens_available']}")
        print(f"Cache stats: {health['cache']}")
```

## Error Recovery

The system handles errors automatically:

1. **Token Expiry**: Automatic refresh using stored refresh token
2. **Rate Limits**: Exponential backoff with circuit breaker
3. **Network Errors**: Falls back to cached data for GET requests
4. **Invalid Credentials**: Clear error message with recovery instructions

## Security Best Practices

1. **Never commit credentials** to version control
2. **Use environment variables** for sensitive data
3. **Tokens are encrypted** at rest using machine-specific keys
4. **Storage location**: `~/.wheel_trading/auth/`
5. **Permissions**: Token files have 0600 permissions (owner read/write only)

## Troubleshooting

### Common Issues

1. **"No refresh token available"**
   - Re-run authentication setup
   - Check token storage permissions

2. **"Circuit breaker is open"**
   - Too many failures detected
   - Wait 60 seconds for recovery
   - Check API status

3. **"Rate limit exceeded"**
   - Automatic retry with backoff
   - Check rate limit configuration

4. **"Authentication timeout"**
   - Complete login within 5 minutes
   - Check browser popup blocker

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("unity_wheel.auth").setLevel(logging.DEBUG)
```

### Manual Token Refresh

Force token refresh:

```python
async with AuthClient(...) as client:
    await client.refresh_token()
```

### Clear Stored Tokens

Reset authentication:

```python
from src.unity_wheel.auth import SecureTokenStorage

storage = SecureTokenStorage()
storage.clear_tokens()
```

## Testing

Run authentication tests:

```bash
pytest tests/test_auth.py -v
```

Test specific scenarios:

```bash
# Test token expiry handling
pytest tests/test_auth.py::TestAuthClient::test_token_refresh_on_expiry -v

# Test rate limiting
pytest tests/test_auth.py::TestRateLimiter -v

# Test graceful degradation
pytest tests/test_auth.py::TestAuthClient::test_graceful_degradation -v
```

## Integration Example

Full integration with wheel trading bot:

```python
from src.unity_wheel.auth import AuthClient
from src.config.loader import get_config_loader

async def run_wheel_bot():
    # Load config
    config = get_config_loader().load()
    
    # Create auth client from config
    auth_client = AuthClient(
        client_id=config.auth.client_id.get_secret_value(),
        client_secret=config.auth.client_secret.get_secret_value(),
        enable_cache=config.auth.enable_cache,
        cache_ttl=config.auth.cache_ttl_seconds,
        rate_limit_rps=config.auth.rate_limit_rps
    )
    
    async with auth_client:
        # Check health
        health = await auth_client.health_check()
        if health["status"] != "healthy":
            print(f"Auth not healthy: {health}")
            return
        
        # Get account data
        accounts = await auth_client.make_request(
            "GET",
            "https://api.schwabapi.com/v1/accounts"
        )
        
        # Get market data
        market_data = await auth_client.make_request(
            "GET",
            "https://api.schwabapi.com/v1/marketdata/quotes",
            params={"symbols": "U"}
        )
        
        # Use cached data if available
        option_chain = await auth_client.make_request(
            "GET",
            "https://api.schwabapi.com/v1/marketdata/chains",
            params={"symbol": "U"},
            cache_ttl=300  # Cache for 5 minutes
        )
        
        # Process data...

asyncio.run(run_wheel_bot())
```

## Performance Metrics

Expected performance:
- Token refresh: <1s
- Cache hit rate: >80% for market data
- Rate limit compliance: 100%
- Circuit breaker recovery: 60s
- Offline operation: Full GET request support

## Support

For issues:
1. Check health status
2. Review debug logs
3. Verify credentials
4. Test network connectivity
5. Check Schwab API status