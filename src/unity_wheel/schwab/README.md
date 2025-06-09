# Schwab Client Module

Reliable Schwab API client with self-validation, error recovery, and intelligent caching.

## Features

- **Automatic retry logic** with exponential backoff for network failures
- **Position validation** with OCC option symbol parsing
- **Corporate action detection** from position anomalies
- **Fallback to cached data** during API outages
- **Self-validation** of all data consistency
- **Rate limit handling** with automatic retry
- **Async/await** interface for efficient concurrent operations

## Usage

```python
from src.unity_wheel.schwab import SchwabClient

# Create client
client = SchwabClient(
    client_id="your_client_id",
    client_secret="your_client_secret",
)

# Use as async context manager
async with client:
    # Get positions (never cached)
    positions = await client.get_positions()
    
    # Get account info (cached for 30 seconds)
    account = await client.get_account()
    
    # Detect corporate actions
    actions = client.detect_corporate_actions(positions)
```

## Error Handling

The client provides comprehensive error handling:

- **Network errors**: Automatic retry with exponential backoff
- **Authentication errors**: Token refresh and retry
- **Rate limiting**: Respects Retry-After headers
- **Data validation errors**: Quarantines bad data, returns valid subset
- **API outages**: Falls back to last known good data

## Caching Strategy

- **Positions**: Never cached (always fresh)
- **Account data**: Cached briefly (30 seconds)
- **Last known good**: Persisted to disk for fallback during outages

## Testing

```bash
# Run all Schwab client tests
pytest tests/test_schwab.py -v

# Run with coverage
pytest tests/test_schwab.py --cov=src.unity_wheel.schwab
```

## OAuth Implementation

The OAuth flow is not yet implemented. To complete the integration:

1. Implement the OAuth authorization code flow
2. Store tokens securely using the cryptography module
3. Implement automatic token refresh

See `client.py:_authenticate()` for the placeholder implementation.