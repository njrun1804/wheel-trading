# Integration Guide

> **⚠️ Note:** The default `config.yaml` is tuned for high risk/high return strategies.
> For a more conservative approach, see `examples/core/conservative_config.yaml`

This guide consolidates all external service integrations for the Unity Wheel Trading Bot v2.0. The system follows a pull-when-asked architecture for all data sources.

## Table of Contents

1. [Secret Management](#secret-management)
2. [Databento Integration](#databento-integration)
3. [Schwab Integration](#schwab-integration)
4. [FRED Integration](#fred-integration)

## Secret Management

All integrations use a unified secret management system with encryption at rest and audit logging.

### Setup

1. **Generate encryption key:**
```bash
python scripts/setup-secrets.py
```

2. **Add service credentials:**
```python
from src.unity_wheel.secrets import SecretManager

manager = SecretManager()

# Add Databento API key
await manager.store_secret("databento", {"api_key": "YOUR_API_KEY"})

# Add Schwab OAuth credentials
await manager.store_secret("schwab", {
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET"
})

# Add FRED API key
await manager.store_secret("fred", {"api_key": "YOUR_API_KEY"})
```

3. **Environment variables:**
```bash
export WHEEL_ENCRYPTION_KEY="your-base64-encoded-key"
export WHEEL_SECRETS_DIR="$HOME/.wheel/secrets"  # Optional, defaults shown
```

### Security Features

- **AES-256-GCM encryption** for all stored secrets
- **Audit logging** with timestamps and access tracking
- **Key rotation** support with versioning
- **Memory protection** - secrets cleared after use
- **Access control** via file permissions (600)

## Databento Integration

Provides options chain data with smart filtering to reduce storage requirements by 80%.

### Architecture

```
User Request → DatentoIntegration → Rate-Limited Client → Smart Filter → Storage
                                           ↓
                                    Validation & Retry
```

### Setup

1. **Get API key** from [databento.com](https://databento.com)
2. **Store credential:**
```python
await manager.store_secret("databento", {"api_key": "YOUR_API_KEY"})
```

3. **Configure in `config.yaml`:**
```yaml
databento:
  enabled: true
  symbols: ["U"]  # Unity only by default
  storage:
    local_days: 30      # Recent data cached locally
    cloud_enabled: false # Set true for GCS/BigQuery
  filters:
    min_volume: 10      # Skip illiquid options
    max_spread_pct: 0.5 # Skip wide spreads
    delta_range: [0.15, 0.50]  # Wheel-relevant deltas
```

### Usage

```python
from src.unity_wheel.databento import DatentoIntegration

# Initialize
integration = DatentoIntegration(client, storage)

# Get wheel candidates
candidates = await integration.get_wheel_candidates(
    underlying="U",
    target_delta=0.30,
    dte_range=(30, 60)
)

# Get live quote
quote = await integration.get_live_quote("U 250117P00210000")
```

### Cost Optimization

- **Smart filtering** reduces data by 80%
- **Local cache** for 30 days (~5GB for Unity)
- **Pull-when-asked** - no continuous streaming
- **Estimated cost**: <$1/month for typical usage

### Data Quality

- Automatic validation of all data points
- Anomaly detection for price spikes
- Stale data warnings after 15 minutes
- Corporate action detection

## Schwab Integration

Read-only access to portfolio positions and account data. No trading execution.

### OAuth Setup

1. **Register app** at [developer.schwab.com](https://developer.schwab.com)
   - Callback URL: `https://127.0.0.1:5000/callback`
   - Scopes: `accounts`, `positions` (read-only)

2. **Initial authorization:**
```python
from src.unity_wheel.schwab import SchwabClient

client = SchwabClient(client_id, client_secret)
auth_url = client.get_authorization_url()
print(f"Visit: {auth_url}")

# After authorization
tokens = await client.exchange_code(auth_code)
```

3. **Store credentials:**
```python
await manager.store_secret("schwab", {
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "refresh_token": tokens["refresh_token"]
})
```

### Usage

```python
async with SchwabClient(client_id, client_secret) as client:
    # Get positions (never cached)
    positions = await client.get_positions()

    # Get account info (cached 30s)
    account = await client.get_account()

    # Detect corporate actions
    actions = client.detect_corporate_actions(positions)
    if actions:
        logger.warning(f"Corporate actions detected: {actions}")
```

### Features

- **Automatic token refresh** before expiration
- **Position validation** with OCC symbol parsing
- **Corporate action detection** from anomalies
- **Graceful degradation** with cached fallback
- **Comprehensive error handling** with retry logic

### Limitations

- Read-only access (no trading)
- 120 requests/minute rate limit
- Positions endpoint never cached
- Account data cached for 30 seconds

## FRED Integration

Federal Reserve Economic Data for risk-free rates and VIX.

### Setup

1. **Get API key** from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
2. **Store credential:**
```python
await manager.store_secret("fred", {"api_key": "YOUR_API_KEY"})
```

### Usage

```python
from src.unity_wheel.fred import FREDClient

client = FREDClient(api_key)

# Get risk-free rate (3-month T-bill)
risk_free_rate = await client.get_risk_free_rate()

# Get VIX
vix = await client.get_vix()

# Get with specific date
historical_rate = await client.get_series_value(
    "DGS3MO",
    date="2024-01-01"
)
```

### Configuration

```yaml
fred:
  cache_dir: "~/.wheel/fred_cache"
  fallback_rate: 0.05  # If FRED unavailable
  series:
    risk_free: "DGS3MO"  # 3-month Treasury
    vix: "VIXCLS"        # CBOE VIX
```

### Features

- **Local caching** to minimize API calls
- **Automatic fallback** values for resilience
- **Business day adjustment** for weekends/holidays
- **Rate limiting** compliance (120 req/min)

## Common Patterns

### Error Handling

All integrations follow consistent error patterns:

```python
try:
    result = await integration.fetch_data()
except RateLimitError:
    # Automatic retry with backoff
    await asyncio.sleep(60)
    result = await integration.fetch_data()
except NetworkError:
    # Use cached/fallback data
    result = integration.get_cached_data()
except ValidationError as e:
    # Log and skip invalid data
    logger.error(f"Invalid data: {e}")
    return None
```

### Monitoring

Built-in observability for all integrations:

```python
# Check integration health
python run.py --diagnose

# View performance metrics
python run.py --performance

# Export metrics
python run.py --export-metrics
```

### Testing

Each integration includes comprehensive tests:

```bash
# Test specific integration
pytest tests/test_databento.py -v
pytest tests/test_schwab.py -v
pytest tests/test_fred.py -v

# Integration tests
pytest tests/test_autonomous_flow.py -v
```

## Troubleshooting

### Databento Issues
- **"Rate limit exceeded"** - Integration handles automatically with backoff
- **"Invalid symbol"** - Check OCC symbol format: "U 250117P00210000"
- **"No data"** - Verify market hours and symbol liquidity

### Schwab Issues
- **"Token expired"** - Automatic refresh, check refresh_token validity
- **"Position mismatch"** - Check for corporate actions
- **"Network timeout"** - Automatic retry with cached fallback

### FRED Issues
- **"No data for date"** - FRED updates with lag, use previous business day
- **"Series not found"** - Verify series ID at fred.stlouisfed.org

### General Issues
- **Check logs**: `~/.wheel/logs/wheel-bot.log`
- **Verify setup**: `python -m unity_wheel.validate`
- **Run diagnostics**: `python run.py --diagnose`
