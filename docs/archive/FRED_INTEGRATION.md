> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# FRED Integration with Unified Storage

## Overview

The FRED (Federal Reserve Economic Data) integration has been updated to use the project's unified storage architecture with Google Secret Manager support and DuckDB caching.

## Architecture

### 1. **Secret Management**

FRED API keys are now managed through the SecretManager:

```python
from src.unity_wheel.secrets.integration import get_ofred_api_key

# Automatically retrieves from SecretManager
api_key = get_ofred_api_key()
```

- No more environment variables needed
- Encrypted storage (local or Google Cloud)
- Set up with: `python scripts/setup-secrets.py`

### 2. **Unified Storage**

FRED data is stored in the DuckDB cache with optional GCS backup:

```python
from src.unity_wheel.storage.storage import Storage
from src.unity_wheel.data import FREDDataManager

# Initialize unified storage
storage = Storage()
await storage.initialize()

# Create FRED manager with storage
manager = FREDDataManager(storage=storage)
```

### 3. **Get-or-Fetch Pattern**

The system implements automatic caching with the get-or-fetch pattern:

```python
# Automatically checks cache, fetches if stale
rate, confidence = await manager.get_or_fetch_risk_free_rate(
    tenor_months=3,
    fetch_if_stale_days=1  # Fetch if data older than 1 day
)
```

## Data Storage

### DuckDB Tables

FRED data is stored in these DuckDB tables:

1. **fred_series** - Series metadata
2. **fred_observations** - Time series data
3. **fred_features** - Calculated features (SMA, volatility)
4. **risk_metrics** - Daily risk metrics cache

### Storage Locations

- **Local Cache**: `~/.wheel_trading/cache/wheel_cache.duckdb`
- **GCS Backup**: `gs://wheel-processed/fred_*` (if enabled)

### Data Series

The system tracks these key series for wheel strategy:

- **Risk-free rates**: DGS3, DGS1, DFF
- **Volatility**: VIXCLS, VXDCLS
- **Liquidity**: TEDRATE, BAMLH0A0HYM2
- **Economic**: UNRATE, CPIAUCSL

## Usage Examples

### Basic Usage

```python
# Initialize (one time)
await manager.initialize_data(lookback_days=1825)  # 5 years

# Get current data
rf_rate, conf = await manager.get_risk_free_rate(3)
regime, vix = await manager.get_volatility_regime()
iv_rank, conf = await manager.calculate_iv_rank(current_iv=0.65)
```

### Integration with Wheel Strategy

```python
# In your wheel strategy code
rf_rate, _ = await manager.get_or_fetch_risk_free_rate(3)

# Use in Black-Scholes calculations
result = black_scholes_price_validated(
    S=stock_price,
    K=strike,
    T=time_to_expiry,
    r=rf_rate,  # Fresh from FRED
    sigma=volatility,
    option_type="put"
)
```

### Health Monitoring

```python
# Check data health
health = await manager.get_data_health_report()
print(f"Health score: {health['health_score']}/100")

# Check storage stats
stats = await storage.get_storage_stats()
print(f"FRED data size: {stats['tables']['fred_observations']['size_mb']:.1f} MB")
```

## Maintenance

### Update Data

The system automatically fetches new data when requested:

```python
# Manual update check
updates = await manager.update_data()
print(f"Updated {len(updates)} series")
```

### Clean Old Data

```python
# Remove observations older than 5 years
await manager.fred_storage.cleanup_old_data(days_to_keep=1825)

# Run storage maintenance
await storage.cleanup_old_data()
```

## Testing

Run the integration tests:

```bash
# Quick integration test
python test_fred_integration.py

# Full example with sanity checks
python example_fred_usage.py

# Unit tests
pytest tests/test_fred.py -v
```

## Performance

- **Initial load**: ~30s for 5 years of data
- **Cache hit**: <1ms for risk-free rate
- **API fetch**: ~100ms per series
- **Storage size**: ~50MB for all series (5 years)

## Error Handling

The system includes:

- Automatic retry with exponential backoff
- Fallback to cached data during outages
- Default values with low confidence scores
- Comprehensive logging for debugging

## Migration from Old System

If you have existing FRED data in SQLite:

```python
# Old system used SQLite at ./data/market_data.db
# New system uses DuckDB at ~/.wheel_trading/cache/wheel_cache.duckdb

# Data will be automatically fetched fresh on first use
# No migration needed - just start using the new system
```

## Cost

- **FRED API**: Free (120 requests/minute limit)
- **Storage**: <50MB local, optional GCS backup
- **Estimated monthly cost**: $0 (local only) or <$0.01 (with GCS)
