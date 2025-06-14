# Data Ingestion Setup for Optimized Database

## Current State

The wheel trading system uses a **pull-when-asked** architecture:
- Data is fetched on-demand when needed
- Caching reduces API calls (5-min for options, 30s for quotes)
- No automated scheduled ingestion currently active

## Data Flow into Optimized Database

### 1. Databento (Options & Market Data)

**Current Setup:**
- Moneyness filtering: ±35% of spot price (reduces data by 80%)
- Cache TTL: 5 minutes for options, 30 seconds for quotes
- Storage: Was using multiple databases, now consolidated

**Where Data Should Be Saved:**
```sql
-- Options data → options.contracts table
INSERT INTO options.contracts (
    symbol, expiration, strike, option_type, timestamp,
    bid, ask, mid, volume, open_interest,
    implied_volatility, delta, gamma, theta, vega, rho,
    moneyness, days_to_expiry, year_month
) VALUES (...)

-- Market data → market.price_data table  
INSERT INTO market.price_data (
    symbol, date, open, high, low, close, volume,
    daily_return, volatility_20d, volatility_60d, year_month
) VALUES (...)
```

**Recommended Schedule:**
- **Ad-hoc/Live**: On-demand with 5-minute cache
- **End-of-day**: 4:30 PM ET for final prices
- **Script**: `scripts/pull_databento_eod.py`

### 2. FRED (Economic Data)

**Current Setup:**
- 24-hour cache TTL
- Series: DGS10, DFF, VIXCLS, DEXUSEU

**Where Data Should Be Saved:**
```sql
-- Economic indicators → analytics.ml_features table
INSERT INTO analytics.ml_features (
    symbol, feature_date, vix_level, market_regime, ...
) VALUES (...)
```

**Recommended Schedule:**
- **Daily**: 6:00 AM ET
- **Script**: `scripts/pull_fred_daily.py`

## Manual Data Pull Commands

### 1. Pull Options Data Now
```python
from src.unity_wheel.data_providers.databento import DatabentoClient
import duckdb

client = DatabentoClient()
conn = duckdb.connect('data/wheel_trading_optimized.duckdb')

# Get option chains
chains = client.get_option_chains('U')

# Insert into database (with moneyness filtering)
for chain in chains:
    for option in chain.options:
        if 0.65 <= option.strike / chain.spot_price <= 1.35:
            conn.execute("INSERT INTO options.contracts ...", [...])
```

### 2. Pull FRED Data Now
```python
from src.unity_wheel.data_providers.fred import FREDClient
client = FREDClient()

# Get VIX
vix_data = client.get_series_observations('VIXCLS')
# Insert into analytics.ml_features
```

## Automated Ingestion (Optional)

### Cron Schedule
```bash
# End-of-day options (4:30 PM ET Mon-Fri)
30 16 * * 1-5 cd /path/to/wheel-trading && python scripts/pull_databento_eod.py

# Daily FRED (6:00 AM ET)
0 6 * * * cd /path/to/wheel-trading && python scripts/pull_fred_daily.py

# Refresh materialized views (every hour during market)
0 9-16 * * 1-5 cd /path/to/wheel-trading && python scripts/refresh_views.py
```

## Integration Points

### 1. Update Storage Configuration
The system currently points to old databases. Update these references:
- `src/unity_wheel/storage/storage.py` → Use `wheel_trading_optimized.duckdb`
- `src/unity_wheel/data_providers/databento/databento_storage_adapter.py` → Insert to new schema
- Config files that reference database paths

### 2. Existing Scripts to Modify
- `tools/analysis/pull_databento_integrated.py` - Update to use new database
- `tools/download_unity_daily_data.py` - Point to optimized database

### 3. Cache Strategy
The optimized database acts as both operational store AND cache:
- Recent data (< 5 min) served directly
- Older data triggers fresh pull
- Materialized views for instant wheel opportunities

## Data Retention

- **Options data**: 90 days (then archive)
- **Market data**: 3 years
- **FRED data**: 1 year
- **Cleanup**: Weekly CHECKPOINT and ANALYZE

## Testing Data Ingestion

```bash
# Test Databento connection
python -c "from src.unity_wheel.data_providers.databento import DatabentoClient; print(DatabentoClient().test_connection())"

# Test FRED connection  
python -c "from src.unity_wheel.data_providers.fred import FREDClient; print(FREDClient().get_series_info('VIXCLS'))"

# Check current data age
duckdb data/wheel_trading_optimized.duckdb -c "
SELECT 
    'Options' as data_type,
    MAX(timestamp) as latest,
    CURRENT_TIMESTAMP - MAX(timestamp) as age
FROM options.contracts
UNION ALL
SELECT 
    'Market',
    MAX(date),
    CURRENT_DATE - MAX(date)
FROM market.price_data;"
```

## Summary

The system is designed for **on-demand data fetching** with intelligent caching. The optimized database is ready to receive:
1. **Databento**: Options chains → `options.contracts`, Stock quotes → `market.price_data`
2. **FRED**: Economic indicators → `analytics.ml_features`

No changes needed to the pull-when-asked architecture - just point data providers to the new optimized database tables.