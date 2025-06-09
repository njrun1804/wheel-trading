# Pull-When-Asked Storage Architecture

## Overview

The Wheel Trading Bot v2.0 uses a lean, cost-effective storage architecture optimized for single-user, on-demand operation. All data fetching happens only when explicitly requested, with intelligent local caching to minimize API calls and costs.

## Architecture Principles

1. **Pull-When-Asked Only** - No streaming, no continuous ingestion
2. **Local-First Caching** - DuckDB for fast SQL queries on local data
3. **Cold Backup Only** - GCS for archival, not hot queries
4. **< $100/month Total** - Including all Google Cloud and API costs

## Storage Layers

### 1. Local DuckDB Cache (~/.wheel_trading/cache/)

Primary storage for all recent data with automatic TTL and eviction.

```sql
-- Option chains table
option_chains (
    symbol VARCHAR,
    expiration DATE,
    timestamp TIMESTAMP,
    spot_price DECIMAL,
    data JSON,  -- Full chain data
    created_at TIMESTAMP
)

-- Position snapshots  
position_snapshots (
    account_id VARCHAR,
    timestamp TIMESTAMP,
    positions JSON,
    account_data JSON,
    created_at TIMESTAMP
)

-- Greeks cache
greeks_cache (
    option_symbol VARCHAR,
    timestamp TIMESTAMP,
    spot_price DECIMAL,
    risk_free_rate DECIMAL,
    delta, gamma, theta, vega, rho DECIMAL,
    iv DECIMAL,
    created_at TIMESTAMP
)

-- Model predictions
predictions_cache (
    prediction_id VARCHAR,
    timestamp TIMESTAMP,
    input_features JSON,
    predictions JSON,
    model_version VARCHAR,
    created_at TIMESTAMP
)
```

**Features:**
- 5GB max size (configurable)
- 30-day TTL with automatic cleanup
- LRU eviction when approaching size limit
- Export to Parquet for GCS backup

### 2. Google Cloud Storage (Optional)

Two buckets for cold storage only:

```
wheel-raw/
├── schwab/
│   └── {date}/
│       └── {timestamp}_schwab.json
├── databento/
│   └── {date}/
│       └── {timestamp}_databento.json
└── fred/
    └── {date}/
        └── {timestamp}_fred.json

wheel-processed/
├── option_chains/
│   └── year={YYYY}/month={MM}/
│       └── option_chains_{YYYYMMDD}.parquet
├── position_snapshots/
│   └── year={YYYY}/month={MM}/
│       └── positions_{YYYYMMDD}.parquet
└── predictions/
    └── year={YYYY}/month={MM}/
        └── predictions_{YYYYMMDD}.parquet
```

**Lifecycle Policies:**
- Standard → Nearline after 30 days
- Delete raw after 365 days
- Delete processed after 730 days

## Data Flow

### 1. User Requests Recommendation

```python
# User clicks "Get Recommendation" or runs CLI
python run_on_demand.py --portfolio 100000
```

### 2. Storage Checks Cache

```python
# Check local DuckDB cache first
storage = Storage()
positions = await storage.get_or_fetch_positions(
    account_id="default",
    fetch_func=fetch_schwab_positions,
    max_age_minutes=30  # Accept 30-min old data
)
```

### 3. Cache Miss → API Call

```python
# If cache miss, fetch fresh data
async def fetch_schwab_positions(account_id):
    async with SchwabClient() as client:
        positions = await client.get_positions()
        account = await client.get_account()
        return {...}
```

### 4. Store in Cache + Optional GCS

```python
# Automatically handled by storage layer
# 1. Store in DuckDB (primary)
# 2. Upload raw JSON to GCS (if enabled)
# 3. Periodic Parquet export to GCS
```

## Usage Examples

### Basic CLI Usage

```bash
# Get recommendation (uses cache if fresh)
python run_on_demand.py --portfolio 100000

# Force fresh data by clearing cache
python run_on_demand.py --clear-cache --portfolio 100000

# Check storage statistics
python run_on_demand.py --storage-stats
```

### Cloud Run Job

```bash
# Deploy job
gcloud run jobs deploy wheel-recommendation \
    --source . \
    --task-timeout 5m \
    --max-retries 1 \
    --memory 2Gi

# Execute job
gcloud run jobs execute wheel-recommendation \
    --env-vars PORTFOLIO_VALUE=100000,ACCOUNT_ID=default
```

### Programmatic Usage

```python
from src.unity_wheel.storage import Storage

# Initialize storage
storage = Storage()
await storage.initialize()

# Get or fetch with custom TTL
data = await storage.get_or_fetch_option_chain(
    symbol="U",
    expiration=datetime(2024, 2, 16),
    fetch_func=my_fetch_function,
    max_age_minutes=5  # Very fresh data
)

# Check what's cached
stats = await storage.get_storage_stats()
print(f"Cached chains: {stats['option_chains_count']}")
print(f"DB size: {stats['db_size_mb']} MB")
```

## Configuration

### Environment Variables

```bash
# Optional GCS backup
export GCP_PROJECT_ID=your-project
export GCS_RAW_BUCKET=wheel-raw
export GCS_PROCESSED_BUCKET=wheel-processed

# Cache settings
export WHEEL_CACHE__MAX_SIZE_GB=5.0
export WHEEL_CACHE__TTL_DAYS=30
```

### Storage Config

```python
from src.unity_wheel.storage import StorageConfig, GCSConfig

config = StorageConfig(
    cache_config=CacheConfig(
        cache_dir=Path.home() / ".wheel_trading" / "cache",
        max_size_gb=5.0,
        ttl_days=30
    ),
    gcs_config=GCSConfig(
        project_id="your-project",
        raw_bucket="wheel-raw",
        processed_bucket="wheel-processed"
    ),
    enable_gcs_backup=True,
    backup_interval_hours=24
)
```

## Cost Breakdown

| Component | Usage | Monthly Cost |
|-----------|-------|--------------|
| **Local Storage** | 5GB DuckDB | $0 |
| **GCS Standard** | 10GB (first 30d) | ~$0.20 |
| **GCS Nearline** | 190GB (30-365d) | ~$7.60 |
| **GCS Operations** | 10K ops | ~$0.10 |
| **Cloud Run Jobs** | 100 runs @ 30s | ~$2 |
| **Databento API** | 100 chains | ~$40 |
| **Total** | | **< $50/month** |

## Maintenance

### Automatic Cleanup

```python
# Runs automatically every 24 hours
await storage._vacuum()
# - Deletes records older than TTL
# - Reclaims disk space
# - Exports old data to GCS
```

### Manual Maintenance

```bash
# Check storage health
python -c "
from src.unity_wheel.storage import Storage
import asyncio

async def check():
    s = Storage()
    await s.initialize()
    stats = await s.get_storage_stats()
    print(f'DB Size: {stats['db_size_mb']} MB')
    print(f'Oldest data: {stats['option_chains_oldest_days']} days')
    
asyncio.run(check())
"

# Force cleanup
python -c "
from src.unity_wheel.storage import Storage
import asyncio

async def cleanup():
    s = Storage()
    await s.initialize()
    await s.cleanup_old_data()
    print('Cleanup complete')
    
asyncio.run(cleanup())
"
```

## Migration from v1

If you have existing data in SQLite or other formats:

```python
# One-time migration script
from src.unity_wheel.storage import Storage

storage = Storage()
await storage.initialize()

# Import from old SQLite
import sqlite3
conn = sqlite3.connect('old_data.db')
positions = pd.read_sql('SELECT * FROM positions', conn)

# Store in new format
for _, row in positions.iterrows():
    await storage.cache.store_positions(
        account_id=row['account_id'],
        positions=[row['positions']],
        account_data={}
    )
```

## Best Practices

1. **Cache Wisely** - Use appropriate max_age for different data types
2. **Batch Fetches** - Fetch all needed data in one go
3. **Monitor Size** - Check storage stats regularly
4. **Enable GCS** - For production use, enable GCS backup
5. **Clean Regularly** - Let automatic cleanup run daily

## Troubleshooting

### Cache Misses

```python
# Check why cache is missing
logger.info("cache_miss_reason", 
    age_minutes=data_age,
    max_age_minutes=max_age)
```

### Storage Full

```bash
# Increase size limit
export WHEEL_CACHE__MAX_SIZE_GB=10.0

# Or clean up manually
await storage.cache.clear_old_data()
```

### GCS Errors

```python
# GCS is optional - disable if issues
config = StorageConfig(enable_gcs_backup=False)
```