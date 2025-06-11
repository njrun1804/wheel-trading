# Schwab Data Collection - Pull-When-Asked Architecture

## Overview

The Schwab data collection system uses an on-demand, pull-when-asked architecture that fetches data only when the user requests a trading recommendation. This approach minimizes API calls, reduces costs, and eliminates the complexity of continuous background synchronization.

## Architecture

### Core Principles

1. **No Continuous Sync** - Data is fetched only when explicitly requested
2. **Local-First Caching** - DuckDB cache checked before any API call
3. **Intelligent TTLs** - Balance data freshness with API efficiency
4. **Zero Background Processes** - No daemons, cron jobs, or polling

### Data Flow

```
User clicks "Get Recommendation"
    ↓
Storage checks DuckDB cache
    ↓
If data is stale (> TTL):
    → SchwabDataFetcher
    → Schwab API (with rate limiting)
    → Store in DuckDB cache
    ↓
Return data to user
```

## Implementation

### 1. **Unified Storage Layer**

```python
from unity_wheel.storage import Storage

# All data access goes through storage
storage = Storage()
await storage.initialize()

# Fetch positions (cache-aware)
positions = await storage.get_or_fetch_positions(
    account_id="default",
    fetch_func=fetch_schwab_data,
    max_age_minutes=30  # Accept 30-min old data
)
```

### 2. **Simple Data Fetcher**

```python
from unity_wheel.schwab import SchwabClient, SchwabDataFetcher

async def fetch_schwab_data(account_id: str):
    async with SchwabClient() as client:
        fetcher = SchwabDataFetcher(client)
        return await fetcher.fetch_all_data()
```

### 3. **DuckDB Cache Schema**

```sql
-- Position snapshots
CREATE TABLE position_snapshots (
    account_id VARCHAR,
    timestamp TIMESTAMP,
    positions JSON,        -- Full position data
    account_data JSON,     -- Account summary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (account_id, timestamp)
)

-- Automatic cleanup after 30 days
DELETE FROM position_snapshots
WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
```

## Data Collection Strategy

### What We Collect

| Data Type | TTL | Purpose |
|-----------|-----|---------|
| **Positions** | 30 minutes | Current holdings for wheel analysis |
| **Account Data** | 30 seconds | Buying power, margin for sizing |
| **Option Chains** | 15 minutes | Strike selection (via Databento) |
| **Corporate Actions** | Real-time | Detected from position anomalies |

### What We DON'T Collect

- ❌ Order history (not needed for recommendations)
- ❌ Transaction history (not needed for forward-looking decisions)
- ❌ Streaming quotes (pull latest when needed)
- ❌ Historical time series (calculate on demand)

## Usage Examples

### Basic Recommendation Flow

```python
async def get_wheel_recommendation(portfolio_value: float):
    storage = Storage()
    await storage.initialize()

    # 1. Get positions (cached if fresh)
    positions = await storage.get_or_fetch_positions(
        account_id="default",
        fetch_func=fetch_schwab_data,
        max_age_minutes=30
    )

    # 2. Check existing wheel positions
    short_puts = [
        p for p in positions['positions']
        if p.get('option_type') == 'PUT' and p.get('quantity', 0) < 0
    ]

    # 3. Get option chain (from Databento)
    chain = await storage.get_or_fetch_option_chain(
        symbol="U",
        expiration=target_expiry,
        fetch_func=fetch_databento_chain,
        max_age_minutes=15
    )

    # 4. Generate recommendation
    return generate_recommendation(positions, chain)
```

### Force Fresh Data

```python
# User wants very fresh data
fresh_positions = await storage.get_or_fetch_positions(
    account_id="default",
    fetch_func=fetch_schwab_data,
    max_age_minutes=1  # Only accept 1-minute old data
)
```

### Check Cache Status

```python
# See what's cached
stats = await storage.get_storage_stats()
print(f"Cached positions: {stats['position_snapshots_count']}")
print(f"Oldest data: {stats['position_snapshots_oldest_days']} days")
print(f"Cache size: {stats['db_size_mb']} MB")
```

## Cost Optimization

### API Call Reduction

| Pattern | API Calls Saved | How |
|---------|-----------------|-----|
| Cache Hit | 100% | Data still fresh in DuckDB |
| Batch Fetch | 50% | Positions + account in one call |
| Smart TTLs | 80% | Longer TTL for stable data |
| No Polling | 99% | Only fetch on user action |

### Monthly Cost Estimate

```
Assumptions:
- 100 recommendations/month
- 50% cache hit rate
- No background sync

API Calls: 50 (after cache)
Schwab API: Free (no cost)
Storage: < 100MB DuckDB
Total: < $1/month
```

## Error Handling

### Graceful Degradation

```python
try:
    # Try to get fresh data
    positions = await fetch_schwab_data()
except SchwabError:
    # Fall back to cached data
    if cached_positions and age < 2_hours:
        logger.warning("Using cached data due to API error")
        return cached_positions
    else:
        raise  # Too stale to use
```

### Rate Limiting

- **Token bucket**: 100 req/min with burst of 10
- **Automatic retry**: Exponential backoff
- **Circuit breaker**: Prevents cascade failures

## Security

### Credential Management

```python
# Credentials from SecretManager (never hardcoded)
async with SchwabClient() as client:  # Auto-loads from secrets
    fetcher = SchwabDataFetcher(client)
```

### Token Storage

- OAuth tokens: Encrypted in `~/.wheel_trading/auth/tokens.enc`
- Refresh handled automatically
- Machine-specific encryption

## Monitoring

### Health Checks

```python
# Check data freshness
health = {
    'positions_age': get_data_age('positions'),
    'cache_size_mb': get_cache_size(),
    'last_api_error': get_last_error(),
    'api_calls_today': get_api_call_count()
}
```

### Alerts (Optional)

- Data staleness > 2 hours
- Cache size > 5GB
- API errors > 5 consecutive

## Migration from Continuous Sync

### Before (v1.0)
```python
# Complex continuous sync
await ingestion.run_continuous_sync(interval_minutes=5)
# SQLite storage with 100+ MB database
# Background processes required
```

### After (v2.0)
```python
# Simple on-demand fetch
data = await storage.get_or_fetch_positions(...)
# DuckDB cache < 5MB typical
# Zero background processes
```

## Best Practices

1. **Set Appropriate TTLs**
   - Positions: 30 minutes (changes slowly)
   - Options: 15 minutes (more volatile)
   - Account: 30 seconds (critical for margin)

2. **Batch Related Calls**
   ```python
   # Good: One API call
   all_data = await fetcher.fetch_all_data()

   # Bad: Multiple API calls
   positions = await fetcher.fetch_positions()
   account = await fetcher.fetch_account()
   ```

3. **Handle Stale Data Gracefully**
   - Show data age to user
   - Provide "Refresh" button
   - Use appropriate TTLs

## Cloud Run Integration

### Job Configuration

```yaml
# Runs on-demand only
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: wheel-recommendation
spec:
  template:
    spec:
      containers:
      - image: gcr.io/PROJECT/wheel:latest
        env:
        - name: PORTFOLIO_VALUE
          value: "100000"
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
      timeoutSeconds: 300  # 5 min max
```

### Execution

```bash
# User triggers job
gcloud run jobs execute wheel-recommendation \
    --env-vars PORTFOLIO_VALUE=100000

# Job fetches data, generates recommendation, exits
```

## Summary

The pull-when-asked architecture provides:

✅ **Simplicity** - No background processes to manage
✅ **Cost Efficiency** - < $1/month for data operations
✅ **Reliability** - Fewer moving parts = fewer failures
✅ **Performance** - Local cache = instant responses
✅ **Scalability** - Stateless jobs scale infinitely

Perfect for a single-user recommendation system that values simplicity and cost-effectiveness over real-time streaming.
