# Databento Integration - Pull-When-Asked Architecture

## Overview

The Databento integration provides on-demand options data for the Unity Wheel Trading Bot. Using a REST-only approach, it fetches option chains only when the user requests a trading recommendation, with intelligent caching to minimize API costs.

## Architecture

### Core Design

1. **No Streaming** - REST API only, no WebSocket subscriptions
2. **No Background Sync** - Data fetched only on user request
3. **DuckDB Cache** - 15-minute TTL for options data
4. **Smart Filtering** - Only ±20% of spot price to reduce data 80%
5. **Pay-Per-Use** - Databento "Blast" pricing model

### Data Flow

```
User requests recommendation
    ↓
Check DuckDB cache (15 min TTL)
    ↓
If stale:
    → DatentoClient.get_option_chain()
    → Filter strikes (±20% of spot)
    → Store in DuckDB
    → Optional GCS backup (raw JSON)
    ↓
Return option chain for analysis
```

## Implementation

### 1. **Pull-When-Asked Pattern**

```python
from unity_wheel.storage import Storage
from unity_wheel.databento import DatentoClient

# Define fetch function
async def fetch_option_chain(symbol: str, expiration: datetime):
    client = DatentoClient()  # API key from SecretManager
    chain = await client.get_option_chain(
        underlying=symbol,
        expiration=expiration,
        timestamp=None  # Latest data
    )
    
    # Convert to storage format
    return {
        'symbol': symbol,
        'expiration': expiration.isoformat(),
        'spot_price': float(chain.spot_price),
        'timestamp': chain.timestamp.isoformat(),
        'calls': [opt.to_dict() for opt in chain.calls],
        'puts': [opt.to_dict() for opt in chain.puts]
    }

# Use unified storage
storage = Storage()
chain = await storage.get_or_fetch_option_chain(
    symbol="U",
    expiration=datetime(2024, 2, 16),
    fetch_func=fetch_option_chain,
    max_age_minutes=15
)
```

### 2. **DuckDB Cache Schema**

```sql
-- Option chains cached locally
CREATE TABLE option_chains (
    symbol VARCHAR,
    expiration DATE,
    timestamp TIMESTAMP,
    spot_price DECIMAL,
    data JSON,  -- Full chain data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, expiration, timestamp)
)

-- Auto cleanup after 30 days
DELETE FROM option_chains 
WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
```

### 3. **Strike Filtering**

```python
def filter_strikes(chain: OptionChain) -> OptionChain:
    """Keep only strikes within 20% of spot."""
    spot = chain.spot_price
    min_strike = spot * 0.80
    max_strike = spot * 1.20
    
    filtered_calls = [
        opt for opt in chain.calls
        if min_strike <= opt.strike <= max_strike
    ]
    
    filtered_puts = [
        opt for opt in chain.puts  
        if min_strike <= opt.strike <= max_strike
    ]
    
    # Reduces data by ~80%
    return OptionChain(
        calls=filtered_calls,
        puts=filtered_puts,
        **chain.dict(exclude={'calls', 'puts'})
    )
```

## Data Requirements

### What We Fetch

| Data Type | When | TTL | Purpose |
|-----------|------|-----|---------|
| **Option Chains** | On demand | 15 min | Strike selection |
| **Instrument Definitions** | With chain | 24 hours | Contract details |
| **Spot Price** | With chain | 5 min | Moneyness calc |

### What We DON'T Fetch

- ❌ Streaming quotes (no WebSocket needed)
- ❌ Historical tick data (not needed for recommendations)
- ❌ Full order book depth (top-of-book sufficient)
- ❌ All strikes (filter to ±20% of spot)
- ❌ All expirations (only target DTE range)

## Cost Optimization

### API Cost Reduction

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| **Caching** | 90%+ | 15-min TTL prevents duplicate calls |
| **Strike Filtering** | 80% | Only ±20% of spot |
| **No Streaming** | 100% | No subscription fees |
| **Batch Requests** | 50% | Multiple expirations per call |

### Monthly Cost Estimate

```
Assumptions:
- 100 recommendations/month
- 2 expirations per recommendation  
- 50% cache hit rate

API Calls:
- 100 option chains (after cache)
- Pay-per-request pricing

Databento cost: ~$40/month
Storage: < 100MB DuckDB
GCS backup: < $1/month
Total: < $50/month
```

## Usage Examples

### Get Wheel Candidates

```python
from unity_wheel.databento.integration import DatentoIntegration

# Initialize with storage
integration = DatentoIntegration(storage)

# Find candidates on demand
candidates = await integration.get_wheel_candidates(
    underlying="U",
    target_delta=0.30,
    dte_range=(30, 60),
    min_premium_pct=1.0,
    use_cache=True  # Check cache first
)

# Returns filtered, validated options
for candidate in candidates:
    print(f"Strike: ${candidate.strike}")
    print(f"Premium: ${candidate.premium:.2f}")
    print(f"Delta: {candidate.delta:.3f}")
    print(f"Annualized return: {candidate.annual_return:.1%}")
```

### Analyze Existing Position

```python
# Check current market for position
analysis = await integration.analyze_position(
    symbol="U  240216P00035000",  # OCC format
    quantity=-5  # Short 5 puts
)

print(f"Current delta: {analysis['current_delta']:.3f}")
print(f"Days to expiry: {analysis['dte']}")
print(f"Profit %: {analysis['profit_pct']:.1%}")
print(f"Should roll: {analysis['should_roll']}")
```

### Data Quality Validation

```python
# Validate chain before using
quality = await client.validate_data_quality(
    chain=option_chain,
    max_spread_pct=5.0,
    min_size=10
)

if quality.confidence_score < 0.8:
    logger.warning(f"Low quality data: {quality.confidence_score:.0%}")
```

## Configuration

Add to `config.yaml`:

```yaml
databento:
  # No API key here - use SecretManager
  cache_ttl_minutes: 15
  moneyness_filter: 0.20  # ±20% of spot
  
  validation:
    max_spread_pct: 5.0
    min_bid_size: 10
    min_ask_size: 10
```

## Error Handling

### Graceful Degradation

```python
try:
    # Try to get fresh data
    chain = await client.get_option_chain(...)
except DatabentoError as e:
    # Check cache even if stale
    cached = await storage.get_cached_chain(symbol, expiration)
    if cached and cached.age_minutes < 120:
        logger.warning("Using stale cache due to API error")
        return cached
    raise  # Too old to use
```

### Rate Limiting

- 100 requests/second limit
- Automatic exponential backoff
- Circuit breaker after 5 failures

## Security

### API Key Management

```python
# Never hardcode - use SecretManager
from unity_wheel.secrets import get_databento_api_key

api_key = get_databento_api_key()
client = DatentoClient(api_key=api_key)
```

### Data Privacy

- All data stored locally
- Optional GCS backup (encrypted)
- No data leaves user's control

## Migration from Streaming

### Before (v1.0)
```python
# Complex streaming setup
await client.stream_live_updates(...)
# Continuous WebSocket management
# Background data sync
# BigQuery for analytics
```

### After (v2.0)
```python
# Simple on-demand fetch
chain = await storage.get_or_fetch_option_chain(...)
# No WebSocket needed
# No background processes
# DuckDB for everything
```

## Best Practices

1. **Use Appropriate TTLs**
   - Option chains: 15 minutes
   - Definitions: 24 hours  
   - Underlying price: 5 minutes

2. **Filter Aggressively**
   - Only strikes you might trade
   - Only expirations in target range
   - Skip illiquid options

3. **Monitor Costs**
   ```python
   stats = await storage.get_api_stats()
   print(f"Databento calls today: {stats['databento_calls']}")
   print(f"Cache hit rate: {stats['cache_hit_rate']:.0%}")
   ```

## Summary

The pull-when-asked Databento integration provides:

✅ **Simplicity** - No streaming complexity  
✅ **Cost Control** - Pay only for what you use  
✅ **Performance** - Local cache for fast responses  
✅ **Reliability** - Fewer failure modes  
✅ **Flexibility** - Easy to add new data sources  

Perfect for a recommendation-only system that values cost efficiency and simplicity over real-time streaming.