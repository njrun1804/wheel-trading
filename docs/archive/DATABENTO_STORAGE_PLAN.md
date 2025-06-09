# Databento Storage Plan for Wheel Trading

## Executive Summary

Based on project analysis, we'll implement a **pull-when-asked** storage strategy using DuckDB as primary cache with optional GCS backup. This approach minimizes costs while maintaining data freshness for wheel strategy recommendations.

## What Data to Store

### 1. **Option Chains** (Primary Data)
```sql
-- Core fields needed for wheel strategy
CREATE TABLE option_chains (
    symbol VARCHAR,              -- Underlying symbol (e.g., 'U')
    expiration DATE,            -- Option expiration date
    strike DECIMAL(10,2),       -- Strike price
    option_type VARCHAR(4),     -- 'CALL' or 'PUT'
    bid DECIMAL(10,4),          -- Bid price
    ask DECIMAL(10,4),          -- Ask price
    mid DECIMAL(10,4),          -- Mid price (calculated)
    volume INTEGER,             -- Daily volume
    open_interest INTEGER,      -- Open interest
    implied_volatility DECIMAL(6,4), -- IV
    delta DECIMAL(5,4),         -- Delta (calculated)
    timestamp TIMESTAMP,        -- Data timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, expiration, strike, option_type, timestamp)
);
```

### 2. **Filtered by Moneyness** (80% Storage Reduction)
Only store options within **20% of spot price**:
- For spot = $30: Store strikes $24-$36
- Reduces ~50 strikes to ~10 strikes per expiration
- Critical for cost control

### 3. **Expiration Range**
Store only relevant expirations:
- **30-60 days out** (wheel strategy sweet spot)
- Maximum 3 monthly expirations at a time
- Auto-purge expired options

### 4. **Update Frequency**
- **Real-time quotes**: 15-minute cache (during market hours)
- **Greeks**: 5-minute cache (fast calculations)
- **Historical data**: Daily snapshots only

## How Much to Store

### Storage Estimates
```python
# Per underlying calculations
strikes_per_expiry = 10       # After moneyness filter
expirations = 3               # Monthly expirations in range
updates_per_day = 26          # Every 15 min during market hours
bytes_per_record = 100        # Compressed in DuckDB

# Daily storage
daily_records = strikes_per_expiry * expirations * updates_per_day * 2  # Calls + Puts
daily_mb = (daily_records * bytes_per_record) / 1_000_000  # ~1.6 MB/day

# Monthly storage
monthly_gb = daily_mb * 30 / 1000  # ~0.05 GB/month per symbol

# Total for 10 symbols
total_monthly_gb = monthly_gb * 10  # ~0.5 GB/month
```

### Retention Policy
```yaml
retention:
  option_chains: 30 days      # Historical analysis
  intraday_quotes: 1 day      # Today's movements only
  greeks_cache: 1 hour        # Recalculate frequently
  predictions: 7 days         # Model outputs
  max_db_size: 5 GB           # Hard limit
```

## Implementation Architecture

### 1. **DuckDB Schema**
```python
# Location: ~/.wheel_trading/cache/wheel_cache.duckdb

# Tables:
- option_chains      # Main options data
- spot_prices        # Underlying prices
- greeks_cache       # Calculated Greeks
- wheel_candidates   # Pre-filtered recommendations
- metrics            # Performance tracking

# Indexes:
- idx_symbol_expiry_timestamp
- idx_moneyness_range
- idx_delta_range
```

### 2. **Storage Layer Integration**
```python
# Modified storage.py integration
class DatabentoStorage:
    async def store_options_data(self, chain_data):
        # 1. Apply moneyness filter
        filtered = self._filter_by_moneyness(chain_data)
        
        # 2. Calculate Greeks if missing
        enriched = await self._enrich_with_greeks(filtered)
        
        # 3. Store in DuckDB
        await self.cache.store_option_chain(enriched)
        
        # 4. Optional GCS backup (daily aggregates only)
        if self.gcs_enabled:
            await self._backup_daily_snapshot(enriched)
    
    def _filter_by_moneyness(self, chain_data):
        """Keep only options within 20% of spot"""
        spot = chain_data['spot_price']
        min_strike = spot * 0.8
        max_strike = spot * 1.2
        
        return {
            'calls': [c for c in chain_data['calls'] 
                     if min_strike <= c['strike'] <= max_strike],
            'puts': [p for p in chain_data['puts'] 
                    if min_strike <= p['strike'] <= max_strike]
        }
```

### 3. **Get-or-Fetch Pattern**
```python
async def get_wheel_candidates(self, symbol, target_delta=0.30):
    # 1. Check cache first
    cached = await self.storage.get_candidates(
        symbol, 
        target_delta,
        max_age_minutes=15
    )
    if cached:
        return cached
    
    # 2. Fetch from Databento
    chain = await self.databento.get_option_chain(symbol)
    
    # 3. Filter and calculate
    candidates = self._find_wheel_candidates(chain, target_delta)
    
    # 4. Store and return
    await self.storage.store_candidates(candidates)
    return candidates
```

### 4. **Cost Optimization**
- **Batch requests**: Fetch all expirations in one call
- **Smart TTLs**: Longer cache for stable data
- **Compression**: DuckDB native compression
- **Selective fields**: Only store needed columns
- **Background cleanup**: Daily vacuum operations

## Migration Steps

1. **Update storage schema** (1 hour)
   - Create DuckDB tables
   - Set up indexes
   - Configure retention

2. **Integrate Databento client** (2 hours)
   - Add to existing Storage class
   - Implement moneyness filtering
   - Add Greeks calculation

3. **Update pull script** (1 hour)
   - Use unified storage
   - Add proper error handling
   - Implement retry logic

4. **Testing** (1 hour)
   - Verify data quality
   - Check storage limits
   - Validate calculations

## Success Metrics

- **Storage < 1GB/month** for typical usage
- **Cache hit rate > 80%** during market hours
- **Data freshness < 15 minutes** for recommendations
- **Query time < 100ms** for candidate selection
- **Monthly cost < $1** (API + storage)

## Next Steps

1. Run `debug_databento.py` to verify connectivity
2. Create DuckDB schema
3. Implement moneyness filtering
4. Add Greeks enrichment
5. Deploy and monitor