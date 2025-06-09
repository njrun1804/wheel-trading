# Schema Recommendations Based on API Verification

## Summary

After reviewing the API inputs from Schwab and Databento against our DuckDB schema, the current design is well-aligned with the data we receive. The schema correctly handles all critical fields needed for the wheel strategy.

## Current Schema Strengths

1. **JSON Flexibility**: Using JSON columns for raw API data allows us to:
   - Preserve all API fields without schema migrations
   - Handle API changes gracefully
   - Debug with complete data context

2. **Proper Data Types**:
   - DECIMAL types for financial values prevent floating-point errors
   - TIMESTAMP fields maintain microsecond precision
   - VARCHAR for symbols handles both stocks and complex option symbols

3. **Efficient Indexing**:
   - Primary keys prevent duplicate data
   - Created_at indexes enable efficient cache expiration
   - Symbol indexes speed up lookups

## Verified API Mappings

### Schwab Position Fields → DuckDB
```
API Field           → Storage Location
symbol              → positions JSON blob
quantity            → positions JSON blob
assetType          → positions JSON blob (mapped to position_type)
marketValue        → positions JSON blob
averagePrice       → positions JSON blob (used to calculate cost_basis)
unrealizedPnL      → positions JSON blob
realizedPnL        → positions JSON blob
```

### Databento Option Chain → DuckDB
```
API Field           → Storage Location          → Transformation
instrument_id       → data JSON blob            → Direct storage
ts_event           → timestamp column           → Nanoseconds to datetime
bid_px             → data JSON blob            → 1e-9 to decimal
ask_px             → data JSON blob            → 1e-9 to decimal
bid_sz             → data JSON blob            → Direct storage
ask_sz             → data JSON blob            → Direct storage
```

## Recommended Schema Enhancements

### 1. Add Option Chain Summary Table (Optional)
```sql
CREATE TABLE IF NOT EXISTS option_chain_summary (
    symbol VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    total_volume INTEGER,
    put_call_ratio DECIMAL(6,4),
    avg_iv DECIMAL(6,4),
    PRIMARY KEY (symbol, expiration, timestamp)
)
```
**Benefit**: Faster analytics without parsing JSON

### 2. Add Materialized Views for Common Queries
```sql
CREATE VIEW active_positions AS
SELECT
    account_id,
    json_extract(positions, '$[*].symbol') as symbols,
    json_extract(positions, '$[*].market_value') as values
FROM position_snapshots
WHERE timestamp = (SELECT MAX(timestamp) FROM position_snapshots);
```
**Benefit**: Simplify common queries

### 3. Add Data Quality Tracking
```sql
CREATE TABLE IF NOT EXISTS data_quality_log (
    id INTEGER PRIMARY KEY,
    source VARCHAR NOT NULL,  -- 'schwab' or 'databento'
    timestamp TIMESTAMP NOT NULL,
    quality_score DECIMAL(4,2),
    issues JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```
**Benefit**: Track data quality over time

## No Critical Changes Needed

The current schema successfully captures all required data from both APIs. The recommended enhancements are optional optimizations that could improve query performance but are not necessary for correct operation.

## Best Practices Confirmed

1. ✅ **Decimal Precision**: All monetary values use DECIMAL to avoid rounding errors
2. ✅ **UTC Timestamps**: All times stored in UTC for consistency
3. ✅ **JSON Flexibility**: Raw API responses preserved for future needs
4. ✅ **Cache Management**: TTL implemented via created_at timestamps
5. ✅ **Primary Keys**: Prevent duplicate data entries
