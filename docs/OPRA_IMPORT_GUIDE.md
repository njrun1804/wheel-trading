# OPRA Data Import Guide

This guide explains how to import OPRA options data into the unified DuckDB storage used by the Unity Wheel Trading system.

## Overview

The OPRA data import integrates options data from Databento into our existing unified storage structure, which already contains:
- Databento options chains
- FRED economic data
- Schwab account data
- Calculated Greeks and risk metrics

## Prerequisites

1. **Required Python packages:**
   ```bash
   pip install zstandard pandas duckdb
   ```

2. **OPRA data files** downloaded from Databento in the expected format:
   - `*.ohlcv-1d.csv.zst` - Compressed daily OHLCV data
   - `symbology.csv` - Symbol mapping file
   - `metadata.json` - Query metadata

## Import Process

### Quick Start

```bash
# Run the import with default settings
python import_opra_to_unified_db.py

# Or specify custom paths
python import_opra_to_unified_db.py \
  --data-dir /path/to/OPRA-data \
  --cache-dir data/cache
```

### What the Import Does

1. **Creates unified tables** in DuckDB:
   - `options_data` - Stores all options OHLCV data
   - `opra_symbology` - Maps instrument IDs to option symbols

2. **Parses OPRA symbols** to extract:
   - Underlying symbol (e.g., "U" for Unity)
   - Expiration date
   - Strike price
   - Option type (PUT/CALL)

3. **Filters for Unity options** - Only imports options where underlying = "U"

4. **Integrates with existing storage** - Uses the same DuckDB instance as other data providers

## Database Schema

### options_data Table
```sql
CREATE TABLE options_data (
    ts_event TIMESTAMP NOT NULL,      -- Event timestamp
    instrument_id BIGINT NOT NULL,    -- Unique option identifier
    symbol VARCHAR NOT NULL,          -- Underlying symbol (e.g., 'U')
    expiration DATE NOT NULL,         -- Option expiration date
    strike DECIMAL(10,2) NOT NULL,    -- Strike price
    option_type VARCHAR(4) NOT NULL,  -- 'PUT' or 'CALL'
    open DECIMAL(10,4),              -- Opening price
    high DECIMAL(10,4),              -- High price
    low DECIMAL(10,4),               -- Low price
    close DECIMAL(10,4),             -- Closing price
    volume BIGINT,                   -- Trading volume
    data_source VARCHAR DEFAULT 'OPRA',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (instrument_id, ts_event)
)
```

### Query Examples

After import, you can query the data:

```sql
-- Connect to database
duckdb data/cache/wheel_cache.duckdb

-- Find most liquid Unity put strikes
SELECT
    strike,
    AVG(volume) as avg_volume,
    COUNT(*) as days_traded
FROM options_data
WHERE symbol = 'U'
    AND option_type = 'PUT'
    AND ts_event >= '2025-01-01'
GROUP BY strike
ORDER BY avg_volume DESC
LIMIT 10;

-- Get recent option chain for specific expiration
SELECT
    strike,
    option_type,
    close as last_price,
    volume
FROM options_data
WHERE symbol = 'U'
    AND expiration = '2025-02-21'
    AND DATE(ts_event) = '2025-01-15'
ORDER BY option_type, strike;

-- Analyze put skew
SELECT
    DATE(ts_event) as date,
    strike,
    close / (SELECT close FROM options_data o2
             WHERE o2.symbol = 'U'
             AND o2.strike = 35
             AND o2.option_type = 'PUT'
             AND DATE(o2.ts_event) = DATE(o1.ts_event)) as relative_price
FROM options_data o1
WHERE symbol = 'U'
    AND option_type = 'PUT'
    AND strike IN (30, 32.5, 35, 37.5, 40)
    AND ts_event >= '2025-01-01'
ORDER BY date DESC, strike;
```

## Integration with Wheel Trading System

The imported data automatically integrates with:

1. **WheelStrategy** - Can use historical option prices for backtesting
2. **DatabentoStorageAdapter** - Unified storage for all options data
3. **RiskAnalytics** - Historical volatility calculations
4. **PerformanceTracker** - Track recommendation accuracy

### Using in Code

```python
from src.unity_wheel.storage.storage import Storage

# Initialize storage
storage = Storage()
await storage.initialize()

# Query historical options
async with storage.cache.connection() as conn:
    df = conn.execute("""
        SELECT * FROM options_data
        WHERE symbol = 'U'
            AND option_type = 'PUT'
            AND expiration = '2025-02-21'
            AND strike BETWEEN 30 AND 40
        ORDER BY ts_event DESC
        LIMIT 100
    """).df()
```

## Performance Considerations

- Import creates indexes on common query patterns
- Data is stored in columnar format for fast analytics
- Automatic compression reduces storage by ~80%
- TTL set to 365 days (configurable)

## Troubleshooting

### Import Errors

1. **"zstandard package required"**
   ```bash
   pip install zstandard
   ```

2. **"No Unity options found"**
   - Check that data files contain Unity (U) options
   - Verify symbology.csv has Unity mappings

3. **Database locked errors**
   - Ensure no other processes are using the database
   - Check file permissions on cache directory

### Verification

After import, verify data:

```python
# Check import status
python -c "
from src.unity_wheel.storage.storage import Storage
import asyncio

async def check():
    storage = Storage()
    await storage.initialize()
    stats = await storage.get_storage_stats()
    print(f'Storage size: {stats}')

asyncio.run(check())
"
```

## Maintenance

- Old data is automatically cleaned up based on TTL
- Run `VACUUM` periodically to reclaim space:
  ```sql
  duckdb data/cache/wheel_cache.duckdb -c "VACUUM;"
  ```

- Monitor storage size:
  ```bash
  du -h data/cache/wheel_cache.duckdb
  ```
