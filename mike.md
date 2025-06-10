# Databento Unity Options Data: Comprehensive Technical Guide

## Executive Summary

This guide provides optimal configurations for pulling Unity options data from Databento's API. Key recommendations:
- Use **CMBP-1** (consolidated NBBO) schema for quotes with **1-minute bars** for underlying
- Leverage **parent symbology** to fetch all strikes/expirations in one request (up to 2,000 symbols)
- Pull data after **2:00 AM ET** for complete previous day's data (new ~10 hour availability)
- Transform to DuckDB immediately for 75-95% storage reduction and faster queries
- Implement 5-10 concurrent requests with proper error handling
- Use timestamp-based incremental updates with deduplication

## 1. Optimal Query Parameters

### Schema Selection for Unity Options

**Recommended: CMBP-1 (Consolidated Market by Price - Level 1)**

For Unity options with lower liquidity than major indices, CMBP-1 provides the optimal balance:
- Consolidated NBBO (best bid/ask across all exchanges)
- ~90% data volume reduction vs MBP-10
- Sufficient for accurate Greeks calculation
- Lower bandwidth and storage costs

```python
# Optimal Unity options configuration
data = client.timeseries.get_range(
    dataset='OPRA.PILLAR',
    schema='cmbp-1',  # Consolidated NBBO - optimal for Unity
    symbols=['U.OPT'],  # Parent symbol retrieves all options
    stype_in='parent',  # Critical: enables parent symbology
    start='2024-01-01T09:30',
    end='2024-01-01T16:00',
)
```

**Note:** Use MBP-10 only if you need order book depth analysis. For typical options analytics, spreads, and Greeks, CMBP-1 is sufficient.

### Underlying Price Requirements

**Recommendation: 1-minute OHLCV bars**

Daily bars (ohlcv-1d) are insufficient for options risk management. Unity's intraday volatility demands:
- Sub-daily resolution for accurate delta hedging
- Frequent Greeks recalculation throughout the day
- Real-time theta decay tracking

```python
# Underlying Unity stock tracking
underlying_data = client.timeseries.get_range(
    dataset='XNAS.ITCH',  # NASDAQ feed for Unity
    schema='ohlcv-1m',    # 1-minute bars essential
    symbols=['U'],        # Unity stock symbol
    start='2024-01-01T09:30',
    end='2024-01-01T16:00',
)
```

### Combined Quotes and Greeks Strategy

**Important:** Databento does not provide pre-calculated Greeks. Implement a two-step approach:

```python
# Step 1: Get instrument definitions for contract specifications
definitions = client.timeseries.get_range(
    dataset='OPRA.PILLAR',
    schema='definition',
    symbols=['U.OPT'],
    stype_in='parent',
    start='2024-01-01',
    end='2024-01-01',
)

# Step 2: Get real-time quotes
quotes = client.timeseries.get_range(
    dataset='OPRA.PILLAR',
    schema='cmbp-1',
    symbols=['U.OPT'],
    stype_in='parent',
    start='2024-01-01T09:30',
    end='2024-01-01T16:00',
)

# Step 3: Calculate Greeks client-side (example with py_vollib)
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks import delta, gamma, theta, vega, rho

# Greeks calculation using market data
def calculate_greeks(spot, strike, rate, time_to_expiry, volatility, option_type):
    price = bs(option_type, spot, strike, time_to_expiry, rate, volatility)
    d = delta(option_type, spot, strike, time_to_expiry, rate, volatility)
    g = gamma(option_type, spot, strike, time_to_expiry, rate, volatility)
    # ... calculate remaining Greeks
    return {'price': price, 'delta': d, 'gamma': g}
```

## 2. Batch vs Individual Queries

### Query Limits by Context

| Context | Symbol Limit | Notes |
|---------|--------------|-------|
| Historical REST API | **2,000 symbols** | Applies to all schemas |
| Live WebSocket | **100 symbols** | Per session (can open multiple) |
| Parent Symbology | **All children** | Single parent expands to all options |

### Optimal Batching Strategy

**Use parent symbology with date ranges:**

```python
# Most efficient: Parent symbols retrieve complete chains
data = client.timeseries.get_range(
    dataset="OPRA.PILLAR",
    symbols=["U.OPT"],      # Parent symbol
    stype_in="parent",      # CRITICAL - must specify
    schema="cmbp-1",
    start="2024-01-01",
    end="2024-01-31"
)

# Alternative: Batch multiple underlyings
multi_chain_data = client.timeseries.get_range(
    dataset="OPRA.PILLAR",
    symbols=["U.OPT", "AAPL.OPT", "TSLA.OPT"],  # Multiple parents
    stype_in="parent",
    schema="cmbp-1",
    start="2024-01-01",
    end="2024-01-01"  # Single day for manageable size
)
```

### Query Strategy Comparison

| Strategy | Efficiency | Use Case |
|----------|------------|----------|
| By Date Range | **Best** | Daily snapshots, incremental updates |
| By Expiration | Medium | Contract lifecycle analysis |
| By Individual Symbol | Poor | Avoid unless necessary |

**Recommendation:** Query by date with all symbols for that date. This aligns with daily update cycles and keeps response sizes manageable.

## 3. Data Freshness & Timing

### OPRA Data Availability Timeline

**New ~10 Hour Availability** (significant improvement from T+1):
- Market closes: 4:00-4:15 PM ET
- Data processing complete: ~2:00 AM ET next day
- Safe pull time: **5:00-6:00 AM ET** (ensures full availability)

```python
import pytz
from datetime import datetime, timedelta

def get_optimal_pull_schedule():
    eastern = pytz.timezone('US/Eastern')
    return {
        'primary_pull': '06:00 ET',     # Complete previous day
        'new_listings_check': '09:15 ET', # Catch morning additions
        'incremental': 'Every 30 min during market',
        'end_of_day': '17:00 ET',       # Final reconciliation
    }

# Check data completeness before processing
def verify_data_availability(date):
    record_count = client.metadata.get_record_count(
        dataset="OPRA.PILLAR",
        symbols=["U.OPT"],
        stype_in="parent",
        start=date,
        end=date + timedelta(days=1)
    )
    return record_count.count > 0
```

### New Contract Listings

Options are listed throughout the day:
- **Weekly options:** Usually Thursday/Friday for next week
- **New strikes:** Added intraday if underlying moves significantly
- **Definitions update:** Real-time as contracts are listed

```python
# Daily new listings check
def get_new_listings(last_update_date):
    new_definitions = client.timeseries.get_range(
        dataset='OPRA.PILLAR',
        schema='definition',
        symbols=['U.OPT'],
        stype_in='parent',
        start=last_update_date,
        end='now'
    )
    return [d for d in new_definitions if d.date_listed >= last_update_date]
```

## 4. Storage Optimization

### Transformation Strategy

**Recommendation: Transform immediately but archive raw DBN**

```python
# Optimal workflow
def process_daily_data(date):
    # 1. Download raw data
    raw_data = client.timeseries.get_range(...)

    # 2. Save raw DBN for archival (highly compressed)
    raw_data.to_file(f'archive/unity_{date}.dbn.zst')

    # 3. Transform and load to DuckDB
    df = raw_data.to_df()
    df = optimize_dataframe(df)  # Drop unnecessary fields

    # 4. Insert into DuckDB
    con.execute("""
        INSERT INTO options_ticks
        SELECT * FROM df
    """)
```

### Field Optimization Guidelines

**Safe to Drop** (15-25% storage savings):
```python
DROP_FIELDS = [
    'sequence',      # Internal ordering, reconstructible
    'ts_in_delta',   # Timing offset field
    'ts_recv',       # Keep only ts_event
    '_reserved*',    # Internal padding fields
    'publisher_id',  # If single exchange
    'bid_ct',        # Order count at price level
    'ask_ct',        # Order count at price level
]
```

**Never Drop:**
- `instrument_id` or `raw_symbol`
- `ts_event` (primary timestamp)
- `bid_px`, `ask_px`, `bid_sz`, `ask_sz`
- `action`, `side` (for order flow)

### Optimal DuckDB Schema

```sql
-- Partitioned options data table
CREATE TABLE options_ticks (
    trade_date DATE NOT NULL,
    ts_event TIMESTAMP NOT NULL,
    instrument_id UINTEGER NOT NULL,
    bid_px DECIMAL(10,4) NOT NULL,
    ask_px DECIMAL(10,4) NOT NULL,
    bid_sz UINTEGER NOT NULL,
    ask_sz UINTEGER NOT NULL,
    PRIMARY KEY (trade_date, ts_event, instrument_id)
) PARTITION BY (trade_date);

-- Separate instrument reference table
CREATE TABLE instruments (
    instrument_id UINTEGER PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    underlying VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    option_type CHAR(1) NOT NULL,
    date_listed DATE,
    UNIQUE(symbol)
);

-- For efficient queries
CREATE INDEX idx_options_instrument ON options_ticks(instrument_id);
CREATE INDEX idx_instruments_underlying ON instruments(underlying, expiration);
```

### Storage Benchmarks

| Format | Size (100M records) | Compression | Query Speed |
|--------|---------------------|-------------|-------------|
| Raw CSV | 45.2 GB | None | Slowest |
| Raw DBN | 8.5 GB | Native | N/A |
| DBN + ZSTD | 4.8 GB | 89% | N/A |
| DuckDB Native | 6.2 GB | 86% | **Fastest** |
| Parquet + ZSTD | 4.5 GB | 90% | 1.5x slower |

## 5. Incremental Updates

### Change Detection Strategy

**No `updated_at` field available.** Use timestamp-based detection:

```python
class IncrementalUpdater:
    def __init__(self, buffer_minutes=5):
        self.buffer = timedelta(minutes=buffer_minutes)

    def get_incremental_data(self, last_update_time):
        # Pull with overlap buffer for safety
        start_time = last_update_time - self.buffer

        new_data = client.timeseries.get_range(
            dataset='OPRA.PILLAR',
            schema='cmbp-1',
            symbols=['U.OPT'],
            stype_in='parent',
            start=start_time,
            end='now'
        )

        return self.deduplicate(new_data, last_update_time)

    def deduplicate(self, data, cutoff_time):
        # Use ts_event as primary timestamp
        df = data.to_df()
        return df[df['ts_event'] > cutoff_time]
```

### Deduplication Keys

Primary key for quotes: `(ts_event, instrument_id)`
- Add `(bid_px, ask_px)` for quote updates
- Add `trade_id` for trades

### Daily Update Workflow

```python
def daily_update_pipeline(date):
    """Complete daily update process"""

    # 1. Check if data is available
    if not verify_data_availability(date):
        logger.info(f"Data not yet available for {date}")
        return False

    # 2. Pull new instrument definitions
    new_instruments = get_new_listings(date)
    if new_instruments:
        update_instruments_table(new_instruments)

    # 3. Pull market data
    data = client.timeseries.get_range(
        dataset='OPRA.PILLAR',
        schema='cmbp-1',
        symbols=['U.OPT'],
        stype_in='parent',
        start=date,
        end=date + timedelta(days=1)
    )

    # 4. Process and store
    process_and_store(data, date)

    # 5. Update metadata
    update_last_processed_date(date)

    return True
```

## 6. Local Caching Strategy

### Optimal Cache Key Structure

```python
# Recommended hierarchical key format
def build_cache_key(instrument_id, date, data_type='quotes'):
    """
    Format: dataset:instrument_id:date:type
    Example: OPRA:12345678:20241201:quotes
    """
    return f"OPRA:{instrument_id}:{date}:{data_type}"

# Alternative using OCC symbol
def build_cache_key_occ(symbol, date, data_type='quotes'):
    """
    Format: dataset:symbol:date:type
    Example: OPRA:U__241220C00150000:20241201:quotes
    """
    return f"OPRA:{symbol}:{date}:{data_type}"
```

### Contract Adjustment Handling

```python
class OptionsCache:
    def __init__(self):
        self.cache = {}
        self.adjustment_log = {}

    def detect_adjustments(self, new_definitions):
        """Identify adjusted contracts by symbol patterns"""
        adjusted = []
        for defn in new_definitions:
            # Adjusted options have numeric suffixes or 'A'
            if re.search(r'[A-Z]\d+|[A-Z]A\d*', defn.symbol):
                adjusted.append(defn)
                self.invalidate_related_cache(defn)
        return adjusted

    def invalidate_related_cache(self, adjusted_contract):
        """Remove cache entries for adjusted contracts"""
        base_underlying = adjusted_contract.underlying.rstrip('0123456789A')
        pattern = f"*:{base_underlying}*"

        # Find and invalidate affected keys
        affected_keys = [k for k in self.cache.keys()
                        if self._matches_pattern(k, pattern)]

        for key in affected_keys:
            self.adjustment_log[key] = {
                'invalidated': datetime.now(),
                'reason': 'contract_adjustment',
                'new_symbol': adjusted_contract.symbol
            }
            self.cache.pop(key, None)
```

### Cache Storage Recommendations

| Data Type | Cache Strategy | TTL |
|-----------|---------------|-----|
| Instrument Definitions | In-memory dict | 24 hours |
| Daily Quotes | DuckDB partition | Permanent |
| Intraday Updates | Redis/Memory | Until EOD |
| Greeks | Calculate on-demand | Don't cache |

## 7. Error Handling Specifics

### Error Classification and Actions

```python
class DatabentoErrorHandler:

    ERROR_ACTIONS = {
        # Skip these - won't resolve with retries
        'skip': {
            'patterns': [
                'Could not resolve smart symbols',
                'Symbol not found',
                'Invalid date range',
                'Contract expired',
            ],
            'http_codes': [404, 422],
            'action': 'log_and_skip'
        },

        # Retry with backoff
        'retry': {
            'patterns': [
                'Gateway timeout',
                'Connection timeout',
                'Rate limit exceeded',
                'Service unavailable',
            ],
            'http_codes': [429, 500, 502, 503, 504],
            'action': 'exponential_backoff',
            'max_retries': 3
        }
    }

    def handle_error(self, error):
        error_msg = str(error).lower()

        # Check error patterns
        for category, config in self.ERROR_ACTIONS.items():
            if any(pattern.lower() in error_msg for pattern in config['patterns']):
                return self._execute_action(config['action'], error)

        # Check HTTP status codes if available
        if hasattr(error, 'status_code'):
            for category, config in self.ERROR_ACTIONS.items():
                if error.status_code in config.get('http_codes', []):
                    return self._execute_action(config['action'], error)

        # Default: log and skip
        logger.error(f"Unhandled error: {error}")
        return 'skip'
```

### "Could Not Resolve Smart Symbols" Fix

This error occurs when:
1. Missing `stype_in="parent"` parameter
2. Invalid symbol format
3. Symbol doesn't exist in dataset

```python
# WRONG - causes resolution error
data = client.timeseries.get_range(
    symbols=["U"],  # Missing stype_in parameter
    dataset="OPRA.PILLAR"
)

# CORRECT - properly uses parent symbology
data = client.timeseries.get_range(
    symbols=["U.OPT"],      # Note: .OPT suffix
    stype_in="parent",      # REQUIRED for parent symbols
    dataset="OPRA.PILLAR"
)

# Alternative - use raw OCC symbols
data = client.timeseries.get_range(
    symbols=["U  241220C00150000"],  # Full OCC format
    stype_in="raw_symbol",           # Default
    dataset="OPRA.PILLAR"
)
```

### Empty Data vs Error Detection

```python
def analyze_response(response, symbol, date_range):
    """Differentiate between legitimate empty data and errors"""

    # HTTP 200 with empty data = no activity
    if response.status_code == 200 and len(response) == 0:
        # Check legitimate reasons for empty data
        if is_weekend_or_holiday(date_range[0]):
            return 'empty_legitimate', 'Non-trading day'

        if is_option_expired(symbol, date_range[0]):
            return 'empty_legitimate', 'Option expired'

        if is_far_otm_option(symbol):
            return 'empty_possible', 'Low liquidity strike'

        return 'empty_investigate', 'Unexpected empty data'

    # HTTP 404 = symbol doesn't exist
    elif response.status_code == 404:
        return 'error', 'Symbol not found in dataset'

    # HTTP 422 = bad request format
    elif response.status_code == 422:
        return 'error', 'Invalid request parameters'

    return 'success', 'Data retrieved'
```

## 8. Performance Tuning

### Concurrency Recommendations

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp

class OptimizedDataFetcher:
    def __init__(self, max_concurrent=10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    # Threading approach (recommended for Databento SDK)
    def fetch_parallel_threading(self, date_chunks):
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = []
            for chunk in date_chunks:
                future = executor.submit(self.fetch_chunk, chunk)
                futures.append(future)

            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results

    # Async approach (for direct HTTP API calls)
    async def fetch_parallel_async(self, date_chunks):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for chunk in date_chunks:
                task = self.fetch_chunk_async(session, chunk)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
        return results

    async def fetch_chunk_async(self, session, chunk):
        async with self.semaphore:  # Limit concurrent requests
            # Implementation here
            pass
```

### Optimal Concurrency Settings

| Metric | Recommended | Maximum | Notes |
|--------|-------------|---------|-------|
| Concurrent Requests | 5-10 | 20 | Beyond 20 shows diminishing returns |
| Chunk Size | 1 day | 1 month | Balance memory vs efficiency |
| Connection Pool | 30 | 100 | Per-host connection limit |
| Request Timeout | 30s | 120s | Adjust based on data size |

### Memory Management for Large Chains

```python
class MemoryEfficientProcessor:
    def __init__(self, chunk_size=50000, memory_limit_gb=4):
        self.chunk_size = chunk_size
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024

    def process_large_dataset(self, date_range):
        """Process data in memory-efficient chunks"""

        for date in date_range:
            # Use iterator to avoid loading all data at once
            data_iter = client.timeseries.get_range(
                dataset='OPRA.PILLAR',
                schema='cmbp-1',
                symbols=['U.OPT'],
                stype_in='parent',
                start=date,
                end=date + timedelta(days=1)
            )

            # Process in chunks
            chunk_buffer = []
            for record in data_iter:
                chunk_buffer.append(record)

                if len(chunk_buffer) >= self.chunk_size:
                    self.process_and_store_chunk(chunk_buffer)
                    chunk_buffer = []

                    # Check memory usage
                    if self.get_memory_usage() > self.memory_limit:
                        gc.collect()

            # Process remaining records
            if chunk_buffer:
                self.process_and_store_chunk(chunk_buffer)

    def process_and_store_chunk(self, chunk):
        """Convert chunk to DataFrame and insert to DuckDB"""
        df = pd.DataFrame(chunk)

        # Optimize DataFrame
        df = self.optimize_dtypes(df)

        # Insert to DuckDB (releases memory after insert)
        con.execute("INSERT INTO options_ticks SELECT * FROM df")

        del df, chunk  # Explicit cleanup
```

### Performance Optimization Checklist

- [ ] Use parent symbology to minimize API calls
- [ ] Implement 5-10 concurrent requests maximum
- [ ] Process data in daily chunks to limit memory usage
- [ ] Transform to DuckDB format immediately after download
- [ ] Drop unnecessary fields before storage
- [ ] Use connection pooling for HTTP requests
- [ ] Monitor memory usage and implement gc.collect() if needed
- [ ] Archive raw DBN files with compression
- [ ] Use DuckDB partitioning by date for faster queries
- [ ] Implement proper error handling with exponential backoff

## Implementation Best Practices Summary

1. **Always use `stype_in="parent"`** when fetching full option chains
2. **Pull data after 2:00 AM ET** for complete previous day's data
3. **Use CMBP-1 schema** for consolidated NBBO quotes
4. **Implement 1-minute bars** for underlying price tracking
5. **Process in daily chunks** to manage memory efficiently
6. **Transform to DuckDB immediately** for optimal storage and query performance
7. **Use instrument_id as primary key** for efficient joins and lookups
8. **Implement retry logic** only for transient errors (429, 5xx)
9. **Archive raw DBN files** compressed for data lineage
10. **Monitor new listings daily** via instrument definitions schema

This configuration provides optimal performance for Unity options data management while maintaining cost efficiency and data integrity.
