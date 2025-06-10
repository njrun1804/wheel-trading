# Databento Unity Integration Guide

## Overview

This guide provides complete instructions for pulling Unity (U) stock and options data from Databento's API and storing it efficiently in local DuckDB storage. The implementation uses a pull-when-asked architecture with no streaming, optimized for cost efficiency.

## 1. Quick Start

### Prerequisites

```bash
# Install required packages
pip install databento pandas duckdb google-cloud-secret-manager

# Set up Google Cloud authentication (one-time)
gcloud auth application-default login
```

### API Key Setup with Google Cloud Secrets

The Databento API key is stored in Google Cloud Secret Manager for security. Here's how to set it up:

```bash
# Create the secret (one-time setup)
echo -n "your-databento-api-key" | gcloud secrets create databento_api_key --data-file=-

# Grant access to your service account
gcloud secrets add-iam-policy-binding databento_api_key \
    --member="user:your-email@domain.com" \
    --role="roles/secretmanager.secretAccessor"
```

The code automatically retrieves the API key using:

```python
from src.unity_wheel.secrets.integration import get_databento_api_key

# This function handles all the Google Cloud authentication
api_key = get_databento_api_key()
```

### Quick Test

```python
# Test Unity options data pull
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from datetime import datetime, timezone

async def test_unity_data():
    client = DatabentoClient()  # Uses Google Secrets for API key

    # Get next monthly expiration
    expiration = datetime(2025, 6, 20, tzinfo=timezone.utc)

    # Pull option chain
    chain = await client.get_option_chain("U", expiration)
    print(f"Found {len(chain.puts)} puts, {len(chain.calls)} calls")
    print(f"Unity spot price: ${chain.spot_price}")

    await client.close()

# Run: asyncio.run(test_unity_data())
```

## 2. Architecture Overview

### Pull-When-Asked Pattern

- **No Streaming**: REST API only, no WebSocket subscriptions
- **On-Demand**: Data fetched only when user requests recommendation
- **Local Cache**: DuckDB with 15-minute TTL for options data
- **Cost Efficient**: Pay-per-request model, ~$50/month for typical usage

### Data Flow

```
User Request → Check Cache → Fetch from Databento → Filter by Moneyness → Store in DuckDB → Return Data
```

## 3. Implementation Details

### 3.1 Correct Symbol Formats

```python
# Unity options (all contracts)
symbols = ["U.OPT"]
stype_in = SType.PARENT

# Unity stock quotes
dataset = "EQUS.MINI"  # Composite NBBO
symbols = ["U"]
stype_in = SType.RAW_SYMBOL

# Specific option contract
symbol = "U250620P00025000"  # Unity Jun 20 2025 $25 Put
stype_in = SType.RAW_SYMBOL
```

### 3.2 Optimal Query Parameters

- **Options Schema**: Use `Schema.DEFINITION` for contract details, `Schema.TRADES` for quotes
- **Stock Schema**: Use `Schema.MBP_1` (top of book) for underlying quotes
- **Date Handling**: OPRA data is T+1, so use `end = yesterday` for historical queries
- **Batch Size**: Maximum 100 symbols per request for options

### 3.3 Data Collection Pattern

```python
async def collect_unity_data(client: DatabentoClient, target_dte_range=(30, 60)):
    """Collect Unity options data with optimal filtering."""

    # 1. Get Unity spot price first
    spot_data = await client._get_underlying_price("U")
    spot_price = float(spot_data.last_price)

    # 2. Calculate target strikes (±20% of spot)
    min_strike = spot_price * 0.80
    max_strike = spot_price * 1.20

    # 3. Get monthly expirations in DTE range
    expirations = get_monthly_expirations_in_dte_range(target_dte_range)

    # 4. Fetch option chains
    all_options = []
    for expiration in expirations:
        try:
            # Get definitions first (lightweight)
            definitions = await client._get_definitions("U", expiration)

            # Filter to target strikes before fetching quotes
            target_ids = [
                d.instrument_id for d in definitions
                if min_strike <= float(d.strike_price) <= max_strike
                and d.option_type == "P"  # Puts for wheel
            ]

            # Get quotes only for filtered instruments
            if target_ids:
                quotes = await client._get_quotes_by_ids(target_ids, None)
                all_options.extend(quotes.values())

        except Exception as e:
            logger.warning(f"Failed to get {expiration}: {e}")
            continue

    return all_options
```

### 3.4 Storage Schema

```sql
-- DuckDB schema optimized for wheel strategy
CREATE TABLE databento_option_chains (
    symbol VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    option_type VARCHAR(4) NOT NULL,
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    mid DECIMAL(10,4),
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility DECIMAL(6,4),
    delta DECIMAL(5,4),
    timestamp TIMESTAMP NOT NULL,
    spot_price DECIMAL(10,2) NOT NULL,
    moneyness DECIMAL(5,4),  -- For efficient filtering
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, expiration, strike, option_type, timestamp)
);

-- Indexes for performance
CREATE INDEX idx_moneyness ON databento_option_chains(symbol, moneyness);
CREATE INDEX idx_delta_range ON databento_option_chains(symbol, delta);
```

## 4. Cost Optimization

### Moneyness Filtering (80% Cost Reduction)

```python
def filter_by_moneyness(chain: OptionChain, moneyness_range: float = 0.20):
    """Keep only options within ±20% of spot price."""
    spot = float(chain.spot_price)
    min_strike = spot * (1 - moneyness_range)
    max_strike = spot * (1 + moneyness_range)

    # Filter strikes
    chain.puts = [p for p in chain.puts
                  if min_strike <= p.strike_price <= max_strike]
    chain.calls = [c for c in chain.calls
                   if min_strike <= c.strike_price <= max_strike]

    return chain
```

### Caching Strategy

```python
# Cache configuration
CACHE_TTL = {
    "option_chains": 15,      # 15 minutes for quotes
    "definitions": 1440,      # 24 hours for contract specs
    "underlying": 5,          # 5 minutes for spot price
    "greeks": 60              # 1 hour for calculated Greeks
}
```

### Monthly Cost Estimate

```
Assumptions:
- 100 recommendations/month
- 2 expirations per recommendation
- 50% cache hit rate
- Moneyness filtering reduces data by 80%

Cost breakdown:
- Definition queries: 100 × $0.01 = $1
- Quote queries: 100 × $0.10 = $10
- Underlying quotes: 100 × $0.01 = $1
- Total with filtering: ~$12/month
- Without filtering: ~$60/month
```

## 5. Usage Examples

### 5.1 Complete Data Pull

```python
#!/usr/bin/env python3
"""Pull Unity options data for wheel strategy."""

import asyncio
from datetime import datetime, timedelta, timezone
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from src.unity_wheel.data_providers.databento.databento_storage_adapter import DatabentoStorageAdapter
from src.unity_wheel.storage.storage import Storage

async def main():
    # Initialize storage
    storage = Storage()
    await storage.initialize()

    adapter = DatabentoStorageAdapter(storage)
    await adapter.initialize()

    # Initialize client (uses Google Secrets)
    client = DatabentoClient()

    try:
        # Find next monthly expiration
        today = datetime.now(timezone.utc)
        expiration = get_next_monthly_expiration(today, min_dte=30)

        print(f"Fetching Unity options for {expiration.date()}")

        # Get option chain
        chain = await client.get_option_chain("U", expiration)

        # Get definitions
        definitions = await client._get_definitions("U", expiration)

        # Store with moneyness filtering
        success = await adapter.store_option_chain(
            chain=chain,
            definitions=definitions,
            enriched=False
        )

        if success:
            print(f"Stored {len(chain.puts)} puts after filtering")

            # Get storage stats
            stats = await adapter.get_storage_stats()
            print(f"Database size: {stats['db_size_mb']:.1f} MB")
            print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 Find Wheel Candidates

```python
async def find_wheel_candidates(underlying="U", target_delta=0.30):
    """Find optimal puts for wheel strategy."""

    client = DatabentoClient()

    # Get spot price
    spot_data = await client._get_underlying_price(underlying)
    spot = float(spot_data.last_price)

    # Find candidates
    candidates = []

    # Get next 3 monthly expirations
    for i in range(3):
        expiration = get_nth_monthly_expiration(i + 1)
        dte = (expiration - datetime.now(timezone.utc)).days

        if 30 <= dte <= 60:  # Target DTE range
            chain = await client.get_option_chain(underlying, expiration)

            for put in chain.puts:
                # Filter by delta (would need Greeks calculation)
                if 0.25 <= abs(put.delta) <= 0.35:
                    expected_return = (put.mid_price / spot) * (365 / dte)

                    candidates.append({
                        'strike': put.strike_price,
                        'expiration': expiration,
                        'dte': dte,
                        'bid': put.bid_price,
                        'ask': put.ask_price,
                        'mid': put.mid_price,
                        'delta': put.delta,
                        'expected_return': expected_return
                    })

    # Sort by expected return
    return sorted(candidates, key=lambda x: x['expected_return'], reverse=True)
```

## 6. Troubleshooting

### Common Issues and Solutions

#### "Could not resolve smart symbols: U.OPT"
- **Cause**: Wrong `stype_in` parameter
- **Fix**: Use `SType.PARENT` not `SType.RAW_SYMBOL`

#### "data_end_after_available_end"
- **Cause**: OPRA historical data is T+1
- **Fix**: Use yesterday as end date for queries

#### Empty option chain
- **Cause**: Unity only has monthly options (3rd Friday)
- **Fix**: Check expiration dates, ensure it's a valid monthly expiration

#### High API costs
- **Cause**: Fetching all strikes without filtering
- **Fix**: Apply moneyness filter before fetching quotes

#### Authentication errors
- **Cause**: Google Cloud credentials not set up
- **Fix**: Run `gcloud auth application-default login`

### Debug Commands

```bash
# Test Databento connection
python tools/debug/debug_databento.py

# Check cached data
sqlite3 ~/.wheel_trading/cache/wheel_cache.duckdb \
  "SELECT COUNT(*) FROM databento_option_chains WHERE symbol='U'"

# View recent errors
grep ERROR logs/wheel.log | tail -20
```

## 7. Performance Optimization

### Concurrent Requests

```python
# Process multiple expirations concurrently
async def fetch_multiple_expirations(expirations):
    tasks = []
    for exp in expirations:
        task = client.get_option_chain("U", exp)
        tasks.append(task)

    # Limit concurrency to respect rate limits
    results = []
    for i in range(0, len(tasks), 5):  # 5 concurrent
        batch = tasks[i:i+5]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)

        if i + 5 < len(tasks):
            await asyncio.sleep(0.5)  # Rate limit pause

    return results
```

### Memory Management

```python
# Process large datasets in chunks
async def process_historical_data(start_date, end_date):
    current = start_date

    while current <= end_date:
        # Process one week at a time
        week_end = min(current + timedelta(days=7), end_date)

        # Fetch and process
        data = await fetch_week_data(current, week_end)
        await store_data(data)

        # Clear memory
        del data

        current = week_end + timedelta(days=1)
```

## 8. Best Practices

1. **Always use moneyness filtering** - Reduces costs by 80%
2. **Cache aggressively** - 15-minute TTL for quotes is usually sufficient
3. **Batch operations** - Process multiple expirations together
4. **Handle errors gracefully** - Unity options may have gaps
5. **Monitor costs** - Track API usage to stay within budget
6. **Use correct datasets** - OPRA.PILLAR for options, EQUS.MINI for stock
7. **Store only what you need** - Drop unnecessary fields before storage

## 9. Integration with Wheel Strategy

The data integrates with the wheel advisor through:

```python
from src.unity_wheel.api.advisor import WheelAdvisor
from src.unity_wheel.data_providers.databento import get_market_snapshot

# Get recommendation using Databento data
advisor = WheelAdvisor()
market_data = await get_market_snapshot("U")  # Uses Databento
recommendation = advisor.advise_position(
    symbol="U",
    market_data=market_data,
    portfolio_value=100000
)
```

## Summary

This guide provides everything needed to efficiently pull Unity options data from Databento:
- ✅ Correct symbol formats and datasets
- ✅ Google Cloud Secrets integration
- ✅ Cost optimization through filtering
- ✅ Local caching with DuckDB
- ✅ Production-ready error handling
- ✅ No streaming complexity

Expected results:
- ~$12/month for daily wheel recommendations
- 15-minute data freshness
- 80% cost reduction through moneyness filtering
- <100MB monthly storage
