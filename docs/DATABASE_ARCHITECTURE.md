# Database Architecture

## Overview

The wheel trading system uses a **two-database architecture** optimized for different use cases:

1. **Operational Database** (`~/.wheel_trading/cache/wheel_cache.duckdb`)
   - Real-time data storage
   - Live trading decisions
   - Caching of API responses
   - Ephemeral data with TTL

2. **Analytical Database** (`data/unified_wheel_trading.duckdb`)
   - Historical data warehouse
   - Backtesting and parameter optimization
   - Pre-calculated features
   - Long-term data retention

## Architecture Diagram

```
┌─────────────────────────┐     ┌──────────────────────┐
│  Operational Database   │     │  Analytical Database │
│ ~/.wheel_trading/cache/ │     │ data/unified_wheel_  │
│                         │     │     trading.duckdb   │
├─────────────────────────┤     ├──────────────────────┤
│ Tables:                 │     │ Tables:              │
│ • option_chains         │ ETL │ • market_data        │
│ • position_snapshots    │ ──> │ • options_metadata   │
│ • greeks_cache          │     │ • economic_indicators│
│ • price_history         │     │ • backtest_features  │
│                         │     │ • backtest_results   │
├─────────────────────────┤     ├──────────────────────┤
│ Purpose:                │     │ Purpose:             │
│ • Real-time trading     │     │ • Historical analysis│
│ • API response caching  │     │ • Backtesting        │
│ • Session management    │     │ • Parameter tuning   │
│ • Temporary storage     │     │ • Research           │
└─────────────────────────┘     └──────────────────────┘
         ↓                               ↓
    Daily Trading                   Backtesting &
   Recommendations                  Optimization
```

## Operational Database Schema

### Core Tables

1. **option_chains**
   - Real-time option chain snapshots
   - TTL: 15 minutes
   - Indexes: symbol, expiration, timestamp

2. **position_snapshots**
   - Account positions and balances
   - TTL: 30 minutes
   - Indexes: account_id, timestamp

3. **greeks_cache**
   - Calculated Greeks values
   - TTL: 1 hour
   - Indexes: option_symbol, timestamp

4. **price_history**
   - Recent price data for risk calculations
   - TTL: 7 days
   - Indexes: symbol, date

## Analytical Database Schema

### Core Tables

1. **market_data**
   - Comprehensive price history (stocks & options)
   - 179,585 records (3+ years)
   - Columns: symbol, date, OHLCV, returns, data_type

2. **options_metadata**
   - Parsed option contract details
   - 9,421 unique contracts
   - Columns: symbol, underlying, expiration, strike, option_type

3. **economic_indicators**
   - FRED data (VIX, risk-free rates, etc.)
   - 8,679 observations
   - Columns: indicator, date, value

4. **backtest_features**
   - Pre-calculated features for fast backtesting
   - 133,602 records
   - Includes: volatility, VaR, risk-free rate

### Optimized Views

1. **stock_price_history**
   - Stock-only data with rolling calculations
   - Pre-computed 20d/250d volatility
   - Average returns over various windows

2. **options_history**
   - Joined option prices with metadata
   - Calculated DTE and moneyness
   - Spot price at each date

3. **latest_market_snapshot**
   - Current market conditions
   - Latest prices, volatility, VIX
   - Used for daily recommendations

4. **liquid_option_strikes**
   - Options with recent volume
   - Filtered for liquidity (volume > 0)
   - Last 5 days of data

## ETL Pipeline

The ETL process (`tools/etl_unified_database.py`) syncs data from operational to analytical:

1. **Extract**: Pull from operational cache
2. **Transform**:
   - Calculate returns
   - Compute rolling volatility
   - Add risk metrics (VaR)
   - Join with economic data
3. **Load**: Insert into analytical database

### Running ETL

```bash
# One-time setup/fix
python tools/etl_unified_database.py

# Regular sync (should be scheduled)
python tools/sync_to_unified_db.py  # TODO: Create this
```

## Performance Optimizations

### Indexes
- Time-series queries: `(symbol, date)`
- Option lookups: `(underlying, strike, expiration)`
- Backtest ranges: `(date)` with date filtering

### Pre-calculations
- Returns calculated once and stored
- Rolling volatility in views
- VaR/CVaR in backtest_features table

### Query Patterns
```sql
-- Fast backtest data retrieval
SELECT * FROM backtest_features
WHERE symbol = 'U'
AND date BETWEEN '2023-01-01' AND '2024-01-01';

-- Option chain for specific expiration
SELECT * FROM liquid_option_strikes
WHERE underlying = 'U'
AND expiration = '2024-02-16'
AND option_type = 'P';
```

## Usage Examples

### For Backtesting
```python
# Backtester automatically uses unified DB
backtester = WheelBacktester(storage)
results = await backtester.backtest_strategy(
    symbol='U',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1)
)
```

### For Daily Recommendations
```python
# Advisor uses operational DB for real-time data
advisor = WheelAdvisor()
recommendation = advisor.advise_position(market_snapshot)
```

### For Research/Analysis
```python
# Direct query to analytical DB
conn = duckdb.connect('data/unified_wheel_trading.duckdb')
df = conn.execute("""
    SELECT date, close, volatility_20d, var_95
    FROM backtest_features
    WHERE symbol = 'U'
""").df()
```

## Maintenance

### Daily Tasks
1. Run ETL to sync new data
2. Vacuum operational DB (auto via TTL)
3. Update economic indicators

### Weekly Tasks
1. Full VACUUM ANALYZE on analytical DB
2. Check index usage statistics
3. Archive old operational data

### Monthly Tasks
1. Review and optimize slow queries
2. Update pre-calculated features
3. Clean up old backtest results

## Best Practices

1. **Never write synthetic data** - All data must come from real sources
2. **Use appropriate database** - Operational for live, analytical for historical
3. **Leverage pre-calculations** - Use views and feature tables
4. **Monitor performance** - Track query times and optimize as needed
5. **Maintain data quality** - Regular validation and consistency checks
