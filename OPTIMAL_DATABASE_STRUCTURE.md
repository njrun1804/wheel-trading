# Optimal Database Structure for Wheel Trading System

## Current Situation
You have data scattered across two DuckDB databases:
1. `~/.wheel_trading/cache/wheel_cache.duckdb` - Unity stock prices & FRED data
2. `data/cache/wheel_cache.duckdb` - Unity options data

## Recommended Unified Structure

### Single Consolidated Database: `data/unified_trading.duckdb`

#### Core Tables

**1. `market_data` (Unified price history)**
```sql
CREATE TABLE market_data (
    symbol VARCHAR,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    returns DOUBLE,
    data_type VARCHAR, -- 'stock' or 'option'
    PRIMARY KEY (symbol, date)
);
```

**2. `options_chain` (Current + Historical)**
```sql
CREATE TABLE options_chain (
    symbol VARCHAR,           -- Full OCC symbol
    underlying VARCHAR,       -- 'U'
    expiration DATE,
    strike DOUBLE,
    option_type VARCHAR,      -- 'P' or 'C'
    date DATE,               -- Quote date
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    open_interest INT,
    bid DOUBLE,
    ask DOUBLE,
    iv DOUBLE,               -- Implied volatility
    PRIMARY KEY (symbol, date)
);
```

**3. `economic_indicators` (FRED data)**
```sql
CREATE TABLE economic_indicators (
    date DATE,
    indicator VARCHAR,
    value DOUBLE,
    PRIMARY KEY (date, indicator)
);
```

**4. `greeks_cache` (Calculated Greeks)**
```sql
CREATE TABLE greeks_cache (
    symbol VARCHAR,
    date DATE,
    underlying_price DOUBLE,
    delta DOUBLE,
    gamma DOUBLE,
    theta DOUBLE,
    vega DOUBLE,
    rho DOUBLE,
    calculated_at TIMESTAMP,
    PRIMARY KEY (symbol, date)
);
```

**5. `model_features` (Pre-calculated features for ML)**
```sql
CREATE TABLE model_features (
    date DATE,
    symbol VARCHAR,
    -- Price features
    price_sma_20 DOUBLE,
    price_sma_50 DOUBLE,
    rsi_14 DOUBLE,
    volatility_20 DOUBLE,
    volatility_60 DOUBLE,
    -- Option features
    put_call_ratio DOUBLE,
    iv_rank DOUBLE,
    iv_percentile DOUBLE,
    -- Market features
    vix DOUBLE,
    risk_free_rate DOUBLE,
    spy_correlation DOUBLE,
    PRIMARY KEY (date, symbol)
);
```

**6. `recommendations` (Model outputs & decisions)**
```sql
CREATE TABLE recommendations (
    recommendation_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP,
    symbol VARCHAR,
    action VARCHAR,           -- 'SELL_PUT', 'BUY_BACK', 'HOLD', etc.
    strike DOUBLE,
    expiration DATE,
    contracts INT,
    confidence DOUBLE,
    expected_return DOUBLE,
    risk_metrics JSON,        -- VaR, CVaR, etc.
    features_used JSON,       -- Feature importance
    actual_outcome DOUBLE,    -- For backtesting
    notes TEXT
);
```

**7. `positions` (Track actual positions)**
```sql
CREATE TABLE positions (
    position_id VARCHAR PRIMARY KEY,
    symbol VARCHAR,
    open_date DATE,
    close_date DATE,
    strike DOUBLE,
    expiration DATE,
    contracts INT,
    premium_collected DOUBLE,
    cost_basis DOUBLE,
    status VARCHAR,           -- 'OPEN', 'CLOSED', 'ASSIGNED'
    pnl DOUBLE,
    recommendation_id VARCHAR -- Link to recommendation
);
```

## Implementation Plan

### Step 1: Consolidate Existing Data
```python
import duckdb

# Create unified database
unified_conn = duckdb.connect('data/unified_trading.duckdb')

# Connect to existing databases
home_conn = duckdb.connect('~/.wheel_trading/cache/wheel_cache.duckdb')
project_conn = duckdb.connect('data/cache/wheel_cache.duckdb')

# Copy Unity stock data
unified_conn.execute("""
    CREATE TABLE market_data AS
    SELECT symbol, date, open, high, low, close, volume, returns, 'stock' as data_type
    FROM home_conn.price_history
    WHERE symbol = 'U'
""")

# Copy Unity options data
unified_conn.execute("""
    INSERT INTO market_data
    SELECT symbol, ts_event as date, open, high, low, close, volume, NULL as returns, 'option' as data_type
    FROM project_conn.unity_options_ohlcv
""")

# Copy FRED data
unified_conn.execute("""
    CREATE TABLE economic_indicators AS
    SELECT date, series_id as indicator, value
    FROM home_conn.fred_observations
""")
```

### Step 2: Create Views for Easy Access
```sql
-- Current options chain
CREATE VIEW current_options AS
SELECT * FROM options_chain
WHERE date = (SELECT MAX(date) FROM options_chain)
  AND expiration > CURRENT_DATE;

-- Risk-free rate view
CREATE VIEW current_risk_free_rate AS
SELECT value FROM economic_indicators
WHERE indicator = 'DGS3'  -- 3-month Treasury
  AND date = (SELECT MAX(date) FROM economic_indicators WHERE indicator = 'DGS3');

-- Unity price history with returns
CREATE VIEW unity_stock AS
SELECT * FROM market_data
WHERE symbol = 'U' AND data_type = 'stock'
ORDER BY date;
```

## Usage Patterns

### For Training (Historical Analysis)
```python
# Get all data for model training
query = """
    SELECT
        m.date,
        m.close as stock_price,
        m.returns,
        e.value as vix,
        f.*
    FROM market_data m
    LEFT JOIN model_features f ON m.date = f.date AND m.symbol = f.symbol
    LEFT JOIN economic_indicators e ON m.date = e.date AND e.indicator = 'VIXCLS'
    WHERE m.symbol = 'U' AND m.data_type = 'stock'
    ORDER BY m.date
"""
```

### For Daily Recommendations
```python
# Get current market state
query = """
    WITH latest_stock AS (
        SELECT * FROM market_data
        WHERE symbol = 'U' AND data_type = 'stock'
        ORDER BY date DESC LIMIT 1
    ),
    latest_vix AS (
        SELECT value FROM economic_indicators
        WHERE indicator = 'VIXCLS'
        ORDER BY date DESC LIMIT 1
    )
    SELECT
        s.close as current_price,
        s.returns,
        v.value as current_vix,
        -- Get relevant options
        (SELECT COUNT(*) FROM current_options WHERE underlying = 'U') as available_options
    FROM latest_stock s, latest_vix v
"""
```

## Benefits of This Structure

1. **Single Source of Truth** - All data in one place
2. **Efficient Queries** - Proper indexes and materialized views
3. **Feature Engineering** - Pre-calculated features for fast model training
4. **Audit Trail** - Track recommendations and outcomes
5. **Backtesting Ready** - Historical data properly organized
6. **Scalable** - Can add more symbols/indicators easily

## Migration Script
I can create a migration script to consolidate your existing data into this structure if you'd like to proceed.
