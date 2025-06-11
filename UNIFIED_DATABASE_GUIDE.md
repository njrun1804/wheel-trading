# Unified Database Guide

## Overview
The unified database (`data/unified_wheel_trading.duckdb`) consolidates all data needed for the wheel trading system into a single, optimized structure.

## Database Contents

### Core Tables

#### 1. `market_data` (179,585 rows)
Unified price history for both stocks and options:
- Unity stock prices: 861 days (2022-2025)
- Unity options prices: 178,724 daily OHLCV records
- Fields: symbol, date, open, high, low, close, volume, returns, data_type

#### 2. `options_metadata` (9,421 rows)
Parsed option contract details:
- Fields: symbol, underlying, expiration, strike, option_type
- Automatically extracted from OCC symbols

#### 3. `economic_indicators` (8,679 rows)
FRED economic data including:
- DGS3: 3-Year Treasury Rate (risk-free rate)
- VIXCLS: CBOE Volatility Index
- DFF: Federal Funds Rate
- Plus 6 other indicators

#### 4. Trading System Tables
- `greeks_cache`: Store calculated Greeks
- `recommendations`: Track model recommendations
- `backtest_results`: Store backtesting outcomes

### Pre-built Views

#### Current Market State
```sql
-- Get current Unity stock price
SELECT * FROM current_unity_stock;

-- Get current risk-free rate
SELECT * FROM current_risk_free_rate;

-- Get current VIX
SELECT * FROM current_vix;
```

#### Available Options
```sql
-- Get all available put options
SELECT * FROM available_puts
WHERE strike BETWEEN 20 AND 30  -- Filter by strike range
  AND days_to_expiry BETWEEN 30 AND 60;  -- Filter by DTE
```

#### Historical Volatility
```sql
-- Get Unity volatility metrics
SELECT * FROM unity_volatility
ORDER BY date DESC
LIMIT 30;
```

## Usage Examples

### 1. Training Data Query
```python
import duckdb

conn = duckdb.connect('data/unified_wheel_trading.duckdb')

# Get training data with features
training_data = conn.execute("""
    SELECT
        s.date,
        s.close as stock_price,
        s.returns,
        v.volatility_20d,
        v.volatility_60d,
        e1.value as vix,
        e2.value as risk_free_rate
    FROM market_data s
    JOIN unity_volatility v ON s.date = v.date
    LEFT JOIN economic_indicators e1 ON s.date = e1.date AND e1.indicator = 'VIXCLS'
    LEFT JOIN economic_indicators e2 ON s.date = e2.date AND e2.indicator = 'DGS3'
    WHERE s.symbol = 'U'
      AND s.data_type = 'stock'
      AND s.date >= '2023-01-01'
    ORDER BY s.date
""").df()
```

### 2. Daily Recommendation Query
```python
# Get current market state for recommendation
current_state = conn.execute("""
    WITH current_stock AS (
        SELECT * FROM current_unity_stock
    ),
    current_market AS (
        SELECT
            (SELECT value FROM current_vix) as vix,
            (SELECT rate FROM current_risk_free_rate) as risk_free_rate
    )
    SELECT
        s.close as current_price,
        s.returns as last_return,
        v.volatility_20d,
        m.vix,
        m.risk_free_rate,
        (SELECT COUNT(*) FROM available_puts) as available_options
    FROM current_stock s
    JOIN unity_volatility v ON s.date = v.date
    CROSS JOIN current_market m
""").fetchone()

# Get option chain
options = conn.execute("""
    SELECT * FROM available_puts
    WHERE strike <= (SELECT close * 0.95 FROM current_unity_stock)
      AND days_to_expiry BETWEEN 30 AND 60
    ORDER BY strike DESC, expiration
""").df()
```

### 3. Backtesting Query
```python
# Get historical option prices for backtesting
backtest_data = conn.execute("""
    SELECT
        md.date,
        om.symbol,
        om.strike,
        om.expiration,
        md.close as premium,
        s.close as stock_price,
        om.expiration - md.date as dte
    FROM market_data md
    JOIN options_metadata om ON md.symbol = om.symbol
    JOIN market_data s ON md.date = s.date AND s.symbol = 'U' AND s.data_type = 'stock'
    WHERE om.option_type = 'P'
      AND om.underlying = 'U'
      AND md.data_type = 'option'
      AND md.date >= '2024-01-01'
    ORDER BY md.date, om.strike
""").df()
```

### 4. Save Recommendations
```python
import uuid
from datetime import datetime

# Save a recommendation
recommendation = {
    'id': str(uuid.uuid4()),
    'timestamp': datetime.now(),
    'symbol': 'U     250620P00025000',
    'action': 'SELL_PUT',
    'strike': 25.0,
    'expiration': '2025-06-20',
    'contracts': 10,
    'confidence': 0.85,
    'expected_return': 0.15
}

conn.execute("""
    INSERT INTO recommendations
    (recommendation_id, timestamp, symbol, action, strike, expiration, contracts, confidence, expected_return)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", list(recommendation.values()))
```

## Integration with Wheel Trading System

The unified database is designed to work seamlessly with:
1. **Risk Analytics** (`src/unity_wheel/risk/analytics.py`)
2. **Wheel Strategy** (`src/unity_wheel/strategy/wheel.py`)
3. **Options Math** (`src/unity_wheel/math/options.py`)
4. **Advisor** (`src/unity_wheel/api/advisor.py`)

Update the database path in your configuration:
```yaml
# config.yaml
storage:
  database_path: "data/unified_wheel_trading.duckdb"
```

## Performance Tips

1. **Use indexes**: The database has indexes on commonly queried columns
2. **Filter early**: Apply date/strike filters in WHERE clause, not in application
3. **Use views**: Pre-built views are optimized for common queries
4. **Batch inserts**: When saving multiple recommendations, use transactions

## Maintenance

```python
# Vacuum database periodically
conn.execute("VACUUM")

# Update statistics
conn.execute("ANALYZE")

# Check database size
size = conn.execute("SELECT current_setting('database_size')").fetchone()
```

## Next Steps

1. **Calculate Greeks**: Populate the greeks_cache table with historical Greeks
2. **Feature Engineering**: Create additional technical indicators
3. **Backtest**: Run wheel strategy backtests using historical data
4. **Monitor**: Set up daily data updates from live sources
