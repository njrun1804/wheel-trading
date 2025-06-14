# Optimal Data Structure for Wheel Trading System

## Executive Summary

This document outlines the optimal data structure for the wheel trading system, consolidating from 3 databases with 30+ tables to a single optimized database with 12 core tables.

## Core Design Principles

1. **Single Source of Truth**: One database (`wheel_trading.duckdb`)
2. **Partitioning**: Monthly partitions for time-series data
3. **Materialized Views**: Pre-computed aggregations for common queries
4. **Smart Indexing**: Covering indexes for 90% of queries
5. **Hardware Optimization**: Tuned for M4 Pro (12 cores, 24GB RAM)

## Optimal Table Structure

### 1. Market Data Layer

```sql
-- Core market data table (partitioned by month)
CREATE TABLE market_data (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    -- Calculated fields for common queries
    daily_return DECIMAL(8,6),
    volatility_20d DECIMAL(8,6),
    volatility_60d DECIMAL(8,6),
    -- Partitioning key
    year_month INTEGER GENERATED ALWAYS AS (YEAR(date) * 100 + MONTH(date)) STORED
) PARTITION BY RANGE (year_month);

-- Indexes
CREATE INDEX idx_market_symbol_date ON market_data(symbol, date DESC);
CREATE INDEX idx_market_volatility ON market_data(symbol, volatility_20d) WHERE date >= CURRENT_DATE - INTERVAL '30 days';
```

### 2. Options Data Layer

```sql
-- Consolidated options table with smart filtering
CREATE TABLE options_data (
    symbol VARCHAR NOT NULL,
    expiration DATE NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('CALL', 'PUT')),
    timestamp TIMESTAMP NOT NULL,
    -- Market data
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    mid DECIMAL(10,4) GENERATED ALWAYS AS ((bid + ask) / 2) STORED,
    volume INTEGER,
    open_interest INTEGER,
    -- Greeks (pre-calculated)
    implied_volatility DECIMAL(8,6),
    delta DECIMAL(6,4),
    gamma DECIMAL(8,6),
    theta DECIMAL(10,4),
    vega DECIMAL(10,4),
    rho DECIMAL(10,4),
    -- Derived fields for faster queries
    moneyness DECIMAL(6,4),
    days_to_expiry INTEGER GENERATED ALWAYS AS (expiration - DATE(timestamp)) STORED,
    -- Partitioning
    year_month INTEGER GENERATED ALWAYS AS (YEAR(timestamp) * 100 + MONTH(timestamp)) STORED
) PARTITION BY RANGE (year_month);

-- Covering indexes for common access patterns
CREATE INDEX idx_options_wheel ON options_data(symbol, expiration, delta, timestamp DESC) 
    INCLUDE (strike, bid, ask, implied_volatility)
    WHERE option_type = 'PUT' AND delta BETWEEN -0.35 AND -0.25;

CREATE INDEX idx_options_covered_call ON options_data(symbol, expiration, strike, timestamp DESC)
    INCLUDE (bid, ask, delta, implied_volatility)  
    WHERE option_type = 'CALL' AND delta BETWEEN 0.15 AND 0.35;

CREATE INDEX idx_options_moneyness ON options_data(symbol, moneyness, expiration)
    WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 hour';
```

### 3. Trading Decision Layer

```sql
-- Active positions with denormalized data
CREATE TABLE positions (
    position_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    position_type VARCHAR(20) NOT NULL,
    -- Position details
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,2) NOT NULL,
    entry_date TIMESTAMP NOT NULL,
    expiration DATE,
    strike DECIMAL(10,2),
    -- Current state
    current_price DECIMAL(10,2),
    unrealized_pnl DECIMAL(12,2),
    status VARCHAR(20) NOT NULL,
    -- Risk metrics
    position_delta DECIMAL(8,4),
    position_theta DECIMAL(10,2),
    margin_required DECIMAL(12,2),
    -- Metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_positions_active ON positions(account_id, symbol) WHERE status = 'ACTIVE';
CREATE INDEX idx_positions_expiring ON positions(expiration, symbol) WHERE status = 'ACTIVE';

-- Trading decisions audit trail
CREATE TABLE trading_decisions (
    decision_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR NOT NULL,
    decision_type VARCHAR(20) NOT NULL,
    recommendation JSON NOT NULL,
    -- Execution details
    executed BOOLEAN DEFAULT FALSE,
    execution_price DECIMAL(10,2),
    execution_timestamp TIMESTAMP,
    -- Analytics
    confidence_score DECIMAL(4,3),
    expected_return DECIMAL(8,4),
    risk_score DECIMAL(4,3)
);

CREATE INDEX idx_decisions_pending ON trading_decisions(symbol, timestamp DESC) WHERE executed = FALSE;
```

### 4. Materialized Views for Performance

```sql
-- Real-time wheel opportunities (refreshed every minute)
CREATE MATERIALIZED VIEW mv_wheel_opportunities AS
SELECT 
    o.symbol,
    o.expiration,
    o.strike,
    o.bid,
    o.ask,
    o.delta,
    o.implied_volatility,
    o.volume,
    -- Calculated metrics
    (o.bid * 100) / (o.strike * 100) as premium_yield,
    (o.ask - o.bid) / NULLIF(o.ask, 0) as spread_pct,
    m.volatility_20d as underlying_volatility,
    -- Scoring
    (o.bid * 100) / (o.strike * 100) * 365 / o.days_to_expiry as annualized_return
FROM options_data o
JOIN market_data m ON o.symbol = m.symbol AND m.date = CURRENT_DATE
WHERE 
    o.option_type = 'PUT'
    AND o.delta BETWEEN -0.35 AND -0.25
    AND o.days_to_expiry BETWEEN 20 AND 45
    AND o.timestamp >= CURRENT_TIMESTAMP - INTERVAL '15 minutes'
    AND o.bid > 0 AND o.ask > 0
    AND (o.ask - o.bid) / o.ask < 0.10  -- Max 10% spread
ORDER BY annualized_return DESC;

-- Portfolio summary (refreshed every 5 minutes)
CREATE MATERIALIZED VIEW mv_portfolio_summary AS
SELECT
    p.account_id,
    p.symbol,
    COUNT(*) as position_count,
    SUM(p.quantity * p.current_price) as market_value,
    SUM(p.unrealized_pnl) as total_unrealized_pnl,
    SUM(p.position_delta * p.quantity * p.current_price) as portfolio_delta,
    SUM(p.margin_required) as total_margin_used,
    MAX(p.last_updated) as last_update
FROM positions p
WHERE p.status = 'ACTIVE'
GROUP BY p.account_id, p.symbol;
```

### 5. Analytics & ML Layer

```sql
-- Feature store for ML models
CREATE TABLE ml_features (
    symbol VARCHAR NOT NULL,
    feature_date DATE NOT NULL,
    -- Price features
    returns_5d DECIMAL(8,6),
    returns_20d DECIMAL(8,6),
    volatility_realized DECIMAL(8,6),
    volatility_garch DECIMAL(8,6),
    -- Option features
    put_call_ratio DECIMAL(8,4),
    iv_rank DECIMAL(4,3),
    iv_percentile DECIMAL(4,3),
    term_structure_slope DECIMAL(8,6),
    -- Market regime
    vix_level DECIMAL(6,2),
    market_regime VARCHAR(20),
    -- Technical indicators
    rsi_14d DECIMAL(4,2),
    bollinger_position DECIMAL(4,3),
    PRIMARY KEY (symbol, feature_date)
);

CREATE INDEX idx_ml_features_recent ON ml_features(symbol, feature_date DESC);

-- Model predictions cache
CREATE TABLE ml_predictions (
    prediction_id VARCHAR PRIMARY KEY,
    model_version VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    prediction_date DATE NOT NULL,
    prediction_type VARCHAR(20),
    predictions JSON,
    confidence DECIMAL(4,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ml_predictions ON ml_predictions(symbol, prediction_date DESC, model_version);
```

### 6. System Tables

```sql
-- Performance tracking
CREATE TABLE query_performance (
    query_hash VARCHAR PRIMARY KEY,
    query_pattern VARCHAR NOT NULL,
    avg_duration_ms DECIMAL(10,2),
    p95_duration_ms DECIMAL(10,2),
    execution_count BIGINT,
    last_execution TIMESTAMP,
    index_used VARCHAR
);

-- Data quality metrics
CREATE TABLE data_quality_metrics (
    check_date DATE NOT NULL,
    table_name VARCHAR NOT NULL,
    metric_name VARCHAR NOT NULL,
    metric_value DECIMAL(10,4),
    status VARCHAR(10),
    PRIMARY KEY (check_date, table_name, metric_name)
);
```

## Index Strategy

### Primary Access Patterns & Indexes

1. **Wheel Candidate Search** (Most Frequent)
   - Index: `idx_options_wheel` 
   - Covers: symbol → expiration → delta range → latest data
   - Includes: price & IV data to avoid table lookup

2. **Portfolio Monitoring**
   - Index: `idx_positions_active`
   - Covers: active positions by account/symbol

3. **Expiration Management**
   - Index: `idx_positions_expiring`
   - Covers: positions nearing expiration

4. **Historical Analysis**
   - Index: `idx_market_symbol_date`
   - Covers: price history lookups

## Performance Optimizations

### 1. Hardware-Specific Tuning
```sql
-- M4 Pro optimizations
PRAGMA memory_limit='8GB';      -- 33% of 24GB RAM
PRAGMA threads=6;               -- Half of 12 cores
PRAGMA checkpoint_threshold='1GB';
PRAGMA enable_profiling=true;
PRAGMA enable_object_cache=true;
```

### 2. Query Optimization Rules
- All queries use covering indexes (90%+ index-only scans)
- Materialized views for complex aggregations
- Partition pruning for date-range queries
- No cross-database JOINs

### 3. Data Retention Policy
- Intraday options data: 7 days
- Daily options data: 90 days  
- Market data: 3 years
- ML features: 1 year
- Audit trail: Indefinite

## Migration Strategy

### Phase 1: Schema Creation (Day 1)
1. Create new `wheel_trading.duckdb` with optimal schema
2. Set up partitions and indexes
3. Configure materialized view refresh jobs

### Phase 2: Data Migration (Day 2-3)
1. Migrate historical market data from parquet files
2. Migrate active positions and recent options data
3. Rebuild ML features from historical data

### Phase 3: Application Updates (Day 4-5)
1. Update connection strings in config
2. Modify queries to use new schema
3. Update data providers to write to new structure

### Phase 4: Validation & Cutover (Day 6-7)
1. Parallel run with data validation
2. Performance testing and optimization
3. Final cutover and old database archival

## Expected Benefits

1. **Performance**: 
   - 80% reduction in query time for wheel candidates
   - 90% reduction for portfolio calculations
   - Near-instant (<10ms) position lookups

2. **Storage**:
   - 40% reduction through deduplication
   - Better compression with partitioning
   - Automatic old data pruning

3. **Maintainability**:
   - Single database to backup/monitor
   - Clear table purposes
   - Self-documenting schema

4. **Scalability**:
   - Ready for 10x data growth
   - Partition-wise operations
   - Parallel query execution