-- Database Redundancy Optimization Script
-- Execute this script to remove redundancies and optimize the database structure

-- ============================================================
-- PHASE 1: Remove Redundancies
-- ============================================================

-- Note: unified_trading.duckdb should be deleted at the file system level
-- rm ./data/unified_trading.duckdb

-- Remove duplicate slice_cache from wheel_trading_master
-- Connect to wheel_trading_master.duckdb first
DROP TABLE IF EXISTS slice_cache;

-- ============================================================
-- PHASE 2: Create Materialized Views
-- ============================================================

-- Create materialized version of todays_opportunities
DROP VIEW IF EXISTS todays_opportunities;
CREATE TABLE todays_opportunities_mat AS
SELECT 
    o.*,
    m.close AS stock_price,
    ((o.strike / m.close) - 1) * 100 AS strike_distance_pct
FROM options_data AS o
INNER JOIN market_data AS m 
    ON m.symbol = o.symbol 
    AND m.date = o.date
WHERE 
    o.date = (SELECT max(date) FROM options_data)
    AND o.volume > 0
    AND o.implied_volatility > 0.30
ORDER BY o.implied_volatility DESC;

-- Create index for performance
CREATE INDEX idx_todays_opportunities_symbol ON todays_opportunities_mat(symbol);
CREATE INDEX idx_todays_opportunities_iv ON todays_opportunities_mat(implied_volatility);

-- Create refresh procedure
CREATE OR REPLACE PROCEDURE refresh_todays_opportunities()
AS
BEGIN
    DROP TABLE IF EXISTS todays_opportunities_mat;
    CREATE TABLE todays_opportunities_mat AS
    SELECT 
        o.*,
        m.close AS stock_price,
        ((o.strike / m.close) - 1) * 100 AS strike_distance_pct
    FROM options_data AS o
    INNER JOIN market_data AS m 
        ON m.symbol = o.symbol 
        AND m.date = o.date
    WHERE 
        o.date = (SELECT max(date) FROM options_data)
        AND o.volume > 0
        AND o.implied_volatility > 0.30
    ORDER BY o.implied_volatility DESC;
END;

-- ============================================================
-- PHASE 3: Create Denormalized Tables
-- ============================================================

-- Create options_enhanced table with commonly joined data
CREATE TABLE IF NOT EXISTS options_enhanced AS
SELECT 
    o.*,
    m.close as underlying_price,
    m.high as underlying_high,
    m.low as underlying_low,
    m.volume as stock_volume,
    (o.strike / m.close - 1) * 100 as strike_distance_pct,
    o.strike - m.close as strike_distance_dollars,
    CASE 
        WHEN o.strike > m.close THEN 'OTM'
        WHEN o.strike < m.close THEN 'ITM'
        ELSE 'ATM'
    END as moneyness
FROM options_data o
LEFT JOIN market_data m 
    ON m.symbol = o.symbol 
    AND m.date = o.date;

-- Create indexes for options_enhanced
CREATE INDEX idx_options_enhanced_symbol_date ON options_enhanced(symbol, date);
CREATE INDEX idx_options_enhanced_expiration ON options_enhanced(expiration);
CREATE INDEX idx_options_enhanced_moneyness ON options_enhanced(moneyness);
CREATE INDEX idx_options_enhanced_iv ON options_enhanced(implied_volatility);

-- ============================================================
-- PHASE 4: Optimize Existing Tables
-- ============================================================

-- Add indexes to frequently queried columns
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date ON market_data(symbol, date);
CREATE INDEX IF NOT EXISTS idx_options_data_symbol_exp ON options_data(symbol, expiration);
CREATE INDEX IF NOT EXISTS idx_options_data_iv ON options_data(implied_volatility);
CREATE INDEX IF NOT EXISTS idx_options_data_delta ON options_data(delta);

-- ============================================================
-- PHASE 5: Create Summary Statistics Table
-- ============================================================

CREATE TABLE IF NOT EXISTS daily_option_summary AS
SELECT 
    date,
    symbol,
    COUNT(*) as total_contracts,
    COUNT(DISTINCT expiration) as unique_expirations,
    AVG(implied_volatility) as avg_iv,
    MAX(implied_volatility) as max_iv,
    MIN(implied_volatility) as min_iv,
    SUM(volume) as total_volume,
    SUM(open_interest) as total_oi,
    AVG(CASE WHEN option_type = 'call' THEN implied_volatility END) as avg_call_iv,
    AVG(CASE WHEN option_type = 'put' THEN implied_volatility END) as avg_put_iv,
    SUM(CASE WHEN option_type = 'call' THEN volume ELSE 0 END) as call_volume,
    SUM(CASE WHEN option_type = 'put' THEN volume ELSE 0 END) as put_volume
FROM options_data
GROUP BY date, symbol;

CREATE INDEX idx_daily_summary_symbol_date ON daily_option_summary(symbol, date);

-- ============================================================
-- PHASE 6: Clean Up Empty Tables in wheel_cache.duckdb
-- ============================================================

-- Connect to wheel_cache.duckdb and run:
-- DROP TABLE IF EXISTS bug_history;
-- DROP TABLE IF EXISTS code_complexity_cache;
-- DROP TABLE IF EXISTS code_performance_metrics;
-- DROP TABLE IF EXISTS greeks_cache;
-- DROP TABLE IF EXISTS optimization_experiments;
-- DROP TABLE IF EXISTS option_chains;  -- Unused, 0 rows
-- DROP TABLE IF EXISTS options_data;    -- Unused, 0 rows
-- DROP TABLE IF EXISTS position_snapshots;
-- DROP TABLE IF EXISTS predictions_cache;

-- ============================================================
-- Verification Queries
-- ============================================================

-- Check table sizes after optimization
SELECT 
    table_name,
    estimated_size
FROM duckdb_tables()
ORDER BY estimated_size DESC;

-- Verify materialized view
SELECT COUNT(*) as opportunity_count 
FROM todays_opportunities_mat;

-- Verify enhanced table
SELECT COUNT(*) as enhanced_count 
FROM options_enhanced;