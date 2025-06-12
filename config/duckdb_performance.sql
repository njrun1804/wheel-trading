-- DuckDB Performance Configuration for Wheel Trading
-- Optimized for M4 Mac with 24GB RAM

-- Memory Configuration (33% of system RAM = 8GB, leaving 16GB for system/Claude)
PRAGMA memory_limit='8GB';
PRAGMA temp_directory='/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/data/cache/duckdb_temp';

-- Thread Configuration (use 6 of 8 performance cores)
PRAGMA threads=6;

-- Enable parallel query execution
PRAGMA enable_parallel_query=true;
PRAGMA parallel_scan=true;

-- Optimize for analytics workloads
PRAGMA enable_optimizer=true;
PRAGMA optimizer_reorder_filter=true;
PRAGMA optimizer_join_threshold=100000;

-- Cache configuration (conservative for 24GB system)
PRAGMA cache_size='2GB';

-- Checkpoint configuration (reduce write amplification)
PRAGMA checkpoint_threshold='1GB';
PRAGMA wal_autocheckpoint='1GB';

-- Statistics and profiling
PRAGMA enable_profiling='json';
PRAGMA profiling_output='/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/logs/duckdb_profiles';

-- Create optimized settings function
CREATE OR REPLACE MACRO apply_performance_settings() AS (
    CASE 
        WHEN current_setting('memory_limit') != '8GB' THEN 
            'Settings not applied - run source config/duckdb_performance.sql'
        ELSE 
            'Performance settings active'
    END
);

-- Index maintenance hints
-- CREATE INDEX IF NOT EXISTS idx_options_symbol_date ON options(symbol, quote_date);
-- CREATE INDEX IF NOT EXISTS idx_options_expiry ON options(expiry_date);
-- CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status, updated_at);