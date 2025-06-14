#!/usr/bin/env python3
"""
Create optimized DuckDB schema with M4 Pro hardware acceleration
Uses all 12 CPU cores and optimized memory settings
"""

import duckdb
import os
from pathlib import Path
from datetime import datetime, timedelta

def create_optimized_database():
    """Create the optimized database with hardware-specific settings"""
    
    db_path = "data/wheel_trading_optimized.duckdb"
    
    # Remove if exists
    if os.path.exists(db_path):
        print(f"⚠️  Removing existing {db_path}")
        os.remove(db_path)
    
    print(f"🚀 Creating optimized database at {db_path}")
    conn = duckdb.connect(db_path)
    
    # Configure for M4 Pro hardware acceleration
    print("⚡ Configuring for M4 Pro (12 cores, 24GB RAM)...")
    conn.execute("PRAGMA memory_limit='8GB'")  # Use 8GB (33% of 24GB)
    conn.execute("PRAGMA threads=10")  # Use 10 cores (leaving 2 for system)
    conn.execute("PRAGMA checkpoint_threshold='1GB'")  # Optimize checkpoints
    conn.execute("PRAGMA enable_profiling='json'")  # Enable query profiling
    conn.execute("PRAGMA enable_object_cache=true")  # Cache parsed objects
    
    # Create schemas
    print("📁 Creating schemas...")
    conn.execute("CREATE SCHEMA IF NOT EXISTS market")
    conn.execute("CREATE SCHEMA IF NOT EXISTS options")  
    conn.execute("CREATE SCHEMA IF NOT EXISTS trading")
    conn.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    conn.execute("CREATE SCHEMA IF NOT EXISTS system")
    
    # Create market data table with partitioning support
    print("📊 Creating market data tables...")
    conn.execute("""
        CREATE TABLE market.price_data (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10,2),
            high DECIMAL(10,2),
            low DECIMAL(10,2),
            close DECIMAL(10,2),
            volume BIGINT,
            daily_return DECIMAL(8,6),
            volatility_20d DECIMAL(8,6),
            volatility_60d DECIMAL(8,6),
            -- Partitioning column (computed on insert)
            year_month INTEGER,
            PRIMARY KEY (symbol, date)
        )
    """)
    
    # Create options table with smart filtering
    print("📈 Creating options tables...")
    conn.execute("""
        CREATE TABLE options.contracts (
            symbol VARCHAR NOT NULL,
            expiration DATE NOT NULL,
            strike DECIMAL(10,2) NOT NULL,
            option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('CALL', 'PUT')),
            timestamp TIMESTAMP NOT NULL,
            -- Market data
            bid DECIMAL(10,4),
            ask DECIMAL(10,4),
            mid DECIMAL(10,4),  -- Will calculate on insert
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
            days_to_expiry INTEGER,  -- Will calculate on insert
            -- Partitioning
            year_month INTEGER,  -- Will calculate on insert
            PRIMARY KEY (symbol, expiration, strike, option_type, timestamp)
        )
    """)
    
    # Create HIGH-PERFORMANCE covering indexes
    print("🏎️  Creating performance indexes...")
    
    # Wheel candidate search - MOST CRITICAL
    conn.execute("""
        CREATE INDEX idx_wheel_candidates ON options.contracts(
            symbol, expiration, delta, timestamp DESC,
            strike, bid, ask, implied_volatility, moneyness, days_to_expiry
        )
        WHERE option_type = 'PUT' 
        AND delta BETWEEN -0.35 AND -0.25
        AND days_to_expiry BETWEEN 20 AND 45
    """)
    
    # Covered call candidates
    conn.execute("""
        CREATE INDEX idx_covered_calls ON options.contracts(
            symbol, expiration, strike, timestamp DESC,
            bid, ask, delta, implied_volatility
        )
        WHERE option_type = 'CALL' 
        AND delta BETWEEN 0.15 AND 0.35
    """)
    
    # Recent data access
    conn.execute("""
        CREATE INDEX idx_recent_options ON options.contracts(
            timestamp DESC, symbol,
            strike, option_type, bid, ask, delta
        )
    """)
    
    # Market data indexes
    conn.execute("""
        CREATE INDEX idx_market_symbol_date ON market.price_data(
            symbol, date DESC, close, volume, volatility_20d
        )
    """)
    
    conn.execute("""
        CREATE INDEX idx_market_volatility ON market.price_data(
            symbol, volatility_20d DESC
        )
        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    """)
    
    # Trading tables
    print("💼 Creating trading tables...")
    conn.execute("""
        CREATE TABLE trading.positions (
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
            option_type VARCHAR(4),
            -- Current state
            current_price DECIMAL(10,2),
            unrealized_pnl DECIMAL(12,2),
            status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
            -- Risk metrics
            position_delta DECIMAL(8,4),
            position_theta DECIMAL(10,2),
            margin_required DECIMAL(12,2),
            -- Metadata
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_active_positions (account_id, symbol) WHERE status = 'ACTIVE',
            INDEX idx_expiring_positions (expiration, symbol) WHERE status = 'ACTIVE'
        )
    """)
    
    conn.execute("""
        CREATE TABLE trading.decisions (
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
            risk_score DECIMAL(4,3),
            INDEX idx_pending_decisions (symbol, timestamp DESC) WHERE executed = FALSE,
            INDEX idx_executed_decisions (symbol, execution_timestamp DESC) WHERE executed = TRUE
        )
    """)
    
    # Analytics tables
    print("🧮 Creating analytics tables...")
    conn.execute("""
        CREATE TABLE analytics.ml_features (
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
            PRIMARY KEY (symbol, feature_date),
            INDEX idx_ml_recent (symbol, feature_date DESC)
        )
    """)
    
    # System tables
    print("⚙️  Creating system tables...")
    conn.execute("""
        CREATE TABLE system.migration_log (
            migration_id INTEGER PRIMARY KEY,
            table_name VARCHAR NOT NULL,
            rows_migrated BIGINT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            status VARCHAR(20),
            error_message TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE system.performance_metrics (
            metric_date DATE NOT NULL,
            metric_name VARCHAR NOT NULL,
            metric_value DECIMAL(10,4),
            PRIMARY KEY (metric_date, metric_name)
        )
    """)
    
    print("✅ Schema creation complete!")
    
    # Show configuration
    result = conn.execute("""
        SELECT 
            current_setting('memory_limit') as memory_limit,
            current_setting('threads') as threads,
            current_setting('checkpoint_threshold') as checkpoint
    """).fetchone()
    
    print(f"\n🔧 Configuration:")
    print(f"   Memory Limit: {result[0]}")
    print(f"   Threads: {result[1]}")
    print(f"   Checkpoint: {result[2]}")
    
    conn.close()
    print(f"\n✨ Database created successfully at {db_path}")

if __name__ == "__main__":
    create_optimized_database()