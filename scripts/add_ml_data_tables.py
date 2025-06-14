#!/usr/bin/env python3
"""
Add ML data tables to optimized database
Supports comprehensive market snapshots for future strategy development
"""

import duckdb
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_ml_tables(db_path: str = "data/wheel_trading_optimized.duckdb"):
    """Add ML-specific tables to the database"""
    
    logger.info("🧠 Adding ML data tables to database...")
    
    conn = duckdb.connect(db_path)
    
    try:
        # Create ML data schema
        conn.execute("CREATE SCHEMA IF NOT EXISTS ml_data")
        logger.info("✅ Created ml_data schema")
        
        # 1. Market snapshots table - high frequency underlying data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_data.market_snapshots (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                collection_type VARCHAR,  -- 'regular', 'high_vol', 'near_expiry', 'event'
                
                -- Price data
                spot_price DECIMAL(10,2),
                bid DECIMAL(10,2),
                ask DECIMAL(10,2),
                bid_size INTEGER,
                ask_size INTEGER,
                
                -- Volume data
                volume BIGINT,
                vwap DECIMAL(10,2),
                dollar_volume DECIMAL(15,2),
                
                -- Microstructure
                bid_ask_spread DECIMAL(10,4),
                effective_spread DECIMAL(10,4),
                order_imbalance DECIMAL(6,4),
                quote_count INTEGER,
                trade_count INTEGER,
                
                -- Intraday metrics
                high DECIMAL(10,2),
                low DECIMAL(10,2),
                open DECIMAL(10,2),
                intraday_range DECIMAL(6,4),
                intraday_trend DECIMAL(6,4),
                
                -- Market regime
                iv_regime VARCHAR,
                skew_level DECIMAL(6,4),
                
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Create indexes for ML queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_market_time 
            ON ml_data.market_snapshots(timestamp DESC)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_market_regime 
            ON ml_data.market_snapshots(symbol, iv_regime, timestamp)
        """)
        
        logger.info("✅ Created market_snapshots table")
        
        # 2. Option snapshots - comprehensive option chain data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_data.option_snapshots (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                expiration DATE NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                
                -- Price data
                bid DECIMAL(10,4),
                ask DECIMAL(10,4),
                last DECIMAL(10,4),
                
                -- Volume/OI
                volume INTEGER,
                open_interest INTEGER,
                volume_oi_ratio DECIMAL(10,4),
                
                -- Greeks
                implied_volatility DECIMAL(8,6),
                delta DECIMAL(6,4),
                gamma DECIMAL(8,6),
                vega DECIMAL(10,4),
                theta DECIMAL(10,4),
                rho DECIMAL(10,4),
                
                -- Derived metrics
                moneyness DECIMAL(6,4),
                days_to_expiry INTEGER,
                bid_ask_spread_pct DECIMAL(6,4),
                
                -- ML features
                iv_rank_contract DECIMAL(4,3),  -- IV rank for this specific contract
                volume_surprise DECIMAL(10,4)   -- Volume vs 20-day average
            )
        """)
        
        # Create indexes separately
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_option_snapshot 
            ON ml_data.option_snapshots(symbol, timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_option_moneyness 
            ON ml_data.option_snapshots(symbol, moneyness, timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_option_expiry 
            ON ml_data.option_snapshots(symbol, days_to_expiry, timestamp)
        """)
        logger.info("✅ Created option_snapshots table")
        
        # 3. IV surface metrics - aggregated surface characteristics
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_data.surface_metrics (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                
                -- Core IV metrics
                atm_iv DECIMAL(8,6),
                atm_iv_change DECIMAL(8,6),  -- vs previous snapshot
                
                -- Skew metrics
                iv_skew_25d DECIMAL(8,6),    -- 25-delta put IV - call IV
                iv_skew_10d DECIMAL(8,6),    -- 10-delta skew (tail risk)
                risk_reversal_25d DECIMAL(8,6),
                
                -- Term structure
                term_structure_slope DECIMAL(8,6),  -- 60d - 30d IV
                term_structure_curvature DECIMAL(8,6),  -- Butterfly
                
                -- Put/Call ratios
                put_call_volume_ratio DECIMAL(10,4),
                put_call_oi_ratio DECIMAL(10,4),
                put_call_dollar_ratio DECIMAL(10,4),  -- Dollar-weighted
                
                -- Market maker positioning
                total_gamma DECIMAL(15,2),
                total_vanna DECIMAL(15,2),
                total_charm DECIMAL(15,2),
                gamma_flip_level DECIMAL(10,2),  -- Where gamma changes sign
                
                -- Spread metrics by moneyness
                avg_spread_otm DECIMAL(6,4),
                avg_spread_atm DECIMAL(6,4),
                avg_spread_itm DECIMAL(6,4),
                
                -- Liquidity metrics
                total_option_volume BIGINT,
                total_option_oi BIGINT,
                option_liquidity_score DECIMAL(6,4),
                
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_surface_time 
            ON ml_data.surface_metrics(timestamp DESC)
        """)
        logger.info("✅ Created surface_metrics table")
        
        # 4. ML predictions table - store model outputs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_data.model_predictions (
                prediction_id VARCHAR PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                model_version VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                
                -- Prediction targets
                prediction_type VARCHAR NOT NULL,  -- 'assignment', 'iv_direction', 'optimal_strike'
                prediction_horizon INTEGER,  -- Hours ahead
                
                -- Predictions
                prediction_value DECIMAL(10,6),
                confidence_score DECIMAL(4,3),
                prediction_json JSON,  -- Full prediction details
                
                -- Feature snapshot
                feature_json JSON,  -- Features used for prediction
                
                -- Outcome tracking
                actual_value DECIMAL(10,6),
                prediction_error DECIMAL(10,6),
                evaluated_at TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_pred_lookup 
            ON ml_data.model_predictions(symbol, prediction_type, timestamp DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_pred_evaluation 
            ON ml_data.model_predictions(evaluated_at)
        """)
        logger.info("✅ Created model_predictions table")
        
        # 5. Training datasets table - curated datasets for model training
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_data.training_datasets (
                dataset_id VARCHAR PRIMARY KEY,
                dataset_name VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Dataset metadata
                symbol VARCHAR,
                start_date DATE,
                end_date DATE,
                row_count INTEGER,
                feature_count INTEGER,
                
                -- Dataset location
                storage_path VARCHAR,  -- Path to parquet/arrow file
                compression VARCHAR,
                
                -- Quality metrics
                missing_data_pct DECIMAL(6,4),
                outlier_pct DECIMAL(6,4),
                
                -- Usage tracking
                last_used TIMESTAMP,
                use_count INTEGER DEFAULT 0
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_dataset_lookup 
            ON ml_data.training_datasets(dataset_name, symbol)
        """)
        logger.info("✅ Created training_datasets table")
        
        # 6. Feature importance tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_data.feature_importance (
                model_name VARCHAR NOT NULL,
                model_version VARCHAR NOT NULL,
                feature_name VARCHAR NOT NULL,
                importance_score DECIMAL(10,6),
                importance_rank INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                PRIMARY KEY (model_name, model_version, feature_name)
            )
        """)
        logger.info("✅ Created feature_importance table")
        
        # Create a view for latest complete snapshot
        conn.execute("""
            CREATE OR REPLACE VIEW ml_data.latest_snapshot AS
            WITH latest_time AS (
                SELECT MAX(timestamp) as max_timestamp
                FROM ml_data.market_snapshots
                WHERE symbol = 'U'
            )
            SELECT 
                ms.*,
                sm.atm_iv,
                sm.iv_skew_25d,
                sm.term_structure_slope,
                sm.put_call_volume_ratio,
                sm.total_gamma
            FROM ml_data.market_snapshots ms
            JOIN ml_data.surface_metrics sm 
                ON ms.symbol = sm.symbol 
                AND ms.timestamp = sm.timestamp
            JOIN latest_time lt 
                ON ms.timestamp = lt.max_timestamp
            WHERE ms.symbol = 'U'
        """)
        logger.info("✅ Created latest_snapshot view")
        
        # Add storage info to system
        conn.execute("USE system")
        conn.execute("""
            INSERT INTO migration_log (
                migration_id, table_name, rows_migrated, 
                started_at, completed_at, status
            ) VALUES (
                nextval('migration_seq'), 'ml_data_tables', 0,
                ?, ?, 'completed'
            )
        """, [datetime.now(), datetime.now()])
        conn.execute("USE main")
        
        conn.commit()
        logger.info("\n✅ ML data tables added successfully!")
        
        # Show table summary
        tables = conn.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = 'ml_data'
            ORDER BY table_name
        """).fetchall()
        
        logger.info("\n📊 ML Data Tables:")
        for schema, table in tables:
            row_count = conn.execute(f"SELECT COUNT(*) FROM {schema}.{table}").fetchone()[0]
            logger.info(f"   {schema}.{table}: {row_count} rows")
        
        # Check total database size
        db_size = conn.execute("""
            SELECT 
                database_name,
                database_size / 1024 / 1024 as size_mb
            FROM duckdb_databases()
            WHERE database_name NOT IN ('system', 'temp')
        """).fetchone()
        
        if db_size:
            logger.info(f"\n💾 Database size: {db_size[1]:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Error adding ML tables: {e}")
        raise
    finally:
        conn.close()

def show_collection_strategy():
    """Show recommended data collection strategy"""
    
    print("""
📊 ML Data Collection Strategy
==============================

1. Regular Market Hours (9:30 AM - 4:00 PM ET)
   - Frequency: Every 30 minutes
   - Data: Full option surface (±50% moneyness)
   - Purpose: Baseline ML training data

2. High Volatility Mode (VIX > 30)
   - Frequency: Every 15 minutes
   - Data: Extended surface + microstructure
   - Purpose: Regime-specific model training

3. Near Expiration (< 2 DTE)
   - Frequency: Every 5 minutes
   - Data: ATM ± 10% strikes
   - Purpose: Assignment prediction model

4. Around Events (Earnings ± 2 hours)
   - Frequency: Every 5 minutes
   - Data: Full surface + trade flow
   - Purpose: Event impact analysis

Storage Estimates:
- Regular day: ~50 MB/day
- High vol day: ~150 MB/day
- With 1 year retention: ~15-45 GB

The schema is designed to be flexible:
- Can adjust collection frequency
- Can filter data after collection
- Can create derived features offline
- Can export to Parquet for ML training
    """)

if __name__ == "__main__":
    # Add ML tables
    add_ml_tables()
    
    # Show collection strategy
    show_collection_strategy()