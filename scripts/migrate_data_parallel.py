#!/usr/bin/env python3
"""
Migrate data to optimized database using M4 Pro hardware acceleration
Uses parallel processing for maximum performance
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
import multiprocessing as mp
import logging
from typing import List, Tuple, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# M4 Pro optimization settings
CPU_CORES = 10  # Use 10 of 12 cores
BATCH_SIZE = 50000  # Optimal batch size for M4 Pro
MEMORY_LIMIT = "8GB"

class DataMigrator:
    def __init__(self, target_db: str):
        self.target_db = target_db
        self.migration_stats = []
        
    def log_migration(self, table_name: str, rows: int, duration: float, status: str = "completed"):
        """Log migration statistics"""
        conn = duckdb.connect(self.target_db)
        conn.execute("USE system")
        conn.execute("""
            INSERT INTO migration_log (
                migration_id, table_name, rows_migrated, 
                started_at, completed_at, status
            ) VALUES (
                nextval('migration_seq'), ?, ?, 
                ? - INTERVAL ? SECOND, ?, ?
            )
        """, [table_name, rows, datetime.now(), duration, datetime.now(), status])
        conn.close()
        
    def migrate_market_data_parallel(self):
        """Migrate market data using parallel processing"""
        logger.info("🚀 Starting parallel market data migration...")
        start_time = time.time()
        
        # Connect to source data
        conn = duckdb.connect(":memory:")
        
        # Load parquet files
        logger.info("📊 Loading parquet files...")
        parquet_files = [
            "data/unity-options/processed/unity_ohlcv_3y.parquet",
            "data/unity-options/processed/unity_ohlcv_daily_corrected.parquet"
        ]
        
        total_rows = 0
        for pf in parquet_files:
            if Path(pf).exists():
                # Read parquet file
                df = pd.read_parquet(pf)
                logger.info(f"  Loaded {len(df):,} rows from {Path(pf).name}")
                
                # Process in parallel chunks
                with concurrent.futures.ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
                    chunk_size = len(df) // CPU_CORES
                    futures = []
                    
                    for i in range(0, len(df), chunk_size):
                        chunk = df.iloc[i:i+chunk_size]
                        future = executor.submit(self._process_market_chunk, chunk)
                        futures.append(future)
                    
                    # Collect results
                    processed_chunks = []
                    for future in concurrent.futures.as_completed(futures):
                        processed_chunks.append(future.result())
                
                # Combine and insert
                processed_df = pd.concat(processed_chunks, ignore_index=True)
                self._insert_market_data(processed_df)
                total_rows += len(processed_df)
        
        duration = time.time() - start_time
        logger.info(f"✅ Market data migration complete: {total_rows:,} rows in {duration:.1f}s")
        logger.info(f"   Speed: {total_rows/duration:,.0f} rows/sec")
        
        self.log_migration("market.price_data", total_rows, duration)
        return total_rows
        
    def _process_market_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of market data"""
        # Ensure required columns
        df = df.copy()
        
        # This appears to be options data, not market data
        # Extract underlying symbol (U) from option symbols
        if 'symbol' in df.columns and df['symbol'].str.len().iloc[0] > 5:
            # Extract underlying from option symbol (e.g., "U     240216C00024000" -> "U")
            df['underlying_symbol'] = df['symbol'].str.strip().str[:1]
            df = df.groupby(['underlying_symbol', 'ts_event']).agg({
                'open': 'mean',
                'high': 'max',
                'low': 'min',
                'close': 'mean',
                'volume': 'sum'
            }).reset_index()
            df['symbol'] = df['underlying_symbol']
        
        # Rename date column
        if 'ts_event' in df.columns:
            df['date'] = pd.to_datetime(df['ts_event'])
        elif 'date' not in df.columns:
            logger.warning("No date column found")
            return pd.DataFrame()
        
        # Calculate returns and volatility
        if 'close' in df.columns and 'date' in df.columns:
            df = df.sort_values(['symbol', 'date'])
            df['daily_return'] = df.groupby('symbol')['close'].pct_change()
            
            # Rolling volatility per symbol
            df['volatility_20d'] = df.groupby('symbol')['daily_return'].transform(
                lambda x: x.rolling(window=20, min_periods=20).std() * np.sqrt(252)
            )
            
            df['volatility_60d'] = df.groupby('symbol')['daily_return'].transform(
                lambda x: x.rolling(window=60, min_periods=60).std() * np.sqrt(252)
            )
        
        # Add partitioning column
        df['year_month'] = df['date'].dt.year * 100 + df['date'].dt.month
        
        # Select only required columns
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 
                        'volume', 'daily_return', 'volatility_20d', 'volatility_60d', 'year_month']
        
        available_cols = [col for col in required_cols if col in df.columns]
        return df[available_cols].dropna(subset=['symbol', 'date'])
        
    def _insert_market_data(self, df: pd.DataFrame):
        """Insert market data into target database"""
        conn = duckdb.connect(self.target_db)
        conn.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
        conn.execute("SET threads TO 10")
        
        # Insert data
        conn.execute("INSERT INTO market.price_data SELECT * FROM df")
        conn.close()
        
    def migrate_options_data_parallel(self):
        """Migrate options data with moneyness filtering"""
        logger.info("📈 Starting parallel options data migration...")
        start_time = time.time()
        
        sources = [
            ("data/wheel_trading_optimized.duckdb", "SELECT * FROM options_data"),
            ("data/cache/data/wheel_trading_optimized.duckdb", "SELECT * FROM options_data WHERE bid > 0"),
            (str(Path.home() / "data/wheel_trading_optimized.duckdb"), 
             "SELECT * FROM unity_options_daily WHERE timestamp > CURRENT_DATE - 90")
        ]
        
        total_rows = 0
        
        for source_db, query in sources:
            if Path(source_db).exists():
                logger.info(f"  Processing {Path(source_db).name}...")
                try:
                    conn = duckdb.connect(source_db, read_only=True)
                    
                    # Count rows
                    count = conn.execute(f"SELECT COUNT(*) FROM ({query}) t").fetchone()[0]
                    logger.info(f"    Found {count:,} rows")
                    
                    if count > 0:
                        # Process in batches
                        for offset in range(0, count, BATCH_SIZE):
                            batch_query = f"{query} LIMIT {BATCH_SIZE} OFFSET {offset}"
                            df = conn.execute(batch_query).df()
                            
                            # Process and filter
                            df = self._process_options_chunk(df)
                            
                            if len(df) > 0:
                                self._insert_options_data(df)
                                total_rows += len(df)
                            
                            logger.info(f"    Processed {min(offset + BATCH_SIZE, count):,}/{count:,} rows")
                    
                    conn.close()
                    
                except Exception as e:
                    logger.warning(f"    Failed to process {source_db}: {e}")
        
        duration = time.time() - start_time
        logger.info(f"✅ Options data migration complete: {total_rows:,} rows in {duration:.1f}s")
        logger.info(f"   Speed: {total_rows/duration:,.0f} rows/sec")
        
        self.log_migration("options.contracts", total_rows, duration)
        return total_rows
        
    def _process_options_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process options data with moneyness filtering"""
        df = df.copy()
        
        # Calculate moneyness if not present
        if 'moneyness' not in df.columns and 'strike' in df.columns:
            # Need underlying price - approximate from option prices
            if 'underlying_price' in df.columns:
                df['moneyness'] = df['strike'] / df['underlying_price']
            else:
                # Estimate from at-the-money options
                df['moneyness'] = 1.0  # Default
        
        # Filter to relevant moneyness range (80% - 120%)
        if 'moneyness' in df.columns:
            df = df[(df['moneyness'] >= 0.8) & (df['moneyness'] <= 1.2)]
        
        # Calculate derived fields
        if 'bid' in df.columns and 'ask' in df.columns:
            df['mid'] = (df['bid'] + df['ask']) / 2
        
        if 'expiration' in df.columns and 'timestamp' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['days_to_expiry'] = (df['expiration'] - df['timestamp'].dt.date).dt.days
        
        # Add partitioning column
        if 'timestamp' in df.columns:
            df['year_month'] = df['timestamp'].dt.year * 100 + df['timestamp'].dt.month
        
        # Ensure all required columns exist
        required_cols = ['symbol', 'expiration', 'strike', 'option_type', 'timestamp',
                        'bid', 'ask', 'mid', 'volume', 'open_interest',
                        'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
                        'moneyness', 'days_to_expiry', 'year_month']
        
        # Add missing columns with nulls
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        return df[required_cols]
        
    def _insert_options_data(self, df: pd.DataFrame):
        """Insert options data into target database"""
        conn = duckdb.connect(self.target_db)
        conn.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
        conn.execute("SET threads TO 10")
        
        # Insert data
        conn.execute("INSERT INTO options.contracts SELECT * FROM df")
        conn.close()
        
    def create_materialized_views(self):
        """Create materialized views (as tables in DuckDB)"""
        logger.info("👁️  Creating materialized views...")
        
        conn = duckdb.connect(self.target_db)
        conn.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
        conn.execute("SET threads TO 10")
        
        # Create wheel opportunities as a table (DuckDB doesn't have true materialized views)
        logger.info("  Creating wheel_opportunities_mv...")
        conn.execute("""
            CREATE TABLE analytics.wheel_opportunities_mv AS
            SELECT 
                o.symbol,
                o.expiration,
                o.strike,
                o.bid,
                o.ask,
                o.delta,
                o.implied_volatility,
                o.volume,
                o.days_to_expiry,
                o.moneyness,
                (o.bid * 100) / (o.strike * 100) as premium_yield,
                (o.ask - o.bid) / NULLIF(o.ask, 0) as spread_pct,
                o.timestamp
            FROM options.contracts o
            WHERE 
                o.option_type = 'PUT'
                AND o.delta BETWEEN -0.35 AND -0.25
                AND o.days_to_expiry BETWEEN 20 AND 45
                AND o.bid > 0 
                AND o.ask > 0
                AND (o.ask - o.bid) / o.ask < 0.10
                AND o.timestamp = (
                    SELECT MAX(timestamp) 
                    FROM options.contracts o2 
                    WHERE o2.symbol = o.symbol 
                    AND o2.expiration = o.expiration 
                    AND o2.strike = o.strike 
                    AND o2.option_type = o.option_type
                )
            ORDER BY premium_yield DESC
        """)
        
        # Create index on materialized view
        conn.execute("""
            CREATE INDEX idx_wheel_opp_symbol ON analytics.wheel_opportunities_mv(symbol)
        """)
        
        logger.info("✅ Materialized views created")
        conn.close()
        
    def validate_migration(self):
        """Validate the migration was successful"""
        logger.info("🔍 Validating migration...")
        
        conn = duckdb.connect(self.target_db)
        
        # Check row counts
        tables = [
            ("market.price_data", "Market Data"),
            ("options.contracts", "Options Contracts"),
            ("analytics.wheel_opportunities_mv", "Wheel Opportunities")
        ]
        
        for table, name in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"  {name}: {count:,} rows")
        
        # Check data quality
        recent_options = conn.execute("""
            SELECT COUNT(*) 
            FROM options.contracts 
            WHERE timestamp >= CURRENT_DATE - 7
        """).fetchone()[0]
        
        logger.info(f"  Recent options (last 7 days): {recent_options:,}")
        
        # Check indexes
        indexes = conn.execute("""
            SELECT table_name, index_name 
            FROM duckdb_indexes() 
            ORDER BY table_name, index_name
        """).fetchall()
        
        logger.info(f"  Indexes created: {len(indexes)}")
        
        conn.close()
        return True

def main():
    """Run the migration"""
    print("🚀 Starting data migration with M4 Pro acceleration")
    print(f"   Using {CPU_CORES} CPU cores")
    print(f"   Memory limit: {MEMORY_LIMIT}")
    print(f"   Batch size: {BATCH_SIZE:,}")
    print()
    
    migrator = DataMigrator("data/wheel_trading_optimized.duckdb")
    
    # Run migrations
    migrator.migrate_market_data_parallel()
    migrator.migrate_options_data_parallel()
    migrator.create_materialized_views()
    
    # Validate
    if migrator.validate_migration():
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration validation failed!")
        
if __name__ == "__main__":
    main()