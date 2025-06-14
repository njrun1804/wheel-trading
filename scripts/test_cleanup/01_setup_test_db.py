#!/usr/bin/env python3
"""Setup isolated test database with real Unity data."""

import os
import sys
import shutil
import duckdb
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def setup_test_database():
    """Create test database with subset of real data."""
    print("Setting up test database...")
    
    # Paths
    # Try to find the right main database
    if os.path.exists("data/wheel_trading_master.duckdb"):
        main_db = "data/wheel_trading_master.duckdb"
    else:
        main_db = "data/wheel_trading_optimized.duckdb"
    test_db = "data/test_wheel_trading.duckdb"
    
    # Remove old test DB
    if os.path.exists(test_db):
        os.remove(test_db)
        print("✓ Removed old test database")
    
    # Connect to test database and attach main database
    test_conn = duckdb.connect(test_db)
    
    # Attach the main database as 'source' (not 'main' which is reserved)
    test_conn.execute(f"ATTACH '{main_db}' AS source (READ_ONLY)")
    
    # Create schemas
    schemas = ['market', 'options', 'trading', 'analytics', 'ml_data', 'system']
    for schema in schemas:
        test_conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    print("✓ Created schemas")
    
    # Try to copy specific tables we know exist
    # First, let's check what tables are available
    try:
        # Create the main tables we need for testing
        test_conn.execute("""
            CREATE TABLE IF NOT EXISTS market.price_data AS 
            SELECT * FROM source.market.price_data WHERE 1=0
        """)
        
        test_conn.execute("""
            CREATE TABLE IF NOT EXISTS options.contracts AS 
            SELECT * FROM source.options.contracts WHERE 1=0
        """)
        
        test_conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics.ml_features AS 
            SELECT * FROM source.analytics.ml_features WHERE 1=0
        """)
        
        print("✓ Created tables")
    except Exception as e:
        print(f"Warning: Could not create all tables: {e}")
        print("Creating minimal test structure...")
    
    # Copy recent data (last 30 days)
    cutoff_date = datetime.now().date() - timedelta(days=30)
    
    # Stock data
    stock_count = 0
    try:
        test_conn.execute(f"""
            INSERT INTO market.price_data
            SELECT * FROM source.market.price_data
            WHERE symbol = 'U' AND date >= '{cutoff_date}'
        """)
        stock_count = test_conn.execute("SELECT COUNT(*) FROM market.price_data").fetchone()[0]
        print(f"✓ Copied {stock_count} stock records")
    except Exception as e:
        print(f"Warning: Could not copy stock data: {e}")
    
    # Options data (recent only)
    options_count = 0
    try:
        test_conn.execute(f"""
            INSERT INTO options.contracts
            SELECT * FROM source.options.contracts  
            WHERE symbol = 'U' AND timestamp >= '{cutoff_date}'
        """)
        options_count = test_conn.execute("SELECT COUNT(*) FROM options.contracts").fetchone()[0]
        print(f"✓ Copied {options_count} option records")
    except Exception as e:
        print(f"Warning: Could not copy options data: {e}")
    
    # ML features
    ml_count = 0
    try:
        test_conn.execute(f"""
            INSERT INTO analytics.ml_features
            SELECT * FROM source.analytics.ml_features
            WHERE symbol = 'U' AND feature_date >= '{cutoff_date}'
        """)
        ml_count = test_conn.execute("SELECT COUNT(*) FROM analytics.ml_features").fetchone()[0]
        print(f"✓ Copied {ml_count} ML feature records")
    except Exception as e:
        print(f"Warning: Could not copy ML features: {e}")
    
    # Close connection
    test_conn.close()
    
    print("\n✅ Test database ready!")
    print(f"   Location: {test_db}")
    print(f"   Stock data: {stock_count} days")
    print(f"   Options: {options_count} contracts")
    print(f"   ML features: {ml_count} records")
    
    return test_db

if __name__ == "__main__":
    setup_test_database()