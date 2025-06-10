#!/usr/bin/env python3
"""
Database cleanup script - removes redundant tables.
"""

import duckdb
from pathlib import Path
from datetime import datetime

def cleanup_database():
    """Remove redundant tables from the database."""
    db_path = Path("~/.wheel_trading/cache/wheel_cache.duckdb").expanduser()
    conn = duckdb.connect(str(db_path))
    
    print("="*60)
    print("DATABASE CLEANUP SCRIPT")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    # Tables to remove (redundant or empty)
    tables_to_remove = [
        'unity_price_history',      # Empty - redundant with price_history
        'unity_daily_stock',        # Empty - redundant
        'unity_daily_summary',      # 207 records - duplicate data
        'unity_daily_summary_real', # 861 records - duplicate data  
        'unity_options_ticks',      # 206K records - old format
        'unity_options_raw',        # Empty - temporary table
        'unity_options_processed',  # Empty - temporary table
        'unity_daily_options',      # Empty - redundant
        'options_ticks',            # 3.3M records - old format
        'instruments',              # May be empty/unused
        'risk_metrics'              # May be empty/unused
    ]
    
    # Check current database size
    db_size_before = db_path.stat().st_size / (1024 * 1024)
    print(f"Database size before cleanup: {db_size_before:.1f} MB")
    
    total_records_removed = 0
    
    for table in tables_to_remove:
        try:
            # Check if table exists and get record count
            exists = conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = '{table}'
            """).fetchone()[0]
            
            if exists:
                record_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                
                if record_count > 0:
                    print(f"\nDropping {table}: {record_count:,} records")
                    total_records_removed += record_count
                else:
                    print(f"\nDropping {table}: Empty table")
                
                conn.execute(f"DROP TABLE {table}")
                print(f"  ✅ {table} removed")
            else:
                print(f"\n❌ {table}: Table not found")
                
        except Exception as e:
            print(f"\n❌ Error removing {table}: {e}")
    
    # Vacuum to reclaim space
    print(f"\nVacuuming database to reclaim space...")
    conn.execute("VACUUM")
    
    # Check final database size
    db_size_after = db_path.stat().st_size / (1024 * 1024)
    space_saved = db_size_before - db_size_after
    
    print(f"\n" + "="*60)
    print("CLEANUP COMPLETE")
    print("="*60)
    print(f"Records removed: {total_records_removed:,}")
    print(f"Database size before: {db_size_before:.1f} MB")
    print(f"Database size after: {db_size_after:.1f} MB") 
    print(f"Space saved: {space_saved:.1f} MB ({space_saved/db_size_before*100:.1f}%)")
    
    # Show remaining tables
    print(f"\nRemaining tables:")
    remaining_tables = ['price_history', 'unity_options_daily', 'unity_stock_1min', 
                       'fred_series', 'fred_observations', 'fred_features']
    
    for table in remaining_tables:
        try:
            exists = conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = '{table}'
            """).fetchone()[0]
            
            if exists:
                records = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  ✅ {table}: {records:,} records")
        except:
            continue
    
    conn.close()
    
    print(f"\n💾 Database optimized and ready for trading analysis!")

if __name__ == "__main__":
    print("Auto-running database cleanup...")
    cleanup_database()