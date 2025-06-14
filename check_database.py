#!/usr/bin/env python3
"""Check database structure and data integrity."""

import duckdb

def check_database():
    # Connect to database
    conn = duckdb.connect('data/wheel_trading_optimized.duckdb')
    
    # Get all tables
    print("Tables in database:")
    tables = conn.execute('SHOW TABLES').fetchall()
    print(tables)
    
    if not tables:
        print("\nERROR: Database is empty! No tables found.")
        # Check the archived master database
        print("\nChecking archived master database...")
        conn_master = duckdb.connect('data/archive/wheel_trading_master.duckdb')
        master_tables = conn_master.execute('SHOW TABLES').fetchall()
        print(f"Master database tables: {master_tables}")
        
        if master_tables:
            # Check Unity data in master
            for table in master_tables:
                table_name = table[0]
                try:
                    count = conn_master.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = 'U'").fetchone()[0]
                    print(f"\n{table_name} Unity records: {count}")
                except:
                    print(f"\n{table_name} - no symbol column or error")
        conn_master.close()
    else:
        # Check each table
        for table in tables:
            table_name = table[0]
            print(f"\n\nTable: {table_name}")
            
            # Get schema
            schema = conn.execute(f'DESCRIBE {table_name}').fetchall()
            print("Schema:")
            for col in schema[:10]:  # Show first 10 columns
                print(f"  {col}")
            
            # Get row count
            count = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
            print(f"Total rows: {count}")
            
            # Check for Unity data if symbol column exists
            if any('symbol' in str(col).lower() for col in schema):
                unity_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = 'U'").fetchone()[0]
                print(f"Unity (U) rows: {unity_count}")
    
    conn.close()

if __name__ == "__main__":
    check_database()