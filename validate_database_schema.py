#!/usr/bin/env python3
"""Validate database schema and data integrity for wheel trading system."""

import duckdb
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name]
    ).fetchone()
    return result[0] > 0

def validate_table_schema(conn: duckdb.DuckDBPyConnection, table_name: str) -> Dict[str, List[str]]:
    """Validate a table's schema and return issues found."""
    issues = []
    
    # Get table schema
    schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
    columns = {col[0]: col[1] for col in schema}
    
    # Define expected schemas
    expected_schemas = {
        'stock_data': {
            'symbol': 'VARCHAR',
            'date': ['DATE', 'TIMESTAMP'],
            'open': ['DOUBLE', 'DECIMAL'],
            'high': ['DOUBLE', 'DECIMAL'],
            'low': ['DOUBLE', 'DECIMAL'],
            'close': ['DOUBLE', 'DECIMAL'],
            'volume': ['BIGINT', 'INTEGER']
        },
        'options_data': {
            'symbol': 'VARCHAR',
            'expiration': ['DATE', 'TIMESTAMP'],
            'strike': ['DOUBLE', 'DECIMAL'],
            'option_type': 'VARCHAR',
            'bid': ['DOUBLE', 'DECIMAL'],
            'ask': ['DOUBLE', 'DECIMAL']
        },
        'market_data': {
            'symbol': 'VARCHAR',
            'timestamp': ['TIMESTAMP', 'DATETIME'],
            'price': ['DOUBLE', 'DECIMAL']
        }
    }
    
    if table_name in expected_schemas:
        expected = expected_schemas[table_name]
        for col, expected_type in expected.items():
            if col not in columns:
                issues.append(f"Missing column: {col}")
            else:
                actual_type = columns[col].upper()
                if isinstance(expected_type, list):
                    if not any(exp.upper() in actual_type for exp in expected_type):
                        issues.append(f"Column {col}: expected one of {expected_type}, got {actual_type}")
                else:
                    if expected_type.upper() not in actual_type:
                        issues.append(f"Column {col}: expected {expected_type}, got {actual_type}")
    
    return {table_name: issues}

def check_data_integrity(conn: duckdb.DuckDBPyConnection, table_name: str) -> Dict[str, any]:
    """Check data integrity for a table."""
    integrity_report = {
        'row_count': 0,
        'null_counts': {},
        'date_range': None,
        'unique_symbols': [],
        'issues': []
    }
    
    # Get row count
    integrity_report['row_count'] = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
    if integrity_report['row_count'] == 0:
        integrity_report['issues'].append("Table is empty")
        return integrity_report
    
    # Get column names
    columns = [col[0] for col in conn.execute(f"DESCRIBE {table_name}").fetchall()]
    
    # Check for nulls in each column
    for col in columns:
        null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
        if null_count > 0:
            integrity_report['null_counts'][col] = null_count
    
    # Check date range if date/timestamp column exists
    date_cols = ['date', 'timestamp', 'ts_event', 'expiration']
    for col in date_cols:
        if col in columns:
            try:
                min_date = conn.execute(f"SELECT MIN({col}) FROM {table_name}").fetchone()[0]
                max_date = conn.execute(f"SELECT MAX({col}) FROM {table_name}").fetchone()[0]
                integrity_report['date_range'] = (min_date, max_date)
                break
            except:
                pass
    
    # Get unique symbols if symbol column exists
    if 'symbol' in columns:
        symbols = conn.execute(f"SELECT DISTINCT symbol FROM {table_name}").fetchall()
        integrity_report['unique_symbols'] = [s[0] for s in symbols]
    
    # Table-specific checks
    if table_name == 'stock_data' and 'symbol' in columns:
        # Check for Unity data
        unity_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = 'U'").fetchone()[0]
        if unity_count == 0:
            integrity_report['issues'].append("No Unity (U) stock data found")
        else:
            integrity_report['unity_count'] = unity_count
            
        # Check for data gaps
        if integrity_report['date_range'] and unity_count > 0:
            expected_days = (integrity_report['date_range'][1] - integrity_report['date_range'][0]).days
            if unity_count < expected_days * 0.7:  # Assuming ~250 trading days per year
                integrity_report['issues'].append(f"Potential data gaps: {unity_count} records for {expected_days} days")
    
    elif table_name == 'options_data' and 'symbol' in columns:
        # Check for Unity options
        unity_options = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = 'U'").fetchone()[0]
        if unity_options == 0:
            integrity_report['issues'].append("No Unity (U) options data found")
        else:
            integrity_report['unity_options_count'] = unity_options
    
    return integrity_report

def validate_database(db_path: str = 'data/wheel_trading_optimized.duckdb'):
    """Run complete database validation."""
    print(f"\n{'='*60}")
    print(f"DATABASE VALIDATION REPORT")
    print(f"Database: {db_path}")
    print(f"{'='*60}\n")
    
    try:
        conn = duckdb.connect(db_path)
        
        # Get all tables
        tables = conn.execute("SHOW TABLES").fetchall()
        
        if not tables:
            print("❌ CRITICAL: Database is EMPTY! No tables found.")
            print("\nRecommendation: Restore from archive using:")
            print("  cp data/archive/wheel_trading_master.duckdb data/wheel_trading_optimized.duckdb")
            return
        
        print(f"✅ Found {len(tables)} tables\n")
        
        # Required tables for wheel trading
        required_tables = ['stock_data', 'options_data', 'market_data', 'active_options']
        missing_tables = []
        
        table_names = [t[0] for t in tables]
        for req_table in required_tables:
            if req_table not in table_names:
                missing_tables.append(req_table)
        
        if missing_tables:
            print(f"❌ Missing required tables: {', '.join(missing_tables)}\n")
        else:
            print("✅ All required tables present\n")
        
        # Validate each table
        total_issues = 0
        for table in tables:
            table_name = table[0]
            print(f"\n{'='*40}")
            print(f"Table: {table_name}")
            print(f"{'='*40}")
            
            # Schema validation
            schema_issues = validate_table_schema(conn, table_name)
            if schema_issues[table_name]:
                print(f"\n❌ Schema Issues:")
                for issue in schema_issues[table_name]:
                    print(f"   - {issue}")
                    total_issues += 1
            else:
                print("\n✅ Schema OK")
            
            # Data integrity
            integrity = check_data_integrity(conn, table_name)
            print(f"\nData Summary:")
            print(f"  - Row count: {integrity['row_count']:,}")
            
            if integrity['date_range']:
                print(f"  - Date range: {integrity['date_range'][0]} to {integrity['date_range'][1]}")
            
            if integrity['unique_symbols']:
                print(f"  - Symbols: {', '.join(integrity['unique_symbols'][:10])}")
                if len(integrity['unique_symbols']) > 10:
                    print(f"    (and {len(integrity['unique_symbols']) - 10} more)")
            
            if 'unity_count' in integrity:
                print(f"  - Unity stock records: {integrity['unity_count']:,}")
            
            if 'unity_options_count' in integrity:
                print(f"  - Unity options records: {integrity['unity_options_count']:,}")
            
            if integrity['null_counts']:
                print(f"\n⚠️  NULL values found:")
                for col, count in integrity['null_counts'].items():
                    print(f"   - {col}: {count:,} nulls")
            
            if integrity['issues']:
                print(f"\n❌ Data Issues:")
                for issue in integrity['issues']:
                    print(f"   - {issue}")
                    total_issues += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        if total_issues == 0:
            print("✅ Database validation PASSED - No issues found")
        else:
            print(f"❌ Database validation FAILED - {total_issues} issues found")
            print("\nNext Steps:")
            print("1. Review the issues above")
            print("2. Consider restoring from archive if database is corrupted")
            print("3. Run data collection scripts to populate missing data")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ ERROR: Failed to validate database: {e}")
        print("\nThis usually means the database file is corrupted or missing.")

if __name__ == "__main__":
    # First check the main database
    validate_database()
    
    # Also check the archive
    print("\n\n" + "="*80)
    print("CHECKING ARCHIVE DATABASE")
    print("="*80)
    validate_database('data/archive/wheel_trading_master.duckdb')