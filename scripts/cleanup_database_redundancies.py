#!/usr/bin/env python3
"""
Database Redundancy Cleanup Script
Removes redundant tables and optimizes database structure
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import duckdb


def backup_database(db_path: str) -> str:
    """Create a backup of the database before making changes"""
    backup_dir = Path("./data/backups")
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{Path(db_path).stem}_{timestamp}.duckdb"

    shutil.copy2(db_path, backup_path)
    print(f"✓ Backed up {db_path} to {backup_path}")
    return str(backup_path)


def cleanup_wheel_trading_master():
    """Remove redundancies from data/wheel_trading_optimized.duckdb"""
    db_path = "./data/wheel_trading_optimized.duckdb"

    if not os.path.exists(db_path):
        print(f"✗ Database not found: {db_path}")
        return

    # Backup first
    backup_database(db_path)

    conn = duckdb.connect(db_path)

    try:
        # Remove duplicate slice_cache
        conn.execute("DROP TABLE IF EXISTS slice_cache")
        print("✓ Removed duplicate slice_cache from wheel_trading_master")

        # Create materialized view for todays_opportunities
        conn.execute(
            """
            DROP VIEW IF EXISTS todays_opportunities
        """
        )

        conn.execute(
            """
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
            ORDER BY o.implied_volatility DESC
        """
        )
        print("✓ Created materialized view todays_opportunities_mat")

        # Create options_enhanced denormalized table
        conn.execute(
            """
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
                AND m.date = o.date
        """
        )
        print("✓ Created options_enhanced denormalized table")

        # Add indexes
        indexes = [
            ("idx_market_data_symbol_date", "market_data(symbol, date)"),
            ("idx_options_data_symbol_exp", "options_data(symbol, expiration)"),
            ("idx_options_data_iv", "options_data(implied_volatility)"),
            ("idx_options_enhanced_symbol_date", "options_enhanced(symbol, date)"),
            ("idx_options_enhanced_moneyness", "options_enhanced(moneyness)"),
        ]

        for idx_name, idx_def in indexes:
            try:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def}")
                print(f"✓ Created index {idx_name}")
            except Exception as e:
                print(f"✗ Failed to create index {idx_name}: {e}")

        conn.commit()

    except Exception as e:
        print(f"✗ Error updating wheel_trading_master: {e}")
        conn.rollback()
    finally:
        conn.close()


def cleanup_wheel_cache():
    """Remove empty tables from data/wheel_trading_optimized.duckdb"""
    db_path = "./data/cache/data/wheel_trading_optimized.duckdb"

    if not os.path.exists(db_path):
        print(f"✗ Database not found: {db_path}")
        return

    # Backup first
    backup_database(db_path)

    conn = duckdb.connect(db_path)

    try:
        # Get list of empty tables
        empty_tables = []
        tables = conn.execute("SHOW TABLES").fetchall()

        for table in tables:
            table_name = table[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            if count == 0:
                empty_tables.append(table_name)

        # Keep slice_cache and options_data as they might be used
        tables_to_remove = [
            t
            for t in empty_tables
            if t not in ["slice_cache", "options_data", "option_chains"]
        ]

        for table in tables_to_remove:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"✓ Removed empty table: {table}")

        conn.commit()

    except Exception as e:
        print(f"✗ Error updating wheel_cache: {e}")
        conn.rollback()
    finally:
        conn.close()


def remove_unified_trading():
    """Remove the unused data/wheel_trading_optimized.duckdb"""
    db_path = "./data/wheel_trading_optimized.duckdb"

    if os.path.exists(db_path):
        # Backup first
        backup_database(db_path)

        # Remove the file
        os.remove(db_path)
        print("✓ Removed unused data/wheel_trading_optimized.duckdb")
    else:
        print("✓ data/wheel_trading_optimized.duckdb already removed")


def print_database_stats():
    """Print statistics about the databases after cleanup"""
    print("\n" + "=" * 60)
    print("Database Statistics After Cleanup")
    print("=" * 60)

    databases = [
        "./data/wheel_trading_optimized.duckdb",
        "./data/cache/data/wheel_trading_optimized.duckdb",
    ]

    for db_path in databases:
        if os.path.exists(db_path):
            print(f"\n{db_path}:")
            print(f"  Size: {os.path.getsize(db_path) / 1024 / 1024:.2f} MB")

            try:
                conn = duckdb.connect(db_path, read_only=True)
                tables = conn.execute("SHOW TABLES").fetchall()
                print(f"  Tables: {len(tables)}")

                # Show table row counts
                for table in sorted(tables):
                    table_name = table[0]
                    try:
                        count = conn.execute(
                            f"SELECT COUNT(*) FROM {table_name}"
                        ).fetchone()[0]
                        if count > 0:
                            print(f"    - {table_name}: {count:,} rows")
                    except:
                        pass

                conn.close()
            except Exception as e:
                print(f"  Error reading database: {e}")


def main():
    """Main cleanup function"""
    print("Database Redundancy Cleanup Script")
    print("=" * 60)

    # Confirm before proceeding
    response = input("\nThis will modify your databases. Continue? (y/N): ")
    if response.lower() != "y":
        print("Cleanup cancelled.")
        return

    print("\nStarting cleanup...")

    # 1. Remove data/wheel_trading_optimized.duckdb
    remove_unified_trading()

    # 2. Clean up wheel_trading_master
    cleanup_wheel_trading_master()

    # 3. Clean up wheel_cache
    cleanup_wheel_cache()

    # 4. Print statistics
    print_database_stats()

    print("\n✓ Cleanup complete!")
    print("\nNext steps:")
    print("1. Test your application to ensure everything works")
    print("2. Update any code that references removed tables/views")
    print("3. Set up a schedule to refresh materialized views")


if __name__ == "__main__":
    main()
