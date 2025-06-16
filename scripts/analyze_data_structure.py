#!/usr/bin/env python3
"""
Comprehensive analysis of database structure and data quality
"""

import os
from datetime import datetime

import duckdb
import pandas as pd


def analyze_database_structure():
    """Analyze all tables and their structures."""
    conn = duckdb.connect("data/unified_wheel_trading.duckdb")

    print("=" * 80)
    print("DATABASE STRUCTURE ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Get all tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"\nTotal tables found: {len(tables)}")
    print("-" * 80)

    # Analyze each table
    for table in sorted(tables):
        table_name = table[0]
        print(f"\n📊 TABLE: {table_name}")
        print("-" * 40)

        try:
            # Get row count
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"Rows: {count:,}")

            # Get schema
            schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
            print("\nSchema:")
            for col in schema:
                print(f"  - {col[0]}: {col[1]}")

            # For non-empty tables, get sample data info
            if count > 0:
                # Check for date columns and get range
                date_cols = [
                    col[0]
                    for col in schema
                    if "date" in col[0].lower() or "time" in col[0].lower()
                ]

                for date_col in date_cols:
                    try:
                        date_range = conn.execute(
                            f"""
                            SELECT
                                MIN({date_col}) as min_date,
                                MAX({date_col}) as max_date
                            FROM {table_name}
                            WHERE {date_col} IS NOT NULL
                        """
                        ).fetchone()

                        if date_range[0] and date_range[1]:
                            print(f"\nDate range ({date_col}):")
                            print(f"  From: {date_range[0]}")
                            print(f"  To:   {date_range[1]}")
                    except:
                        pass

                # For options tables, check strikes and expirations
                if "option" in table_name.lower():
                    try:
                        # Check for strike column
                        if any("strike" in col[0].lower() for col in schema):
                            strike_col = next(
                                col[0] for col in schema if "strike" in col[0].lower()
                            )
                            strike_stats = conn.execute(
                                f"""
                                SELECT
                                    COUNT(DISTINCT {strike_col}) as unique_strikes,
                                    MIN({strike_col}) as min_strike,
                                    MAX({strike_col}) as max_strike
                                FROM {table_name}
                                WHERE {strike_col} IS NOT NULL
                            """
                            ).fetchone()

                            print("\nStrikes:")
                            print(f"  Unique: {strike_stats[0]}")
                            print(f"  Range: ${strike_stats[1]} - ${strike_stats[2]}")

                        # Check for expiration column
                        if any("expir" in col[0].lower() for col in schema):
                            exp_col = next(
                                col[0] for col in schema if "expir" in col[0].lower()
                            )
                            exp_count = conn.execute(
                                f"""
                                SELECT COUNT(DISTINCT {exp_col}) as unique_expirations
                                FROM {table_name}
                                WHERE {exp_col} IS NOT NULL
                            """
                            ).fetchone()[0]

                            print(f"  Unique expirations: {exp_count}")
                    except Exception as e:
                        print(f"  Error analyzing options data: {e}")

                # Show sample records
                print("\nSample record:")
                sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchone()
                for i, col in enumerate(schema):
                    if sample[i] is not None:
                        print(f"  {col[0]}: {sample[i]}")

        except Exception as e:
            print(f"Error analyzing table: {e}")

    # Analyze relationships and redundancy
    print("\n" + "=" * 80)
    print("DATA STRUCTURE ISSUES & RECOMMENDATIONS")
    print("=" * 80)

    # Check for Unity-related tables
    unity_tables = [
        t[0] for t in tables if "unity" in t[0].lower() or t[0].upper() == "U"
    ]
    print(f"\nUnity-related tables found: {len(unity_tables)}")
    for table in unity_tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  - {table}: {count:,} rows")

    # Check for options tables
    options_tables = [t[0] for t in tables if "option" in t[0].lower()]
    print(f"\nOptions tables found: {len(options_tables)}")
    for table in options_tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  - {table}: {count:,} rows")

    # Check for price/stock tables
    price_tables = [
        t[0] for t in tables if "price" in t[0].lower() or "stock" in t[0].lower()
    ]
    print(f"\nPrice/stock tables found: {len(price_tables)}")
    for table in price_tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  - {table}: {count:,} rows")

    conn.close()

    # Check parquet files
    print("\n" + "=" * 80)
    print("PARQUET FILES IN DATA DIRECTORY")
    print("=" * 80)

    if os.path.exists("data"):
        parquet_files = [f for f in os.listdir("data") if f.endswith(".parquet")]
        print(f"\nParquet files found: {len(parquet_files)}")
        for file in sorted(parquet_files):
            size_mb = os.path.getsize(f"data/{file}") / 1e6
            print(f"  - {file}: {size_mb:.1f} MB")

            # Try to read and analyze
            try:
                df = pd.read_parquet(f"data/{file}")
                print(f"    Rows: {len(df):,}")
                print(f"    Columns: {', '.join(df.columns[:5])}...")

                # Check date range if available
                date_cols = [
                    col
                    for col in df.columns
                    if "date" in col.lower() or "time" in col.lower()
                ]
                if date_cols:
                    date_col = date_cols[0]
                    print(
                        f"    Date range: {df[date_col].min()} to {df[date_col].max()}"
                    )
            except Exception as e:
                print(f"    Error reading: {e}")


if __name__ == "__main__":
    analyze_database_structure()
