#!/usr/bin/env python3
"""Comprehensive Data Quality Assessment for Wheel Trading System."""

import json
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataQualityAssessment:
    def __init__(self, db_path="data/cache/wheel_cache.duckdb"):
        self.db_path = db_path
        self.issues = defaultdict(list)
        self.stats = {}

    def run_full_assessment(self):
        """Run complete data quality assessment."""
        print("=" * 80)
        print("COMPREHENSIVE DATA QUALITY ASSESSMENT")
        print("=" * 80)
        print(f"Assessment Date: {datetime.now()}")
        print(f"Database: {self.db_path}")
        print("=" * 80)

        # Connect to database
        conn = duckdb.connect(self.db_path)

        # 1. List all tables
        print("\n1. DATABASE STRUCTURE")
        print("-" * 40)
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Tables found: {len(tables)}")
        for table in tables:
            table_name = table[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  - {table_name}: {count:,} records")
            self.stats[table_name] = {"count": count}

        # 2. Check each table for data quality
        print("\n2. DATA QUALITY CHECKS BY TABLE")
        print("-" * 40)

        for table in tables:
            table_name = table[0]
            print(f"\n=== {table_name} ===")
            self._assess_table(conn, table_name)

        # 3. Check for synthetic/dummy data
        print("\n3. SYNTHETIC/DUMMY DATA DETECTION")
        print("-" * 40)
        self._check_for_synthetic_data(conn)

        # 4. Check data gaps
        print("\n4. DATA GAPS ANALYSIS")
        print("-" * 40)
        self._check_data_gaps(conn)

        # 5. Cross-table consistency
        print("\n5. CROSS-TABLE CONSISTENCY")
        print("-" * 40)
        self._check_consistency(conn)

        # 6. Summary
        print("\n6. SUMMARY OF ISSUES")
        print("-" * 40)
        self._print_summary()

        conn.close()

    def _assess_table(self, conn, table_name):
        """Assess quality of a specific table."""
        try:
            # Get column info
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            print(f"  Columns: {len(columns)}")

            # Get sample data
            sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").df()

            # Check for NULL values
            null_counts = conn.execute(
                f"""
                SELECT {', '.join([f'SUM(CASE WHEN {col[0]} IS NULL THEN 1 ELSE 0 END) as {col[0]}_nulls' for col in columns])}
                FROM {table_name}
            """
            ).fetchone()

            null_cols = []
            for i, col in enumerate(columns):
                if null_counts[i] > 0:
                    null_cols.append(f"{col[0]} ({null_counts[i]:,} nulls)")

            if null_cols:
                print(f"  NULL values found in: {', '.join(null_cols)}")
                self.issues[table_name].append(f"NULL values in columns: {', '.join(null_cols)}")

            # Check for duplicates (if there's an ID or unique column)
            if table_name == "unity_options_ohlcv":
                dupes = conn.execute(
                    """
                    SELECT symbol, ts_event, COUNT(*) as cnt
                    FROM unity_options_ohlcv
                    GROUP BY symbol, ts_event
                    HAVING COUNT(*) > 1
                """
                ).fetchall()
                if dupes:
                    print(f"  DUPLICATES: {len(dupes)} duplicate symbol/date combinations")
                    self.issues[table_name].append(f"{len(dupes)} duplicate entries")

            # Check date ranges
            date_cols = [
                col[0]
                for col in columns
                if "date" in col[0].lower() or "timestamp" in col[0].lower() or "ts_" in col[0]
            ]
            for date_col in date_cols:
                try:
                    date_range = conn.execute(
                        f"""
                        SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date
                        FROM {table_name}
                        WHERE {date_col} IS NOT NULL
                    """
                    ).fetchone()
                    if date_range[0] and date_range[1]:
                        print(f"  Date range ({date_col}): {date_range[0]} to {date_range[1]}")

                        # Check for future dates
                        future_count = conn.execute(
                            f"""
                            SELECT COUNT(*) FROM {table_name}
                            WHERE {date_col} > CURRENT_DATE
                        """
                        ).fetchone()[0]
                        if future_count > 0:
                            print(f"  WARNING: {future_count} records with future dates!")
                            self.issues[table_name].append(
                                f"{future_count} future dates in {date_col}"
                            )
                except:
                    pass

            # Check for data anomalies
            numeric_cols = [
                col[0]
                for col in columns
                if col[1] in ["DOUBLE", "FLOAT", "INTEGER", "BIGINT", "DECIMAL"]
            ]
            for col in numeric_cols:
                try:
                    stats = conn.execute(
                        f"""
                        SELECT
                            MIN({col}) as min_val,
                            MAX({col}) as max_val,
                            AVG({col}) as avg_val,
                            COUNT(DISTINCT {col}) as unique_vals
                        FROM {table_name}
                        WHERE {col} IS NOT NULL
                    """
                    ).fetchone()

                    # Check for suspicious values
                    if stats[0] is not None:
                        # Check for negative prices
                        if "price" in col.lower() or col in ["open", "high", "low", "close"]:
                            neg_count = conn.execute(
                                f"SELECT COUNT(*) FROM {table_name} WHERE {col} < 0"
                            ).fetchone()[0]
                            if neg_count > 0:
                                print(f"  ERROR: {neg_count} negative values in {col}!")
                                self.issues[table_name].append(
                                    f"{neg_count} negative prices in {col}"
                                )

                        # Check for zeros in price columns
                        if col in ["open", "high", "low", "close", "volume"]:
                            zero_count = conn.execute(
                                f"SELECT COUNT(*) FROM {table_name} WHERE {col} = 0"
                            ).fetchone()[0]
                            if zero_count > 0:
                                pct = zero_count / self.stats[table_name]["count"] * 100
                                if pct > 5:  # More than 5% zeros is suspicious
                                    print(
                                        f"  WARNING: {zero_count:,} ({pct:.1f}%) zero values in {col}"
                                    )
                                    self.issues[table_name].append(f"{pct:.1f}% zeros in {col}")
                except:
                    pass

        except Exception as e:
            print(f"  Error assessing table: {e}")
            self.issues[table_name].append(f"Assessment error: {str(e)}")

    def _check_for_synthetic_data(self, conn):
        """Check for synthetic or dummy data patterns."""

        # Check Unity options for synthetic patterns
        if "unity_options_ohlcv" in [t[0] for t in conn.execute("SHOW TABLES").fetchall()]:
            print("\nChecking Unity options for synthetic data...")

            # Check for perfect round numbers
            round_numbers = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM unity_options_ohlcv
                WHERE (close * 100) % 100 = 0  -- Exactly X.00
                  AND close > 0
            """
            ).fetchone()[0]

            total = self.stats.get("unity_options_ohlcv", {}).get("count", 1)
            round_pct = round_numbers / total * 100
            if round_pct > 20:  # More than 20% round numbers is suspicious
                print(f"  WARNING: {round_pct:.1f}% of prices are round numbers (X.00)")
                self.issues["synthetic"].append(f"Unity options: {round_pct:.1f}% round prices")

            # Check for repeated exact values
            repeated = conn.execute(
                """
                WITH price_counts AS (
                    SELECT close, COUNT(*) as cnt
                    FROM unity_options_ohlcv
                    WHERE close > 0
                    GROUP BY close
                    HAVING COUNT(*) > 100
                )
                SELECT COUNT(*) FROM price_counts
            """
            ).fetchone()[0]

            if repeated > 10:
                print(f"  WARNING: {repeated} prices appear more than 100 times")
                self.issues["synthetic"].append(
                    f"Unity options: {repeated} frequently repeated prices"
                )

            # Check for mathematical relationships (synthetic data often has patterns)
            patterns = conn.execute(
                """
                SELECT COUNT(*) as pattern_count
                FROM unity_options_ohlcv
                WHERE ABS(high - low - 0.01) < 0.001  -- High - Low = exactly 0.01
                   OR ABS(open - close) < 0.0001  -- Open = Close
                   OR (volume % 100 = 0 AND volume > 0)  -- Volume is multiple of 100
            """
            ).fetchone()[0]

            pattern_pct = patterns / total * 100
            if pattern_pct > 10:
                print(f"  WARNING: {pattern_pct:.1f}% of records show synthetic patterns")
                self.issues["synthetic"].append(
                    f"Unity options: {pattern_pct:.1f}% synthetic patterns"
                )

        # Check FRED data
        if "fred_data" in [t[0] for t in conn.execute("SHOW TABLES").fetchall()]:
            print("\nChecking FRED data...")
            fred_check = conn.execute(
                """
                SELECT series_id, COUNT(*) as cnt,
                       COUNT(DISTINCT value) as unique_vals,
                       MIN(date) as min_date,
                       MAX(date) as max_date
                FROM fred_data
                GROUP BY series_id
            """
            ).fetchall()

            for series in fred_check:
                if series[2] < 10:  # Less than 10 unique values
                    print(f"  WARNING: {series[0]} has only {series[2]} unique values")
                    self.issues["synthetic"].append(
                        f"FRED {series[0]}: only {series[2]} unique values"
                    )

    def _check_data_gaps(self, conn):
        """Check for gaps in time series data."""

        # Check Unity options for missing dates
        if "unity_options_ohlcv" in [t[0] for t in conn.execute("SHOW TABLES").fetchall()]:
            print("\nChecking Unity options for date gaps...")

            gaps = conn.execute(
                """
                WITH date_series AS (
                    SELECT DISTINCT DATE(ts_event) as trading_date
                    FROM unity_options_ohlcv
                    ORDER BY trading_date
                ),
                date_gaps AS (
                    SELECT
                        trading_date,
                        LAG(trading_date) OVER (ORDER BY trading_date) as prev_date,
                        trading_date - LAG(trading_date) OVER (ORDER BY trading_date) as gap_days
                    FROM date_series
                )
                SELECT * FROM date_gaps
                WHERE gap_days > 5  -- More than 5 days (weekend + holiday)
                ORDER BY gap_days DESC
            """
            ).fetchall()

            if gaps:
                print(f"  Found {len(gaps)} significant date gaps:")
                for gap in gaps[:5]:  # Show top 5
                    print(f"    - {gap[2]} days gap between {gap[0]} and {gap[1]}")
                self.issues["gaps"].append(f"Unity options: {len(gaps)} date gaps > 5 days")

            # Check for missing strikes
            print("\nChecking for missing strikes...")
            strike_coverage = conn.execute(
                """
                WITH recent_dates AS (
                    SELECT DISTINCT ts_event
                    FROM unity_options_ohlcv
                    WHERE ts_event >= CURRENT_DATE - INTERVAL 30 DAY
                ),
                strike_counts AS (
                    SELECT
                        ts_event,
                        COUNT(DISTINCT CAST(SUBSTRING(symbol, 14, 8) AS FLOAT) / 1000) as num_strikes
                    FROM unity_options_ohlcv
                    WHERE ts_event IN (SELECT ts_event FROM recent_dates)
                      AND SUBSTRING(symbol, 13, 1) = 'P'  -- Puts only
                    GROUP BY ts_event
                )
                SELECT
                    MIN(num_strikes) as min_strikes,
                    MAX(num_strikes) as max_strikes,
                    AVG(num_strikes) as avg_strikes
                FROM strike_counts
            """
            ).fetchone()

            if strike_coverage[0] and strike_coverage[1]:
                if strike_coverage[0] < strike_coverage[1] * 0.5:  # Min is less than 50% of max
                    print(
                        f"  WARNING: Strike coverage varies significantly: {strike_coverage[0]} to {strike_coverage[1]} strikes"
                    )
                    self.issues["gaps"].append(
                        f"Strike coverage varies: {strike_coverage[0]}-{strike_coverage[1]} strikes"
                    )

        # Check FRED data for gaps
        if "fred_data" in [t[0] for t in conn.execute("SHOW TABLES").fetchall()]:
            print("\nChecking FRED data for gaps...")
            fred_gaps = conn.execute(
                """
                SELECT
                    series_id,
                    MAX(date) as last_date,
                    CURRENT_DATE - MAX(date) as days_stale
                FROM fred_data
                GROUP BY series_id
                HAVING CURRENT_DATE - MAX(date) > 30
            """
            ).fetchall()

            if fred_gaps:
                for gap in fred_gaps:
                    print(f"  WARNING: {gap[0]} last updated {gap[2]} days ago (last: {gap[1]})")
                    self.issues["gaps"].append(f"FRED {gap[0]}: {gap[2]} days stale")

    def _check_consistency(self, conn):
        """Check cross-table consistency."""
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]

        # Check if we have complementary data
        if "unity_options_ohlcv" in tables:
            # Check date overlaps with other data sources
            unity_dates = conn.execute(
                """
                SELECT MIN(ts_event) as min_date, MAX(ts_event) as max_date
                FROM unity_options_ohlcv
            """
            ).fetchone()

            print(f"Unity options date range: {unity_dates[0]} to {unity_dates[1]}")

            # Check if we have price history for the underlying
            if "price_history" in tables:
                underlying_check = conn.execute(
                    """
                    SELECT COUNT(*) FROM price_history
                    WHERE symbol = 'U'
                """
                ).fetchone()[0]

                if underlying_check == 0:
                    print("  WARNING: No underlying Unity (U) price history found!")
                    self.issues["consistency"].append("No Unity stock price history")

        # Check for orphaned data
        if "positions" in tables and "option_chains" in tables:
            orphaned = conn.execute(
                """
                SELECT COUNT(DISTINCT p.symbol) as orphaned_positions
                FROM positions p
                LEFT JOIN option_chains oc ON p.symbol = oc.symbol
                WHERE oc.symbol IS NULL
                  AND p.symbol LIKE 'U%'  -- Unity options
            """
            ).fetchone()

            if orphaned and orphaned[0] > 0:
                print(f"  WARNING: {orphaned[0]} positions without corresponding option chain data")
                self.issues["consistency"].append(f"{orphaned[0]} orphaned positions")

    def _print_summary(self):
        """Print summary of all issues found."""
        total_issues = sum(len(issues) for issues in self.issues.values())

        if total_issues == 0:
            print("\n✅ NO SIGNIFICANT DATA QUALITY ISSUES FOUND!")
        else:
            print(f"\n⚠️  TOTAL ISSUES FOUND: {total_issues}")
            print("\nIssues by category:")

            for category, issues in self.issues.items():
                if issues:
                    print(f"\n{category.upper()}:")
                    for issue in issues:
                        print(f"  - {issue}")

        # Print recommendations
        print("\n7. RECOMMENDATIONS")
        print("-" * 40)

        if "synthetic" in self.issues and self.issues["synthetic"]:
            print("- Investigate potential synthetic data patterns in Unity options")

        if "gaps" in self.issues and self.issues["gaps"]:
            print("- Fill data gaps for complete historical analysis")
            print("- Update stale FRED data series")

        if "consistency" in self.issues and self.issues["consistency"]:
            print("- Ensure all options positions have corresponding market data")
            print("- Add Unity stock price history for complete analysis")

        if any("negative" in str(issue) for issues in self.issues.values() for issue in issues):
            print("- Fix negative price values in historical data")

        if any("zero" in str(issue) for issues in self.issues.values() for issue in issues):
            print("- Investigate and clean zero values in price/volume columns")


if __name__ == "__main__":
    assessment = DataQualityAssessment()
    assessment.run_full_assessment()
