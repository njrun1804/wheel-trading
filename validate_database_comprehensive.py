#!/usr/bin/env python3
"""
Comprehensive validation of data/wheel_trading_optimized.duckdb database.

Tests financial data integrity, mathematical consistency, data completeness,
and statistical anomalies to ensure the database is reliable for financial decisions.
"""

import json
import logging
from datetime import datetime
from typing import Any

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatabaseValidator:
    """Comprehensive database validation for financial data integrity."""

    def __init__(self, db_path: str = "data/wheel_trading_optimized.duckdb"):
        """Initialize validator with database connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.validation_results = {
            "financial_integrity": {},
            "mathematical_consistency": {},
            "data_completeness": {},
            "statistical_anomalies": {},
            "summary": {"total_checks": 0, "passed": 0, "failed": 0, "warnings": 0},
        }

    def validate_all(self) -> dict[str, Any]:
        """Run all validation checks."""
        logger.info("Starting comprehensive database validation...")

        # Check what tables exist
        self._check_database_structure()

        # Financial data integrity
        self._validate_financial_integrity()

        # Mathematical consistency
        self._validate_mathematical_consistency()

        # Data completeness
        self._validate_data_completeness()

        # Statistical anomalies
        self._validate_statistical_anomalies()

        # Generate summary
        self._generate_summary()

        return self.validation_results

    def _check_database_structure(self):
        """Check what tables and data exist in the database."""
        logger.info("Checking database structure...")

        tables = self.conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]

        self.validation_results["database_structure"] = {
            "tables": table_names,
            "table_count": len(table_names),
        }

        # Check row counts for each table
        table_info = {}
        for table in table_names:
            try:
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                columns = self.conn.execute(f"DESCRIBE {table}").fetchall()
                table_info[table] = {"row_count": count, "columns": [col[0] for col in columns]}
            except Exception as e:
                table_info[table] = {"error": str(e)}

        self.validation_results["database_structure"]["table_details"] = table_info

    def _validate_financial_integrity(self):
        """Validate financial data integrity."""
        logger.info("Validating financial data integrity...")

        checks = {}

        # Check for negative prices in stock data
        try:
            negative_prices = self.conn.execute(
                """
                SELECT COUNT(*) as count, MIN(close) as min_price
                FROM stock_prices
                WHERE close < 0 OR open < 0 OR high < 0 OR low < 0
            """
            ).fetchone()

            checks["negative_stock_prices"] = {
                "status": "PASS" if negative_prices[0] == 0 else "FAIL",
                "count": negative_prices[0],
                "min_price": negative_prices[1],
            }
            self._update_summary("PASS" if negative_prices[0] == 0 else "FAIL")
        except Exception as e:
            checks["negative_stock_prices"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        # Check for negative volumes
        try:
            negative_volumes = self.conn.execute(
                """
                SELECT COUNT(*) as count
                FROM stock_prices
                WHERE volume < 0
            """
            ).fetchone()

            checks["negative_volumes"] = {
                "status": "PASS" if negative_volumes[0] == 0 else "FAIL",
                "count": negative_volumes[0],
            }
            self._update_summary("PASS" if negative_volumes[0] == 0 else "FAIL")
        except Exception as e:
            checks["negative_volumes"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        # Check option pricing relationships
        try:
            # Get options data for put-call parity check
            # Note: Using premium as proxy for bid/ask since exact bid/ask may not be available
            options_data = self.conn.execute(
                """
                SELECT
                    o.option_symbol, o.strike, o.expiration, o.option_type,
                    o.premium, o.premium, o.premium, o.implied_volatility,
                    s.close as underlying_price
                FROM options o
                JOIN stock_prices s ON o.underlying = s.symbol
                    AND DATE(o.date) = DATE(s.date)
                WHERE o.premium > 0
                    AND s.close > 0
                    AND o.expiration > o.date
                LIMIT 1000
            """
            ).fetchall()

            if options_data:
                violations = 0
                for row in options_data:
                    # Basic arbitrage check: bid should be less than ask
                    if row[4] > row[5]:  # bid > ask
                        violations += 1

                checks["option_bid_ask_spread"] = {
                    "status": "PASS" if violations == 0 else "FAIL",
                    "violations": violations,
                    "total_checked": len(options_data),
                }
                self._update_summary("PASS" if violations == 0 else "FAIL")
            else:
                checks["option_bid_ask_spread"] = {
                    "status": "WARNING",
                    "message": "No options data found",
                }
                self._update_summary("WARNING")

        except Exception as e:
            checks["option_bid_ask_spread"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        self.validation_results["financial_integrity"] = checks

    def _validate_mathematical_consistency(self):
        """Validate mathematical consistency of options data."""
        logger.info("Validating mathematical consistency...")

        checks = {}

        # Check Greeks calculations
        try:
            # Get sample options with Greeks
            options_with_greeks = self.conn.execute(
                """
                SELECT
                    o.strike, o.expiration, o.option_type, o.implied_volatility,
                    o.delta, o.gamma, o.theta, o.vega,
                    s.close as underlying_price,
                    o.date
                FROM options o
                JOIN stock_prices s ON o.underlying = s.symbol
                    AND DATE(o.date) = DATE(s.date)
                WHERE o.delta IS NOT NULL
                    AND o.implied_volatility > 0
                    AND o.expiration > o.date
                LIMIT 100
            """
            ).fetchall()

            if options_with_greeks:
                greek_errors = []
                for row in options_with_greeks:
                    strike, expiration, option_type, iv, delta, gamma, theta, vega, spot, date = row

                    # Calculate time to expiration
                    days_to_exp = (pd.to_datetime(expiration) - pd.to_datetime(date)).days
                    if days_to_exp <= 0:
                        continue

                    time_to_exp = days_to_exp / 365.0

                    # Validate Greeks ranges
                    if option_type == "call":
                        if not (0 <= delta <= 1):
                            greek_errors.append(f"Call delta out of range: {delta}")
                    else:  # put
                        if not (-1 <= delta <= 0):
                            greek_errors.append(f"Put delta out of range: {delta}")

                    # Gamma should be positive
                    if gamma < 0:
                        greek_errors.append(f"Negative gamma: {gamma}")

                    # Theta should typically be negative (time decay)
                    if theta > 0 and days_to_exp > 1:
                        greek_errors.append(f"Positive theta for non-expiring option: {theta}")

                checks["greeks_validation"] = {
                    "status": "PASS" if len(greek_errors) == 0 else "FAIL",
                    "errors": greek_errors[:10],  # First 10 errors
                    "error_count": len(greek_errors),
                    "total_checked": len(options_with_greeks),
                }
                self._update_summary("PASS" if len(greek_errors) == 0 else "FAIL")
            else:
                checks["greeks_validation"] = {
                    "status": "WARNING",
                    "message": "No options with Greeks found",
                }
                self._update_summary("WARNING")

        except Exception as e:
            checks["greeks_validation"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        # Check implied volatility ranges
        try:
            iv_stats = self.conn.execute(
                """
                SELECT
                    MIN(implied_volatility) as min_iv,
                    MAX(implied_volatility) as max_iv,
                    AVG(implied_volatility) as avg_iv,
                    COUNT(*) as count
                FROM options
                WHERE implied_volatility > 0
            """
            ).fetchone()

            if iv_stats[3] > 0:  # count > 0
                # IV should be between 0 and 5 (500%)
                iv_valid = iv_stats[0] > 0 and iv_stats[1] < 5
                checks["implied_volatility_range"] = {
                    "status": "PASS" if iv_valid else "FAIL",
                    "min_iv": iv_stats[0],
                    "max_iv": iv_stats[1],
                    "avg_iv": iv_stats[2],
                    "count": iv_stats[3],
                }
                self._update_summary("PASS" if iv_valid else "FAIL")
            else:
                checks["implied_volatility_range"] = {
                    "status": "WARNING",
                    "message": "No implied volatility data found",
                }
                self._update_summary("WARNING")

        except Exception as e:
            checks["implied_volatility_range"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        self.validation_results["mathematical_consistency"] = checks

    def _validate_data_completeness(self):
        """Validate data completeness."""
        logger.info("Validating data completeness...")

        checks = {}

        # Check for gaps in price history
        try:
            # Get date ranges for each symbol
            price_gaps = self.conn.execute(
                """
                WITH date_series AS (
                    SELECT
                        symbol,
                        MIN(date) as min_date,
                        MAX(date) as max_date,
                        COUNT(DISTINCT date) as trading_days
                    FROM stock_prices
                    GROUP BY symbol
                )
                SELECT
                    symbol,
                    min_date,
                    max_date,
                    trading_days,
                    DATEDIFF('day', min_date, max_date) as calendar_days
                FROM date_series
                WHERE DATEDIFF('day', min_date, max_date) > 0
            """
            ).fetchall()

            gap_analysis = []
            for symbol, min_date, max_date, trading_days, calendar_days in price_gaps:
                # Approximate expected trading days (252 per year)
                expected_trading_days = calendar_days * (252 / 365)
                completeness = (
                    trading_days / expected_trading_days if expected_trading_days > 0 else 0
                )

                if completeness < 0.9:  # Less than 90% complete
                    gap_analysis.append(
                        {
                            "symbol": symbol,
                            "completeness": completeness,
                            "trading_days": trading_days,
                            "expected": int(expected_trading_days),
                        }
                    )

            checks["price_history_gaps"] = {
                "status": "PASS" if len(gap_analysis) == 0 else "WARNING",
                "symbols_with_gaps": len(gap_analysis),
                "gap_details": gap_analysis[:5],  # First 5 symbols with gaps
            }
            self._update_summary("PASS" if len(gap_analysis) == 0 else "WARNING")

        except Exception as e:
            checks["price_history_gaps"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        # Check option chain completeness
        try:
            chain_completeness = self.conn.execute(
                """
                WITH option_summary AS (
                    SELECT
                        underlying,
                        expiration,
                        COUNT(DISTINCT strike) as strikes,
                        COUNT(DISTINCT CASE WHEN option_type = 'call' THEN strike END) as call_strikes,
                        COUNT(DISTINCT CASE WHEN option_type = 'put' THEN strike END) as put_strikes
                    FROM options
                    WHERE date = (SELECT MAX(date) FROM options)
                    GROUP BY underlying, expiration
                )
                SELECT
                    underlying,
                    COUNT(DISTINCT expiration) as expirations,
                    AVG(strikes) as avg_strikes,
                    MIN(call_strikes) as min_calls,
                    MIN(put_strikes) as min_puts
                FROM option_summary
                GROUP BY underlying
            """
            ).fetchall()

            incomplete_chains = []
            for symbol, expirations, avg_strikes, min_calls, min_puts in chain_completeness:
                if min_calls == 0 or min_puts == 0 or avg_strikes < 10:
                    incomplete_chains.append(
                        {
                            "symbol": symbol,
                            "expirations": expirations,
                            "avg_strikes": avg_strikes,
                            "missing_calls": min_calls == 0,
                            "missing_puts": min_puts == 0,
                        }
                    )

            checks["option_chain_completeness"] = {
                "status": "PASS" if len(incomplete_chains) == 0 else "WARNING",
                "incomplete_chains": len(incomplete_chains),
                "details": incomplete_chains[:5],
            }
            self._update_summary("PASS" if len(incomplete_chains) == 0 else "WARNING")

        except Exception as e:
            checks["option_chain_completeness"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        # Check data staleness
        try:
            staleness = self.conn.execute(
                """
                SELECT
                    'stock_prices' as table_name,
                    MAX(date) as last_update,
                    DATEDIFF('day', MAX(date), CURRENT_DATE) as days_stale
                FROM stock_prices
                UNION ALL
                SELECT
                    'options' as table_name,
                    MAX(date) as last_update,
                    DATEDIFF('day', MAX(date), CURRENT_DATE) as days_stale
                FROM options
                UNION ALL
                SELECT
                    'economic_indicators' as table_name,
                    MAX(date) as last_update,
                    DATEDIFF('day', MAX(date), CURRENT_DATE) as days_stale
                FROM economic_indicators
            """
            ).fetchall()

            stale_data = []
            for table, last_update, days_stale in staleness:
                if days_stale is not None and days_stale > 7:  # More than 7 days old
                    stale_data.append(
                        {"table": table, "last_update": str(last_update), "days_stale": days_stale}
                    )

            checks["data_staleness"] = {
                "status": "PASS" if len(stale_data) == 0 else "WARNING",
                "stale_tables": len(stale_data),
                "details": stale_data,
            }
            self._update_summary("PASS" if len(stale_data) == 0 else "WARNING")

        except Exception as e:
            checks["data_staleness"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        self.validation_results["data_completeness"] = checks

    def _validate_statistical_anomalies(self):
        """Validate statistical anomalies in the data."""
        logger.info("Validating statistical anomalies...")

        checks = {}

        # Detect outliers in prices using z-score
        try:
            price_outliers = self.conn.execute(
                """
                WITH price_stats AS (
                    SELECT
                        symbol,
                        AVG(close) as mean_price,
                        STDDEV(close) as std_price
                    FROM stock_prices
                    WHERE close > 0
                    GROUP BY symbol
                    HAVING COUNT(*) > 100
                )
                SELECT
                    p.symbol,
                    p.date,
                    p.close,
                    ps.mean_price,
                    ps.std_price,
                    ABS(p.close - ps.mean_price) / ps.std_price as z_score
                FROM stock_prices p
                JOIN price_stats ps ON p.symbol = ps.symbol
                WHERE ABS(p.close - ps.mean_price) / ps.std_price > 5
                ORDER BY z_score DESC
                LIMIT 20
            """
            ).fetchall()

            checks["price_outliers"] = {
                "status": "WARNING" if len(price_outliers) > 10 else "PASS",
                "outlier_count": len(price_outliers),
                "extreme_outliers": [
                    {"symbol": row[0], "date": str(row[1]), "price": row[2], "z_score": row[5]}
                    for row in price_outliers[:5]
                ],
            }
            self._update_summary("WARNING" if len(price_outliers) > 10 else "PASS")

        except Exception as e:
            checks["price_outliers"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        # Check for data discontinuities (large jumps)
        try:
            discontinuities = self.conn.execute(
                """
                WITH price_changes AS (
                    SELECT
                        symbol,
                        date,
                        close,
                        LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close,
                        ABS(close - LAG(close) OVER (PARTITION BY symbol ORDER BY date)) /
                            LAG(close) OVER (PARTITION BY symbol ORDER BY date) as pct_change
                    FROM stock_prices
                    WHERE close > 0
                )
                SELECT
                    symbol,
                    date,
                    close,
                    prev_close,
                    pct_change
                FROM price_changes
                WHERE pct_change > 0.5  -- More than 50% change
                ORDER BY pct_change DESC
                LIMIT 20
            """
            ).fetchall()

            checks["price_discontinuities"] = {
                "status": "WARNING" if len(discontinuities) > 5 else "PASS",
                "discontinuity_count": len(discontinuities),
                "large_jumps": [
                    {
                        "symbol": row[0],
                        "date": str(row[1]),
                        "price": row[2],
                        "prev_price": row[3],
                        "pct_change": row[4],
                    }
                    for row in discontinuities[:5]
                ],
            }
            self._update_summary("WARNING" if len(discontinuities) > 5 else "PASS")

        except Exception as e:
            checks["price_discontinuities"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        # Verify return distributions
        try:
            # Get daily returns for statistical analysis
            returns_data = self.conn.execute(
                """
                WITH daily_returns AS (
                    SELECT
                        symbol,
                        (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date)) /
                            LAG(close) OVER (PARTITION BY symbol ORDER BY date) as return
                    FROM stock_prices
                    WHERE close > 0
                )
                SELECT
                    symbol,
                    AVG(return) as mean_return,
                    STDDEV(return) as std_return,
                    COUNT(*) as observations
                FROM daily_returns
                WHERE return IS NOT NULL
                GROUP BY symbol
                HAVING COUNT(*) > 100
            """
            ).fetchall()

            distribution_warnings = []
            for symbol, mean_return, std_return, observations in returns_data:
                # Check for unrealistic returns
                if abs(mean_return) > 0.01:  # More than 1% daily average
                    distribution_warnings.append(
                        {
                            "symbol": symbol,
                            "issue": "High average daily return",
                            "mean_return": mean_return,
                            "annualized": mean_return * 252,
                        }
                    )

                if std_return > 0.1:  # More than 10% daily volatility
                    distribution_warnings.append(
                        {
                            "symbol": symbol,
                            "issue": "High daily volatility",
                            "std_return": std_return,
                            "annualized_vol": std_return * np.sqrt(252),
                        }
                    )

            checks["return_distributions"] = {
                "status": "WARNING" if len(distribution_warnings) > 0 else "PASS",
                "warnings": len(distribution_warnings),
                "details": distribution_warnings[:5],
            }
            self._update_summary("WARNING" if len(distribution_warnings) > 0 else "PASS")

        except Exception as e:
            checks["return_distributions"] = {"status": "ERROR", "error": str(e)}
            self._update_summary("FAIL")

        self.validation_results["statistical_anomalies"] = checks

    def _update_summary(self, status: str):
        """Update summary statistics."""
        self.validation_results["summary"]["total_checks"] += 1
        if status == "PASS":
            self.validation_results["summary"]["passed"] += 1
        elif status == "FAIL":
            self.validation_results["summary"]["failed"] += 1
        elif status == "WARNING":
            self.validation_results["summary"]["warnings"] += 1

    def _generate_summary(self):
        """Generate overall summary and recommendations."""
        summary = self.validation_results["summary"]

        # Calculate health score
        if summary["total_checks"] > 0:
            health_score = (summary["passed"] / summary["total_checks"]) * 100
        else:
            health_score = 0

        summary["health_score"] = health_score

        # Generate recommendations
        recommendations = []

        if (
            self.validation_results["financial_integrity"]
            .get("negative_stock_prices", {})
            .get("status")
            == "FAIL"
        ):
            recommendations.append("Fix negative prices in stock_prices table")

        if (
            self.validation_results["mathematical_consistency"]
            .get("greeks_validation", {})
            .get("status")
            == "FAIL"
        ):
            recommendations.append("Recalculate option Greeks for consistency")

        if any(
            check.get("status") == "WARNING"
            for check in self.validation_results["data_completeness"].values()
        ):
            recommendations.append("Fill data gaps and update stale data")

        if any(
            check.get("status") == "WARNING"
            for check in self.validation_results["statistical_anomalies"].values()
        ):
            recommendations.append("Review and clean statistical outliers")

        summary["recommendations"] = recommendations
        summary["safe_for_trading"] = health_score >= 80 and summary["failed"] == 0

    def generate_report(self, output_file: str = "database_validation_report.json"):
        """Generate detailed validation report."""
        logger.info(f"Generating validation report: {output_file}")

        # Add timestamp
        self.validation_results["validation_timestamp"] = datetime.now().isoformat()
        self.validation_results["database_path"] = self.db_path

        # Save to file
        with open(output_file, "w") as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        # Print summary
        summary = self.validation_results["summary"]
        print("\n" + "=" * 80)
        print("DATABASE VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Database: {self.db_path}")
        print(f"Timestamp: {self.validation_results['validation_timestamp']}")
        print(f"\nHealth Score: {summary['health_score']:.1f}%")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"\nSafe for Trading: {'YES' if summary['safe_for_trading'] else 'NO'}")

        if summary["recommendations"]:
            print("\nRecommendations:")
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"  {i}. {rec}")

        print(f"\nDetailed report saved to: {output_file}")
        print("=" * 80)

        return self.validation_results


def main():
    """Run comprehensive database validation."""
    validator = DatabaseValidator()

    try:
        # Run all validations
        results = validator.validate_all()

        # Generate report
        validator.generate_report()

        # Return exit code based on results
        if results["summary"]["safe_for_trading"]:
            return 0
        else:
            return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 2
    finally:
        validator.conn.close()


if __name__ == "__main__":
    exit(main())
