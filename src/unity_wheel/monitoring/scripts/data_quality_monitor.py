#!/usr/bin/env python3
"""Real-time data quality monitoring dashboard."""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple

import duckdb

logger = logging.getLogger(__name__)

from src.config.loader import get_config

# Get Unity ticker once
_config = get_config()
UNITY_TICKER = _config.unity.ticker


# Colors for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def clear_screen():
    """Clear terminal screen."""
    import subprocess

    subprocess.run(["clear" if os.name == "posix" else "cls"], shell=False, check=False)


def get_status_icon(is_good: bool, is_warning: bool = False) -> str:
    """Get colored status icon."""
    if is_good:
        return f"{Colors.GREEN}✅{Colors.RESET}"
    elif is_warning:
        return f"{Colors.YELLOW}⚠️{Colors.RESET}"
    else:
        return f"{Colors.RED}❌{Colors.RESET}"


def check_data_freshness(conn) -> Dict[str, Dict]:
    """Check freshness of all data sources."""
    freshness = {}

    # Unity prices
    try:
        result = conn.execute(
            """
            SELECT MAX(date) as latest,
                   CURRENT_DATE - MAX(date) as days_old,
                   COUNT(*) as records
            FROM price_history WHERE symbol = ?
        """,
            (UNITY_TICKER,),
        ).fetchone()

        if result[0]:
            freshness["unity_prices"] = {
                "latest": result[0],
                "days_old": result[1],
                "records": result[2],
                "status": "good" if result[1] <= 1 else ("warning" if result[1] <= 7 else "error"),
            }
    except duckdb.Error as exc:
        logger.error("unity_price_query_failed", extra={"error": str(exc)})
        freshness["unity_prices"] = {"status": "error", "records": 0}

    # Options data
    try:
        # Check if table exists by trying to query it
        result = conn.execute(
            """
            SELECT MAX(timestamp) as latest,
                   COUNT(*) as records,
                   COUNT(DISTINCT DATE(timestamp)) as days_with_data
            FROM options_data WHERE underlying = ?
        """,
            (UNITY_TICKER,),
        ).fetchone()

        if result[0]:
            latest = datetime.fromisoformat(str(result[0]))
            hours_old = (datetime.now() - latest).total_seconds() / 3600
            freshness["options"] = {
                "latest": latest,
                "hours_old": hours_old,
                "records": result[1],
                "days_with_data": result[2],
                "status": (
                    "good" if hours_old <= 24 else ("warning" if hours_old <= 72 else "error")
                ),
            }
        else:
            freshness["options"] = {"status": "error", "records": 0}
    except duckdb.Error as exc:
        logger.error("options_query_failed", extra={"error": str(exc)})
        freshness["options"] = {"status": "error", "records": 0}

    # FRED data
    try:
        result = conn.execute(
            """
            SELECT MAX(observation_date) as latest,
                   CURRENT_DATE - MAX(observation_date) as days_old,
                   COUNT(DISTINCT series_id) as series_count
            FROM fred_observations
        """
        ).fetchone()

        if result[0]:
            freshness["fred"] = {
                "latest": result[0],
                "days_old": result[1],
                "series_count": result[2],
                "status": "good" if result[1] <= 7 else ("warning" if result[1] <= 30 else "error"),
            }
    except duckdb.Error as exc:
        logger.error("fred_query_failed", extra={"error": str(exc)})
        freshness["fred"] = {"status": "error", "series_count": 0}

    return freshness


def check_data_quality(conn) -> Dict[str, any]:
    """Check data quality metrics."""
    quality = {}

    # Check for data gaps in Unity prices
    try:
        gaps = conn.execute(
            """
            WITH date_series AS (
                SELECT date, LAG(date) OVER (ORDER BY date) as prev_date
                FROM price_history WHERE symbol = ?
            )
            SELECT COUNT(*) as total_gaps,
                   COUNT(CASE WHEN date - prev_date > 10 THEN 1 END) as large_gaps
            FROM date_series WHERE date - prev_date > 1
        """,
            (UNITY_TICKER,),
        ).fetchone()

        quality["price_gaps"] = {
            "total": gaps[0],
            "large": gaps[1],
            "status": "good" if gaps[1] == 0 else ("warning" if gaps[1] < 5 else "error"),
        }
    except duckdb.Error as exc:
        logger.error("price_gap_query_failed", extra={"error": str(exc)})
        quality["price_gaps"] = {"status": "error"}

    # Check Unity volatility
    try:
        vol = conn.execute(
            """
            SELECT STDDEV(returns) * SQRT(252) as annual_vol,
                   MAX(ABS(returns)) as max_move
            FROM price_history
            WHERE symbol = ?
                AND date >= CURRENT_DATE - 30
                AND returns IS NOT NULL
        """,
            (UNITY_TICKER,),
        ).fetchone()

        if vol[0]:
            quality["volatility"] = {
                "annual": vol[0] * 100,
                "max_daily_move": vol[1] * 100,
                "status": "good" if vol[0] < 1.0 else ("warning" if vol[0] < 1.5 else "error"),
            }
    except duckdb.Error as exc:
        logger.error("volatility_query_failed", extra={"error": str(exc)})
        quality["volatility"] = {"status": "error"}

    # Check options bid-ask spreads
    try:
        spreads = conn.execute(
            """
            SELECT AVG((ask - bid) / bid) as avg_spread,
                   MAX((ask - bid) / bid) as max_spread,
                   COUNT(CASE WHEN (ask - bid) / bid > 0.5 THEN 1 END) as wide_spreads
            FROM options_data
            WHERE underlying = ?
                AND bid > 0
                AND ask > 0
                AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
        """,
            (UNITY_TICKER,),
        ).fetchone()

        if spreads[0]:
            quality["spreads"] = {
                "average": spreads[0] * 100,
                "max": spreads[1] * 100,
                "wide_count": spreads[2],
                "status": (
                    "good" if spreads[0] < 0.1 else ("warning" if spreads[0] < 0.3 else "error")
                ),
            }
    except duckdb.Error as exc:
        logger.error("spread_query_failed", extra={"error": str(exc)})
        quality["spreads"] = {"status": "none"}

    return quality


def calculate_health_score(freshness: Dict, quality: Dict) -> Tuple[int, str]:
    """Calculate overall data health score."""
    score = 100

    # Freshness scoring
    for source, data in freshness.items():
        if data["status"] == "warning":
            score -= 10
        elif data["status"] == "error":
            score -= 20

    # Quality scoring
    for metric, data in quality.items():
        if data.get("status") == "warning":
            score -= 5
        elif data.get("status") == "error":
            score -= 10

    # Determine status
    if score >= 80:
        status = f"{Colors.GREEN}GOOD{Colors.RESET}"
    elif score >= 60:
        status = f"{Colors.YELLOW}FAIR{Colors.RESET}"
    else:
        status = f"{Colors.RED}POOR{Colors.RESET}"

    return max(score, 0), status


def display_dashboard(freshness: Dict, quality: Dict, refresh_count: int):
    """Display the monitoring dashboard."""
    clear_screen()

    # Header
    print(f"\n{Colors.BOLD}📊 DATA QUALITY MONITOR{Colors.RESET}")
    print("=" * 60)
    print(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Update #{refresh_count})")

    # Data Freshness Section
    print(f"\n{Colors.CYAN}⏰ DATA FRESHNESS{Colors.RESET}")
    print("-" * 40)

    # Unity Prices
    up = freshness.get("unity_prices", {})
    if up.get("records", 0) > 0:
        icon = get_status_icon(up["status"] == "good", up["status"] == "warning")
        print(f"{icon} Unity Prices: {up['latest']} ({up['days_old']} days old)")
        print(f"   Records: {up['records']:,}")
    else:
        print(f"{get_status_icon(False)} Unity Prices: No data")

    # Options Data
    opt = freshness.get("options", {})
    if opt.get("records", 0) > 0:
        icon = get_status_icon(opt["status"] == "good", opt["status"] == "warning")
        hours = opt["hours_old"]
        if hours < 24:
            age_str = f"{hours:.1f} hours"
        else:
            age_str = f"{hours/24:.1f} days"
        print(f"\n{icon} Options Data: {opt['latest'].strftime('%Y-%m-%d %H:%M')} ({age_str} old)")
        print(f"   Contracts: {opt['records']:,} | Days with data: {opt['days_with_data']}")
    else:
        print(f"\n{get_status_icon(False)} Options Data: No data")

    # FRED Data
    fred = freshness.get("fred", {})
    if fred.get("series_count", 0) > 0:
        icon = get_status_icon(fred["status"] == "good", fred["status"] == "warning")
        print(f"\n{icon} Economic Data: {fred['latest']} ({fred['days_old']} days old)")
        print(f"   Series tracked: {fred['series_count']}")
    else:
        print(f"\n{get_status_icon(False)} Economic Data: No data")

    # Data Quality Section
    print(f"\n{Colors.MAGENTA}🔍 DATA QUALITY{Colors.RESET}")
    print("-" * 40)

    # Price Gaps
    gaps = quality.get("price_gaps", {})
    if "total" in gaps:
        icon = get_status_icon(gaps["status"] == "good", gaps["status"] == "warning")
        print(f"{icon} Price Gaps: {gaps['total']} total, {gaps['large']} large (>10 days)")

    # Volatility
    vol = quality.get("volatility", {})
    if "annual" in vol:
        icon = get_status_icon(vol["status"] == "good", vol["status"] == "warning")
        print(
            f"{icon} Volatility: {vol['annual']:.1f}% annual, {vol['max_daily_move']:.1f}% max daily"
        )

    # Spreads
    spreads = quality.get("spreads", {})
    if "average" in spreads:
        icon = get_status_icon(spreads["status"] == "good", spreads["status"] == "warning")
        print(
            f"{icon} Option Spreads: {spreads['average']:.1f}% avg, {spreads['wide_count']} wide (>50%)"
        )

    # Overall Health Score
    score, status = calculate_health_score(freshness, quality)
    print(f"\n{Colors.BOLD}🏆 OVERALL HEALTH SCORE: {score}/100 - {status}{Colors.RESET}")

    # Recommendations
    print(f"\n{Colors.YELLOW}💡 RECOMMENDATIONS{Colors.RESET}")
    print("-" * 40)

    recommendations = []

    # Check freshness issues
    if freshness.get("unity_prices", {}).get("status") == "error":
        recommendations.append("🔴 Update Unity price data urgently")
    elif freshness.get("unity_prices", {}).get("status") == "warning":
        recommendations.append("🟡 Consider updating Unity price data")

    if freshness.get("options", {}).get("records", 0) == 0:
        recommendations.append("🔴 No options data - run: python tools/data/fetch_unity_options.py")
    elif freshness.get("options", {}).get("status") == "error":
        recommendations.append("🔴 Options data is stale - refresh needed")

    # Check quality issues
    if quality.get("volatility", {}).get("annual", 0) > 100:
        recommendations.append("🟡 High volatility detected - review risk limits")

    if quality.get("price_gaps", {}).get("large", 0) > 5:
        recommendations.append("🟡 Many data gaps - consider backfilling historical data")

    if not recommendations:
        recommendations.append("✅ All systems operational")

    for rec in recommendations:
        print(f"  {rec}")

    # Footer
    print(f"\n{Colors.CYAN}Press Ctrl+C to exit | Refreshing every 60 seconds...{Colors.RESET}")


def main():
    """Main monitoring loop."""
    db_path = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")

    if not os.path.exists(db_path):
        print(f"{Colors.RED}❌ Database not found at {db_path}{Colors.RESET}")
        print("Run data collection first: python tools/analysis/pull_unity_prices.py")
        sys.exit(1)

    refresh_count = 0

    try:
        while True:
            refresh_count += 1

            # Connect to database
            conn = duckdb.connect(db_path, read_only=True)

            try:
                # Check data freshness
                freshness = check_data_freshness(conn)

                # Check data quality
                quality = check_data_quality(conn)

                # Display dashboard
                display_dashboard(freshness, quality, refresh_count)

            finally:
                conn.close()

            # Wait before next refresh
            time.sleep(60)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}Monitoring stopped.{Colors.RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
