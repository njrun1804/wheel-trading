#!/usr/bin/env python3
"""
Quick daily health check for the wheel trading database.

This script performs essential checks that should be run before trading each day.
"""

from datetime import datetime

import duckdb

from unity_wheel.config.unified_config import get_config
config = get_config()



def daily_health_check():
    """Perform essential daily health checks."""

    print("=" * 60)
    print("DAILY WHEEL TRADING DATABASE HEALTH CHECK")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Connect to database
    try:
        conn = duckdb.connect("data/wheel_trading_optimized.duckdb", read_only=True)
    except Exception as e:
        print(f"❌ CRITICAL: Cannot connect to database: {e}")
        return False

    health_score = 0
    total_checks = 0
    issues = []

    # Check 1: Data freshness
    total_checks += 1
    try:
        latest_data = conn.execute(
            """
            SELECT
                (SELECT MAX(date) FROM stock_prices) as latest_stock,
                (SELECT MAX(date) FROM options) as latest_options,
                (SELECT MAX(date) FROM economic_indicators) as latest_economic
        """
        ).fetchone()

        days_old = (datetime.now().date() - latest_data[0]).days
        if days_old <= 2:  # Allow weekend gap
            health_score += 1
            print(f"✅ Data Freshness: {days_old} days old")
        else:
            print(f"❌ Data Freshness: {days_old} days old (stale)")
            issues.append(f"Data is {days_old} days old")
    except Exception as e:
        print(f"❌ Data Freshness Check Failed: {e}")
        issues.append("Cannot check data freshness")

    # Check 2: Core option data integrity
    total_checks += 1
    try:
        option_stats = conn.execute(
            """
            SELECT
                COUNT(*) as total_options,
                COUNT(CASE WHEN delta IS NULL THEN 1 END) as missing_greeks,
                COUNT(CASE WHEN option_type = 'P' AND delta > 0 THEN 1 END) as wrong_put_delta,
                COUNT(CASE WHEN option_type = 'C' AND delta < 0 THEN 1 END) as wrong_call_delta,
                COUNT(CASE WHEN theta > 0 AND dte > 1 THEN 1 END) as positive_theta
            FROM options
        """
        ).fetchone()

        total_opts, missing_greeks, wrong_puts, wrong_calls, pos_theta = option_stats

        # Calculate issues percentage
        issues_pct = ((wrong_puts + wrong_calls) / total_opts * 100) if total_opts > 0 else 100

        if issues_pct < 1:  # Less than 1% issues
            health_score += 1
            print(f"✅ Option Greeks: {total_opts:,} options, {issues_pct:.2f}% issues")
        else:
            print(f"❌ Option Greeks: {issues_pct:.2f}% have incorrect delta signs")
            issues.append(f"{wrong_puts + wrong_calls} options with wrong delta signs")

        # Report positive theta (warning, not failure)
        pos_theta_pct = (pos_theta / total_opts * 100) if total_opts > 0 else 0
        if pos_theta_pct > 15:
            print(f"⚠️  Positive Theta: {pos_theta_pct:.1f}% of options ({pos_theta:,})")

    except Exception as e:
        print(f"❌ Option Data Check Failed: {e}")
        issues.append("Cannot verify option data integrity")

    # Check 3: Price data sanity
    total_checks += 1
    try:
        price_stats = conn.execute(
            """
            SELECT
                COUNT(*) as total_prices,
                COUNT(CASE WHEN close <= 0 THEN 1 END) as negative_prices,
                COUNT(CASE WHEN volume < 0 THEN 1 END) as negative_volume
            FROM stock_prices
        """
        ).fetchone()

        total_prices, neg_prices, neg_vol = price_stats

        if neg_prices == 0 and neg_vol == 0:
            health_score += 1
            print(f"✅ Price Data: {total_prices:,} records, no negative values")
        else:
            print(f"❌ Price Data: {neg_prices} negative prices, {neg_vol} negative volumes")
            issues.append(f"Price data has {neg_prices + neg_vol} invalid values")

    except Exception as e:
        print(f"❌ Price Data Check Failed: {e}")
        issues.append("Cannot verify price data")

    # Check 4: Current market data for Unity
    total_checks += 1
    try:
        unity_data = conn.execute(
            """
            SELECT close, volume, date
            FROM stock_prices
            WHERE symbol = config.trading.symbol
            ORDER BY date DESC
            LIMIT 1
        """
        ).fetchone()

        if unity_data:
            close, volume, date = unity_data
            days_old = (datetime.now().date() - date).days

            if days_old <= 2 and close > 0 and volume > 0:
                health_score += 1
                print(f"✅ Unity Data: ${close:.2f}, volume {volume:,}, {days_old} days old")
            else:
                print(f"❌ Unity Data: Stale or invalid (${close:.2f}, {days_old} days old)")
                issues.append("Unity data is stale or invalid")
        else:
            print("❌ Unity Data: No recent data found")
            issues.append("No Unity data available")

    except Exception as e:
        print(f"❌ Unity Data Check Failed: {e}")
        issues.append("Cannot verify Unity data")

    # Check 5: Option chain availability for Unity
    total_checks += 1
    try:
        chain_data = conn.execute(
            """
            SELECT
                COUNT(DISTINCT expiration) as expirations,
                COUNT(*) as total_options,
                MIN(dte) as min_dte,
                MAX(dte) as max_dte
            FROM options
            WHERE underlying = 'U'
                AND date = (SELECT MAX(date) FROM options WHERE underlying = 'U')
        """
        ).fetchone()

        if chain_data:
            expirations, total_opts, min_dte, max_dte = chain_data

            if expirations >= 3 and total_opts >= 50:  # At least 3 expirations, 50 options
                health_score += 1
                print(f"✅ Unity Options: {expirations} expirations, {total_opts} options")
            else:
                print(f"❌ Unity Options: Only {expirations} expirations, {total_opts} options")
                issues.append("Insufficient Unity option coverage")
        else:
            print("❌ Unity Options: No option data found")
            issues.append("No Unity option data available")

    except Exception as e:
        print(f"❌ Unity Options Check Failed: {e}")
        issues.append("Cannot verify Unity options")

    conn.close()

    # Calculate final health score
    final_score = (health_score / total_checks * 100) if total_checks > 0 else 0

    print("\n" + "=" * 60)
    print("HEALTH CHECK SUMMARY")
    print("=" * 60)
    print(f"Overall Health Score: {final_score:.1f}% ({health_score}/{total_checks} checks passed)")

    # Determine trading readiness
    if final_score >= 80 and len(issues) == 0:
        print("🟢 TRADING STATUS: READY")
        print("✅ Database is healthy and ready for trading")
        trading_ready = True
    elif final_score >= 60:
        print("🟡 TRADING STATUS: CAUTION")
        print("⚠️  Database has some issues but may be usable with extra care")
        trading_ready = False
    else:
        print("🔴 TRADING STATUS: NOT READY")
        print("❌ Database has critical issues - do not trade")
        trading_ready = False

    if issues:
        print(f"\n🚨 ISSUES FOUND ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    print("\n" + "=" * 60)

    return trading_ready


if __name__ == "__main__":
    trading_ready = daily_health_check()
    exit(0 if trading_ready else 1)
