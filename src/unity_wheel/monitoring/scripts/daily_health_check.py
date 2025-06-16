#!/usr/bin/env python3
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


Daily health check for Unity Wheel Trading System.
Run this each morning to verify everything is working properly.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.loader import get_config
from unity_wheel.analytics import IntegratedDecisionEngine
from unity_wheel.analytics.performance_tracker import PerformanceTracker
from unity_wheel.secrets import SecretManager
from unity_wheel.storage import UnifiedStorage
from unity_wheel.utils import get_logger

logger = get_logger(__name__)


async def check_data_freshness(storage: UnifiedStorage) -> dict:
    """Check if we have recent data."""
    results = {"status": "✅", "issues": []}

    # Check Unity price data
    config = get_config()
    latest = await storage.get_latest_price(config.unity.ticker)
    if latest:
        days_old = (datetime.now() - latest["date"]).days
        if days_old > 1:
            results["status"] = "⚠️"
            results["issues"].append(f"Price data is {days_old} days old")
    else:
        results["status"] = "❌"
        results["issues"].append("No price data found")

    # Check if we have enough historical data
    historical = await storage.get_price_history(config.unity.ticker, days=750)
    if len(historical) < 500:
        results["status"] = "⚠️"
        results["issues"].append(f"Only {len(historical)} days of history (need 750)")

    return results


def check_configuration() -> dict:
    """Check configuration health."""
    from src.config.loader import get_config_loader

    results = {"status": "✅", "issues": []}

    try:
        loader = get_config_loader()
        config = loader.config

        # Check critical parameters
        if config.strategy.delta_target < 0.1 or config.strategy.delta_target > 0.5:
            results["issues"].append(
                f"Delta target {config.strategy.delta_target} seems unusual"
            )

        if config.risk.kelly_fraction > 0.5:
            results["issues"].append(
                f"Kelly fraction {config.risk.kelly_fraction} is aggressive"
            )

        # Get health report
        health = loader.generate_health_report()
        warnings = [line for line in health if "WARNING" in line]
        if warnings:
            results["status"] = "⚠️"
            results["issues"].extend(warnings)

    except (ValueError, KeyError, AttributeError) as e:
        results["status"] = "❌"
        results["issues"].append(f"Configuration error: {e}")

    return results


def check_credentials() -> dict:
    """Check API credentials."""
    results = {"status": "✅", "issues": []}

    try:
        secrets = SecretManager()

        # Check Databento
        if not secrets.get_secret("DATABENTO_API_KEY"):
            results["status"] = "⚠️"
            results["issues"].append("Databento API key not configured")

        # Check Schwab (optional)
        if not secrets.get_secret("SCHWAB_CLIENT_ID"):
            results["issues"].append("Schwab credentials not configured (optional)")

    except (ValueError, KeyError, AttributeError) as e:
        results["status"] = "❌"
        results["issues"].append(f"Credential error: {e}")

    return results


async def test_decision_engine() -> dict:
    """Test if decision engine can make recommendations."""
    results = {"status": "✅", "issues": []}

    try:
        config = get_config()
        IntegratedDecisionEngine(config.unity.ticker, 100000)

        # Try to fetch real Unity data from Databento
        try:
            from unity_wheel.cli.databento_integration import get_market_data_sync

            # Get real market data
            market_data, confidence = get_market_data_sync(100000, config.unity.ticker)
            logger.info(
                f"Decision engine check using real {config.unity.ticker} data",
                extra={"confidence": confidence},
            )
            results["issues"].append(
                "Decision engine initialized with real market data"
            )

        except (ValueError, KeyError, AttributeError) as e:
            # FAIL if real data is not available - no fallbacks allowed
            results["status"] = "❌"
            results["issues"].append(f"CRITICAL: Cannot get real market data: {e}")
            logger.error(f"Decision engine check failed - no real data available: {e}")
            return results

    except (ValueError, KeyError, AttributeError) as e:
        results["status"] = "❌"
        results["issues"].append(f"Decision engine error: {e}")

    return results


def check_performance() -> dict:
    """Check recent trading performance."""
    results = {"status": "✅", "issues": []}

    try:
        tracker = PerformanceTracker()
        stats = tracker.get_performance_stats(30)

        if isinstance(stats, dict) and "win_rate" in stats:
            if stats["win_rate"] < 0.5:
                results["status"] = "⚠️"
                results["issues"].append(f"Low win rate: {stats['win_rate']:.1%}")

            suggestions = tracker.suggest_improvements()
            if suggestions:
                results["issues"].extend(suggestions[:2])  # Top 2 suggestions

    except (ValueError, KeyError, AttributeError):
        # Performance tracking is optional
        results["issues"].append("No performance data yet")

    return results


async def main():
    """Run all health checks."""
    logger.info("🏥 Unity Wheel Trading System - Daily Health Check")
    print("=" * 60)
    logger.info("Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize storage
    storage = UnifiedStorage()

    # Run all checks
    checks = {
        "📊 Data Freshness": await check_data_freshness(storage),
        "⚙️  Configuration": check_configuration(),
        "🔑 Credentials": check_credentials(),
        "🧠 Decision Engine": await test_decision_engine(),
        "📈 Performance": check_performance(),
    }

    # Display results
    all_good = True
    for _name, result in checks.items():
        logger.info("{name}: {result['status']}")
        for _issue in result["issues"]:
            logger.info("   → {issue}")
            if result["status"] != "✅":
                all_good = False
        print()

    # Overall status
    print("=" * 60)
    if all_good:
        logger.info("✅ System Status: HEALTHY")
        logger.info("   All systems operational. Ready for trading.")
    else:
        logger.info("⚠️  System Status: NEEDS ATTENTION")
        logger.info("   Review issues above before trading.")

    # Quick recommendations
    logger.info("\n💡 Quick Actions:")
    if any("data" in str(r["issues"]).lower() for r in checks.values()):
        logger.info("   1. Run: python pull_databento_integrated.py")
    if any("config" in str(r["issues"]).lower() for r in checks.values()):
        logger.info("   2. Review: config.yaml")
    if any("credential" in str(r["issues"]).lower() for r in checks.values()):
        logger.info("   3. Check: .env file")

    logger.info("\n📋 Next Steps:")
    logger.info("   1. Fix any issues above")
    logger.info("   2. Run: python run.py --portfolio 100000")
    logger.info("   3. Review recommendation carefully")
    print()


if __name__ == "__main__":
    asyncio.run(main())
