#!/usr/bin/env python3
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


Live monitoring script for Unity wheel strategy.
Displays real-time status and alerts for anomalies.
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ..config.loader import get_config, get_config_loader
from ....analytics import IntegratedDecisionEngine
from ....analytics.performance_tracker import PerformanceTracker
from ....risk.limits import RiskLimitChecker, TradingLimits
from ....storage import UnifiedStorage
from ....utils import get_logger

logger = get_logger(__name__)

# Global flag for graceful shutdown
running = True


async def get_realized_volatility_from_databento(ticker: str) -> float:
    """Fetch realized volatility from Databento or calculate from historical data."""
    try:
        from unity_wheel.data_providers.databento import DatabentoClient
        from unity_wheel.data_providers.databento.price_history_loader import (
            OptimizedPriceHistoryLoader,
        )

        client = DatabentoClient()
        loader = OptimizedPriceHistoryLoader(client)

        # Get 30 days of historical data for volatility calculation
        historical_data = await loader.load_price_history(ticker, days=30)

        if historical_data and len(historical_data) > 1:
            # Calculate realized volatility from returns
            import numpy as np

            closes = [float(d["close"]) for d in historical_data]
            returns = np.diff(np.log(closes))

            # Annualized volatility
            daily_vol = np.std(returns)
            annual_vol = daily_vol * np.sqrt(252)  # 252 trading days

            logger.info(f"Calculated {ticker} realized volatility: {annual_vol:.2f}")
            return annual_vol
        else:
            raise ValueError(
                f"CRITICAL: Insufficient data for {ticker} volatility calculation - cannot proceed without real data"
            )

    except (ValueError, KeyError, AttributeError) as e:
        logger.error(f"CRITICAL: Failed to fetch volatility from Databento: {e}")
        raise ValueError(f"Cannot calculate volatility without real market data: {e}")


def signal_handler(signum, frame) -> None:
    """Handle shutdown signals gracefully."""
    global running
    running = False
    logger.info("\n\n🛑 Shutting down monitor...")


async def display_dashboard(
    storage: UnifiedStorage,
    engine: IntegratedDecisionEngine,
    risk_checker: RiskLimitChecker,
    tracker: PerformanceTracker,
):
    """Display live monitoring dashboard."""
    # Clear screen (works on most terminals)
    import subprocess

    subprocess.run(["clear" if os.name == "posix" else "cls"], shell=False, check=False)

    logger.info("🎯 Unity Wheel Strategy Monitor")
    print("=" * 60)
    logger.info("Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Current market data
    config = get_config()
    latest = await storage.get_latest_price(config.unity.ticker)
    if latest:
        logger.info("📊 Market Data:")
        logger.info("   Price: ${latest.get('close', 0):.2f}")
        logger.info("   Volume: {latest.get('volume', 0):,.0f}")
        logger.info("   Change: {latest.get('returns', 0)*100:+.2f}%")

        # Calculate realized vol
        history = await storage.get_price_history(config.unity.ticker, days=20)
        if len(history) >= 20:
            import numpy as np

            returns = [h["returns"] for h in history]
            vol = np.std(returns) * np.sqrt(252)
            logger.info("   Volatility: {vol:.1%}")
    else:
        logger.info("❌ No market data available")
    print()

    # 2. Risk limits status
    restrictions = risk_checker.get_current_restrictions()
    logger.info("🛡️  Risk Status:")
    if restrictions["can_trade"]:
        logger.info("   ✅ Trading allowed")
    else:
        logger.info("   🚫 Trading blocked")
        for reason in restrictions["reasons"]:
            logger.info("      - {reason}")

    if restrictions["reduced_size"]:
        logger.info("   ⚠️  Size reduced to {restrictions['max_position_size']:.1%}")
    print()

    # 3. Recent performance
    stats = tracker.get_performance_stats(30)
    if isinstance(stats, dict) and "total_trades" in stats:
        logger.info("📈 Performance (30 days):")
        logger.info("   Trades: {stats['total_trades']}")
        logger.info("   Win Rate: {stats.get('win_rate', 0):.1%}")
        logger.info("   Avg Return: {stats.get('avg_return', 0):.2%}")
    else:
        logger.info("📈 Performance: No recent trades")
    print()

    # 4. System health
    config_loader = get_config_loader()
    unused = config_loader.get_unused_parameters()
    logger.info("⚙️  System Health:")
    logger.info("   Config parameters: {len(config_loader.config.__dict__)} loaded")
    logger.info("   Unused parameters: {len(unused)}")

    # Check for recent errors in logs
    print()

    # 5. Active alerts
    logger.info("🚨 Active Alerts:")
    alerts = []

    # Check volatility
    if latest and "vol" in locals() and vol > 1.0:
        alerts.append(f"High volatility: {vol:.1%}")

    # Check for gaps
    if latest and "open" in latest and "prev_close" in latest:
        gap = abs(latest["open"] - latest["prev_close"]) / latest["prev_close"]
        if gap > 0.05:
            alerts.append(f"Price gap: {gap:.1%}")

    if alerts:
        for alert in alerts:
            logger.info("   ⚠️  {alert}")
    else:
        logger.info("   ✅ No active alerts")
    print()

    print("-" * 60)
    logger.info("Press Ctrl+C to stop monitoring")


async def check_for_opportunities(
    engine: IntegratedDecisionEngine,
    storage: UnifiedStorage,
    risk_checker: RiskLimitChecker,
    tracker: PerformanceTracker,
):
    """Check for trading opportunities."""
    try:
        # Get current data
        config = get_config()
        latest = await storage.get_latest_price(config.unity.ticker)
        if not latest:
            return

        historical = await storage.get_price_history(config.unity.ticker, days=750)
        if len(historical) < 500:
            return

        # Prepare market data
        current_prices = {
            "close": float(latest["close"]),
            "open": float(latest.get("open", latest["close"])),
            "prev_close": float(
                historical[-2]["close"] if len(historical) > 1 else latest["close"]
            ),
            "volume": float(latest.get("volume", 0)),
            "realized_vol": await get_realized_volatility_from_databento(config.unity.ticker),
        }

        # Mock option chain for monitoring
        # In production, this would fetch real options data
        option_chain = None

        # Get recommendation
        recommendation = await engine.get_recommendation(
            current_prices=current_prices,
            historical_data=None,  # Engine will handle
            option_chain=option_chain,
            current_positions=None,
            event_calendar=[],
        )

        # Check risk limits
        breaches = risk_checker.check_all_limits(
            recommendation._asdict(), portfolio_value = config.trading.portfolio_value, market_data=current_prices  # Default
        )

        # Log if interesting opportunity
        if recommendation.action != "NO_TRADE" and recommendation.confidence > 0.7:
            if risk_checker.should_allow_trade(breaches):
                logger.info(
                    "Trading opportunity detected",
                    action=recommendation.action,
                    confidence=recommendation.confidence,
                    objective=recommendation.objective_value,
                )

                # Record for tracking (but don't execute)
                tracker.record_recommendation(recommendation._asdict())
            else:
                logger.warning(
                    "Opportunity blocked by risk limits",
                    action=recommendation.action,
                    breaches=[b.name for b in breaches],
                )

    except (ValueError, KeyError, AttributeError) as e:
        logger.error(f"Error checking opportunities: {e}")


async def main():
    """Main monitoring loop."""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("🚀 Starting Unity Wheel Strategy Monitor...")
    logger.info("Loading components...")

    # Initialize components
    storage = UnifiedStorage()
    config = get_config()
    engine = IntegratedDecisionEngine(config.unity.ticker, 100000)
    risk_checker = RiskLimitChecker()
    tracker = PerformanceTracker()

    # Main loop
    check_counter = 0
    while running:
        try:
            # Display dashboard
            await display_dashboard(storage, engine, risk_checker, tracker)

            # Check for opportunities every 5 updates
            check_counter += 1
            if check_counter % 5 == 0:
                await check_for_opportunities(engine, storage, risk_checker, tracker)

            # Wait before next update
            await asyncio.sleep(10)  # Update every 10 seconds

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Monitor error: {e}")
            await asyncio.sleep(30)  # Wait longer on error

    logger.info("\n✅ Monitor stopped gracefully")


if __name__ == "__main__":
    asyncio.run(main())