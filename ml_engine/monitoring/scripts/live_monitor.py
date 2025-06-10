#!/usr/bin/env python3
"""
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

from src.config.loader import get_config, get_config_loader
from src.unity_wheel.analytics import IntegratedDecisionEngine
from src.unity_wheel.analytics.performance_tracker import PerformanceTracker
from src.unity_wheel.risk.limits import RiskLimitChecker
from src.unity_wheel.storage import UnifiedStorage
from src.unity_wheel.utils import get_logger

logger = get_logger(__name__)

# Global flag for graceful shutdown
running = True


async def get_realized_volatility_from_databento(ticker: str) -> float:
    """Fetch realized volatility from Databento or calculate from historical data."""
    try:
        from src.unity_wheel.data_providers.databento import DatabentoClient
        from src.unity_wheel.data_providers.databento.price_history_loader import (
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

    except Exception as e:
        logger.error(f"CRITICAL: Failed to fetch volatility from Databento: {e}")
        raise ValueError(f"Cannot calculate volatility without real market data: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    running = False
    print("\n\n🛑 Shutting down monitor...")


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

    print("🎯 Unity Wheel Strategy Monitor")
    print("=" * 60)
    print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Current market data
    config = get_config()
    latest = await storage.get_latest_price(config.unity.ticker)
    if latest:
        print("📊 Market Data:")
        print(f"   Price: ${latest.get('close', 0):.2f}")
        print(f"   Volume: {latest.get('volume', 0):,.0f}")
        print(f"   Change: {latest.get('returns', 0)*100:+.2f}%")

        # Calculate realized vol
        history = await storage.get_price_history(config.unity.ticker, days=20)
        if len(history) >= 20:
            import numpy as np

            returns = [h["returns"] for h in history]
            vol = np.std(returns) * np.sqrt(252)
            print(f"   Volatility: {vol:.1%}")
    else:
        print("❌ No market data available")
    print()

    # 2. Risk limits status
    restrictions = risk_checker.get_current_restrictions()
    print("🛡️  Risk Status:")
    if restrictions["can_trade"]:
        print("   ✅ Trading allowed")
    else:
        print("   🚫 Trading blocked")
        for reason in restrictions["reasons"]:
            print(f"      - {reason}")

    if restrictions["reduced_size"]:
        print(f"   ⚠️  Size reduced to {restrictions['max_position_size']:.1%}")
    print()

    # 3. Recent performance
    stats = tracker.get_performance_stats(30)
    if isinstance(stats, dict) and "total_trades" in stats:
        print("📈 Performance (30 days):")
        print(f"   Trades: {stats['total_trades']}")
        print(f"   Win Rate: {stats.get('win_rate', 0):.1%}")
        print(f"   Avg Return: {stats.get('avg_return', 0):.2%}")
    else:
        print("📈 Performance: No recent trades")
    print()

    # 4. System health
    config_loader = get_config_loader()
    unused = config_loader.get_unused_parameters()
    print("⚙️  System Health:")
    print(f"   Config parameters: {len(config_loader.config.__dict__)} loaded")
    print(f"   Unused parameters: {len(unused)}")

    # Check for recent errors in logs
    print()

    # 5. Active alerts
    print("🚨 Active Alerts:")
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
            print(f"   ⚠️  {alert}")
    else:
        print("   ✅ No active alerts")
    print()

    print("-" * 60)
    print("Press Ctrl+C to stop monitoring")


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
            recommendation._asdict(), portfolio_value=100000, market_data=current_prices  # Default
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

    except Exception as e:
        logger.error(f"Error checking opportunities: {e}")


async def main():
    """Main monitoring loop."""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🚀 Starting Unity Wheel Strategy Monitor...")
    print("Loading components...")

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

        except Exception as e:
            logger.error(f"Monitor error: {e}")
            await asyncio.sleep(30)  # Wait longer on error

    print("\n✅ Monitor stopped gracefully")


if __name__ == "__main__":
    asyncio.run(main())
