#!/usr/bin/env python3
"""
Load minimal historical price data needed for wheel strategy risk calculations.
This is a one-time setup - only needs 250 days of daily stock prices.
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.unity import TICKER
from unity_wheel.databento import DatabentoClient
from unity_wheel.databento.price_history_loader import PriceHistoryLoader
from unity_wheel.storage import Storage, StorageConfig
from unity_wheel.utils import setup_structured_logging


async def load_historical_data():
    """One-time load of historical prices for risk calculations."""

    setup_structured_logging()

    print("📊 Loading Historical Price Data for Wheel Strategy")
    print("=" * 60)
    print("\nBased on DATABENTO_HISTORICAL_REQUIREMENTS.md:")
    print("- Need: 250 days of daily stock prices")
    print("- Purpose: VaR/CVaR risk calculations")
    print("- No historical options data needed!")
    print()

    # Initialize components
    storage = Storage(StorageConfig())
    await storage.initialize()

    client = DatabentoClient()
    loader = PriceHistoryLoader(client, storage)

    try:
        # Define symbols to load
        symbols = [TICKER]  # Start with Unity
        # Could add more: ['SPY', 'QQQ', 'IWM'] for diversification

        print(f"Loading price history for: {', '.join(symbols)}")
        print()

        for symbol in symbols:
            print(f"\n{'='*30}")
            print(f"Loading {symbol}...")
            print(f"{'='*30}")

            # Check existing data
            availability = await loader.check_data_availability(symbol)
            print(f"\nCurrent data: {availability['days_available']} days")
            print(f"Date range: {availability['date_range']}")

            if availability["optimal_data"]:
                print("✅ Already have sufficient data!")
                if availability["days_available"] > 0:
                    print(f"Annualized return: {availability['annualized_return']:.1%}")
                    print(f"Annualized volatility: {availability['annualized_volatility']:.1%}")
            else:
                # Load historical data
                print(f"\n⏳ Loading {loader.REQUIRED_DAYS} days of history...")

                success = await loader.load_price_history(symbol)

                if success:
                    # Verify loaded data
                    new_availability = await loader.check_data_availability(symbol)
                    print(f"\n✅ Loaded {new_availability['days_available']} days")
                    print(f"Date range: {new_availability['date_range']}")
                    print(f"Annualized return: {new_availability['annualized_return']:.1%}")
                    print(f"Annualized volatility: {new_availability['annualized_volatility']:.1%}")
                else:
                    print("❌ Failed to load price history")

        # Show total storage used
        print(f"\n{'='*60}")
        print("Storage Summary:")

        db_stats = await storage.cache.get_stats()
        print(f"Database size: {db_stats['size_mb']:.2f} MB")
        print(f"Tables: {db_stats['tables']}")

        # Show risk calculation readiness
        print(f"\n{'='*60}")
        print("Risk Calculation Readiness:")

        for symbol in symbols:
            availability = await loader.check_data_availability(symbol)
            status = "✅" if availability["sufficient_for_risk"] else "❌"
            print(f"{status} {symbol}: {availability['days_available']} days")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await client.close()

    print(f"\n{'='*60}")
    print("✅ Historical data setup complete!")
    print("\nNext steps:")
    print("1. Run wheel strategy recommendations (no more historical data needed)")
    print("2. Set up daily cron job to append latest prices")
    print("3. Risk metrics will use this data automatically")


async def show_data_requirements():
    """Show exactly what historical data is needed."""

    print("\n📋 Historical Data Requirements Summary")
    print("=" * 60)
    print()
    print("STOCK PRICES (Required):")
    print("- What: Daily OHLCV bars")
    print("- How much: 250 days (~1 year)")
    print("- Why: Risk calculations (VaR, CVaR, Sharpe)")
    print("- Size: ~2KB per symbol")
    print()
    print("OPTIONS DATA (Not Required):")
    print("- What: Current chains only")
    print("- How much: 0 days (no history needed)")
    print("- Why: Real-time recommendations only")
    print("- Size: Handled by 15-minute cache")
    print()
    print("TOTAL STORAGE:")
    print("- 10 symbols × 2KB = ~20KB for all historical data")
    print("- Compare to 5GB DuckDB limit = 0.0004% usage")
    print()
    print("This is a recommendation system, not a backtesting platform!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--requirements":
        asyncio.run(show_data_requirements())
    else:
        asyncio.run(load_historical_data())
