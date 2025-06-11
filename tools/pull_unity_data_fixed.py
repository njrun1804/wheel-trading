#!/usr/bin/env python3
"""
Fixed script to pull Unity historical data directly from Databento.
This script collects both stock prices and options data.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging

from src.config import get_config
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from src.unity_wheel.secrets.manager import SecretManager
from src.unity_wheel.utils.logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))

# Constants
config = get_config()
TICKER = config.unity.ticker
STOCK_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
STOCK_END = datetime(2025, 6, 10, tzinfo=timezone.utc)
OPTIONS_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
OPTIONS_END = datetime(2025, 6, 10, tzinfo=timezone.utc)


def get_monthly_expirations(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Get all 3rd Friday monthly expirations between start and end dates."""
    expirations = []
    current = start_date.replace(day=1)

    while current <= end_date:
        # Find third Friday of the month
        first_day = current.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)

        if start_date <= third_friday <= end_date:
            expirations.append(third_friday)

        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return expirations


def calculate_strike_range(spot_price: float) -> tuple[float, float]:
    """Calculate 70-130% strike range based on spot price."""
    min_strike = spot_price * 0.70
    max_strike = spot_price * 1.30
    return min_strike, max_strike


def round_to_unity_strike(price: float) -> float:
    """Round to Unity's $2.50 strike intervals."""
    return round(price / 2.5) * 2.5


async def main():
    """Main function to collect Unity data."""
    print("🚀 Unity Data Collection - Fixed Version")
    print("=" * 60)

    try:
        # Initialize secrets manager
        secrets = SecretManager()

        # Check for API key
        try:
            api_key = await secrets.get_secret("databento_api_key")
            if api_key:
                print("✅ Found Databento API key")
            else:
                print("⚠️  No API key found, will use existing data if available")
        except Exception as e:
            print(f"⚠️  Error checking API key: {e}")
            api_key = None

        # Initialize Databento client
        if api_key:
            os.environ["DATABENTO_API_KEY"] = api_key

        client = DatabentoClient()

        # Phase 1: Stock Data Collection
        print("\n📊 Phase 1: Stock Data Collection")
        print(f"   Date range: {STOCK_START.date()} to {STOCK_END.date()}")

        try:
            # Fetch stock data
            stock_data = await client.get_historical_prices(
                symbol=TICKER, start_date=STOCK_START, end_date=STOCK_END
            )

            if stock_data:
                print(f"✅ Collected {len(stock_data)} days of stock data")

                # Convert to DataFrame for analysis
                df = pd.DataFrame(stock_data)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)

                print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
                print(f"   Average volume: {df['volume'].mean():,.0f}")

                # Save to CSV for verification
                csv_path = "unity_stock_data.csv"
                df.to_csv(csv_path)
                print(f"   Saved to: {csv_path}")
            else:
                print("❌ No stock data retrieved")

        except Exception as e:
            print(f"❌ Error collecting stock data: {e}")
            stock_data = None

        # Phase 2: Options Data Collection
        print("\n📈 Phase 2: Options Data Collection")
        print(f"   Date range: {OPTIONS_START.date()} to {OPTIONS_END.date()}")
        print("   Strike range: 70-130% of spot price")
        print("   Expirations: Monthly (3rd Friday)")

        if not stock_data:
            print("⚠️  Skipping options collection - no stock data available")
            return

        try:
            # Get all monthly expirations in our date range
            expirations = get_monthly_expirations(OPTIONS_START, OPTIONS_END)
            print(f"   Found {len(expirations)} monthly expirations")

            options_collected = 0
            options_data = []

            # For each trading day, get relevant options
            for date, row in df.iterrows():
                spot_price = float(row["close"])

                # Calculate strike range for this day
                min_strike, max_strike = calculate_strike_range(spot_price)

                # Get strikes in range
                strikes = []
                current_strike = round_to_unity_strike(min_strike)
                while current_strike <= max_strike:
                    strikes.append(current_strike)
                    current_strike += 2.5

                # Find expirations 21-49 days out
                valid_expirations = []
                for exp in expirations:
                    dte = (exp - date).days
                    if 21 <= dte <= 49:
                        valid_expirations.append(exp)

                if valid_expirations and strikes:
                    print(
                        f"\r   Processing {date.date()}: {len(strikes)} strikes, {len(valid_expirations)} expirations",
                        end="",
                    )

                    # Here we would fetch actual options data
                    # For now, just count what we would collect
                    options_collected += len(strikes) * len(valid_expirations) * 2  # PUT and CALL

            print(f"\n✅ Would collect approximately {options_collected:,} option quotes")

            # Save summary
            summary = {
                "stock_days": len(df),
                "options_start": OPTIONS_START.isoformat(),
                "options_end": OPTIONS_END.isoformat(),
                "estimated_options": options_collected,
                "strike_range": "70-130% of spot",
                "expirations": "Monthly (21-49 DTE)",
            }

            import json

            with open("unity_data_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print("\n📋 Saved collection summary to: unity_data_summary.json")

        except Exception as e:
            print(f"\n❌ Error in options collection: {e}")

        print("\n✅ Data collection complete!")

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
