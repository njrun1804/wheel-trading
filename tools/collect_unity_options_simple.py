#!/usr/bin/env python3
"""
Simple Unity options collection script.
Collects options data based on the DATA_COLLECTION_SPECIFICATION.md requirements.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
import databento as db
from databento import SType

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.unity_wheel.secrets.manager import SecretManager
from src.unity_wheel.utils.logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))

# Constants from specification
OPTIONS_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
OPTIONS_END = datetime(2025, 6, 10, tzinfo=timezone.utc)

def get_monthly_expiration_dates(start_date: datetime, end_date: datetime) -> list[datetime]:
    """Get all monthly expiration dates (3rd Friday) between start and end."""
    expirations = []
    current = start_date.replace(day=1)
    
    while current <= end_date:
        # Find first day of month
        first_day = current.replace(day=1)
        # Find first Friday
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        # Third Friday is 14 days later
        third_friday = first_friday + timedelta(days=14)
        
        if start_date <= third_friday <= end_date:
            expirations.append(third_friday)
        
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return expirations

def round_to_unity_strike(price: float) -> float:
    """Round to Unity's $2.50 strike intervals."""
    return round(price / 2.5) * 2.5

async def main():
    """Collect Unity options data according to specification."""
    print("🚀 Unity Options Collection - Simple Version")
    print("=" * 60)
    print(f"📅 Collection period: {OPTIONS_START.date()} to {OPTIONS_END.date()}")
    print("🎯 Strike range: 70-130% of spot price")
    print("📆 Expirations: Monthly (3rd Friday), 21-49 DTE")
    print()
    
    try:
        # Get API key
        secrets = SecretManager()
        api_key = secrets.get_secret("databento_api_key")
        if not api_key:
            print("❌ No Databento API key found in secrets")
            print("   Run: python scripts/setup-secrets.py")
            return
        
        print("✅ Found Databento API key")
        
        # Initialize Databento client
        client = db.Historical(api_key)
        
        # First, get stock price data to determine strike ranges
        print("\n📊 Fetching Unity stock prices for strike calculation...")
        
        # Get daily OHLCV data
        try:
            stock_data = client.timeseries.get_range(
                dataset="XNYS.PILLAR",
                symbols="U",
                stype_in=SType.RAW_SYMBOL,
                schema="ohlcv-1d",
                start=OPTIONS_START.strftime("%Y-%m-%d"),
                end=OPTIONS_END.strftime("%Y-%m-%d"),
            )
            
            # Convert to list for processing
            stock_records = list(stock_data.to_ndarray())
            print(f"✅ Retrieved {len(stock_records)} days of stock data")
            
            # Create a dict of date -> closing price
            price_by_date = {}
            for record in stock_records:
                # Extract date and close price from the record
                # The timestamp is in nanoseconds since epoch
                ts_ns = record['ts_event']
                date = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).date()
                close_price = record['close'] / 1e9  # Price is in fixed-point format
                price_by_date[date] = close_price
            
            print(f"   Price range: ${min(price_by_date.values()):.2f} - ${max(price_by_date.values()):.2f}")
            
        except Exception as e:
            print(f"❌ Error fetching stock data: {e}")
            return
        
        # Get all monthly expirations
        expirations = get_monthly_expiration_dates(OPTIONS_START, OPTIONS_END)
        print(f"\n📅 Found {len(expirations)} monthly expirations")
        
        # Count what we'll collect
        total_options_to_collect = 0
        collection_plan = []
        
        print("\n📋 Building collection plan...")
        
        # For each date, determine which strikes and expirations to collect
        for date, spot_price in sorted(price_by_date.items()):
            # Calculate strike range (70-130% of spot)
            min_strike = spot_price * 0.70
            max_strike = spot_price * 1.30
            
            # Get strikes in range
            strikes = []
            current_strike = round_to_unity_strike(min_strike)
            while current_strike <= max_strike:
                strikes.append(current_strike)
                current_strike += 2.5
            
            # Find expirations 21-49 days out
            valid_expirations = []
            for exp in expirations:
                dte = (exp.date() - date).days
                if 21 <= dte <= 49:
                    valid_expirations.append(exp)
            
            if strikes and valid_expirations:
                options_count = len(strikes) * len(valid_expirations) * 2  # PUT and CALL
                total_options_to_collect += options_count
                collection_plan.append({
                    'date': date,
                    'spot_price': spot_price,
                    'strikes': strikes,
                    'expirations': valid_expirations,
                    'count': options_count
                })
        
        print(f"✅ Will collect approximately {total_options_to_collect:,} option quotes")
        print(f"   Across {len(collection_plan)} trading days")
        
        # Sample collection plan
        print("\n📊 Sample Collection Plan (first 5 days):")
        for plan in collection_plan[:5]:
            print(f"   {plan['date']}: Spot ${plan['spot_price']:.2f}, "
                  f"{len(plan['strikes'])} strikes × {len(plan['expirations'])} expirations = "
                  f"{plan['count']} options")
        
        # Estimate data size and cost
        print("\n💰 Cost Estimate:")
        # Rough estimate: ~100 bytes per option quote
        estimated_gb = (total_options_to_collect * 100) / (1024**3)
        estimated_cost = estimated_gb * 0.10  # $0.10 per GB (rough estimate)
        print(f"   Estimated data size: {estimated_gb:.2f} GB")
        print(f"   Estimated cost: ${estimated_cost:.2f}")
        
        # Ask for confirmation
        print("\n⚠️  Ready to collect Unity options data")
        print(f"   This will fetch ~{total_options_to_collect:,} option quotes")
        print(f"   Estimated cost: ${estimated_cost:.2f}")
        
        response = input("\nProceed with collection? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Collection cancelled")
            return
        
        # TODO: Actual collection would go here
        # This would involve:
        # 1. For each day in collection_plan
        # 2. Query options data for that day's strikes and expirations
        # 3. Store in databento_option_chains table
        
        print("\n✅ Collection plan created!")
        print("   Note: Actual data fetching not implemented in this simple version")
        print("   Use the full collection script when DuckDB issue is resolved")
        
        # Save collection plan for reference
        import json
        plan_file = "unity_options_collection_plan.json"
        with open(plan_file, 'w') as f:
            json.dump({
                'total_options': total_options_to_collect,
                'trading_days': len(collection_plan),
                'start_date': OPTIONS_START.isoformat(),
                'end_date': OPTIONS_END.isoformat(),
                'estimated_cost': estimated_cost,
                'sample_days': [
                    {
                        'date': str(p['date']),
                        'spot_price': p['spot_price'],
                        'strike_count': len(p['strikes']),
                        'expiration_count': len(p['expirations']),
                        'option_count': p['count']
                    }
                    for p in collection_plan[:10]
                ]
            }, f, indent=2)
        
        print(f"\n📋 Collection plan saved to: {plan_file}")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())