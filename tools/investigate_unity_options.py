#!/usr/bin/env python3
"""
Investigate Unity options availability on Databento.

This script checks:
1. Unity stock data availability 
2. Unity options existence and format
3. Available expirations and strikes
4. Subscription coverage
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.unity_wheel.data_providers.databento.client import DatabentoClient
import logging

logger = logging.getLogger(__name__)

async def investigate_unity():
    """Investigate Unity options on Databento."""
    print("🔍 INVESTIGATING UNITY OPTIONS ON DATABENTO")
    print("=" * 60)
    
    client = DatabentoClient()
    
    try:
        # 1. Verify Unity stock exists and get current price
        print("📊 Step 1: Checking Unity stock data...")
        try:
            spot_data = await client._get_underlying_price("U")
            print(f"✅ Unity stock found: ${spot_data.last_price:.2f}")
            print(f"   Timestamp: {spot_data.timestamp}")
            print(f"   Bid: ${spot_data.bid_price:.2f}" if spot_data.bid_price else "   Bid: N/A")
            print(f"   Ask: ${spot_data.ask_price:.2f}" if spot_data.ask_price else "   Ask: N/A")
        except Exception as e:
            print(f"❌ Unity stock not found: {e}")
            return
            
        # 2. Check if Unity has options by looking at definitions
        print(f"\n📋 Step 2: Checking Unity options definitions...")
        
        # Try to get definitions for the past week to see if Unity has any options at all
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        try:
            definitions = await client._get_definitions("U", end_date)
            if definitions:
                print(f"✅ Found {len(definitions)} Unity option definitions")
                
                # Show sample definitions
                for i, defn in enumerate(definitions[:5]):
                    print(f"   Option {i+1}: {defn.option_type} ${defn.strike_price} exp {defn.expiration}")
                    
                if len(definitions) > 5:
                    print(f"   ... and {len(definitions) - 5} more")
                    
            else:
                print(f"❌ No Unity option definitions found")
                
        except Exception as e:
            print(f"❌ Error getting definitions: {e}")
            
        # 3. Check specific known expiration dates
        print(f"\n📅 Step 3: Checking known expiration dates...")
        
        # Standard monthly expirations for next few months
        test_expirations = []
        today = datetime.now(timezone.utc)
        
        for month_offset in range(1, 6):  # Next 5 months
            test_date = today + timedelta(days=30 * month_offset)
            year, month = test_date.year, test_date.month
            
            # Calculate 3rd Friday
            first_day = datetime(year, month, 1, tzinfo=timezone.utc)
            days_to_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_to_friday)
            third_friday = first_friday + timedelta(days=14)
            
            test_expirations.append(third_friday)
            
        for exp_date in test_expirations:
            try:
                print(f"   Testing {exp_date.strftime('%Y-%m-%d')}...")
                chain = await client.get_option_chain("U", exp_date)
                
                puts_count = len(chain.puts)
                calls_count = len(chain.calls)
                total = puts_count + calls_count
                
                if total > 0:
                    print(f"   ✅ Found {puts_count} puts, {calls_count} calls")
                    
                    # Show some sample strikes
                    if chain.puts:
                        strikes = sorted([float(p.strike_price) for p in chain.puts[:3]])
                        print(f"      Sample put strikes: {strikes}")
                    if chain.calls:
                        strikes = sorted([float(c.strike_price) for c in chain.calls[:3]])
                        print(f"      Sample call strikes: {strikes}")
                else:
                    print(f"   ❌ No options found for this date")
                    
            except Exception as e:
                print(f"   ❌ Error for {exp_date.strftime('%Y-%m-%d')}: {e}")
                
        # 4. Check if Unity is the correct symbol
        print(f"\n🏢 Step 4: Verifying Unity symbol information...")
        
        # Unity Software Inc. should be NYSE:U
        print(f"   Symbol: U")
        print(f"   Expected: Unity Software Inc. (NYSE:U)")
        print(f"   Current price: ${spot_data.last_price:.2f}")
        
        # Sanity check - Unity Software typically trades in $20-200 range
        price = float(spot_data.last_price)
        if 10 <= price <= 300:
            print(f"   ✅ Price range looks reasonable for Unity Software")
        else:
            print(f"   ⚠️  Price might be for different Unity symbol")
            
        # 5. Check subscription and datasets
        print(f"\n📊 Step 5: Checking subscription coverage...")
        
        try:
            # Try different datasets for options
            datasets_to_test = [
                "OPRA.PILLAR",   # Main options dataset
                "OPRA.COMPLEX",  # Complex options
            ]
            
            for dataset in datasets_to_test:
                print(f"   Testing dataset: {dataset}")
                try:
                    # Just try to get any data to test subscription
                    response = client.client.timeseries.get_range(
                        dataset=dataset,
                        schema="definition",
                        start=start_date,
                        end=end_date,
                        symbols=["U.OPT"],
                        stype_in="parent"
                    )
                    
                    count = sum(1 for _ in response)
                    print(f"   ✅ {dataset}: {count} records found")
                    
                except Exception as e:
                    if "subscription" in str(e).lower():
                        print(f"   ❌ {dataset}: Subscription issue - {e}")
                    else:
                        print(f"   ❌ {dataset}: {e}")
                        
        except Exception as e:
            print(f"   ❌ Error testing datasets: {e}")
            
        # 6. Alternative symbols to check
        print(f"\n🔄 Step 6: Testing alternative Unity symbols...")
        
        alternative_symbols = [
            "UNITY",    # Alternative ticker
            "U.OPT",    # Direct options symbol
            "U ",       # With space
        ]
        
        for alt_symbol in alternative_symbols:
            try:
                print(f"   Testing symbol: '{alt_symbol}'")
                alt_spot = await client._get_underlying_price(alt_symbol.strip())
                print(f"   ✅ Found: ${alt_spot.last_price:.2f}")
            except Exception as e:
                print(f"   ❌ '{alt_symbol}': {e}")
                
        # 7. Summary and recommendations
        print(f"\n📋 INVESTIGATION SUMMARY")
        print("=" * 60)
        
        if definitions and len(definitions) > 0:
            print(f"✅ Unity options ARE available on Databento")
            print(f"   Found {len(definitions)} option contracts")
            print(f"   Recommended approach: Use existing integration")
        else:
            print(f"❌ Unity options NOT found on Databento")
            print(f"   Possible reasons:")
            print(f"   1. Unity may not have actively traded options")
            print(f"   2. Subscription may not include Unity options")
            print(f"   3. Unity options may be thinly traded")
            print(f"   4. Need different symbol format")
            
            print(f"\n🔧 RECOMMENDATIONS:")
            print(f"   1. Check if Unity Software (U) actually has listed options")
            print(f"   2. Verify Databento subscription includes all option symbols")
            print(f"   3. Consider using a different underlying with more liquid options")
            print(f"   4. Contact Databento support for Unity options coverage")
            
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(investigate_unity())