#!/usr/bin/env python3
"""
Find Unity options using CORRECT Databento structure.

Options are in different datasets/schemas than stocks.
This script uses the proper Databento options API structure.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import databento as db
from databento_dbn import Schema, SType
from src.unity_wheel.secrets.integration import get_databento_api_key

async def find_unity_options_properly():
    """Find Unity options using correct Databento API structure."""
    print("🔍 FINDING UNITY OPTIONS - CORRECT DATABENTO STRUCTURE")
    print("=" * 70)
    
    # Get API key
    api_key = get_databento_api_key()
    client = db.Historical(api_key)
    
    # 1. First, let's understand the correct date range for OPRA data
    print("📅 Step 1: Finding correct date range for OPRA options data...")
    
    # OPRA data is typically T+1, so use yesterday as end date
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)
    
    # Use trading hours (market close)
    end_time = yesterday.replace(hour=21, minute=0, second=0, microsecond=0)  # 4 PM ET = 21 UTC
    start_time = end_time - timedelta(hours=1)  # 1 hour window
    
    print(f"   Start: {start_time}")
    print(f"   End: {end_time}")
    
    # 2. Try to get Unity option definitions using different approaches
    print(f"\n📋 Step 2: Getting Unity option definitions...")
    
    # Method 1: Use parent symbol U.OPT
    try:
        print(f"   Method 1: U.OPT with PARENT symbol type...")
        response = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema=Schema.DEFINITION,
            start=start_time,
            end=end_time,
            symbols=["U.OPT"],
            stype_in=SType.PARENT
        )
        
        definitions = list(response)
        print(f"   ✅ Found {len(definitions)} definitions with U.OPT")
        
        if definitions:
            for i, defn in enumerate(definitions[:3]):
                print(f"      Def {i+1}: {defn}")
                
    except Exception as e:
        print(f"   ❌ U.OPT failed: {e}")
        definitions = []
    
    # Method 2: If no definitions found, try broader date range
    if not definitions:
        print(f"\n   Method 2: Trying broader date range (last week)...")
        
        week_end = yesterday.replace(hour=21, minute=0, second=0, microsecond=0)
        week_start = week_end - timedelta(days=7)
        
        try:
            response = client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.DEFINITION,
                start=week_start,
                end=week_end,
                symbols=["U.OPT"],
                stype_in=SType.PARENT
            )
            
            definitions = list(response)
            print(f"   ✅ Found {len(definitions)} definitions in past week")
            
        except Exception as e:
            print(f"   ❌ Week range failed: {e}")
    
    # Method 3: Try getting any recent options to see format
    if not definitions:
        print(f"\n   Method 3: Testing with known liquid options symbol...")
        
        # Try SPY first to verify our approach works
        try:
            response = client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.DEFINITION,
                start=week_start,
                end=week_end,
                symbols=["SPY.OPT"],
                stype_in=SType.PARENT
            )
            
            spy_definitions = list(response)
            print(f"   ✅ SPY.OPT test: Found {len(spy_definitions)} definitions")
            print(f"      This confirms our API approach is correct")
            
            # Now try Unity with exact same pattern
            response = client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.DEFINITION,
                start=week_start,
                end=week_end,
                symbols=["U.OPT"],
                stype_in=SType.PARENT
            )
            
            definitions = list(response)
            print(f"   Result for U.OPT: {len(definitions)} definitions")
            
        except Exception as e:
            print(f"   ❌ SPY test failed: {e}")
    
    # Method 4: Try different Unity symbol formats
    if not definitions:
        print(f"\n   Method 4: Trying alternative Unity symbol formats...")
        
        unity_formats = [
            "U     .OPT",  # Unity with spaces
            "U.OPT",       # Standard
            "U",           # Just U
        ]
        
        for symbol_format in unity_formats:
            try:
                print(f"      Testing: '{symbol_format}'...")
                response = client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema=Schema.DEFINITION,
                    start=week_start,
                    end=week_end,
                    symbols=[symbol_format],
                    stype_in=SType.PARENT
                )
                
                test_definitions = list(response)
                if test_definitions:
                    print(f"      ✅ '{symbol_format}': Found {len(test_definitions)} definitions")
                    definitions = test_definitions
                    break
                else:
                    print(f"      ❌ '{symbol_format}': No definitions")
                    
            except Exception as e:
                print(f"      ❌ '{symbol_format}': {e}")
    
    # 3. If we found definitions, get recent quotes
    if definitions:
        print(f"\n📊 Step 3: Getting Unity option quotes...")
        
        # Get instrument IDs from definitions
        instrument_ids = [d.instrument_id for d in definitions[:10]]  # First 10
        
        try:
            # Get recent trades/quotes for these instruments
            response = client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.TRADES,  # Or try Schema.MBP_1 for quotes
                start=start_time,
                end=end_time,
                symbols=None,
                instrument_ids=instrument_ids
            )
            
            quotes = list(response)
            print(f"   ✅ Found {len(quotes)} recent quotes/trades")
            
            if quotes:
                for i, quote in enumerate(quotes[:3]):
                    print(f"      Quote {i+1}: {quote}")
                    
        except Exception as e:
            print(f"   ❌ Error getting quotes: {e}")
    
    # 4. Try live/current data approach
    print(f"\n🔴 Step 4: Checking live Unity options data...")
    
    try:
        # For live data, use current time
        live_end = datetime.now(timezone.utc)
        live_start = live_end - timedelta(minutes=30)
        
        response = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema=Schema.DEFINITION,
            start=live_start,
            end=live_end,
            symbols=["U.OPT"],
            stype_in=SType.PARENT
        )
        
        live_definitions = list(response)
        print(f"   ✅ Live Unity options: {len(live_definitions)} definitions")
        
    except Exception as e:
        print(f"   ❌ Live data error: {e}")
    
    # 5. Final summary
    print(f"\n📋 SUMMARY")
    print("=" * 70)
    
    total_found = len(definitions) if definitions else 0
    
    if total_found > 0:
        print(f"✅ SUCCESS: Found {total_found} Unity option contracts")
        print(f"   Unity DOES have options available on Databento")
        print(f"   Ready to implement real data collection")
        
        # Show sample contracts
        print(f"\n📄 Sample Unity Options:")
        for i, defn in enumerate(definitions[:5]):
            try:
                exp_date = defn.expiration if hasattr(defn, 'expiration') else 'Unknown'
                strike = defn.strike_price if hasattr(defn, 'strike_price') else 'Unknown'  
                opt_type = defn.option_type if hasattr(defn, 'option_type') else 'Unknown'
                print(f"   {i+1}. {opt_type} ${strike} exp {exp_date}")
            except:
                print(f"   {i+1}. {defn}")
                
    else:
        print(f"❌ No Unity options found with current approach")
        print(f"   Need to investigate further or check subscription")
        
    print(f"\n🎯 NEXT STEPS:")
    if total_found > 0:
        print(f"   1. Update data collection script with working parameters")
        print(f"   2. Implement full Unity options data pull")
        print(f"   3. Store real Unity options in database")
    else:
        print(f"   1. Check Databento subscription coverage")
        print(f"   2. Try different date ranges") 
        print(f"   3. Contact Databento support for Unity options")

if __name__ == "__main__":
    asyncio.run(find_unity_options_properly())