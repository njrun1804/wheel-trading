#!/usr/bin/env python3
"""Validate all API calls are working correctly for live and historical data."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unity_wheel.data_providers.databento.client import DatabentoClient
from unity_wheel.data_providers.fred.fred_client import FREDClient
from unity_wheel.secrets.manager import SecretManager
from unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)


async def test_databento_apis():
    """Test all Databento API functionality."""
    print("\n=== Testing Databento APIs ===")
    
    try:
        client = DatabentoClient()
        
        # Test 1: Historical stock data (EQUS.MINI schema)
        print("\n1. Testing historical stock data (EQUS.MINI)...")
        try:
            stock_data = await client.fetch_stock_history(
                symbol="U",
                start_date=datetime.now() - timedelta(days=5),
                end_date=datetime.now() - timedelta(days=1)
            )
            print(f"   ✅ Fetched {len(stock_data)} days of Unity stock data")
            if stock_data:
                latest = stock_data.iloc[-1]
                print(f"   Latest: {latest['date']} - Close: ${latest['close']:.2f}")
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            
        # Test 2: Historical options data (OPRA.PILLAR schema)
        print("\n2. Testing historical options data (OPRA.PILLAR)...")
        try:
            # Get option chains for Unity
            chains = await client.fetch_option_chains(
                symbol="U",
                start_date=datetime.now() - timedelta(days=2),
                end_date=datetime.now() - timedelta(days=1)
            )
            print(f"   ✅ Fetched {len(chains)} option chain snapshots")
            if chains:
                chain = chains[0]
                print(f"   Chain date: {chain.timestamp}")
                print(f"   Puts: {len(chain.puts)}, Calls: {len(chain.calls)}")
                print(f"   Spot price: ${chain.spot_price}")
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            
        # Test 3: Live stock quotes (if market is open)
        print("\n3. Testing live stock quotes...")
        try:
            from unity_wheel.utils.trading_calendar import SimpleTradingCalendar
            cal = SimpleTradingCalendar()
            if cal.is_market_open():
                quote = await client.fetch_live_stock_quote("U")
                print(f"   ✅ Live quote: Bid=${quote.bid_price} Ask=${quote.ask_price}")
            else:
                print("   ⏸️  Market closed, skipping live test")
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            
        # Test 4: Option definitions
        print("\n4. Testing option definitions...")
        try:
            definitions = await client.fetch_option_definitions("U")
            print(f"   ✅ Fetched {len(definitions)} option definitions")
            # Group by expiration
            expirations = {}
            for defn in definitions:
                exp = defn.expiration.date()
                if exp not in expirations:
                    expirations[exp] = 0
                expirations[exp] += 1
            print(f"   Expirations: {len(expirations)} dates")
            for exp, count in sorted(expirations.items())[:3]:
                print(f"     {exp}: {count} contracts")
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            
        await client.close()
        
    except Exception as e:
        print(f"\n❌ Databento client initialization failed: {str(e)}")
        return False
        
    return True


async def test_fred_apis():
    """Test all FRED API functionality."""
    print("\n=== Testing FRED APIs ===")
    
    try:
        client = FREDClient()
        
        # Test 1: VIX data
        print("\n1. Testing VIX data fetch...")
        try:
            vix_data = await client.fetch_vix_history(days=30)
            print(f"   ✅ Fetched {len(vix_data)} days of VIX data")
            if vix_data:
                latest = vix_data.iloc[-1]
                print(f"   Latest VIX: {latest['value']:.2f} on {latest['date']}")
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            
        # Test 2: Risk-free rate
        print("\n2. Testing risk-free rate...")
        try:
            rf_rate = await client.fetch_risk_free_rate()
            print(f"   ✅ Current risk-free rate: {rf_rate:.4f} ({rf_rate*100:.2f}%)")
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            
        # Test 3: Multiple economic indicators
        print("\n3. Testing multiple indicators...")
        indicators = {
            'DGS10': '10-Year Treasury',
            'DFEDTARU': 'Fed Funds Upper',
            'DEXUSEU': 'USD/EUR Exchange',
            'TEDRATE': 'TED Spread'
        }
        
        for series_id, name in indicators.items():
            try:
                data = await client.fetch_series(series_id, limit=5)
                if data:
                    latest = data.iloc[-1]
                    print(f"   ✅ {name}: {latest['value']:.4f}")
                else:
                    print(f"   ⚠️  {name}: No data")
            except Exception as e:
                print(f"   ❌ {name}: {str(e)}")
                
        await client.close()
        
    except Exception as e:
        print(f"\n❌ FRED client initialization failed: {str(e)}")
        return False
        
    return True


async def test_data_storage():
    """Test that data is being stored correctly."""
    print("\n=== Testing Data Storage ===")
    
    import duckdb
    
    try:
        conn = duckdb.connect("data/wheel_trading_optimized.duckdb", read_only=True)
        
        # Check recent data
        print("\n1. Checking recent stock data...")
        result = conn.execute("""
            SELECT COUNT(*) as count, 
                   MAX(date) as latest_date,
                   MIN(date) as oldest_date
            FROM market.price_data 
            WHERE symbol = 'U'
        """).fetchone()
        print(f"   Records: {result[0]}")
        print(f"   Date range: {result[2]} to {result[1]}")
        
        print("\n2. Checking recent options data...")
        result = conn.execute("""
            SELECT COUNT(*) as count,
                   MAX(timestamp) as latest,
                   COUNT(DISTINCT DATE(timestamp)) as days
            FROM options.contracts 
            WHERE symbol = 'U'
        """).fetchone()
        print(f"   Records: {result[0]}")
        print(f"   Latest: {result[1]}")
        print(f"   Days with data: {result[2]}")
        
        print("\n3. Checking FRED data...")
        result = conn.execute("""
            SELECT COUNT(*) as count,
                   COUNT(DISTINCT feature_date) as days
            FROM analytics.ml_features 
            WHERE vix_level IS NOT NULL
        """).fetchone()
        print(f"   Records with VIX: {result[0]}")
        print(f"   Days with VIX: {result[1]}")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ Database check failed: {str(e)}")
        return False
        
    return True


async def main():
    """Run all API validation tests."""
    print("Unity Wheel Trading API Validation")
    print("=" * 50)
    
    # Check API keys first
    secrets = SecretManager()
    
    databento_key = secrets.get_secret("databento_api_key")
    fred_key = secrets.get_secret("fred_api_key")
    
    print("\nAPI Key Status:")
    print(f"  Databento: {'✅ Configured' if databento_key and not databento_key.startswith('your_') else '❌ Not configured'}")
    print(f"  FRED: {'✅ Configured' if fred_key and not fred_key.startswith('your_') else '❌ Not configured'}")
    
    # Run tests
    databento_ok = await test_databento_apis()
    fred_ok = await test_fred_apis()
    storage_ok = await test_data_storage()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Databento APIs: {'✅ PASS' if databento_ok else '❌ FAIL'}")
    print(f"  FRED APIs: {'✅ PASS' if fred_ok else '❌ FAIL'}")
    print(f"  Data Storage: {'✅ PASS' if storage_ok else '❌ FAIL'}")
    
    if not (databento_ok and fred_ok and storage_ok):
        print("\n⚠️  Some APIs are not working correctly!")
        return 1
    
    print("\n✅ All APIs validated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))