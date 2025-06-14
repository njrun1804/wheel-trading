#!/usr/bin/env python3
"""Validate live API connections for Databento and FRED."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from unity_wheel.data_providers.databento.client import DatabentoClient
from unity_wheel.data_providers.fred.fred_client import FREDClient
from unity_wheel.secrets.integration import get_databento_api_key, get_fred_api_key


async def test_databento_connection() -> Dict[str, any]:
    """Test Databento API connection and data retrieval."""
    print("\n" + "="*60)
    print("DATABENTO API VALIDATION")
    print("="*60)
    
    results = {
        'api_key_present': False,
        'connection_successful': False,
        'can_retrieve_data': False,
        'unity_data_available': False,
        'rate_limiting_works': False,
        'errors': []
    }
    
    try:
        # Check API key
        api_key = get_databento_api_key()
        if api_key and api_key != 'dummy' and api_key != 'test':
            results['api_key_present'] = True
            print("✅ API key found (not dummy/test)")
        else:
            print("❌ No valid API key found")
            results['errors'].append("Invalid or missing API key")
            return results
        
        # Initialize client
        client = DatabentoClient(api_key=api_key)
        print("✅ Client initialized")
        
        # Test basic connection
        try:
            # Get available datasets
            datasets = await client.get_datasets()
            if datasets:
                results['connection_successful'] = True
                print(f"✅ Connection successful - {len(datasets)} datasets available")
            else:
                print("⚠️  Connected but no datasets found")
        except Exception as e:
            results['errors'].append(f"Connection failed: {str(e)}")
            print(f"❌ Connection failed: {e}")
            return results
        
        # Test data retrieval for Unity
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            print(f"\nTesting Unity (U) data retrieval...")
            print(f"  Date range: {start_date.date()} to {end_date.date()}")
            
            # Get stock price
            stock_data = await client.get_stock_bars(
                symbol='U',
                start=start_date,
                end=end_date
            )
            
            if stock_data and len(stock_data) > 0:
                results['can_retrieve_data'] = True
                print(f"✅ Retrieved {len(stock_data)} stock price records")
                
                # Show sample data
                latest = stock_data.iloc[-1]
                print(f"  Latest price: ${latest['close']:.2f} on {latest.name}")
            else:
                print("❌ No stock data retrieved")
                results['errors'].append("No Unity stock data available")
        except Exception as e:
            results['errors'].append(f"Data retrieval failed: {str(e)}")
            print(f"❌ Data retrieval failed: {e}")
        
        # Test options data
        try:
            print("\nTesting Unity options data...")
            options = await client.get_option_chain(
                underlying='U',
                expiration_date=end_date + timedelta(days=30)
            )
            
            if options and len(options) > 0:
                results['unity_data_available'] = True
                print(f"✅ Retrieved {len(options)} options contracts")
            else:
                print("⚠️  No options data found")
        except Exception as e:
            print(f"⚠️  Options retrieval failed: {e}")
        
        # Test rate limiting
        try:
            print("\nTesting rate limiting...")
            start_time = datetime.now()
            
            # Make 5 rapid requests
            for i in range(5):
                await client.get_stock_bars('U', start_date, end_date)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > 0.05:  # Should have some delay due to rate limiting
                results['rate_limiting_works'] = True
                print(f"✅ Rate limiting active (5 requests took {elapsed:.2f}s)")
            else:
                print("⚠️  Rate limiting may not be working")
        except Exception as e:
            print(f"⚠️  Rate limit test failed: {e}")
        
    except Exception as e:
        results['errors'].append(f"Unexpected error: {str(e)}")
        print(f"\n❌ CRITICAL ERROR: {e}")
    
    return results


async def test_fred_connection() -> Dict[str, any]:
    """Test FRED API connection and data retrieval."""
    print("\n" + "="*60)
    print("FRED API VALIDATION")
    print("="*60)
    
    results = {
        'api_key_present': False,
        'connection_successful': False,
        'can_retrieve_data': False,
        'rate_limiting_works': False,
        'caching_works': False,
        'errors': []
    }
    
    try:
        # Check API key
        api_key = get_fred_api_key()
        if api_key and api_key != 'dummy' and api_key != 'test':
            results['api_key_present'] = True
            print("✅ API key found (not dummy/test)")
        else:
            print("❌ No valid API key found")
            results['errors'].append("Invalid or missing API key")
            return results
        
        # Initialize client
        client = FREDClient(api_key=api_key)
        print("✅ Client initialized")
        
        # Test basic connection with DGS10 (10-year treasury)
        try:
            print("\nTesting data retrieval (DGS10 - 10Y Treasury)...")
            
            observations = await client.get_series_observations('DGS10', limit=10)
            
            if observations and len(observations) > 0:
                results['connection_successful'] = True
                results['can_retrieve_data'] = True
                print(f"✅ Retrieved {len(observations)} observations")
                
                # Show latest data
                latest = observations[-1]
                print(f"  Latest 10Y yield: {latest.value}% on {latest.date}")
            else:
                print("❌ No data retrieved")
                results['errors'].append("No FRED data available")
        except Exception as e:
            results['errors'].append(f"Connection failed: {str(e)}")
            print(f"❌ Connection failed: {e}")
            return results
        
        # Test multiple series
        try:
            print("\nTesting multiple series retrieval...")
            series_ids = ['DGS10', 'DFF', 'UNRATE']  # 10Y, Fed Funds, Unemployment
            
            for series_id in series_ids:
                data = await client.get_series_observations(series_id, limit=1)
                if data:
                    print(f"  ✅ {series_id}: {data[0].value}")
                else:
                    print(f"  ❌ {series_id}: No data")
        except Exception as e:
            print(f"⚠️  Multiple series test failed: {e}")
        
        # Test rate limiting
        try:
            print("\nTesting rate limiting...")
            start_time = datetime.now()
            
            # Make 5 rapid requests
            for i in range(5):
                await client.get_series_observations('DGS10', limit=1)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > 0.5:  # FRED rate limit is stricter
                results['rate_limiting_works'] = True
                print(f"✅ Rate limiting active (5 requests took {elapsed:.2f}s)")
            else:
                print("⚠️  Rate limiting may not be working")
        except Exception as e:
            print(f"⚠️  Rate limit test failed: {e}")
        
        # Test caching
        try:
            print("\nTesting caching...")
            
            # First request (should hit API)
            start1 = datetime.now()
            data1 = await client.get_series_observations('DGS10', limit=10)
            time1 = (datetime.now() - start1).total_seconds()
            
            # Second request (should hit cache)
            start2 = datetime.now()
            data2 = await client.get_series_observations('DGS10', limit=10)
            time2 = (datetime.now() - start2).total_seconds()
            
            if time2 < time1 * 0.1:  # Cache should be 10x+ faster
                results['caching_works'] = True
                print(f"✅ Caching works (API: {time1:.3f}s, Cache: {time2:.3f}s)")
            else:
                print(f"⚠️  Caching may not be working (API: {time1:.3f}s, Cache: {time2:.3f}s)")
        except Exception as e:
            print(f"⚠️  Cache test failed: {e}")
        
    except Exception as e:
        results['errors'].append(f"Unexpected error: {str(e)}")
        print(f"\n❌ CRITICAL ERROR: {e}")
    
    return results


async def validate_all_apis():
    """Run all API validation tests."""
    print("\n" + "="*80)
    print("WHEEL TRADING SYSTEM - API CONNECTION VALIDATION")
    print("="*80)
    print(f"Timestamp: {datetime.now()}")
    
    # Test both APIs
    databento_results = await test_databento_connection()
    fred_results = await test_fred_connection()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = True
    
    print("\nDatabento API:")
    for key, value in databento_results.items():
        if key != 'errors':
            status = "✅" if value else "❌"
            print(f"  {status} {key.replace('_', ' ').title()}: {value}")
            if not value and key != 'unity_data_available':  # Options data is optional
                all_passed = False
    
    print("\nFRED API:")
    for key, value in fred_results.items():
        if key != 'errors':
            status = "✅" if value else "❌"
            print(f"  {status} {key.replace('_', ' ').title()}: {value}")
            if not value:
                all_passed = False
    
    # Show all errors
    all_errors = databento_results['errors'] + fred_results['errors']
    if all_errors:
        print("\n❌ ERRORS ENCOUNTERED:")
        for error in all_errors:
            print(f"  - {error}")
    
    # Final verdict
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL API VALIDATIONS PASSED")
        print("\nThe system is ready for live trading with real market data.")
    else:
        print("❌ API VALIDATION FAILED")
        print("\nNext Steps:")
        print("1. Check API keys in .env file or environment variables")
        print("2. Ensure DATABENTO_API_KEY and FRED_API_KEY are set correctly")
        print("3. Verify network connectivity and firewall settings")
        print("4. Check API subscription status and limits")
    
    print("="*80)


if __name__ == "__main__":
    # Run validation
    asyncio.run(validate_all_apis())