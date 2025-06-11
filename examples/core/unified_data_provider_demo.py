#!/usr/bin/env python3
"""Demonstration of the unified data provider system.

This script shows how the unified data provider consolidates access to all
market data sources through a single, consistent interface with automatic
fallback and caching.
"""

import asyncio
from datetime import datetime, timedelta

from src.unity_wheel.data_providers.unified_provider import (
    DataRequest,
    DataSourceType,
    get_risk_free_rate,
    get_unified_provider,
    get_unity_market_data,
    get_unity_options_chain,
)


async def demonstrate_basic_data_access():
    """Show basic data access through unified provider."""
    print("📊 Basic Data Access Demo")
    print("=" * 30)
    print()
    
    provider = get_unified_provider()
    
    print("🔍 Registered providers:")
    stats = provider.get_performance_stats()
    for provider_name in stats["registered_providers"]:
        print(f"  ✅ {provider_name}")
    print()
    
    # Test market data access
    print("📈 Getting Unity market data...")
    try:
        market_response = await provider.get_market_data(
            symbol="U",
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now()
        )
        
        print(f"  Source: {market_response.source}")
        print(f"  Quality: {market_response.quality.value}")
        print(f"  Data type: {type(market_response.data).__name__}")
        
        if hasattr(market_response.data, 'shape'):
            print(f"  Data shape: {market_response.data.shape}")
            if not market_response.data.empty:
                print(f"  Latest price: ${market_response.data['close'].iloc[-1]:.2f}")
        
    except Exception as e:
        print(f"  ❌ Market data failed: {e}")
    
    print()
    
    # Test options data access
    print("🎯 Getting Unity options chain...")
    try:
        options_response = await provider.get_options_chain(
            symbol="U",
            expiration=datetime.now() + timedelta(days=45)
        )
        
        print(f"  Source: {options_response.source}")
        print(f"  Quality: {options_response.quality.value}")
        print(f"  Data type: {type(options_response.data).__name__}")
        
        if isinstance(options_response.data, list) and options_response.data:
            print(f"  Options found: {len(options_response.data)}")
            first_option = options_response.data[0]
            print(f"  Example option: ${first_option.get('strike', 'N/A')} strike, "
                  f"${first_option.get('bid', 'N/A'):.2f} bid")
        
    except Exception as e:
        print(f"  ❌ Options data failed: {e}")
    
    print()
    
    # Test economic data access
    print("💰 Getting risk-free rate...")
    try:
        econ_response = await provider.get_economic_indicator("risk_free_rate")
        
        print(f"  Source: {econ_response.source}")
        print(f"  Quality: {econ_response.quality.value}")
        
        if hasattr(econ_response.data, 'iloc') and not econ_response.data.empty:
            rate = econ_response.data["value"].iloc[0]
            print(f"  Risk-free rate: {rate:.2%}")
        
    except Exception as e:
        print(f"  ❌ Economic data failed: {e}")
    
    print()


async def demonstrate_fallback_behavior():
    """Show automatic fallback behavior when providers fail."""
    print("🔄 Provider Fallback Demo")
    print("=" * 30)
    print()
    
    provider = get_unified_provider()
    
    print("🔍 Testing provider health...")
    health_results = await provider.health_check_all()
    
    for provider_name, health in health_results.items():
        status = health.get("status", "unknown")
        if status == "healthy":
            print(f"  ✅ {provider_name}: {status}")
        else:
            print(f"  ❌ {provider_name}: {status}")
            if "error" in health:
                print(f"     Error: {health['error']}")
    
    print()
    
    # Demonstrate fallback by requesting data
    print("📊 Testing fallback chain for market data...")
    
    # Show the fallback chain
    source_mapping = provider.get_performance_stats()["source_mapping"]
    market_providers = source_mapping.get("market_data", [])
    print(f"  Fallback chain: {' → '.join(market_providers)}")
    
    # Make request and see which provider responds
    try:
        request = DataRequest(
            source_type=DataSourceType.MARKET_DATA,
            symbol="U",
            start_time=datetime.now() - timedelta(days=7)
        )
        
        response = await provider.get_data(request)
        print(f"  ✅ Data retrieved from: {response.source}")
        print(f"  Data quality: {response.quality.value}")
        
    except Exception as e:
        print(f"  ❌ All providers failed: {e}")
    
    print()


async def demonstrate_convenience_functions():
    """Show convenience functions for common operations."""
    print("🛠️ Convenience Functions Demo")
    print("=" * 35)
    print()
    
    print("⚡ Using convenience functions for Unity wheel strategy...")
    
    # Unity market data
    print("📈 Unity market data (last 30 days):")
    try:
        market_data = await get_unity_market_data(days_back=30)
        
        if hasattr(market_data.data, 'shape') and not market_data.data.empty:
            print(f"  ✅ Retrieved {len(market_data.data)} days of data")
            print(f"  Latest close: ${market_data.data['close'].iloc[-1]:.2f}")
            
            # Calculate simple metrics
            returns = market_data.data['close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized
            print(f"  30-day annualized volatility: {volatility:.1%}")
        else:
            print("  📊 Using mock/simulated data")
            
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    print()
    
    # Unity options chain
    print("🎯 Unity options chain (45 DTE):")
    try:
        options_data = await get_unity_options_chain(expiration_days=45)
        
        if isinstance(options_data.data, list) and options_data.data:
            print(f"  ✅ Retrieved {len(options_data.data)} options")
            
            # Analyze options for wheel strategy
            put_candidates = []
            for option in options_data.data:
                strike = option.get('strike', 0)
                bid = option.get('bid', 0)
                
                if strike and bid and strike < 40:  # Focus on puts below $40
                    premium_yield = (bid / strike) * 100
                    put_candidates.append({
                        'strike': strike,
                        'bid': bid,
                        'premium_yield': premium_yield
                    })
            
            if put_candidates:
                # Sort by premium yield
                put_candidates.sort(key=lambda x: x['premium_yield'], reverse=True)
                best = put_candidates[0]
                print(f"  Best wheel candidate: ${best['strike']} strike")
                print(f"  Premium: ${best['bid']:.2f} ({best['premium_yield']:.2f}% yield)")
        else:
            print("  📊 Using mock/simulated options data")
            
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    print()
    
    # Risk-free rate
    print("💰 Current risk-free rate:")
    try:
        rate = await get_risk_free_rate()
        print(f"  ✅ Risk-free rate: {rate:.2%}")
        print(f"  Suitable for options pricing models")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    print()


async def demonstrate_custom_requests():
    """Show custom data requests with specific parameters."""
    print("🔧 Custom Data Requests Demo")
    print("=" * 35)
    print()
    
    provider = get_unified_provider()
    
    # Custom market data request
    print("📊 Custom market data request:")
    custom_request = DataRequest(
        source_type=DataSourceType.MARKET_DATA,
        symbol="U",
        start_time=datetime.now() - timedelta(days=7),
        end_time=datetime.now(),
        parameters={
            "frequency": "1H",  # Hourly data
            "fields": ["open", "high", "low", "close", "volume"]
        }
    )
    
    try:
        response = await provider.get_data(custom_request)
        print(f"  ✅ Data from: {response.source}")
        print(f"  Quality: {response.quality.value}")
        print(f"  Metadata: {response.metadata}")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    print()
    
    # Custom options request with filtering
    print("🎯 Custom options request with filtering:")
    options_request = DataRequest(
        source_type=DataSourceType.OPTIONS_DATA,
        symbol="U",
        parameters={
            "expiration": datetime.now() + timedelta(days=30),
            "min_strike": 30,
            "max_strike": 40,
            "option_type": "put",
            "min_volume": 100
        }
    )
    
    try:
        response = await provider.get_data(options_request)
        print(f"  ✅ Options from: {response.source}")
        print(f"  Quality: {response.quality.value}")
        
        if isinstance(response.data, list):
            filtered_count = len([
                opt for opt in response.data 
                if opt.get('volume', 0) >= 100
            ])
            print(f"  Options meeting volume criteria: {filtered_count}")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    print()


async def demonstrate_performance_monitoring():
    """Show performance monitoring and caching statistics."""
    print("📊 Performance Monitoring Demo")
    print("=" * 35)
    print()
    
    provider = get_unified_provider()
    
    # Get initial stats
    initial_stats = provider.get_performance_stats()
    print("📈 Initial performance statistics:")
    print(f"  Total requests: {initial_stats['total_requests']}")
    print(f"  Cache hits: {initial_stats['cache_hits']}")
    print(f"  Cache hit rate: {initial_stats['cache_hit_rate']:.1%}")
    print()
    
    # Make several requests to demonstrate caching
    print("🔄 Making repeated requests to test caching...")
    
    for i in range(3):
        print(f"  Request {i+1}:")
        
        start_time = asyncio.get_event_loop().time()
        
        # Make the same request multiple times
        market_data = await provider.get_market_data(
            symbol="U",
            start_time=datetime.now() - timedelta(days=7)
        )
        
        elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
        print(f"    Response time: {elapsed:.1f}ms")
        print(f"    Source: {market_data.source}")
        print(f"    Cached: {market_data.cached}")
    
    print()
    
    # Get final stats
    final_stats = provider.get_performance_stats()
    print("📊 Final performance statistics:")
    print(f"  Total requests: {final_stats['total_requests']}")
    print(f"  Cache hits: {final_stats['cache_hits']}")
    print(f"  Cache hit rate: {final_stats['cache_hit_rate']:.1%}")
    
    requests_made = final_stats['total_requests'] - initial_stats['total_requests']
    print(f"  Requests made in demo: {requests_made}")
    
    print()


async def demonstrate_error_handling():
    """Show robust error handling across providers."""
    print("🛡️ Error Handling Demo")
    print("=" * 25)
    print()
    
    provider = get_unified_provider()
    
    # Test with invalid symbol
    print("❌ Testing with invalid symbol:")
    try:
        response = await provider.get_market_data("INVALID_SYMBOL_XYZ")
        print(f"  ✅ Got data from: {response.source} (fallback to mock)")
        
    except Exception as e:
        print(f"  ❌ All providers failed: {e}")
    
    print()
    
    # Test with unsupported data type
    print("❌ Testing unsupported data source:")
    try:
        request = DataRequest(
            source_type=DataSourceType.FUNDAMENTAL_DATA,  # Not widely supported
            symbol="U"
        )
        response = await provider.get_data(request)
        print(f"  ✅ Got data from: {response.source}")
        
    except Exception as e:
        print(f"  ❌ Expected failure: {e}")
    
    print()
    
    # Test cache invalidation
    print("🗑️ Testing cache invalidation:")
    await provider.invalidate_cache(
        source_type=DataSourceType.MARKET_DATA,
        symbol="U"
    )
    print("  ✅ Cache invalidation requested")
    
    print()


async def main():
    """Run the complete unified data provider demonstration."""
    print("🌐 Unity Wheel Trading Bot - Unified Data Provider Demo")
    print("=" * 65)
    print()
    print("This demo shows how the unified data provider consolidates")
    print("access to all market data sources with automatic fallback,")
    print("caching, and error handling.")
    print()
    
    # Run all demonstrations
    await demonstrate_basic_data_access()
    await demonstrate_fallback_behavior()
    await demonstrate_convenience_functions()
    await demonstrate_custom_requests()
    await demonstrate_performance_monitoring()
    await demonstrate_error_handling()
    
    print("🎉 Unified Data Provider Benefits:")
    print("  ✅ Single interface for all data sources")
    print("  ✅ Automatic fallback when providers fail")
    print("  ✅ Built-in caching for performance")
    print("  ✅ Consistent error handling")
    print("  ✅ Health monitoring and diagnostics")
    print("  ✅ Performance tracking and statistics")
    print("  ✅ Extensible provider architecture")
    print()
    
    print("💡 Perfect for Unity Wheel Strategy:")
    print("  📊 Seamless access to market data from multiple sources")
    print("  🎯 Unified options chain retrieval and analysis")
    print("  💰 Economic indicators for risk-free rate modeling")
    print("  🔄 Automatic failover ensures data availability")
    print("  ⚡ Caching optimizes performance for repeated requests")
    print("  🛡️ Robust error handling prevents strategy failures")
    print()
    
    print("🚀 The unified data provider is ready for production use!")


if __name__ == "__main__":
    asyncio.run(main())