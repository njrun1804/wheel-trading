#!/usr/bin/env python3
"""Efficient FRED data pull with progress tracking."""

import asyncio
import time
from datetime import date as Date, timedelta

from src.unity_wheel.storage.storage import Storage
from src.unity_wheel.data import FREDDataManager, WheelStrategyFREDSeries


async def pull_data_efficient():
    """Pull FRED data efficiently with progress tracking."""
    print("FRED Data Pull - Efficient Version")
    print("=" * 50)
    
    start_time = time.time()
    
    # Initialize storage
    print("\n1. Initializing storage...")
    storage = Storage()
    await storage.initialize()
    print("   ✓ Storage initialized")
    
    # Initialize FRED manager
    print("\n2. Setting up FRED manager...")
    manager = FREDDataManager(storage=storage)
    print("   ✓ Manager initialized")
    
    # We'll pull data for different periods based on series update frequency
    print("\n3. Determining data requirements...")
    
    # Daily series need 5 years
    daily_series = ["DGS3", "DGS1", "DFF", "VIXCLS", "VXDCLS", "TEDRATE", "BAMLH0A0HYM2"]
    # Monthly series need less data 
    monthly_series = ["UNRATE", "CPIAUCSL"]
    
    print(f"   - Daily series (5 years): {len(daily_series)} series")
    print(f"   - Monthly series (5 years): {len(monthly_series)} series")
    
    # Calculate expected data points
    daily_points = len(daily_series) * (5 * 252)  # ~252 trading days per year
    monthly_points = len(monthly_series) * (5 * 12)  # 12 months per year
    total_expected = daily_points + monthly_points
    
    print(f"\n   Expected data points: ~{total_expected:,}")
    print(f"   Estimated storage: ~{total_expected * 100 / 1024 / 1024:.1f} MB")
    
    print("\n4. Pulling data (this will take 1-2 minutes)...")
    
    try:
        # Pull all data
        counts = await manager.initialize_data(lookback_days=1825)  # 5 years
        
        # Calculate actual data pulled
        total_obs = sum(counts.values())
        elapsed = time.time() - start_time
        
        print(f"\n✅ Data pull complete in {elapsed:.1f} seconds!")
        print(f"\nData pulled:")
        print(f"{'Series':<15} {'Count':>8} {'Description'}")
        print("-" * 60)
        
        for series_id, count in sorted(counts.items()):
            series_enum = WheelStrategyFREDSeries(series_id)
            print(f"{series_id:<15} {count:>8,}  {series_enum.description}")
        
        print(f"\nTotal observations: {total_obs:,}")
        print(f"Pull rate: {total_obs/elapsed:.0f} observations/second")
        
        # Test retrieval speed
        print("\n5. Testing data retrieval...")
        test_start = time.time()
        
        # Get latest values
        latest = await manager.get_latest_values()
        
        # Get risk-free rate (tests caching)
        rf_rate, _ = await manager.get_risk_free_rate(3)
        
        # Get volatility regime
        regime, vix = await manager.get_volatility_regime()
        
        test_elapsed = (time.time() - test_start) * 1000
        
        print(f"   ✓ Retrieved latest values in {test_elapsed:.1f}ms")
        print(f"\nCurrent market data:")
        print(f"   3-month risk-free rate: {rf_rate*100:.2f}%")
        print(f"   VIX: {vix:.2f} ({regime} volatility)")
        
        # Show storage location
        print(f"\n6. Storage location:")
        print(f"   {storage.config.cache_config.cache_dir}")
        
        # Get storage stats
        stats = await storage.get_storage_stats()
        if 'db_size_mb' in stats:
            print(f"   Database size: {stats['db_size_mb']:.1f} MB")
        
        print("\n✅ All data successfully pulled and cached!")
        print("\nRecommendations:")
        print("   - Daily series will auto-update when accessed after 24 hours")
        print("   - Risk-free rates cached for fast Black-Scholes calculations")
        print("   - Ready for wheel strategy analysis")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Disable verbose logging for cleaner output
    import logging
    logging.getLogger("src.unity_wheel").setLevel(logging.WARNING)
    
    asyncio.run(pull_data_efficient())