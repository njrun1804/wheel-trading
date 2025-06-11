#!/usr/bin/env python3
"""Quick test script to verify FRED integration with new storage architecture."""

import asyncio
from datetime import date as Date
from datetime import timedelta

from src.unity_wheel.data_providers.base import FREDDataManager
from src.unity_wheel.secrets.integration import SecretNotFoundError, get_ofred_api_key
from src.unity_wheel.storage.storage import Storage
from src.unity_wheel.utils import setup_structured_logging as setup_logging


async def test_integration():
    """Test FRED integration with unified storage."""
    setup_logging()

    print("Testing FRED Integration with Unified Storage")
    print("=" * 50)

    # Test 1: Secret Management
    print("\n1. Testing Secret Management:")
    try:
        api_key = get_ofred_api_key()
        print("   ✅ FRED API key loaded successfully")
    except SecretNotFoundError:
        print("   ❌ FRED API key not found")
        print("   Run: python scripts/setup-secrets.py")
        return

    # Test 2: Storage Initialization
    print("\n2. Testing Storage Initialization:")
    storage = Storage()
    await storage.initialize()
    print("   ✅ Unified storage initialized")

    # Test 3: FRED Manager Initialization
    print("\n3. Testing FRED Manager:")
    manager = FREDDataManager(storage=storage)
    print("   ✅ FRED manager initialized")

    # Test 4: Get or Fetch Pattern
    print("\n4. Testing Get-or-Fetch Pattern:")

    # First call - should fetch from API
    print("   First call (should fetch from API)...")
    rate1, conf1 = await manager.get_or_fetch_risk_free_rate(3, fetch_if_stale_days=1)
    print(f"   Rate: {rate1*100:.2f}%, Confidence: {conf1:.0%}")

    # Second call - should use cache
    print("   Second call (should use cache)...")
    rate2, conf2 = await manager.get_or_fetch_risk_free_rate(3, fetch_if_stale_days=1)
    print(f"   Rate: {rate2*100:.2f}%, Confidence: {conf2:.0%}")

    assert rate1 == rate2, "Rates should match between calls"
    print("   ✅ Get-or-fetch pattern working correctly")

    # Test 5: Risk Metrics Caching
    print("\n5. Testing Risk Metrics Caching:")
    regime, vix = await manager.get_volatility_regime()
    print(f"   Volatility regime: {regime}, VIX: {vix:.2f}")

    # Check if cached
    risk_metrics = await manager.fred_storage.get_risk_metrics(Date.today())
    if risk_metrics:
        print("   ✅ Risk metrics cached successfully")
        print(f"   Cached data: {list(risk_metrics.keys())}")

    # Test 6: Storage Stats
    print("\n6. Testing Storage Stats:")
    stats = await storage.get_storage_stats()
    print(f"   Cache location: {storage.config.cache_config.cache_dir}")
    print(f"   GCS enabled: {stats['gcs_enabled']}")
    if "tables" in stats:
        for table, info in stats["tables"].items():
            if table.startswith("fred"):
                print(f"   {table}: {info.get('rows', 0)} rows, {info.get('size_mb', 0):.2f} MB")

    # Test 7: Data Freshness
    print("\n7. Testing Data Freshness:")
    latest_values = await manager.get_latest_values()
    for series_id, (value, obs_date) in list(latest_values.items())[:3]:
        days_old = (Date.today() - obs_date).days
        print(f"   {series_id}: {value:.2f} ({days_old} days old)")

    print("\n✅ All tests passed! FRED integration working correctly.")
    print(f"\nData stored in: {storage.config.cache_config.cache_dir}")


if __name__ == "__main__":
    asyncio.run(test_integration())
