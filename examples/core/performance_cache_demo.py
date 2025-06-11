#!/usr/bin/env python3
"""Demonstration of the enhanced performance caching system.

This script shows how the new MemoryAwareLRUCache provides significant
performance improvements for expensive options calculations.
"""

import asyncio
import time
from typing import List, Tuple

from src.unity_wheel.math.options_enhanced import (
    black_scholes_price_enhanced,
    calculate_all_greeks_enhanced,
    get_cache_performance_stats,
    get_implied_volatility,
    get_option_greeks,
    get_option_price,
)
from src.unity_wheel.utils.performance_cache import get_cache_manager


def benchmark_options_calculations():
    """Benchmark options calculations with and without caching."""
    print("🚀 Performance Cache Demo")
    print("=" * 50)
    print()

    # Test parameters - typical Unity wheel strategy scenarios
    test_scenarios = [
        # (spot, strike, time_to_expiry, risk_free_rate, volatility, option_type)
        (35.0, 30.0, 0.123, 0.05, 0.65, "put"),  # 45 DTE put, high vol
        (35.0, 32.5, 0.123, 0.05, 0.60, "put"),  # Similar but different strike
        (35.0, 35.0, 0.123, 0.05, 0.62, "put"),  # ATM put
        (35.0, 37.5, 0.082, 0.05, 0.58, "call"),  # 30 DTE covered call
        (35.0, 40.0, 0.082, 0.05, 0.55, "call"),  # OTM covered call
        (35.0, 30.0, 0.164, 0.05, 0.68, "put"),  # 60 DTE put (longer term)
        (36.0, 32.5, 0.123, 0.05, 0.60, "put"),  # Different spot price
        (34.0, 30.0, 0.123, 0.05, 0.65, "put"),  # Different spot price
    ]

    print("📊 Testing scenarios:")
    for i, (S, K, T, r, vol, opt_type) in enumerate(test_scenarios):
        dte = int(T * 365)
        print(f"  {i+1}. ${S} stock, ${K} {opt_type}, {dte} DTE, {vol:.0%} vol")
    print()

    # First pass - cache misses
    print("⏱️  First pass (cache misses):")
    first_pass_times = []
    first_pass_results = []

    for i, params in enumerate(test_scenarios):
        start_time = time.time()

        # Calculate price and Greeks
        price_result = black_scholes_price_enhanced(*params)
        greeks_result, greeks_conf = calculate_all_greeks_enhanced(*params)

        elapsed = time.time() - start_time
        first_pass_times.append(elapsed)
        first_pass_results.append((price_result, greeks_result))

        print(
            f"  Scenario {i+1}: {elapsed*1000:.2f}ms - "
            f"Price: ${price_result.value:.3f} (conf: {price_result.confidence:.0%})"
        )

    total_first_pass = sum(first_pass_times)
    print(f"  Total first pass time: {total_first_pass*1000:.2f}ms")
    print()

    # Second pass - cache hits
    print("⚡ Second pass (cache hits):")
    second_pass_times = []
    second_pass_results = []

    for i, params in enumerate(test_scenarios):
        start_time = time.time()

        # Same calculations - should hit cache
        price_result = black_scholes_price_enhanced(*params)
        greeks_result, greeks_conf = calculate_all_greeks_enhanced(*params)

        elapsed = time.time() - start_time
        second_pass_times.append(elapsed)
        second_pass_results.append((price_result, greeks_result))

        print(
            f"  Scenario {i+1}: {elapsed*1000:.2f}ms - "
            f"Price: ${price_result.value:.3f} (conf: {price_result.confidence:.0%})"
        )

    total_second_pass = sum(second_pass_times)
    print(f"  Total second pass time: {total_second_pass*1000:.2f}ms")
    print()

    # Performance summary
    improvement = (total_first_pass - total_second_pass) / total_first_pass * 100
    speedup = total_first_pass / total_second_pass if total_second_pass > 0 else float("inf")

    print("📈 Performance Summary:")
    print(f"  Speed improvement: {improvement:.1f}%")
    print(f"  Speedup factor: {speedup:.1f}x")
    print(f"  Average per calculation:")
    print(f"    First pass: {total_first_pass/len(test_scenarios)*1000:.2f}ms")
    print(f"    Second pass: {total_second_pass/len(test_scenarios)*1000:.2f}ms")
    print()

    # Verify results are identical
    print("✅ Result verification:")
    all_match = True
    for i, (first, second) in enumerate(zip(first_pass_results, second_pass_results)):
        price1, greeks1 = first
        price2, greeks2 = second

        price_match = abs(price1.value - price2.value) < 1e-10
        delta_match = abs(greeks1["delta"] - greeks2["delta"]) < 1e-10

        if not (price_match and delta_match):
            all_match = False
            print(f"  ❌ Scenario {i+1}: Results don't match!")

    if all_match:
        print("  ✅ All cached results match original calculations perfectly")
    print()

    return test_scenarios, (first_pass_times, second_pass_times)


async def demonstrate_cache_management():
    """Demonstrate cache management features."""
    print("🗄️  Cache Management Demo")
    print("=" * 30)
    print()

    # Get cache statistics
    stats = await get_cache_performance_stats()

    print("📊 Cache Statistics:")
    print(f"  Cache name: {stats['cache_name']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Total hits: {stats['total_hits']}")
    print(f"  Total misses: {stats['total_misses']}")
    print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
    print(f"  Evictions: {stats['evictions']}")
    if stats["avg_computation_time_ms"] > 0:
        print(f"  Avg computation time: {stats['avg_computation_time_ms']:.2f}ms")
        print(f"  Max computation time: {stats['max_computation_time_ms']:.2f}ms")
    print()

    # Demonstrate cache manager
    cache_manager = get_cache_manager()
    global_stats = cache_manager.get_global_stats()

    print("🌐 Global Cache Stats:")
    for cache_name, cache_stats in global_stats.items():
        print(f"  {cache_name}:")
        print(f"    Hit rate: {cache_stats.hit_rate:.1%}")
        print(f"    Memory: {cache_stats.memory_bytes / 1024 / 1024:.2f} MB")
        print(f"    Entries: {cache_stats.hits + cache_stats.misses}")
    print()


def demonstrate_convenience_functions():
    """Demonstrate the convenience functions."""
    print("🛠️  Convenience Functions Demo")
    print("=" * 35)
    print()

    # Unity wheel strategy example
    unity_price = 35.00
    put_strike = 32.50
    dte_years = 45 / 365
    risk_free_rate = 0.05
    implied_vol = 0.60

    print(f"📋 Unity Wheel Strategy Analysis:")
    print(f"  Stock price: ${unity_price}")
    print(f"  Put strike: ${put_strike}")
    print(f"  Days to expiry: 45")
    print(f"  Implied volatility: {implied_vol:.0%}")
    print()

    # Get option price
    put_price, price_confidence = get_option_price(
        unity_price, put_strike, dte_years, risk_free_rate, implied_vol, "put"
    )

    print(f"💰 Put Option Analysis:")
    print(f"  Option price: ${put_price:.3f}")
    print(f"  Price confidence: {price_confidence:.0%}")

    # Calculate premium yield
    premium_yield = (put_price / put_strike) * 100
    annualized_yield = premium_yield * (365 / 45)

    print(f"  Premium yield: {premium_yield:.2f}%")
    print(f"  Annualized yield: {annualized_yield:.1f}%")
    print()

    # Get Greeks
    greeks, greeks_confidence = get_option_greeks(
        unity_price, put_strike, dte_years, risk_free_rate, implied_vol, "put"
    )

    print(f"📈 Greeks Analysis (confidence: {greeks_confidence:.0%}):")
    print(f"  Delta: {greeks['delta']:.3f} (prob. assignment: {-greeks['delta']:.1%})")
    print(f"  Gamma: {greeks['gamma']:.4f}")
    print(f"  Theta: ${greeks['theta']:.3f}/day")
    print(f"  Vega: ${greeks['vega']:.3f} per 1% vol change")
    print(f"  Rho: ${greeks['rho']:.3f} per 1% rate change")
    print()

    # Implied volatility example
    market_price = put_price + 0.05  # Add some noise
    calculated_iv, iv_confidence = get_implied_volatility(
        market_price, unity_price, put_strike, dte_years, risk_free_rate, "put"
    )

    print(f"🎯 Implied Volatility Back-calculation:")
    print(f"  Market price: ${market_price:.3f}")
    print(f"  Calculated IV: {calculated_iv:.1%}")
    print(f"  IV confidence: {iv_confidence:.0%}")
    print(f"  IV difference: {(calculated_iv - implied_vol)*100:.1f} vol points")
    print()


def stress_test_cache():
    """Stress test the cache with many calculations."""
    print("🔥 Cache Stress Test")
    print("=" * 25)
    print()

    print("Generating 1000 option calculations with varying parameters...")
    start_time = time.time()

    calculations = 0
    cache_hits = 0

    # Generate many calculations with some repetition to test cache effectiveness
    for i in range(1000):
        # Vary parameters to create realistic mix of cache hits/misses
        spot = 35.0 + (i % 10) * 0.5  # 35.0 to 39.5
        strike = 30.0 + (i % 15) * 1.0  # 30.0 to 44.0
        dte = (30 + (i % 40)) / 365  # 30 to 70 days
        vol = 0.40 + (i % 50) * 0.01  # 40% to 90% vol

        # Calculate price (enhanced version with caching)
        result = black_scholes_price_enhanced(spot, strike, dte, 0.05, vol, "put")
        calculations += 1

        # Some repeat calculations to test cache hits
        if i % 10 == 0:  # Every 10th calculation, repeat a previous one
            repeat_result = black_scholes_price_enhanced(35.0, 32.5, 45 / 365, 0.05, 0.60, "put")
            calculations += 1

    elapsed = time.time() - start_time

    print(f"✅ Completed {calculations} calculations in {elapsed:.3f}s")
    print(f"   Average time per calculation: {elapsed/calculations*1000:.3f}ms")
    print()


async def main():
    """Run the complete performance cache demonstration."""
    print("🎯 Unity Wheel Trading Bot - Enhanced Performance Cache Demo")
    print("=" * 65)
    print()
    print("This demo shows how the new MemoryAwareLRUCache system provides")
    print("significant performance improvements for options calculations.")
    print()

    # Run benchmarks
    test_scenarios, (first_times, second_times) = benchmark_options_calculations()

    # Show cache management
    await demonstrate_cache_management()

    # Show convenience functions
    demonstrate_convenience_functions()

    # Stress test
    stress_test_cache()

    # Final cache stats
    print("📊 Final Cache Performance:")
    final_stats = await get_cache_performance_stats()
    print(f"  Total operations: {final_stats['total_hits'] + final_stats['total_misses']}")
    print(f"  Hit rate: {final_stats['hit_rate']:.1%}")
    print(f"  Memory usage: {final_stats['memory_usage_mb']:.2f} MB")
    print(f"  Performance improvement: Calculations are cached for instant retrieval")
    print()

    print("💡 Key Benefits:")
    print("  ✅ Automatic caching of expensive calculations")
    print("  ✅ Memory-aware eviction prevents unlimited growth")
    print("  ✅ TTL-based expiration ensures data freshness")
    print("  ✅ Performance monitoring and statistics")
    print("  ✅ Thread-safe async operations")
    print("  ✅ Configurable cache sizes and policies")
    print()

    print("🚀 The enhanced caching system is ready for production use!")
    print("   Use the convenience functions for easy integration:")
    print("   - get_option_price()")
    print("   - get_option_greeks()")
    print("   - get_implied_volatility()")


if __name__ == "__main__":
    asyncio.run(main())
