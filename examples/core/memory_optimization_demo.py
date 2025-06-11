#!/usr/bin/env python3
"""Demonstration of memory optimization for Unity Wheel Trading Bot.

This script shows how the memory optimization tools can handle large-scale
market data processing while keeping memory usage within reasonable bounds.
"""

import time
from typing import Iterator, List

import numpy as np

from src.unity_wheel.math.vectorized_options import vectorized_black_scholes, vectorized_greeks
from src.unity_wheel.utils.memory_optimizer import (
    MemoryEfficientArray,
    MemoryPool,
    StreamingDataProcessor,
    create_memory_efficient_dict,
    get_memory_monitor,
    get_memory_summary,
    memory_profiler,
    optimize_numpy_memory,
    set_memory_thresholds,
)


def generate_market_data_stream(num_days: int = 1000) -> Iterator[dict]:
    """Generate simulated market data stream."""
    base_price = 35.0
    for day in range(num_days):
        # Simulate daily price movement
        price_change = np.random.normal(0, 0.02)  # 2% daily volatility
        current_price = base_price * (1 + price_change)

        # Generate options data for this day
        yield {
            "day": day,
            "underlying_price": current_price,
            "timestamp": f"2024-01-{(day % 30) + 1:02d}",
            "volatility": 0.60 + np.random.normal(0, 0.05),  # IV around 60%
            "strikes": list(np.arange(current_price * 0.8, current_price * 1.2, 2.5)),
            "expirations": [30, 45, 60, 75],  # Days to expiry
        }


def demonstrate_memory_monitoring():
    """Show memory monitoring capabilities."""
    print("📊 Memory Monitoring Demo")
    print("=" * 30)
    print()

    # Set reasonable thresholds for demo
    set_memory_thresholds(warning_mb=150.0, critical_mb=300.0)

    monitor = get_memory_monitor()

    # Show initial memory state
    initial_stats = monitor.get_memory_stats()
    print(f"🔍 Initial Memory State:")
    print(f"  {initial_stats}")
    print()

    # Track some objects
    print("📈 Creating and tracking objects...")

    large_arrays = []
    for i in range(5):
        # Create progressively larger arrays
        size = 1000 * (i + 1)
        array = np.random.random(size).astype(np.float64)
        large_arrays.append(array)
        monitor.track_object(array)

        current_stats = monitor.get_memory_stats()
        print(f"  Array {i+1}: {size:,} elements, " f"Memory: {current_stats.process_mb:.1f}MB")

    print()

    # Show object tracking stats
    object_stats = monitor.get_object_stats()
    print("🔍 Object Tracking Stats:")
    for obj_type, stats in object_stats.items():
        print(f"  {obj_type}: {stats.count} objects, " f"{stats.total_size_mb:.2f}MB total")

    print()

    # Memory optimization
    print("⚡ Optimizing memory usage...")

    optimized_arrays = []
    total_saved = 0.0

    for i, array in enumerate(large_arrays):
        original_size = array.nbytes / (1024 * 1024)
        optimized = optimize_numpy_memory(array)
        optimized_size = optimized.nbytes / (1024 * 1024)
        saved = original_size - optimized_size
        total_saved += saved

        optimized_arrays.append(optimized)

        print(
            f"  Array {i+1}: {original_size:.2f}MB → {optimized_size:.2f}MB "
            f"(saved {saved:.2f}MB)"
        )

    print(f"  Total memory saved: {total_saved:.2f}MB")
    print()

    # Test memory cleanup
    print("🧹 Testing memory cleanup...")
    cleanup_triggered, reason = monitor.check_memory_usage(force_cleanup=True)

    if cleanup_triggered:
        print(f"  ✅ Cleanup triggered: {reason}")
        final_stats = monitor.get_memory_stats()
        print(f"  Memory after cleanup: {final_stats.process_mb:.1f}MB")

    print()


def demonstrate_memory_efficient_arrays():
    """Show memory-efficient array usage for price storage."""
    print("📊 Memory-Efficient Arrays Demo")
    print("=" * 35)
    print()

    # Create memory-efficient arrays for different data types
    price_history = MemoryEfficientArray(
        initial_capacity=1000, max_size=10000, dtype=np.float32  # Use float32 for price data
    )

    volume_history = MemoryEfficientArray(
        initial_capacity=1000, max_size=10000, dtype=np.int32  # Use int32 for volume data
    )

    print("📈 Simulating market data collection...")

    with memory_profiler("market_data_collection"):
        # Simulate collecting market data over time
        for day in range(5000):  # 5000 days of data
            # Daily price (Unity stock around $35)
            price = 35.0 + np.random.normal(0, 2.0)
            price_history.append(price)

            # Daily volume
            volume = int(np.random.exponential(1000000))  # Average 1M volume
            volume_history.append(volume)

            # Occasionally add batch data (e.g., intraday prices)
            if day % 100 == 0:
                intraday_prices = 35.0 + np.random.normal(0, 0.5, 50)  # 50 intraday points
                price_history.append(intraday_prices)

                print(
                    f"  Day {day}: Added {len(intraday_prices)} intraday prices, "
                    f"Total prices: {len(price_history):,}"
                )

    print()
    print(f"📊 Final Statistics:")
    print(f"  Price data points: {len(price_history):,}")
    print(f"  Volume data points: {len(volume_history):,}")
    print(f"  Price array capacity: {price_history._capacity:,}")
    print(f"  Estimated price array size: {price_history._estimate_size():.2f}MB")
    print(f"  Estimated volume array size: {volume_history._estimate_size():.2f}MB")

    # Demonstrate data access
    recent_prices = price_history.get_data()[-100:]  # Last 100 prices
    avg_recent_price = np.mean(recent_prices)
    print(f"  Average recent price: ${avg_recent_price:.2f}")

    print()


def demonstrate_streaming_processing():
    """Show streaming data processing for large datasets."""
    print("🌊 Streaming Data Processing Demo")
    print("=" * 40)
    print()

    processor = StreamingDataProcessor(chunk_size=100)

    print("📊 Processing large options dataset in chunks...")

    # Simulate processing a year of market data
    market_data_stream = generate_market_data_stream(num_days=365)

    def process_day_data(day_chunk: List[dict]) -> dict:
        """Process a chunk of daily market data."""
        total_calculations = 0
        total_premium = 0.0

        for day_data in day_chunk:
            spot = day_data["underlying_price"]
            vol = day_data["volatility"]
            strikes = day_data["strikes"]

            # Calculate option prices for all strikes and expirations
            for dte in day_data["expirations"]:
                dte_years = dte / 365

                # Vectorized calculation for all strikes
                result = vectorized_black_scholes(spot, strikes, dte_years, 0.05, vol, "put")

                total_calculations += len(strikes)
                total_premium += np.sum(result.values)

        return {
            "days_processed": len(day_chunk),
            "total_calculations": total_calculations,
            "total_premium": total_premium,
            "avg_premium": total_premium / total_calculations if total_calculations > 0 else 0,
        }

    def accumulate_results(accumulated: dict, chunk_result: dict) -> dict:
        """Accumulate results from multiple chunks."""
        if accumulated is None:
            return chunk_result.copy()

        return {
            "days_processed": accumulated["days_processed"] + chunk_result["days_processed"],
            "total_calculations": accumulated["total_calculations"]
            + chunk_result["total_calculations"],
            "total_premium": accumulated["total_premium"] + chunk_result["total_premium"],
            "avg_premium": 0,  # Will recalculate at the end
        }

    # Process the stream
    start_time = time.time()

    with memory_profiler("streaming_options_processing"):
        results = list(
            processor.process_chunks(
                data_source=market_data_stream,
                processor_func=process_day_data,
                accumulator_func=accumulate_results,
            )
        )

    processing_time = time.time() - start_time

    # Extract final result
    final_result = results[0] if results else {}

    if final_result.get("total_calculations", 0) > 0:
        final_result["avg_premium"] = (
            final_result["total_premium"] / final_result["total_calculations"]
        )

    print(f"⚡ Processing Results:")
    print(f"  Days processed: {final_result.get('days_processed', 0):,}")
    print(f"  Total calculations: {final_result.get('total_calculations', 0):,}")
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Rate: {final_result.get('total_calculations', 0) / processing_time:,.0f} calc/sec")
    print(f"  Average premium: ${final_result.get('avg_premium', 0):.3f}")
    print()


def demonstrate_memory_pools():
    """Show memory pool usage for object reuse."""
    print("🔄 Memory Pool Demo")
    print("=" * 25)
    print()

    def create_options_calculation_context():
        """Factory for options calculation context."""
        return {
            "prices": np.empty(100, dtype=np.float32),
            "greeks": {
                "delta": np.empty(100, dtype=np.float32),
                "gamma": np.empty(100, dtype=np.float32),
                "theta": np.empty(100, dtype=np.float32),
                "vega": np.empty(100, dtype=np.float32),
            },
            "metadata": {},
        }

    def reset_calculation_context(context):
        """Reset context for reuse."""
        context["prices"].fill(0)
        for greek_array in context["greeks"].values():
            greek_array.fill(0)
        context["metadata"].clear()

    # Create memory pool for calculation contexts
    calculation_pool = MemoryPool(
        factory_func=create_options_calculation_context,
        max_size=20,
        reset_func=reset_calculation_context,
    )

    print("⚡ Simulating repeated options calculations with object reuse...")

    # Simulate many calculation cycles
    with memory_profiler("memory_pool_calculations"):
        for cycle in range(500):
            # Get context from pool
            context = calculation_pool.get()

            # Simulate options calculation
            spot = 35.0 + np.random.normal(0, 1.0)
            strikes = np.linspace(spot * 0.9, spot * 1.1, 100)

            # Fill in calculated values (simulated)
            context["prices"][:] = np.random.uniform(0.5, 5.0, 100)
            context["greeks"]["delta"][:] = np.random.uniform(-1, 1, 100)
            context["greeks"]["gamma"][:] = np.random.uniform(0, 0.1, 100)
            context["greeks"]["theta"][:] = np.random.uniform(-0.5, 0, 100)
            context["greeks"]["vega"][:] = np.random.uniform(0, 2.0, 100)

            context["metadata"]["spot"] = spot
            context["metadata"]["calculation_cycle"] = cycle

            # Simulate using the results (e.g., finding best strike)
            best_strike_idx = np.argmax(context["prices"])
            best_price = context["prices"][best_strike_idx]

            # Return context to pool for reuse
            calculation_pool.put(context)

            if cycle % 100 == 0:
                print(f"  Cycle {cycle}: Best price ${best_price:.3f}")

    # Show pool statistics
    pool_stats = calculation_pool.get_stats()
    print()
    print(f"🔍 Pool Statistics:")
    print(f"  Objects created: {pool_stats['created_count']}")
    print(f"  Objects reused: {pool_stats['reused_count']}")
    print(f"  Reuse rate: {pool_stats['reuse_rate']:.1%}")
    print(f"  Pool size: {pool_stats['pool_size']}/{pool_stats['max_size']}")
    print()


def demonstrate_memory_efficient_caching():
    """Show memory-efficient caching for options data."""
    print("💾 Memory-Efficient Caching Demo")
    print("=" * 35)
    print()

    # Create memory-efficient cache for options calculations
    options_cache = create_memory_efficient_dict(max_size=1000)

    print("📊 Building options cache with automatic eviction...")

    cache_hits = 0
    cache_misses = 0

    with memory_profiler("options_caching"):
        # Simulate caching options calculations
        for i in range(2000):  # More than cache size to trigger eviction
            # Generate cache key
            spot = 35.0 + (i % 20) * 0.5  # Vary spot price
            strike = 30.0 + (i % 30) * 1.0  # Vary strike
            vol = 0.5 + (i % 25) * 0.02  # Vary volatility

            cache_key = f"bs_{spot:.1f}_{strike:.1f}_{vol:.2f}"

            # Check cache first
            if cache_key in options_cache:
                cached_result = options_cache[cache_key]
                cache_hits += 1

                if i % 500 == 0:
                    print(f"  Cache hit {i}: {cache_key} → ${cached_result:.3f}")
            else:
                # Calculate and cache
                result = vectorized_black_scholes(spot, strike, 45 / 365, 0.05, vol, "put")

                price = result.values.item() if hasattr(result.values, "item") else result.values
                options_cache[cache_key] = price
                cache_misses += 1

                if i % 500 == 0:
                    print(f"  Cache miss {i}: {cache_key} → ${price:.3f} (calculated)")

    hit_rate = cache_hits / (cache_hits + cache_misses) * 100

    print()
    print(f"📈 Cache Performance:")
    print(f"  Cache hits: {cache_hits:,}")
    print(f"  Cache misses: {cache_misses:,}")
    print(f"  Hit rate: {hit_rate:.1f}%")
    print(f"  Final cache size: {len(options_cache)}")
    print()


def demonstrate_comprehensive_memory_management():
    """Show comprehensive memory management across all systems."""
    print("🎯 Comprehensive Memory Management Demo")
    print("=" * 45)
    print()

    print("🏁 Running all memory optimization systems together...")
    print()

    # Initialize all systems
    price_array = MemoryEfficientArray(max_size=5000, dtype=np.float32)
    processor = StreamingDataProcessor(chunk_size=50)
    calculation_pool = MemoryPool(
        factory_func=lambda: {"temp_data": np.empty(50)},
        max_size=10,
        reset_func=lambda ctx: ctx["temp_data"].fill(0),
    )
    results_cache = create_memory_efficient_dict(max_size=500)

    # Monitor memory throughout
    monitor = get_memory_monitor()
    initial_stats = monitor.get_memory_stats()

    print(f"📊 Starting memory: {initial_stats.process_mb:.1f}MB")

    # Simulate complex workflow
    with memory_profiler("comprehensive_workflow"):
        market_stream = generate_market_data_stream(num_days=200)

        def complex_processing_func(day_chunk: List[dict]) -> dict:
            """Complex processing using all memory optimization features."""
            chunk_results = []

            for day_data in day_chunk:
                # Get reusable calculation context
                calc_context = calculation_pool.get()

                spot = day_data["underlying_price"]
                vol = day_data["volatility"]

                # Store price in efficient array
                price_array.append(spot)

                # Process options with caching
                for strike in day_data["strikes"][:10]:  # Limit to first 10 strikes
                    cache_key = f"{spot:.1f}_{strike:.1f}_{vol:.2f}"

                    if cache_key in results_cache:
                        option_price = results_cache[cache_key]
                    else:
                        # Calculate with vectorized operations
                        result = vectorized_black_scholes(spot, strike, 45 / 365, 0.05, vol, "put")
                        option_price = float(result.values)
                        results_cache[cache_key] = option_price

                    chunk_results.append(option_price)

                # Return context to pool
                calculation_pool.put(calc_context)

            return {
                "processed_days": len(day_chunk),
                "avg_option_price": np.mean(chunk_results) if chunk_results else 0,
                "price_count": len(price_array),
            }

        # Process stream
        results = list(
            processor.process_chunks(
                data_source=market_stream, processor_func=complex_processing_func
            )
        )

    # Final statistics
    final_stats = monitor.get_memory_stats()
    memory_summary = get_memory_summary()

    print()
    print(f"📊 Final Results:")
    print(f"  Starting memory: {initial_stats.process_mb:.1f}MB")
    print(f"  Final memory: {final_stats.process_mb:.1f}MB")
    print(f"  Memory change: {final_stats.process_mb - initial_stats.process_mb:+.1f}MB")
    print(f"  Peak memory: {memory_summary['peak_memory_mb']:.1f}MB")
    print()

    print(f"  Price array size: {len(price_array):,} points")
    print(f"  Cache entries: {len(results_cache)}")
    print(f"  Pool reuse rate: {calculation_pool.get_stats()['reuse_rate']:.1%}")
    print(f"  Processed chunks: {len(results)}")
    print()


def main():
    """Run the complete memory optimization demonstration."""
    print("🧠 Unity Wheel Trading Bot - Memory Optimization Demo")
    print("=" * 60)
    print()
    print("This demo shows how memory optimization tools enable processing")
    print("large-scale market data while keeping memory usage efficient.")
    print()

    # Run all demonstrations
    demonstrate_memory_monitoring()
    demonstrate_memory_efficient_arrays()
    demonstrate_streaming_processing()
    demonstrate_memory_pools()
    demonstrate_memory_efficient_caching()
    demonstrate_comprehensive_memory_management()

    # Final memory summary
    final_summary = get_memory_summary()

    print("🎉 Memory Optimization Benefits:")
    print("  ✅ Automatic memory monitoring and cleanup")
    print("  ✅ Memory-efficient arrays with automatic recycling")
    print("  ✅ Streaming processing for unlimited dataset sizes")
    print("  ✅ Object pooling to reduce allocation overhead")
    print("  ✅ Smart caching with automatic eviction")
    print("  ✅ Numpy memory optimization (float64→float32, etc.)")
    print("  ✅ Comprehensive memory profiling and tracking")
    print()

    print("💡 Perfect for Unity Wheel Strategy:")
    print("  📊 Process years of historical data efficiently")
    print("  ⚡ Handle real-time market data streams")
    print("  🔄 Reuse calculation contexts for performance")
    print("  💾 Cache expensive options calculations intelligently")
    print("  🧹 Automatic cleanup prevents memory leaks")
    print("  📈 Monitor memory usage in production")
    print()

    print(f"🏁 Demo completed with {final_summary['current_stats'].process_mb:.1f}MB memory usage")
    print("   Memory optimization system is ready for production!")


if __name__ == "__main__":
    main()
