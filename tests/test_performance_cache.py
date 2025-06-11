"""Test enhanced performance caching system."""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.unity_wheel.math.options_enhanced import (
    black_scholes_price_enhanced,
    calculate_all_greeks_enhanced,
    get_cache_performance_stats,
    implied_volatility_enhanced,
)
from src.unity_wheel.utils.performance_cache import (
    CacheManager,
    MemoryAwareLRUCache,
    get_cache_manager,
)


class TestMemoryAwareLRUCache:
    """Test the memory-aware LRU cache implementation."""

    @pytest.fixture
    def cache(self):
        """Create a test cache."""
        return MemoryAwareLRUCache(
            max_size=100, max_memory_mb=1.0, ttl_seconds=10, name="test_cache"
        )

    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, cache):
        """Test basic cache get/put operations."""
        # Test miss
        value, found = await cache.get("key1")
        assert not found
        assert value is None

        # Test put and hit
        await cache.put("key1", "value1", computation_time_ms=5.0)
        value, found = await cache.get("key1")
        assert found
        assert value == "value1"

        # Check stats
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.avg_computation_time_ms == 5.0

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache):
        """Test TTL-based cache expiration."""
        # Use very short TTL
        short_cache = MemoryAwareLRUCache(ttl_seconds=0.1, name="short_ttl")

        await short_cache.put("key1", "value1")

        # Should be found immediately
        value, found = await short_cache.get("key1")
        assert found

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be expired
        value, found = await short_cache.get("key1")
        assert not found

    @pytest.mark.asyncio
    async def test_memory_limit_eviction(self, cache):
        """Test eviction based on memory limits."""
        # Fill cache with large objects
        large_value = "x" * 10000  # ~10KB string

        # Add multiple large values
        for i in range(20):
            await cache.put(f"key{i}", large_value)

        # Check that some items were evicted
        stats = cache.get_stats()
        assert stats.evictions > 0
        assert stats.memory_bytes <= cache.max_memory_bytes

    @pytest.mark.asyncio
    async def test_size_limit_eviction(self, cache):
        """Test eviction based on size limits."""
        # Fill beyond max size
        for i in range(150):  # More than max_size of 100
            await cache.put(f"key{i}", f"value{i}")

        # Check evictions occurred
        stats = cache.get_stats()
        assert stats.evictions > 0
        assert len(cache._cache) <= cache.max_size

    @pytest.mark.asyncio
    async def test_lru_ordering(self, cache):
        """Test that LRU ordering works correctly."""
        # Add initial items
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add many more items to trigger eviction
        for i in range(98):
            await cache.put(f"new_key{i}", f"new_value{i}")

        # key1 should still be there (was accessed recently)
        value, found = await cache.get("key1")
        assert found

        # key2 and key3 should be evicted (least recently used)
        value2, found2 = await cache.get("key2")
        value3, found3 = await cache.get("key3")
        assert not found2 or not found3  # At least one should be evicted


class TestCacheManager:
    """Test the global cache manager."""

    def test_get_default_cache(self):
        """Test getting the default cache."""
        manager = get_cache_manager()
        cache = manager.get_cache()
        assert cache.name == "default"

    def test_create_named_cache(self):
        """Test creating named caches."""
        manager = get_cache_manager()

        # Create custom cache
        custom_cache = manager.create_cache(
            name="custom", max_size=500, max_memory_mb=25.0, ttl_seconds=600
        )

        assert custom_cache.name == "custom"
        assert custom_cache.max_size == 500
        assert custom_cache.ttl_seconds == 600

        # Should be retrievable
        retrieved = manager.get_cache("custom")
        assert retrieved is custom_cache

    @pytest.mark.asyncio
    async def test_global_stats(self):
        """Test getting global cache statistics."""
        manager = get_cache_manager()

        # Add some data to default cache
        default_cache = manager.get_cache()
        await default_cache.put("test_key", "test_value")

        # Get global stats
        stats = manager.get_global_stats()
        assert "default" in stats
        assert stats["default"].hits + stats["default"].misses > 0

    @pytest.mark.asyncio
    async def test_clear_all_caches(self):
        """Test clearing all caches."""
        manager = get_cache_manager()

        # Add data to multiple caches
        default_cache = manager.get_cache()
        custom_cache = manager.create_cache("test_clear")

        await default_cache.put("key1", "value1")
        await custom_cache.put("key2", "value2")

        # Clear all
        await manager.clear_all()

        # Check all are empty
        value1, found1 = await default_cache.get("key1")
        value2, found2 = await custom_cache.get("key2")

        assert not found1
        assert not found2


class TestEnhancedOptionsCalculations:
    """Test the enhanced options calculations with caching."""

    def test_black_scholes_cache_performance(self):
        """Test that caching improves Black-Scholes performance."""
        params = (100.0, 105.0, 0.25, 0.05, 0.20, "call")

        # First call (cache miss)
        start_time = time.time()
        result1 = black_scholes_price_enhanced(*params)
        first_call_time = time.time() - start_time

        # Second call (cache hit)
        start_time = time.time()
        result2 = black_scholes_price_enhanced(*params)
        second_call_time = time.time() - start_time

        # Results should be identical
        assert abs(result1.value - result2.value) < 1e-10
        assert result1.confidence == result2.confidence

        # Second call should be faster (though may be minimal for simple calculation)
        # Main benefit is avoiding redundant calculations
        assert second_call_time <= first_call_time * 2  # Allow some variance

    def test_greeks_cache_performance(self):
        """Test that caching improves Greeks calculation performance."""
        params = (100.0, 105.0, 0.25, 0.05, 0.20, "call")

        # First call
        greeks1, conf1 = calculate_all_greeks_enhanced(*params)

        # Second call (should hit cache)
        greeks2, conf2 = calculate_all_greeks_enhanced(*params)

        # Results should be identical
        for greek in ["delta", "gamma", "theta", "vega", "rho"]:
            assert abs(greeks1[greek] - greeks2[greek]) < 1e-10
        assert conf1 == conf2

    def test_implied_volatility_cache_performance(self):
        """Test that IV caching works correctly."""
        # Calculate a known option price first
        bs_result = black_scholes_price_enhanced(100, 105, 0.25, 0.05, 0.25, "call")
        option_price = bs_result.value

        # First IV calculation
        iv_result1 = implied_volatility_enhanced(option_price, 100, 105, 0.25, 0.05, "call")

        # Second calculation (cache hit)
        iv_result2 = implied_volatility_enhanced(option_price, 100, 105, 0.25, 0.05, "call")

        # Should be close to our input volatility of 0.25
        assert abs(iv_result1.value - 0.25) < 0.01
        assert abs(iv_result2.value - iv_result1.value) < 1e-10

    @pytest.mark.asyncio
    async def test_cache_performance_stats(self):
        """Test that performance statistics are collected."""
        # Make some calculations to populate cache
        for strike in [95, 100, 105, 110]:
            black_scholes_price_enhanced(100, strike, 0.25, 0.05, 0.20, "call")
            calculate_all_greeks_enhanced(100, strike, 0.25, 0.05, 0.20, "call")

        # Get performance stats
        stats = await get_cache_performance_stats()

        # Check stats structure
        assert "cache_name" in stats
        assert "hit_rate" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "memory_usage_mb" in stats

        # Should have some cache activity
        assert stats["total_hits"] + stats["total_misses"] > 0

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly for options."""
        from src.unity_wheel.utils.performance_cache import cache_key_for_options

        # Same parameters should generate same key
        key1 = cache_key_for_options(100.0, 105.0, 0.25, 0.05, 0.20, "call")
        key2 = cache_key_for_options(100.0, 105.0, 0.25, 0.05, 0.20, "call")
        assert key1 == key2

        # Different parameters should generate different keys
        key3 = cache_key_for_options(100.0, 110.0, 0.25, 0.05, 0.20, "call")
        assert key1 != key3

        # Different option types should generate different keys
        key4 = cache_key_for_options(100.0, 105.0, 0.25, 0.05, 0.20, "put")
        assert key1 != key4

    def test_cache_invalidation_on_different_inputs(self):
        """Test that different inputs don't hit the same cache entry."""
        # Different spot prices
        result1 = black_scholes_price_enhanced(100, 105, 0.25, 0.05, 0.20, "call")
        result2 = black_scholes_price_enhanced(101, 105, 0.25, 0.05, 0.20, "call")

        # Should be different values
        assert abs(result1.value - result2.value) > 0.01

        # Different strikes
        result3 = black_scholes_price_enhanced(100, 105, 0.25, 0.05, 0.20, "call")
        result4 = black_scholes_price_enhanced(100, 110, 0.25, 0.05, 0.20, "call")

        # Should be different values
        assert abs(result3.value - result4.value) > 0.01


class TestCacheDecoratorIntegration:
    """Test the @cached decorator integration."""

    def test_cached_decorator_with_custom_key_function(self):
        """Test cached decorator with custom key generation."""
        from src.unity_wheel.utils.performance_cache import cached

        call_count = 0

        @cached(
            cache_name="test_decorator", ttl_seconds=60, key_func=lambda x, y: f"custom_{x}_{y}"
        )
        def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call
        result1 = test_function(5, 10)
        assert result1 == 15
        assert call_count == 1

        # Second call with same params (cache hit)
        result2 = test_function(5, 10)
        assert result2 == 15
        assert call_count == 1  # Should not increment

        # Third call with different params (cache miss)
        result3 = test_function(5, 11)
        assert result3 == 16
        assert call_count == 2  # Should increment

    @pytest.mark.asyncio
    async def test_async_cached_decorator(self):
        """Test cached decorator with async functions."""
        from src.unity_wheel.utils.performance_cache import cached

        call_count = 0

        @cached(cache_name="test_async", ttl_seconds=60)
        async def async_test_function(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return x * 2

        # First call
        result1 = await async_test_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call (cache hit)
        result2 = await async_test_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for the enhanced caching system."""

    def test_options_calculation_performance_improvement(self):
        """Benchmark the performance improvement from caching."""
        import statistics

        # Test parameters
        test_cases = [
            (100, 95, 0.25, 0.05, 0.20, "call"),
            (100, 100, 0.25, 0.05, 0.20, "call"),
            (100, 105, 0.25, 0.05, 0.20, "call"),
            (100, 95, 0.25, 0.05, 0.20, "put"),
            (100, 100, 0.25, 0.05, 0.20, "put"),
        ]

        # Time first pass (cache misses)
        first_pass_times = []
        for params in test_cases:
            start_time = time.time()
            black_scholes_price_enhanced(*params)
            calculate_all_greeks_enhanced(*params)
            elapsed = time.time() - start_time
            first_pass_times.append(elapsed)

        # Time second pass (cache hits)
        second_pass_times = []
        for params in test_cases:
            start_time = time.time()
            black_scholes_price_enhanced(*params)
            calculate_all_greeks_enhanced(*params)
            elapsed = time.time() - start_time
            second_pass_times.append(elapsed)

        # Calculate averages
        avg_first_pass = statistics.mean(first_pass_times)
        avg_second_pass = statistics.mean(second_pass_times)

        # Second pass should be faster or equal (due to caching)
        assert avg_second_pass <= avg_first_pass * 1.1  # Allow 10% variance

        print(f"Average first pass time: {avg_first_pass:.6f}s")
        print(f"Average second pass time: {avg_second_pass:.6f}s")
        print(
            f"Performance improvement: {(avg_first_pass - avg_second_pass) / avg_first_pass * 100:.1f}%"
        )

    @pytest.mark.asyncio
    async def test_memory_usage_stays_bounded(self):
        """Test that memory usage stays within bounds."""
        cache_manager = get_cache_manager()

        # Generate many different calculations
        for i in range(1000):
            spot = 100 + (i % 20)  # Vary spot price
            strike = 95 + (i % 30)  # Vary strike
            vol = 0.15 + (i % 50) * 0.01  # Vary volatility

            black_scholes_price_enhanced(spot, strike, 0.25, 0.05, vol, "call")

        # Check memory usage
        stats = await get_cache_performance_stats()

        # Should stay within reasonable bounds (< 100MB for options cache)
        assert stats["memory_usage_mb"] < 100.0

        # Should have some evictions due to size/memory limits
        print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
        print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        print(f"Total evictions: {stats['evictions']}")
