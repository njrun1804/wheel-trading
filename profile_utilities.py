#!/usr/bin/env python3
"""Profile utility performance using Bolt's capabilities"""

import asyncio
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

from einstein.memory_optimizer import MemoryOptimizer
from meta_fast_pattern_cache import FastPatternCache
from src.unity_wheel.storage.duckdb_cache import AsyncConnectionPool

# Import utilities to profile
from src.unity_wheel.utils.logging import get_logger, timed_operation
from src.unity_wheel.utils.recovery import with_recovery
from src.unity_wheel.utils.validation import validate


class UtilityProfiler:
    def __init__(self):
        self.results = {}
        self.logger = get_logger(__name__)

    async def profile_memory_utilities(self):
        """Profile memory-related utilities"""
        print("\n=== PROFILING MEMORY UTILITIES ===")

        # Test 1: Connection pooling performance
        start = time.time()
        pool = AsyncConnectionPool(pool_size=10)

        # Simulate concurrent connections
        async def test_connection():
            async with pool.connection() as conn:
                await asyncio.sleep(0.001)  # Simulate work

        tasks = [test_connection() for _ in range(100)]
        await asyncio.gather(*tasks)

        conn_pool_time = time.time() - start
        print(f"Connection Pool (100 concurrent): {conn_pool_time:.3f}s")

        # Test 2: Memory optimizer
        start = time.time()
        mem_opt = MemoryOptimizer()
        for _ in range(1000):
            mem_opt.get_memory_usage()

        mem_opt_time = time.time() - start
        print(f"Memory Optimizer (1000 checks): {mem_opt_time:.3f}s")

        # Test 3: Pattern cache
        start = time.time()
        cache = FastPatternCache(max_patterns=100)

        # Add patterns
        for i in range(100):
            cache.add_pattern(f"pattern_{i}", {"data": f"value_{i}"})

        # Lookup patterns
        for i in range(1000):
            cache.get_pattern(f"pattern_{i % 100}")

        cache_time = time.time() - start
        print(f"Pattern Cache (100 patterns, 1000 lookups): {cache_time:.3f}s")

        self.results["memory"] = {
            "connection_pool": conn_pool_time,
            "memory_optimizer": mem_opt_time,
            "pattern_cache": cache_time,
        }

    async def profile_logging_utilities(self):
        """Profile logging utilities"""
        print("\n=== PROFILING LOGGING UTILITIES ===")

        # Test 1: Logger creation
        start = time.time()
        loggers = []
        for i in range(100):
            loggers.append(get_logger(f"test.module.{i}"))
        logger_creation_time = time.time() - start
        print(f"Logger Creation (100 loggers): {logger_creation_time:.3f}s")

        # Test 2: Logging performance
        logger = get_logger("perf_test")
        start = time.time()
        for i in range(10000):
            logger.debug(f"Debug message {i}")
        logging_time = time.time() - start
        print(f"Logging (10k messages): {logging_time:.3f}s")

        # Test 3: Timed operations
        @timed_operation("test_operation")
        async def test_func():
            await asyncio.sleep(0.001)

        start = time.time()
        for _ in range(100):
            await test_func()
        timed_op_time = time.time() - start
        print(f"Timed Operations (100 calls): {timed_op_time:.3f}s")

        self.results["logging"] = {
            "logger_creation": logger_creation_time,
            "logging_messages": logging_time,
            "timed_operations": timed_op_time,
        }

    async def profile_validation_utilities(self):
        """Profile validation utilities"""
        print("\n=== PROFILING VALIDATION UTILITIES ===")

        # Test data
        test_data = {
            "strike": 400.0,
            "expiry": "2024-01-19",
            "option_type": "put",
            "quantity": 10,
        }

        # Test 1: Simple validation
        start = time.time()
        for _ in range(10000):
            validate(test_data["strike"], float, "strike")
            validate(test_data["option_type"], str, "option_type")
        simple_validation_time = time.time() - start
        print(f"Simple Validation (20k calls): {simple_validation_time:.3f}s")

        # Test 2: Complex validation with recovery
        @with_recovery
        async def validate_complex(data):
            if data["strike"] < 0:
                raise ValueError("Invalid strike")
            return True

        start = time.time()
        for _ in range(1000):
            await validate_complex(test_data)
        complex_validation_time = time.time() - start
        print(
            f"Complex Validation with Recovery (1k calls): {complex_validation_time:.3f}s"
        )

        self.results["validation"] = {
            "simple_validation": simple_validation_time,
            "complex_validation": complex_validation_time,
        }

    def analyze_import_chains(self):
        """Analyze import performance and dead code"""
        print("\n=== ANALYZING IMPORT CHAINS ===")

        # Profile imports
        pr = cProfile.Profile()
        pr.enable()

        # Import heavy modules
        import_times = {}

        modules = [
            "src.unity_wheel.utils",
            "bolt.unified_memory",
            "einstein.memory_optimizer",
            "jarvis2.core.memory_manager",
        ]

        for module in modules:
            start = time.time()
            try:
                __import__(module)
                import_times[module] = time.time() - start
            except Exception as e:
                import_times[module] = f"Error: {e}"

        pr.disable()

        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)

        print("\nTop import chain bottlenecks:")
        for module, time_taken in sorted(
            import_times.items(),
            key=lambda x: x[1] if isinstance(x[1], float) else 0,
            reverse=True,
        ):
            print(f"  {module}: {time_taken}")

        self.results["imports"] = import_times

    async def run_full_profile(self):
        """Run complete utility profiling"""
        print("UTILITY PERFORMANCE PROFILING")
        print("=" * 50)

        await self.profile_memory_utilities()
        await self.profile_logging_utilities()
        await self.profile_validation_utilities()
        self.analyze_import_chains()

        # Save results
        output_file = Path("utility_profiling_results.json")
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n\nProfiling results saved to: {output_file}")

        # Generate recommendations
        self.generate_recommendations()

    def generate_recommendations(self):
        """Generate optimization recommendations"""
        print("\n\n=== OPTIMIZATION RECOMMENDATIONS ===")

        # Memory recommendations
        if self.results.get("memory", {}).get("connection_pool", 0) > 0.1:
            print("\n1. CONNECTION POOLING:")
            print("   - Consider using a single shared pool instance")
            print("   - Implement connection recycling")

        if self.results.get("memory", {}).get("pattern_cache", 0) > 0.05:
            print("\n2. PATTERN CACHE:")
            print("   - Use LRU eviction for better memory usage")
            print("   - Consider async cache warming")

        # Logging recommendations
        if self.results.get("logging", {}).get("logging_messages", 0) > 1.0:
            print("\n3. LOGGING PERFORMANCE:")
            print("   - Use lazy formatting with %s instead of f-strings")
            print("   - Implement log level filtering earlier")

        # Validation recommendations
        if self.results.get("validation", {}).get("simple_validation", 0) > 0.5:
            print("\n4. VALIDATION:")
            print("   - Cache validation schemas")
            print("   - Use compiled regex patterns")


if __name__ == "__main__":
    profiler = UtilityProfiler()
    asyncio.run(profiler.run_full_profile())
