#!/usr/bin/env python3
"""
Cached Query Router for Einstein

Ultra-fast query routing with:
- In-memory query plan cache with <1ms lookup
- Bloom filter for negative cache hits
- Pre-computed query features
- Lock-free concurrent access
- Automatic cache warming from frequent queries
"""

import asyncio
import contextlib
import hashlib
import logging
import pickle
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from einstein.einstein_config import get_einstein_config

from .adaptive_router import AdaptiveQueryRouter, QueryFeatures
from .query_router import QueryPlan

logger = logging.getLogger(__name__)


class BloomFilter:
    """Space-efficient probabilistic data structure for negative caching."""

    def __init__(self, size: int = 10000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = np.zeros(size, dtype=bool)
        self.count = 0

    def _hash(self, item: str, seed: int) -> int:
        """Generate hash with seed."""
        h = hashlib.md5((item + str(seed)).encode()).digest()
        return int.from_bytes(h[:4], "big") % self.size

    def add(self, item: str):
        """Add item to bloom filter."""
        for i in range(self.num_hashes):
            idx = self._hash(item, i)
            self.bit_array[idx] = True
        self.count += 1

    def contains(self, item: str) -> bool:
        """Check if item might be in the set (no false negatives)."""
        for i in range(self.num_hashes):
            idx = self._hash(item, i)
            if not self.bit_array[idx]:
                return False
        return True

    def clear(self):
        """Clear the bloom filter."""
        self.bit_array.fill(False)
        self.count = 0


@dataclass
class CachedQueryPlan:
    """Cached query plan with metadata."""

    plan: QueryPlan
    features: QueryFeatures
    creation_time: float
    hit_count: int = 0
    last_access: float = field(default_factory=time.time)
    avg_latency_ms: float = 0.0
    avg_result_count: float = 0.0


class QueryPlanCache:
    """High-performance cache for query plans."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: dict[str, CachedQueryPlan] = {}
        self.access_order = deque(maxlen=max_size)
        self.bloom_filter = BloomFilter(size=max_size * 2)

        # Statistics
        self.hits = 0
        self.misses = 0
        self.negative_hits = 0  # Bloom filter prevented lookup

        # Query frequency tracking for warming
        self.query_frequency = Counter()
        self.frequency_window = deque(maxlen=1000)

    def get(self, query: str) -> CachedQueryPlan | None:
        """Get cached plan with O(1) lookup."""

        # Check bloom filter first (fast negative cache)
        if not self.bloom_filter.contains(query):
            self.negative_hits += 1
            self.misses += 1
            return None

        # Lookup in cache
        if query in self.cache:
            cached = self.cache[query]
            cached.hit_count += 1
            cached.last_access = time.time()

            # Update access order
            with contextlib.suppress(ValueError):
                self.access_order.remove(query)
            self.access_order.append(query)

            self.hits += 1
            return cached

        self.misses += 1
        return None

    def put(self, query: str, plan: QueryPlan, features: QueryFeatures):
        """Store plan in cache."""

        # Update frequency tracking
        self.query_frequency[query] += 1
        self.frequency_window.append(query)

        # Add to bloom filter
        self.bloom_filter.add(query)

        # Create cached entry
        cached = CachedQueryPlan(
            plan=plan, features=features, creation_time=time.time()
        )

        # Store in cache
        self.cache[query] = cached
        self.access_order.append(query)

        # Evict if necessary
        while len(self.cache) > self.max_size:
            self._evict_lru()

    def _evict_lru(self):
        """Evict least recently used entry."""
        if self.access_order:
            lru_query = self.access_order.popleft()
            if lru_query in self.cache:
                del self.cache[lru_query]

    def update_performance(self, query: str, latency_ms: float, result_count: int):
        """Update performance metrics for cached plan."""

        if query in self.cache:
            cached = self.cache[query]

            # Update moving average
            alpha = 0.2  # Learning rate
            cached.avg_latency_ms = (
                1 - alpha
            ) * cached.avg_latency_ms + alpha * latency_ms
            cached.avg_result_count = (
                1 - alpha
            ) * cached.avg_result_count + alpha * result_count

    def get_frequent_queries(self, top_n: int = 100) -> list[str]:
        """Get most frequent queries for cache warming."""

        return [query for query, _ in self.query_frequency.most_common(top_n)]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""

        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": round(hit_rate * 100, 1),
            "hits": self.hits,
            "misses": self.misses,
            "negative_hits": self.negative_hits,
            "bloom_filter_size": self.bloom_filter.count,
            "unique_queries": len(self.query_frequency),
        }


class CachedQueryRouter(AdaptiveQueryRouter):
    """Query router with intelligent caching for <1ms response time."""

    def __init__(self, model_path: Path = None):
        super().__init__(model_path)

        config = get_einstein_config()

        # Initialize caches
        self.plan_cache = QueryPlanCache(max_size=config.cache.search_cache_size)
        self.feature_cache = {}  # Query -> pre-computed features

        # Parallel processing pool for batch operations
        self.processing_pool = asyncio.Semaphore(8)  # For 8 concurrent agents

        # Cache warming
        self._warm_cache_task = None
        self._warming_interval = 300  # 5 minutes

        # Performance tracking
        self.router_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_routing_time_ms": 0.0,
            "p99_routing_time_ms": 0.0,
        }

        self.routing_times = deque(maxlen=1000)

    async def analyze_query_cached(
        self, query: str, context: dict[str, Any] = None
    ) -> QueryPlan:
        """Analyze query with caching for ultra-fast response."""

        start_time = time.time()

        # Check cache first
        cached_plan = self.plan_cache.get(query)

        if cached_plan:
            # Cache hit - return immediately
            self.router_stats["cache_hits"] += 1

            routing_time = (time.time() - start_time) * 1000
            self._update_routing_stats(routing_time)

            logger.debug(f"Cache hit for query: {query[:50]}... ({routing_time:.1f}ms)")
            return cached_plan.plan

        # Cache miss - compute plan
        self.router_stats["cache_misses"] += 1

        # Check if features are cached
        if query in self.feature_cache:
            features = self.feature_cache[query]
        else:
            features = self._extract_features(query)
            self.feature_cache[query] = features

        # Get adaptive plan
        plan = await self._compute_adaptive_plan(query, features, context)

        # Cache the plan
        self.plan_cache.put(query, plan, features)

        routing_time = (time.time() - start_time) * 1000
        self._update_routing_stats(routing_time)

        logger.debug(
            f"Computed new plan for query: {query[:50]}... ({routing_time:.1f}ms)"
        )

        return plan

    async def _compute_adaptive_plan(
        self, query: str, features: QueryFeatures, context: dict[str, Any] = None
    ) -> QueryPlan:
        """Compute adaptive plan using parent class logic."""

        # Get bandit recommendation
        arm = self.bandit.select_arm(features)
        recommended_modalities = self.bandit.arms[arm]

        # Create base plan
        base_plan = super().analyze_query(query)

        # Override with adaptive recommendation
        adaptive_plan = QueryPlan(
            query=query,
            query_type=base_plan.query_type,
            search_modalities=recommended_modalities,
            confidence=base_plan.confidence * 0.9,
            estimated_time_ms=sum(
                self.modality_performance[m] for m in recommended_modalities
            ),
            reasoning=f"Cached/Adaptive: {base_plan.reasoning} [Arm {arm}]",
        )

        return adaptive_plan

    async def analyze_batch(self, queries: list[str]) -> list[QueryPlan]:
        """Analyze multiple queries in parallel for agent burst support."""

        async def analyze_single(q: str) -> QueryPlan:
            async with self.processing_pool:
                return await self.analyze_query_cached(q)

        # Process all queries in parallel
        plans = await asyncio.gather(*[analyze_single(q) for q in queries])

        return plans

    def _update_routing_stats(self, routing_time_ms: float):
        """Update routing performance statistics."""

        self.routing_times.append(routing_time_ms)

        # Update moving average
        alpha = 0.1
        self.router_stats["avg_routing_time_ms"] = (1 - alpha) * self.router_stats[
            "avg_routing_time_ms"
        ] + alpha * routing_time_ms

        # Calculate P99
        if len(self.routing_times) >= 100:
            sorted_times = sorted(self.routing_times)
            p99_idx = int(len(sorted_times) * 0.99)
            self.router_stats["p99_routing_time_ms"] = sorted_times[p99_idx]

    async def warm_cache(self):
        """Warm cache with frequent queries."""

        logger.info("🔥 Warming query cache...")

        # Get frequent queries
        frequent_queries = self.plan_cache.get_frequent_queries(top_n=100)

        if not frequent_queries:
            logger.info("No frequent queries to warm cache with")
            return

        # Warm cache in parallel
        warm_tasks = []
        for query in frequent_queries:
            if query not in self.plan_cache.cache:
                warm_tasks.append(self.analyze_query_cached(query))

        if warm_tasks:
            await asyncio.gather(*warm_tasks)
            logger.info(f"Warmed cache with {len(warm_tasks)} queries")

    async def start_cache_warming(self):
        """Start periodic cache warming."""

        async def warming_loop():
            while True:
                try:
                    await asyncio.sleep(self._warming_interval)
                    await self.warm_cache()
                except Exception as e:
                    logger.error(f"Cache warming error: {e}")

        self._warm_cache_task = asyncio.create_task(warming_loop())

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""

        base_stats = super().get_learning_stats()

        return {
            **base_stats,
            "cache_stats": self.plan_cache.get_stats(),
            "router_performance": self.router_stats,
            "feature_cache_size": len(self.feature_cache),
        }

    def persist_cache(self, cache_path: Path = None):
        """Persist cache to disk for fast startup."""

        if cache_path is None:
            cache_path = self.model_path.parent / "query_cache.pkl"

        try:
            cache_data = {
                "plan_cache": {
                    query: cached
                    for query, cached in self.plan_cache.cache.items()
                    if cached.hit_count > 1  # Only persist frequently used
                },
                "feature_cache": self.feature_cache,
                "query_frequency": dict(self.plan_cache.query_frequency),
                "timestamp": time.time(),
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)

            logger.info(f"Persisted query cache to {cache_path}")

        except Exception as e:
            logger.error(f"Failed to persist cache: {e}")

    def load_cache(self, cache_path: Path = None):
        """Load cache from disk."""

        if cache_path is None:
            cache_path = self.model_path.parent / "query_cache.pkl"

        try:
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                # Restore plan cache
                for query, cached in cache_data.get("plan_cache", {}).items():
                    self.plan_cache.cache[query] = cached
                    self.plan_cache.bloom_filter.add(query)

                # Restore feature cache
                self.feature_cache.update(cache_data.get("feature_cache", {}))

                # Restore frequency data
                self.plan_cache.query_frequency.update(
                    cache_data.get("query_frequency", {})
                )

                logger.info(f"Loaded query cache from {cache_path}")
                logger.info(f"  Cached plans: {len(self.plan_cache.cache)}")
                logger.info(f"  Cached features: {len(self.feature_cache)}")

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")


async def benchmark_cached_router():
    """Benchmark the cached query router."""

    print("🚀 Benchmarking Cached Query Router...")

    router = CachedQueryRouter()

    # Test queries
    test_queries = [
        "WheelStrategy class implementation",
        "calculate_delta function in options.py",
        "complex functions with high cyclomatic complexity",
        '"import pandas" statements',
        "Black-Scholes pricing formula",
        "optimize trading performance bottlenecks",
        "find all TODO comments",
        "database connection pooling",
    ] * 10  # Repeat to test caching

    # Warm up
    for query in test_queries[:8]:
        await router.analyze_query_cached(query)

    print("\n📊 Single Query Performance:")

    # Benchmark individual queries
    for i, query in enumerate(test_queries[:8]):
        start = time.time()
        await router.analyze_query_cached(query)
        elapsed = (time.time() - start) * 1000

        cache_status = "HIT" if i >= 8 else "MISS"
        print(f"  Query {i+1} [{cache_status}]: {elapsed:.2f}ms - {query[:30]}...")

    print("\n🎯 Burst Performance (8 concurrent agents):")

    # Test burst of 8 queries
    burst_queries = test_queries[:8]

    start = time.time()
    plans = await router.analyze_batch(burst_queries)
    elapsed = (time.time() - start) * 1000

    print(f"  Total time: {elapsed:.1f}ms")
    print(f"  Average per query: {elapsed/len(burst_queries):.1f}ms")
    print(f"  Queries processed: {len(plans)}")

    # Show statistics
    stats = router.get_performance_stats()
    cache_stats = stats["cache_stats"]
    router_perf = stats["router_performance"]

    print("\n📈 Performance Statistics:")
    print(f"  Cache hit rate: {cache_stats['hit_rate']}%")
    print(f"  Average routing time: {router_perf['avg_routing_time_ms']:.1f}ms")
    print(f"  P99 routing time: {router_perf['p99_routing_time_ms']:.1f}ms")
    print(f"  Cache size: {cache_stats['size']}/{cache_stats['max_size']}")

    # Test cache persistence
    print("\n💾 Testing cache persistence...")

    cache_file = Path("/tmp/test_query_cache.pkl")
    router.persist_cache(cache_file)

    # Create new router and load cache
    new_router = CachedQueryRouter()
    new_router.load_cache(cache_file)

    # Test loaded cache
    start = time.time()
    await new_router.analyze_query_cached(test_queries[0])
    elapsed = (time.time() - start) * 1000

    print(f"  First query after load: {elapsed:.2f}ms (should be <1ms)")

    # Cleanup
    cache_file.unlink(missing_ok=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(benchmark_cached_router())
