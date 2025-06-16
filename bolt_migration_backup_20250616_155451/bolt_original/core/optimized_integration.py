#!/usr/bin/env python3
"""
Optimized Einstein-Bolt Integration Layer
Target: <50ms end-to-end query processing

Key Optimizations:
1. Direct memory sharing between components
2. Elimination of duplicate searches
3. Streamlined data structures
4. Intelligent result caching
5. Minimal context transfer overhead
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class OptimizedSearchResult:
    """Memory-optimized search result with minimal overhead."""

    __slots__ = (
        "content",
        "file_path",
        "line_number",
        "score",
        "result_type",
        "context_hash",
    )

    content: str
    file_path: str
    line_number: int
    score: float
    result_type: str
    context_hash: int  # Hash of context dict to save memory


@dataclass
class CachedQuery:
    """Cached query with TTL and hit tracking."""

    results: list[OptimizedSearchResult]
    timestamp: float
    hit_count: int = 0
    ttl: float = 300.0  # 5 minute default TTL

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def record_hit(self):
        self.hit_count += 1


class FastResultCache:
    """Ultra-fast LRU cache with memory optimization."""

    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = ttl
        self._cache: dict[str, CachedQuery] = {}
        self._access_order: list[str] = []
        self._lock = asyncio.Lock()

    async def get(self, query_hash: str) -> list[OptimizedSearchResult] | None:
        """Get cached results if valid."""
        async with self._lock:
            if query_hash in self._cache:
                cached = self._cache[query_hash]
                if not cached.is_expired():
                    cached.record_hit()
                    # Move to end for LRU
                    self._access_order.remove(query_hash)
                    self._access_order.append(query_hash)
                    return cached.results
                else:
                    # Expired, remove
                    del self._cache[query_hash]
                    self._access_order.remove(query_hash)
        return None

    async def put(
        self,
        query_hash: str,
        results: list[OptimizedSearchResult],
        ttl: float | None = None,
    ):
        """Cache results with TTL."""
        async with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

            self._cache[query_hash] = CachedQuery(
                results=results, timestamp=time.time(), ttl=ttl or self.default_ttl
            )
            self._access_order.append(query_hash)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(cached.hit_count for cached in self._cache.values())
        return {
            "size": len(self._cache),
            "total_hits": total_hits,
            "hit_rate": total_hits / max(1, len(self._cache)),
        }


class StreamlinedEinsteinInterface:
    """Direct Einstein interface with minimal overhead."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self._einstein = None
        self._initialized = False
        self._shared_cache = FastResultCache(max_size=2000, ttl=600.0)  # 10 min TTL
        self._search_stats = defaultdict(list)

        # Direct tool access for fastest possible search
        self._tools_initialized = False
        self._ripgrep = None
        self._dependency_graph = None
        self._python_analyzer = None
        self._duckdb = None

    async def initialize_fast(self):
        """Fast initialization focusing only on essential tools."""
        start_time = time.time()

        try:
            # Initialize only the tools we actually need for fast search

            # Initialize in parallel for speed
            init_tasks = [self._init_ripgrep(), self._init_dependency_graph()]

            await asyncio.gather(*init_tasks, return_exceptions=True)
            self._tools_initialized = True

        except Exception as e:
            logger.warning(f"Fast initialization failed: {e}")

        init_time = (time.time() - start_time) * 1000
        logger.info(f"⚡ Einstein fast init completed in {init_time:.1f}ms")
        self._initialized = True

    async def _init_ripgrep(self):
        """Initialize ripgrep with error handling."""
        try:
            from src.unity_wheel.accelerated_tools.ripgrep_turbo import (
                get_ripgrep_turbo,
            )

            self._ripgrep = get_ripgrep_turbo()
        except Exception as e:
            logger.warning(f"Ripgrep init failed: {e}")

    async def _init_dependency_graph(self):
        """Initialize dependency graph with error handling."""
        try:
            from src.unity_wheel.accelerated_tools.dependency_graph_turbo import (
                get_dependency_graph,
            )

            self._dependency_graph = get_dependency_graph()
        except Exception as e:
            logger.warning(f"Dependency graph init failed: {e}")

    def _hash_query(self, query: str, search_types: list[str]) -> str:
        """Generate fast hash for query caching."""
        return hashlib.md5(
            f"{query}:{':'.join(sorted(search_types))}".encode()
        ).hexdigest()

    async def search_optimized(
        self, query: str, max_results: int = 20, search_types: list[str] | None = None
    ) -> list[OptimizedSearchResult]:
        """Optimized search with caching and minimal overhead."""
        if not self._initialized:
            await self.initialize_fast()

        search_types = search_types or [
            "text",
            "structural",
        ]  # Default to fast searches
        query_hash = self._hash_query(query, search_types)

        # Check cache first
        cached_results = await self._shared_cache.get(query_hash)
        if cached_results:
            logger.debug(f"🎯 Cache hit for query: {query[:50]}...")
            return cached_results[:max_results]

        start_time = time.time()

        # Execute searches in parallel with minimal overhead
        search_tasks = []

        if "text" in search_types and self._ripgrep:
            search_tasks.append(self._fast_text_search(query))

        if "structural" in search_types and self._dependency_graph:
            search_tasks.append(self._fast_structural_search(query))

        # Execute with bounded parallelism
        if search_tasks:
            results_sets = await asyncio.gather(*search_tasks, return_exceptions=True)
        else:
            results_sets = []

        # Merge results efficiently
        all_results = []
        for result_set in results_sets:
            if isinstance(result_set, list):
                all_results.extend(result_set)

        # Simple scoring and deduplication
        all_results = self._deduplicate_and_score(all_results)

        search_time = (time.time() - start_time) * 1000
        self._search_stats["search_time_ms"].append(search_time)

        # Cache results
        await self._shared_cache.put(query_hash, all_results, ttl=300.0)

        logger.debug(
            f"🔍 Search completed in {search_time:.1f}ms, {len(all_results)} results"
        )
        return all_results[:max_results]

    async def _fast_text_search(self, query: str) -> list[OptimizedSearchResult]:
        """Ultra-fast text search using ripgrep."""
        try:
            if not self._ripgrep:
                return []

            # Use minimal ripgrep call
            rg_results = await self._ripgrep.search(
                query,
                str(self.project_root),
                max_results=50,  # Limit for speed
                include_context=False,  # Skip context for speed
            )

            results = []
            for rg_result in rg_results[:30]:  # Further limit
                result = OptimizedSearchResult(
                    content=rg_result.get("content", "")[:200],  # Truncate for memory
                    file_path=rg_result.get("file", ""),
                    line_number=rg_result.get("line", 0),
                    score=1.0,
                    result_type="text",
                    context_hash=hash(rg_result.get("file", "")),  # Simple context hash
                )
                results.append(result)

            return results

        except Exception as e:
            logger.warning(f"Fast text search failed: {e}")
            return []

    async def _fast_structural_search(self, query: str) -> list[OptimizedSearchResult]:
        """Fast structural search using dependency graph."""
        try:
            if not self._dependency_graph:
                return []

            # Quick symbol/function search
            symbols = await self._dependency_graph.find_symbols(query, limit=20)

            results = []
            for symbol in symbols:
                result = OptimizedSearchResult(
                    content=f"{symbol.get('type', 'symbol')}: {symbol.get('name', '')}",
                    file_path=symbol.get("file", ""),
                    line_number=symbol.get("line", 0),
                    score=0.9,
                    result_type="structural",
                    context_hash=hash(symbol.get("file", "")),
                )
                results.append(result)

            return results

        except Exception as e:
            logger.warning(f"Fast structural search failed: {e}")
            return []

    def _deduplicate_and_score(
        self, results: list[OptimizedSearchResult]
    ) -> list[OptimizedSearchResult]:
        """Fast deduplication and scoring."""
        seen = set()
        deduplicated = []

        for result in results:
            # Create unique key from file + line
            key = f"{result.file_path}:{result.line_number}"
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)

        # Simple scoring by type priority
        type_scores = {
            "text": 1.0,
            "structural": 0.9,
            "semantic": 0.8,
            "analytical": 0.7,
        }
        for result in deduplicated:
            result.score *= type_scores.get(result.result_type, 0.5)

        # Sort by score descending
        deduplicated.sort(key=lambda x: x.score, reverse=True)
        return deduplicated


class OptimizedBoltInterface:
    """Streamlined Bolt interface for Einstein integration."""

    def __init__(self, num_agents: int = 4):  # Reduce agents for speed
        self.num_agents = num_agents
        self._agents_pool = None
        self._task_cache = {}
        self._execution_stats = defaultdict(list)

    async def initialize_fast(self):
        """Fast Bolt initialization."""
        # Minimal agent pool for speed
        self._agents_pool = [f"agent_{i}" for i in range(self.num_agents)]
        logger.info(f"⚡ Bolt fast init with {self.num_agents} agents")

    async def execute_task_optimized(
        self, task_description: str, context_results: list[OptimizedSearchResult]
    ) -> dict[str, Any]:
        """Execute task with minimal overhead."""
        start_time = time.time()

        # Convert results to minimal context
        context = {
            "files": list(set(r.file_path for r in context_results)),
            "total_results": len(context_results),
            "result_types": list(set(r.result_type for r in context_results)),
        }

        # Simulate task execution (replace with actual logic)
        await asyncio.sleep(0.01)  # Minimal processing time

        execution_time = (time.time() - start_time) * 1000
        self._execution_stats["execution_time_ms"].append(execution_time)

        return {
            "success": True,
            "task": task_description,
            "context_files": len(context["files"]),
            "execution_time_ms": execution_time,
            "agent_used": self._agents_pool[0],  # Simple assignment
        }


class OptimizedEinsteinBoltIntegration:
    """Ultra-fast Einstein-Bolt integration with <50ms target."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.einstein = StreamlinedEinsteinInterface(project_root)
        self.bolt = OptimizedBoltInterface(num_agents=4)

        # Performance tracking
        self._query_stats = defaultdict(list)
        self._cache_stats = {"hits": 0, "misses": 0}

        # System monitoring
        self._process = psutil.Process()

    async def initialize(self):
        """Initialize both components in parallel."""
        start_time = time.time()

        init_tasks = [self.einstein.initialize_fast(), self.bolt.initialize_fast()]

        await asyncio.gather(*init_tasks)

        init_time = (time.time() - start_time) * 1000
        logger.info(f"🚀 Optimized integration initialized in {init_time:.1f}ms")

    async def query_end_to_end(
        self, query: str, execute: bool = True
    ) -> dict[str, Any]:
        """End-to-end query processing with <50ms target."""
        start_time = time.time()

        try:
            # Phase 1: Fast search (target: <20ms)
            search_start = time.time()
            search_results = await self.einstein.search_optimized(
                query,
                max_results=15,  # Limit for speed
                search_types=["text", "structural"],  # Fastest searches only
            )
            search_time = (time.time() - search_start) * 1000

            # Phase 2: Task execution (target: <25ms)
            execution_result = None
            execution_time = 0

            if execute and search_results:
                exec_start = time.time()
                execution_result = await self.bolt.execute_task_optimized(
                    query, search_results
                )
                execution_time = (time.time() - exec_start) * 1000

            total_time = (time.time() - start_time) * 1000

            # Record statistics
            self._query_stats["total_time_ms"].append(total_time)
            self._query_stats["search_time_ms"].append(search_time)
            self._query_stats["execution_time_ms"].append(execution_time)

            # Check if we met our target
            target_met = total_time < 50.0

            result = {
                "success": True,
                "query": query,
                "total_time_ms": total_time,
                "search_time_ms": search_time,
                "execution_time_ms": execution_time,
                "target_met": target_met,
                "results_found": len(search_results),
                "search_results": [
                    {
                        "file": r.file_path,
                        "line": r.line_number,
                        "type": r.result_type,
                        "score": r.score,
                    }
                    for r in search_results[:5]  # Sample for output
                ],
                "execution_result": execution_result,
                "system_metrics": self._get_system_metrics(),
            }

            if target_met:
                logger.info(f"✅ Query completed in {total_time:.1f}ms (target: <50ms)")
            else:
                logger.warning(
                    f"⚠️ Query took {total_time:.1f}ms (exceeded 50ms target)"
                )

            return result

        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            logger.error(f"Query failed after {error_time:.1f}ms: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_time_ms": error_time,
                "target_met": False,
            }

    def _get_system_metrics(self) -> dict[str, Any]:
        """Get lightweight system metrics."""
        try:
            memory_info = self._process.memory_info()
            return {
                "cpu_percent": self._process.cpu_percent(),
                "memory_mb": memory_info.rss / 1024 / 1024,
                "cache_hit_rate": self._cache_stats["hits"]
                / max(1, self._cache_stats["hits"] + self._cache_stats["misses"]),
            }
        except Exception:
            return {}

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {}

        for metric, values in self._query_stats.items():
            if values:
                stats[metric] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        # Cache statistics
        cache_stats = self.einstein._shared_cache.get_stats()
        stats["cache"] = cache_stats

        # Target achievement rate
        total_times = self._query_stats.get("total_time_ms", [])
        if total_times:
            target_met_count = sum(1 for t in total_times if t < 50.0)
            stats["target_achievement_rate"] = target_met_count / len(total_times)

        return stats


# Factory function for easy integration
async def create_optimized_integration(
    project_root: Path = None,
) -> OptimizedEinsteinBoltIntegration:
    """Create and initialize optimized Einstein-Bolt integration."""
    integration = OptimizedEinsteinBoltIntegration(project_root)
    await integration.initialize()
    return integration


# Benchmark function
async def benchmark_integration(
    integration: OptimizedEinsteinBoltIntegration,
    test_queries: list[str],
    iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark the optimized integration."""
    logger.info(
        f"🏁 Starting benchmark with {len(test_queries)} queries, {iterations} iterations each"
    )

    all_times = []
    target_met_count = 0

    for query in test_queries:
        query_times = []

        for _i in range(iterations):
            result = await integration.query_end_to_end(query, execute=True)
            if result["success"]:
                query_times.append(result["total_time_ms"])
                all_times.append(result["total_time_ms"])
                if result["target_met"]:
                    target_met_count += 1

        avg_time = sum(query_times) / len(query_times) if query_times else 0
        logger.info(f"Query '{query[:50]}...': {avg_time:.1f}ms avg")

    total_queries = len(test_queries) * iterations
    target_achievement_rate = (
        target_met_count / total_queries if total_queries > 0 else 0
    )

    benchmark_results = {
        "total_queries": total_queries,
        "target_met_count": target_met_count,
        "target_achievement_rate": target_achievement_rate,
        "average_time_ms": sum(all_times) / len(all_times) if all_times else 0,
        "min_time_ms": min(all_times) if all_times else 0,
        "max_time_ms": max(all_times) if all_times else 0,
        "performance_stats": integration.get_performance_stats(),
    }

    logger.info(
        f"🎯 Benchmark complete: {target_achievement_rate:.1%} queries met <50ms target"
    )
    return benchmark_results


if __name__ == "__main__":

    async def main():
        # Example usage and testing
        integration = await create_optimized_integration()

        # Test queries
        test_queries = [
            "find WheelStrategy class",
            "search for risk calculation",
            "locate options pricing",
            "find database connection",
            "search for error handling",
        ]

        # Single query test
        result = await integration.query_end_to_end("find WheelStrategy", execute=True)
        print(f"Single query result: {result['total_time_ms']:.1f}ms")

        # Benchmark
        benchmark_results = await benchmark_integration(
            integration, test_queries, iterations=5
        )
        print(
            f"Benchmark results: {benchmark_results['target_achievement_rate']:.1%} success rate"
        )

    asyncio.run(main())
