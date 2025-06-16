"""Unified Search System - Single entry point for all search operations.

Consolidates Einstein's multi-modal orchestration, Bolt's hardware acceleration,
and accelerated tools' performance optimizations into one high-performance system.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from .hybrid_cache_system import HybridCacheSystem
from .search_orchestrator import SearchOrchestrator
from .unified_query_router import UnifiedQueryRouter

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result format."""

    content: str
    file_path: str
    line_number: int
    score: float
    result_type: str  # text, semantic, code, analytical
    context: dict[str, Any]
    timestamp: float
    engine: str  # Which search engine provided this result


@dataclass
class SearchMetrics:
    """Search performance metrics."""

    query: str
    total_time_ms: float
    engine_times_ms: dict[str, float]
    cache_hits: dict[str, bool]
    result_count: int
    engines_used: list[str]
    optimization_target: str
    timestamp: float


@dataclass
class SearchResults:
    """Container for search results and metadata."""

    results: list[SearchResult]
    metrics: SearchMetrics
    total_results: int
    engines_used: list[str]
    cache_hit_rate: float


class UnifiedSearchSystem:
    """Unified search system combining Einstein, Bolt, and accelerated tools."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.orchestrator: SearchOrchestrator | None = None
        self.cache_system: HybridCacheSystem | None = None
        self.query_router: UnifiedQueryRouter | None = None

        # Performance tracking
        self.total_searches = 0
        self.total_time_ms = 0.0
        self.engine_usage = {}
        self.cache_stats = {"hits": 0, "misses": 0}

        # Hardware optimization settings for M4 Pro
        self.hardware_config = {
            "cpu_cores": 12,  # 8P + 4E cores
            "gpu_cores": 20,  # Metal GPU cores
            "max_concurrent_searches": 12,
            "unified_memory_gb": 24,
            "thermal_monitoring": True,
        }

        self.initialized = False

    async def initialize(self):
        """Initialize the unified search system."""
        if self.initialized:
            return

        start_time = time.perf_counter()

        logger.info("🚀 Initializing Unified Search System...")

        # Initialize core components
        self.cache_system = HybridCacheSystem(
            l1_size_mb=512, l2_size_gb=2, l3_size_gb=1
        )

        self.query_router = UnifiedQueryRouter(cache_system=self.cache_system)

        self.orchestrator = SearchOrchestrator(
            cache_system=self.cache_system,
            query_router=self.query_router,
            hardware_config=self.hardware_config,
        )

        # Initialize components
        await self.cache_system.initialize()
        await self.query_router.initialize()
        await self.orchestrator.initialize()

        # Warmup for optimal performance
        await self._warmup_system()

        self.initialized = True

        init_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"✅ Unified Search System initialized in {init_time:.1f}ms")

    async def search(
        self,
        query: str,
        search_types: list[str] | None = None,
        max_results: int = 50,
        optimization_target: str = "balanced",
        context: dict | None = None,
    ) -> SearchResults:
        """
        Execute unified search across all available engines.

        Args:
            query: Search query string
            search_types: List of search types to use ['text', 'semantic', 'code', 'analytical']
            max_results: Maximum results to return
            optimization_target: 'speed', 'accuracy', or 'balanced'
            context: Additional context for search optimization

        Returns:
            SearchResults with aggregated results and performance metrics
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.perf_counter()

        try:
            # Route query to determine optimal engines
            query_plan = await self.query_router.route_query(
                query=query,
                search_types=search_types,
                optimization_target=optimization_target,
                context=context,
            )

            # Execute search through orchestrator
            orchestrator_results = await self.orchestrator.execute_search(
                query=query, query_plan=query_plan, max_results=max_results
            )

            # Build unified results
            results = self._convert_to_unified_results(orchestrator_results)

            # Calculate metrics
            total_time = (time.perf_counter() - start_time) * 1000

            metrics = SearchMetrics(
                query=query,
                total_time_ms=total_time,
                engine_times_ms=orchestrator_results.engine_times,
                cache_hits=orchestrator_results.cache_hits,
                result_count=len(results),
                engines_used=orchestrator_results.engines_used,
                optimization_target=optimization_target,
                timestamp=time.time(),
            )

            # Update statistics
            self._update_stats(metrics)

            # Calculate cache hit rate
            cache_hit_rate = sum(
                1 for hit in orchestrator_results.cache_hits.values() if hit
            ) / max(1, len(orchestrator_results.cache_hits))

            search_results = SearchResults(
                results=results[:max_results],
                metrics=metrics,
                total_results=len(results),
                engines_used=orchestrator_results.engines_used,
                cache_hit_rate=cache_hit_rate,
            )

            # Log performance
            logger.debug(
                f"Search completed: {total_time:.1f}ms, "
                f"{len(results)} results, "
                f"engines: {orchestrator_results.engines_used}, "
                f"cache hit rate: {cache_hit_rate:.1%}"
            )

            return search_results

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}", exc_info=True)

            # Return empty results with error metrics
            error_time = (time.perf_counter() - start_time) * 1000
            return SearchResults(
                results=[],
                metrics=SearchMetrics(
                    query=query,
                    total_time_ms=error_time,
                    engine_times_ms={},
                    cache_hits={},
                    result_count=0,
                    engines_used=[],
                    optimization_target=optimization_target,
                    timestamp=time.time(),
                ),
                total_results=0,
                engines_used=[],
                cache_hit_rate=0.0,
            )

    async def search_burst(
        self,
        queries: list[str],
        max_concurrent: int = None,
        max_results_per_query: int = 20,
        optimization_target: str = "speed",
    ) -> list[SearchResults]:
        """
        Execute burst search for multiple queries concurrently.

        Optimized for M4 Pro with intelligent concurrency management.
        """
        if not self.initialized:
            await self.initialize()

        max_concurrent = (
            max_concurrent or self.hardware_config["max_concurrent_searches"]
        )

        logger.info(
            f"🎯 Processing burst of {len(queries)} queries with {max_concurrent} concurrent workers"
        )

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def search_with_semaphore(query: str) -> SearchResults:
            async with semaphore:
                return await self.search(
                    query=query,
                    max_results=max_results_per_query,
                    optimization_target=optimization_target,
                )

        # Execute all searches concurrently
        start_time = time.perf_counter()

        search_tasks = [search_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for query '{queries[i]}': {result}")
                # Create empty result for failed query
                valid_results.append(
                    SearchResults(
                        results=[],
                        metrics=SearchMetrics(
                            query=queries[i],
                            total_time_ms=0,
                            engine_times_ms={},
                            cache_hits={},
                            result_count=0,
                            engines_used=[],
                            optimization_target=optimization_target,
                            timestamp=time.time(),
                        ),
                        total_results=0,
                        engines_used=[],
                        cache_hit_rate=0.0,
                    )
                )
            else:
                valid_results.append(result)

        total_time = (time.perf_counter() - start_time) * 1000
        avg_time = total_time / len(queries) if queries else 0

        logger.info(
            f"✅ Burst complete: {len(queries)} queries, "
            f"avg {avg_time:.1f}ms/query, total {total_time:.1f}ms"
        )

        return valid_results

    def _convert_to_unified_results(self, orchestrator_results) -> list[SearchResult]:
        """Convert orchestrator results to unified format."""
        unified_results = []

        for (
            engine_name,
            engine_results,
        ) in orchestrator_results.results_by_engine.items():
            for result in engine_results:
                unified_result = SearchResult(
                    content=result.get("content", ""),
                    file_path=result.get("file_path", result.get("file", "")),
                    line_number=result.get("line_number", result.get("line", 0)),
                    score=result.get("score", result.get("similarity", 0.0)),
                    result_type=self._map_engine_to_type(engine_name),
                    context=result.get("context", {}),
                    timestamp=time.time(),
                    engine=engine_name,
                )
                unified_results.append(unified_result)

        # Sort by score descending
        unified_results.sort(key=lambda x: x.score, reverse=True)

        return unified_results

    def _map_engine_to_type(self, engine_name: str) -> str:
        """Map engine name to result type."""
        engine_type_map = {
            "text": "text",
            "ripgrep": "text",
            "semantic": "semantic",
            "vector": "semantic",
            "code": "code",
            "dependency": "code",
            "analytical": "analytical",
            "python": "analytical",
        }

        for key, result_type in engine_type_map.items():
            if key in engine_name.lower():
                return result_type

        return "unknown"

    def _update_stats(self, metrics: SearchMetrics):
        """Update system performance statistics."""
        self.total_searches += 1
        self.total_time_ms += metrics.total_time_ms

        # Update engine usage stats
        for engine in metrics.engines_used:
            self.engine_usage[engine] = self.engine_usage.get(engine, 0) + 1

        # Update cache stats
        cache_hits = sum(1 for hit in metrics.cache_hits.values() if hit)
        cache_total = len(metrics.cache_hits)

        self.cache_stats["hits"] += cache_hits
        self.cache_stats["misses"] += cache_total - cache_hits

    async def _warmup_system(self):
        """Warmup system for optimal performance."""
        logger.info("🔥 Warming up search engines...")

        # Warmup queries to compile kernels and load caches
        warmup_queries = [
            "def hello_world",  # Code search
            "import numpy",  # Text search
            "calculate options",  # Semantic search
            "TODO optimization",  # Mixed search
        ]

        warmup_tasks = []
        for query in warmup_queries:
            task = self.search(query=query, max_results=5, optimization_target="speed")
            warmup_tasks.append(task)

        # Execute warmup searches
        await asyncio.gather(*warmup_tasks, return_exceptions=True)

        logger.info("✅ System warmup complete")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        avg_time = self.total_time_ms / max(1, self.total_searches)

        cache_total = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / max(1, cache_total)

        stats = {
            "total_searches": self.total_searches,
            "average_latency_ms": avg_time,
            "total_time_ms": self.total_time_ms,
            "cache_hit_rate": cache_hit_rate,
            "engine_usage": dict(self.engine_usage),
            "hardware_config": self.hardware_config,
            "initialized": self.initialized,
        }

        # Add cache system stats if available
        if self.cache_system:
            stats["cache_system"] = self.cache_system.get_stats()

        # Add orchestrator stats if available
        if self.orchestrator:
            stats["orchestrator"] = self.orchestrator.get_stats()

        return stats

    async def optimize_performance(self):
        """Optimize system performance based on usage patterns."""
        if not self.initialized:
            return

        logger.info("🚀 Optimizing search system performance...")

        # Optimize cache system
        if self.cache_system:
            await self.cache_system.optimize()

        # Optimize query router
        if self.query_router:
            await self.query_router.optimize()

        # Optimize orchestrator
        if self.orchestrator:
            await self.orchestrator.optimize()

        logger.info("✅ Performance optimization complete")

    async def cleanup(self):
        """Cleanup resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()

        if self.cache_system:
            await self.cache_system.cleanup()

        if self.query_router:
            await self.query_router.cleanup()

        self.initialized = False
        logger.info("🧹 Unified search system cleaned up")
