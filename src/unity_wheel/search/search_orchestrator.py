"""Search Orchestrator - Coordinates multi-modal search execution.

Combines Einstein's optimized_unified_search orchestration with Bolt's hardware
acceleration and intelligent routing for optimal performance.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .engines import (
    AnalyticalSearchEngine,
    CodeAnalysisEngine,
    SemanticSearchEngine,
    TextSearchEngine,
)
from .hybrid_cache_system import HybridCacheSystem
from .unified_query_router import QueryPlan, UnifiedQueryRouter

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResults:
    """Results from search orchestrator."""

    results_by_engine: dict[str, list[dict]]
    engine_times: dict[str, float]
    cache_hits: dict[str, bool]
    engines_used: list[str]
    total_time_ms: float


class SearchOrchestrator:
    """Orchestrates search execution across multiple engines."""

    def __init__(
        self,
        cache_system: HybridCacheSystem,
        query_router: UnifiedQueryRouter,
        hardware_config: dict[str, Any],
    ):
        self.cache_system = cache_system
        self.query_router = query_router
        self.hardware_config = hardware_config

        # Search engines
        self.engines: dict[str, Any] = {}

        # Performance tracking
        self.execution_stats = {
            "total_executions": 0,
            "engine_performance": {},
            "cache_effectiveness": {},
        }

        # Concurrency management for M4 Pro
        self.max_concurrent = hardware_config.get("max_concurrent_searches", 12)
        self.engine_semaphore = asyncio.Semaphore(self.max_concurrent)

        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=hardware_config.get("cpu_cores", 12)
        )

        self.initialized = False

    async def initialize(self):
        """Initialize search engines."""
        if self.initialized:
            return

        logger.info("🔧 Initializing Search Orchestrator...")

        # Initialize search engines
        self.engines = {
            "text": TextSearchEngine(
                cache_system=self.cache_system, hardware_config=self.hardware_config
            ),
            "semantic": SemanticSearchEngine(
                cache_system=self.cache_system, hardware_config=self.hardware_config
            ),
            "code": CodeAnalysisEngine(
                cache_system=self.cache_system, hardware_config=self.hardware_config
            ),
            "analytical": AnalyticalSearchEngine(
                cache_system=self.cache_system, hardware_config=self.hardware_config
            ),
        }

        # Initialize all engines in parallel
        init_tasks = []
        for engine_name, engine in self.engines.items():
            task = self._initialize_engine_with_error_handling(engine_name, engine)
            init_tasks.append(task)

        await asyncio.gather(*init_tasks)

        self.initialized = True
        logger.info("✅ Search Orchestrator initialized")

    async def _initialize_engine_with_error_handling(self, name: str, engine):
        """Initialize engine with error handling."""
        try:
            await engine.initialize()
            logger.debug(f"✅ {name} engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {name} engine: {e}")
            # Keep engine but mark as unavailable
            if hasattr(engine, "available"):
                engine.available = False

    async def execute_search(
        self, query: str, query_plan: QueryPlan, max_results: int = 50
    ) -> OrchestratorResults:
        """Execute search across planned engines."""
        if not self.initialized:
            await self.initialize()

        start_time = time.perf_counter()

        # Results containers
        results_by_engine = {}
        engine_times = {}
        cache_hits = {}
        engines_used = []

        # Create search tasks for each planned engine
        search_tasks = []

        for engine_config in query_plan.engines:
            engine_name = engine_config["name"]
            engine_params = engine_config.get("params", {})

            if engine_name in self.engines:
                task = self._execute_engine_search(
                    engine_name=engine_name,
                    query=query,
                    max_results=max_results,
                    params=engine_params,
                )
                search_tasks.append((engine_name, task))

        # Execute searches in parallel with proper error handling
        task_results = await asyncio.gather(
            *[task for _, task in search_tasks], return_exceptions=True
        )

        # Process results
        for (engine_name, _), result in zip(search_tasks, task_results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Search failed for {engine_name}: {result}")
                results_by_engine[engine_name] = []
                engine_times[engine_name] = 0.0
                cache_hits[engine_name] = False
            else:
                engine_results, engine_time, cache_hit = result
                results_by_engine[engine_name] = engine_results
                engine_times[engine_name] = engine_time
                cache_hits[engine_name] = cache_hit
                engines_used.append(engine_name)

        total_time = (time.perf_counter() - start_time) * 1000

        # Update statistics
        self._update_execution_stats(engines_used, engine_times, cache_hits)

        return OrchestratorResults(
            results_by_engine=results_by_engine,
            engine_times=engine_times,
            cache_hits=cache_hits,
            engines_used=engines_used,
            total_time_ms=total_time,
        )

    async def _execute_engine_search(
        self, engine_name: str, query: str, max_results: int, params: dict
    ) -> tuple:
        """Execute search on a specific engine with caching and error handling."""

        # Check cache first
        cache_key = f"{engine_name}:{query}:{max_results}:{hash(str(params))}"
        cached_result = await self.cache_system.get(cache_key)

        if cached_result is not None:
            return cached_result["results"], cached_result["time_ms"], True

        # Execute search with concurrency control
        async with self.engine_semaphore:
            start_time = time.perf_counter()

            try:
                engine = self.engines[engine_name]

                # Check if engine is available
                if hasattr(engine, "available") and not engine.available:
                    return [], 0.0, False

                # Execute search
                results = await engine.search(
                    query=query, max_results=max_results, **params
                )

                search_time = (time.perf_counter() - start_time) * 1000

                # Cache results
                cache_data = {"results": results, "time_ms": search_time}
                await self.cache_system.put(cache_key, cache_data, ttl_seconds=300)

                return results, search_time, False

            except Exception as e:
                logger.error(f"Engine {engine_name} search failed: {e}")
                search_time = (time.perf_counter() - start_time) * 1000
                return [], search_time, False

    def _update_execution_stats(
        self,
        engines_used: list[str],
        engine_times: dict[str, float],
        cache_hits: dict[str, bool],
    ):
        """Update execution statistics."""
        self.execution_stats["total_executions"] += 1

        # Update engine performance stats
        for engine_name in engines_used:
            if engine_name not in self.execution_stats["engine_performance"]:
                self.execution_stats["engine_performance"][engine_name] = {
                    "total_calls": 0,
                    "total_time_ms": 0.0,
                    "avg_time_ms": 0.0,
                    "error_count": 0,
                }

            stats = self.execution_stats["engine_performance"][engine_name]
            stats["total_calls"] += 1
            stats["total_time_ms"] += engine_times.get(engine_name, 0.0)
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_calls"]

        # Update cache effectiveness stats
        for engine_name, cache_hit in cache_hits.items():
            if engine_name not in self.execution_stats["cache_effectiveness"]:
                self.execution_stats["cache_effectiveness"][engine_name] = {
                    "hits": 0,
                    "misses": 0,
                    "hit_rate": 0.0,
                }

            cache_stats = self.execution_stats["cache_effectiveness"][engine_name]
            if cache_hit:
                cache_stats["hits"] += 1
            else:
                cache_stats["misses"] += 1

            total = cache_stats["hits"] + cache_stats["misses"]
            cache_stats["hit_rate"] = cache_stats["hits"] / total if total > 0 else 0.0

    async def warmup_engines(self):
        """Warmup all engines for optimal performance."""
        if not self.initialized:
            await self.initialize()

        logger.info("🔥 Warming up search engines...")

        # Warmup queries for each engine type
        warmup_queries = {
            "text": ["def main", "import sys", "TODO"],
            "semantic": ["calculate function", "options pricing", "machine learning"],
            "code": ["WheelStrategy", "calculate_delta", "risk_management"],
            "analytical": ["numpy.mean", "pandas.DataFrame", "statistics"],
        }

        warmup_tasks = []
        for engine_name, queries in warmup_queries.items():
            if engine_name in self.engines:
                for query in queries:
                    task = self._execute_engine_search(
                        engine_name=engine_name, query=query, max_results=5, params={}
                    )
                    warmup_tasks.append(task)

        # Execute warmup in parallel
        await asyncio.gather(*warmup_tasks, return_exceptions=True)

        logger.info("✅ Engine warmup complete")

    async def optimize(self):
        """Optimize orchestrator performance based on usage patterns."""
        logger.info("🚀 Optimizing Search Orchestrator...")

        # Analyze engine performance and adjust priorities
        best_performing_engines = []
        worst_performing_engines = []

        for engine_name, stats in self.execution_stats["engine_performance"].items():
            if stats["total_calls"] > 10:  # Only consider engines with sufficient data
                if stats["avg_time_ms"] < 50:  # Fast engines
                    best_performing_engines.append(engine_name)
                elif stats["avg_time_ms"] > 200:  # Slow engines
                    worst_performing_engines.append(engine_name)

        # Optimize engine configurations
        for engine_name in best_performing_engines:
            if engine_name in self.engines:
                engine = self.engines[engine_name]
                if hasattr(engine, "increase_priority"):
                    await engine.increase_priority()

        for engine_name in worst_performing_engines:
            if engine_name in self.engines:
                engine = self.engines[engine_name]
                if hasattr(engine, "optimize_performance"):
                    await engine.optimize_performance()

        # Optimize individual engines
        optimization_tasks = []
        for engine in self.engines.values():
            if hasattr(engine, "optimize"):
                optimization_tasks.append(engine.optimize())

        if optimization_tasks:
            await asyncio.gather(*optimization_tasks, return_exceptions=True)

        logger.info("✅ Orchestrator optimization complete")

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        stats = {
            "initialized": self.initialized,
            "total_executions": self.execution_stats["total_executions"],
            "available_engines": list(self.engines.keys()),
            "engine_performance": dict(self.execution_stats["engine_performance"]),
            "cache_effectiveness": dict(self.execution_stats["cache_effectiveness"]),
            "hardware_config": self.hardware_config,
        }

        # Add individual engine stats
        engine_stats = {}
        for engine_name, engine in self.engines.items():
            if hasattr(engine, "get_stats"):
                try:
                    engine_stats[engine_name] = engine.get_stats()
                except Exception as e:
                    engine_stats[engine_name] = {"error": str(e)}

        stats["individual_engine_stats"] = engine_stats

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all engines."""
        health_status = {"orchestrator_healthy": self.initialized, "engines": {}}

        for engine_name, engine in self.engines.items():
            try:
                if hasattr(engine, "health_check"):
                    engine_health = await engine.health_check()
                else:
                    # Basic health check - try a simple search
                    await engine.search("test", max_results=1)
                    engine_health = {
                        "healthy": True,
                        "response_time_ms": 0.0,
                        "available": True,
                    }

                health_status["engines"][engine_name] = engine_health

            except Exception as e:
                health_status["engines"][engine_name] = {
                    "healthy": False,
                    "error": str(e),
                    "available": False,
                }

        return health_status

    async def cleanup(self):
        """Cleanup orchestrator resources."""
        logger.info("🧹 Cleaning up Search Orchestrator...")

        # Cleanup engines
        cleanup_tasks = []
        for engine in self.engines.values():
            if hasattr(engine, "cleanup"):
                cleanup_tasks.append(engine.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)

        self.initialized = False
        logger.info("✅ Search Orchestrator cleanup complete")
