#!/usr/bin/env python3
"""
High-Performance Einstein Search System

Optimized for >20 ops/sec with M4 Pro hardware acceleration.
Combines all search strategies with intelligent caching and batching.
"""

import asyncio
import hashlib
import logging
import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.unity_wheel.accelerated_tools.dependency_graph_turbo import (
    get_dependency_graph,
)
from src.unity_wheel.accelerated_tools.python_analysis_turbo import get_python_analyzer

# Import accelerated tools
from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo

logger = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    """High-performance search request with batching support."""

    query: str
    search_type: str  # 'text', 'semantic', 'structural', 'batch'
    file_patterns: list[str] = None
    max_results: int = 100
    timeout_ms: int = 50  # Target: sub-50ms (configurable via EinsteinConfig)
    request_id: str = None
    timestamp: float = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = hashlib.md5(
                f"{self.query}:{self.search_type}:{time.time()}".encode()
            ).hexdigest()[:8]
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class SearchResult:
    """High-performance search result."""

    query: str
    file_path: str
    line_number: int
    content: str
    score: float
    search_type: str
    context_lines: list[str] = None
    metadata: dict[str, Any] = None
    processing_time_ms: float = 0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HighPerformanceCache:
    """Ultra-fast cache optimized for M4 Pro with automatic eviction."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, tuple[Any, float]] = {}
        self.access_order = deque()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "size": 0}

    def get(self, key: str) -> Any | None:
        """Get cached value with TTL check."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (most recent)
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
                self.access_order.append(key)
                self.stats["hits"] += 1
                return value
            else:
                # Expired
                del self.cache[key]

        self.stats["misses"] += 1
        return None

    def put(self, key: str, value: Any):
        """Store value with automatic eviction."""
        current_time = time.time()

        # Remove if already exists
        if key in self.cache:
            try:
                self.access_order.remove(key)
            except ValueError:
                pass

        # Evict if at capacity
        while len(self.cache) >= self.max_size and self.access_order:
            old_key = self.access_order.popleft()
            if old_key in self.cache:
                del self.cache[old_key]
                self.stats["evictions"] += 1

        # Store new value
        self.cache[key] = (value, current_time)
        self.access_order.append(key)
        self.stats["size"] = len(self.cache)

    def clear_expired(self):
        """Clear expired entries."""
        current_time = time.time()
        expired_keys = []

        for key, (value, timestamp) in self.cache.items():
            if current_time - timestamp >= self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]
            try:
                self.access_order.remove(key)
            except ValueError:
                pass

        self.stats["size"] = len(self.cache)
        return len(expired_keys)


class BatchSearchProcessor:
    """Batches multiple search requests for optimal throughput."""

    def __init__(self, batch_size: int = 10, batch_timeout_ms: int = 25):
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.pending_requests: list[SearchRequest] = []
        self.batch_lock = asyncio.Lock()
        self.processing_task = None

    async def add_request(self, request: SearchRequest) -> list[SearchResult]:
        """Add request to batch and return results."""
        # Create future for this request
        future = asyncio.Future()

        async with self.batch_lock:
            self.pending_requests.append((request, future))

            # Process if batch is full or timeout
            if len(self.pending_requests) >= self.batch_size:
                await self._process_batch()

        # Start timeout task if not already running
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._batch_timeout())

        return await future

    async def _process_batch(self):
        """Process current batch of requests."""
        if not self.pending_requests:
            return

        batch = self.pending_requests.copy()
        self.pending_requests.clear()

        # Cancel timeout task
        if self.processing_task:
            self.processing_task.cancel()
            self.processing_task = None

        # Process batch in parallel
        tasks = []
        for request, future in batch:
            task = asyncio.create_task(self._process_single_request(request, future))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_request(
        self, request: SearchRequest, future: asyncio.Future
    ):
        """Process a single request and set future result."""
        try:
            # This would be implemented to call the actual search
            results = await self._execute_search(request)
            future.set_result(results)
        except Exception as e:
            future.set_exception(e)

    async def _execute_search(self, request: SearchRequest) -> list[SearchResult]:
        """Execute the actual search using high-performance search engine."""
        try:
            # Get high-performance search instance
            search_engine = await get_high_performance_search(self.project_root)

            # Execute search using the engine
            results = await search_engine.search(
                query=request.query,
                search_type=request.search_type,
                max_results=request.max_results,
            )

            return results

        except Exception as e:
            logger.error(f"High-performance search failed: {e}")
            # Fallback to basic search
            return await self._fallback_search(request)

    async def _fallback_search(self, request: SearchRequest) -> list[SearchResult]:
        """Fallback search implementation."""
        # Basic pattern-based search
        results = []
        try:
            # Use basic file search if available
            import os

            for root, dirs, files in os.walk(self.project_root):
                for file in files:
                    if file.endswith((".py", ".md", ".txt", ".yml", ".yaml", ".json")):
                        file_path = os.path.join(root, file)
                        try:
                            with open(
                                file_path, encoding="utf-8", errors="ignore"
                            ) as f:
                                content = f.read()
                                if request.query.lower() in content.lower():
                                    # Find line number
                                    lines = content.split("\n")
                                    for i, line in enumerate(lines):
                                        if request.query.lower() in line.lower():
                                            results.append(
                                                SearchResult(
                                                    query=request.query,
                                                    file_path=file_path,
                                                    line_number=i + 1,
                                                    content=line.strip(),
                                                    score=0.8,
                                                    search_type=request.search_type,
                                                    processing_time_ms=5.0,
                                                )
                                            )
                                            if len(results) >= request.max_results:
                                                break
                        except Exception:
                            continue

                        if len(results) >= request.max_results:
                            break

                if len(results) >= request.max_results:
                    break

        except Exception as e:
            logger.error(f"Fallback search failed: {e}")

        return results[: request.max_results]

    async def _batch_timeout(self):
        """Process batch after timeout."""
        try:
            await asyncio.sleep(self.batch_timeout_ms / 1000.0)
            async with self.batch_lock:
                if self.pending_requests:
                    await self._process_batch()
        except asyncio.CancelledError:
            pass
        finally:
            self.processing_task = None


class HighPerformanceEinsteinSearch:
    """Ultra-fast Einstein search system targeting >20 ops/sec."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.cpu_count = mp.cpu_count()

        # CRITICAL FIX: Limit executors to prevent CPU overload
        # Use only 50% of cores for search and 25% for analysis to prevent system saturation
        self.search_executor = ThreadPoolExecutor(
            max_workers=max(2, min(6, self.cpu_count // 2))
        )
        self.analysis_executor = ProcessPoolExecutor(
            max_workers=max(1, min(3, self.cpu_count // 4))
        )

        # Ultra-fast caching
        self.search_cache = HighPerformanceCache(max_size=50000, ttl_seconds=600)
        self.file_cache = HighPerformanceCache(max_size=20000, ttl_seconds=1800)

        # Batch processing
        self.batch_processor = BatchSearchProcessor(batch_size=8, batch_timeout_ms=15)

        # Performance tracking
        self.performance_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_response_time_ms": 0,
            "ops_per_second": 0,
            "peak_ops_per_second": 0,
            "last_minute_ops": deque(maxlen=60),
            "response_times": deque(maxlen=100),
        }

        # Accelerated tool instances
        self.ripgrep = None
        self.dependency_graph = None
        self.python_analyzer = None

        # Search strategy cache
        self.strategy_cache = {}

    async def initialize(self):
        """Initialize high-performance search components."""
        logger.info("Initializing high-performance Einstein search...")

        # Initialize accelerated tools in parallel
        tasks = [
            self._init_ripgrep(),
            self._init_dependency_graph(),
            self._init_python_analyzer(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for initialization errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to initialize component {i}: {result}")

        logger.info("High-performance Einstein search initialized")

    async def _init_ripgrep(self):
        """Initialize ripgrep turbo."""
        try:
            self.ripgrep = get_ripgrep_turbo()
            logger.debug("Ripgrep turbo initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ripgrep turbo: {e}")
            raise

    async def _init_dependency_graph(self):
        """Initialize dependency graph turbo."""
        try:
            self.dependency_graph = get_dependency_graph(str(self.project_root))
            logger.debug("Dependency graph turbo initialized")
        except Exception as e:
            logger.error(f"Failed to initialize dependency graph turbo: {e}")
            raise

    async def _init_python_analyzer(self):
        """Initialize Python analyzer turbo."""
        try:
            self.python_analyzer = get_python_analyzer()
            logger.debug("Python analyzer turbo initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Python analyzer turbo: {e}")
            raise

    async def search(
        self, query: str, search_type: str = "auto", max_results: int = 100
    ) -> list[SearchResult]:
        """High-performance search with automatic optimization."""
        start_time = time.time()

        # Generate cache key
        cache_key = f"{search_type}:{query}:{max_results}"

        # Check cache first
        cached_results = self.search_cache.get(cache_key)
        if cached_results:
            self.performance_stats["cache_hits"] += 1
            self._update_performance_stats(time.time() - start_time, from_cache=True)
            return cached_results

        # Determine optimal search strategy
        if search_type == "auto":
            search_type = self._determine_search_strategy(query)

        # Create search request
        from einstein.einstein_config import get_einstein_config

        config = get_einstein_config()

        request = SearchRequest(
            query=query,
            search_type=search_type,
            max_results=max_results,
            timeout_ms=config.performance.high_performance_timeout_ms,  # Configurable timeout
        )

        # Execute search
        try:
            results = await self._execute_optimized_search(request)

            # Cache results
            self.search_cache.put(cache_key, results)

            # Update performance stats
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)

            return results

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

    def _determine_search_strategy(self, query: str) -> str:
        """Determine optimal search strategy based on query patterns."""
        # Cache strategy determination
        if query in self.strategy_cache:
            return self.strategy_cache[query]

        strategy = "text"  # Default

        # Pattern-based strategy selection
        if query.startswith(("class ", "def ", "async def ", "import ", "from ")):
            strategy = "structural"
        elif any(
            term in query.lower()
            for term in ["error", "exception", "bug", "todo", "fixme"]
        ):
            strategy = "text"
        elif len(query.split()) > 3:
            strategy = "semantic"
        elif query.startswith(("@", "self.", "cls.")):
            strategy = "structural"

        # Cache the strategy
        self.strategy_cache[query] = strategy
        return strategy

    async def _execute_optimized_search(
        self, request: SearchRequest
    ) -> list[SearchResult]:
        """Execute optimized search based on type."""

        if request.search_type == "text":
            return await self._text_search(request)
        elif request.search_type == "structural":
            return await self._structural_search(request)
        elif request.search_type == "semantic":
            return await self._semantic_search(request)
        else:
            # Fallback to text search
            return await self._text_search(request)

    async def _text_search(self, request: SearchRequest) -> list[SearchResult]:
        """Ultra-fast text search using ripgrep turbo."""
        if not self.ripgrep:
            return []

        try:
            # Use ripgrep turbo with parallel processing
            results = await self.ripgrep.parallel_search(
                patterns=[request.query],
                directory=str(self.project_root),
                max_results=request.max_results,
            )

            # Convert to SearchResult objects
            search_results = []
            for result in results[: request.max_results]:
                search_results.append(
                    SearchResult(
                        query=request.query,
                        file_path=result.get("file", ""),
                        line_number=result.get("line", 0),
                        content=result.get("content", ""),
                        score=result.get("score", 1.0),
                        search_type="text",
                        processing_time_ms=result.get("time_ms", 0),
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def _structural_search(self, request: SearchRequest) -> list[SearchResult]:
        """Structural search using dependency graph turbo."""
        if not self.dependency_graph:
            return []

        try:
            # Use dependency graph for structural searches
            symbols = await self.dependency_graph.find_symbol(request.query)

            search_results = []
            for symbol in symbols[: request.max_results]:
                search_results.append(
                    SearchResult(
                        query=request.query,
                        file_path=symbol.get("file", ""),
                        line_number=symbol.get("line", 0),
                        content=symbol.get("definition", ""),
                        score=symbol.get("confidence", 0.8),
                        search_type="structural",
                        metadata=symbol.get("metadata", {}),
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Structural search failed: {e}")
            return []

    async def _semantic_search(self, request: SearchRequest) -> list[SearchResult]:
        """Semantic search using Python analyzer turbo."""
        if not self.python_analyzer:
            return []

        try:
            # Use Python analyzer for semantic understanding
            analysis = await self.python_analyzer.semantic_search(
                query=request.query, max_results=request.max_results
            )

            search_results = []
            for item in analysis[: request.max_results]:
                search_results.append(
                    SearchResult(
                        query=request.query,
                        file_path=item.get("file", ""),
                        line_number=item.get("line", 0),
                        content=item.get("content", ""),
                        score=item.get("similarity", 0.7),
                        search_type="semantic",
                        metadata=item.get("context", {}),
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _update_performance_stats(
        self, processing_time: float, from_cache: bool = False
    ):
        """Update performance statistics."""
        processing_time_ms = processing_time * 1000

        self.performance_stats["total_searches"] += 1
        self.performance_stats["response_times"].append(processing_time_ms)

        # Calculate average response time
        if self.performance_stats["response_times"]:
            self.performance_stats["avg_response_time_ms"] = sum(
                self.performance_stats["response_times"]
            ) / len(self.performance_stats["response_times"])

        # Track ops per second
        current_time = time.time()
        self.performance_stats["last_minute_ops"].append(current_time)

        # Calculate current ops/sec (last minute)
        minute_ago = current_time - 60
        recent_ops = [
            t for t in self.performance_stats["last_minute_ops"] if t > minute_ago
        ]
        self.performance_stats["ops_per_second"] = len(recent_ops) / 60

        # Update peak
        if (
            self.performance_stats["ops_per_second"]
            > self.performance_stats["peak_ops_per_second"]
        ):
            self.performance_stats["peak_ops_per_second"] = self.performance_stats[
                "ops_per_second"
            ]

        # Log performance if significant
        if not from_cache and processing_time_ms > 50:
            logger.warning(f"Search took {processing_time_ms:.1f}ms (target: <50ms)")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics."""
        return {
            **self.performance_stats,
            "cache_stats": self.search_cache.stats,
            "file_cache_stats": self.file_cache.stats,
            "target_ops_per_second": 20,
            "target_met": self.performance_stats["ops_per_second"] >= 20,
        }

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up high-performance Einstein search...")

        # Shutdown executors
        self.search_executor.shutdown(wait=False)
        self.analysis_executor.shutdown(wait=False)

        # Clear caches
        self.search_cache.cache.clear()
        self.file_cache.cache.clear()

        logger.info("High-performance Einstein search cleanup complete")


# Global instance
_high_performance_search = None


async def get_high_performance_search(
    project_root: Path = None,
) -> HighPerformanceEinsteinSearch:
    """Get global high-performance search instance."""
    global _high_performance_search

    if _high_performance_search is None:
        _high_performance_search = HighPerformanceEinsteinSearch(project_root)
        await _high_performance_search.initialize()

    return _high_performance_search


if __name__ == "__main__":

    async def test_performance():
        """Test high-performance search system."""
        print("🚀 High-Performance Einstein Search Test")
        print("=" * 50)

        search = await get_high_performance_search()

        # Test queries
        test_queries = [
            "class WheelStrategy",
            "def calculate_delta",
            "import pandas",
            "async def process",
            "logger.info",
            "TODO",
            "Exception",
            "test_",
            "@property",
            "if __name__",
        ]

        print("Testing search performance...")
        start_time = time.time()

        # Run searches
        for query in test_queries:
            results = await search.search(query)
            print(f"  '{query}': {len(results)} results")

        # Test repeated searches (should be cached)
        print("\nTesting cache performance...")
        for query in test_queries[:3]:
            results = await search.search(query)
            print(f"  '{query}': {len(results)} results (cached)")

        total_time = time.time() - start_time
        total_ops = len(test_queries) + 3
        ops_per_second = total_ops / total_time

        print("\n📊 Performance Results:")
        print(f"  Total operations: {total_ops}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Ops/second: {ops_per_second:.1f}")
        print(f"  Target met: {'✅' if ops_per_second >= 20 else '❌'}")

        # Get detailed stats
        stats = search.get_performance_stats()
        print("\n📈 Detailed Stats:")
        print(f"  Average response: {stats['avg_response_time_ms']:.1f}ms")
        print(
            f"  Cache hit rate: {stats['cache_stats']['hits']}/{stats['cache_stats']['hits'] + stats['cache_stats']['misses']}"
        )
        print(f"  Peak ops/sec: {stats['peak_ops_per_second']:.1f}")

        await search.cleanup()

    asyncio.run(test_performance())
