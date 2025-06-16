"""Unified Search API - Single entry point for all search operations.

Consolidates Einstein, Bolt, Jarvis2, and accelerated tools into one high-performance API.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .cache_manager import get_cache_manager
from .search_orchestrator import SearchOrchestrator

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Available search types."""

    TEXT = "text"  # Fast text search (ripgrep)
    SEMANTIC = "semantic"  # Vector similarity search
    CODE = "code"  # Code-specific search with AST analysis
    ANALYTICAL = "analytical"  # Python analysis and dependency graphs
    HYBRID = "hybrid"  # Combined multi-modal search


class OptimizationTarget(Enum):
    """Search optimization targets."""

    SPEED = "speed"  # Fastest possible results
    ACCURACY = "accuracy"  # Most relevant results
    BALANCED = "balanced"  # Balance speed and accuracy


@dataclass
class SearchRequest:
    """Unified search request format."""

    query: str
    search_types: list[SearchType] = field(default_factory=lambda: [SearchType.HYBRID])
    max_results: int = 50
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED
    file_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    context: dict[str, Any] | None = None
    timeout_ms: int | None = None


@dataclass
class SearchMatch:
    """Individual search match result."""

    content: str
    file_path: str
    line_number: int
    column_start: int | None = None
    column_end: int | None = None
    score: float = 1.0
    match_type: str = "text"
    context_before: list[str] | None = None
    context_after: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Unified search response format."""

    request: SearchRequest
    matches: list[SearchMatch]
    total_matches: int
    search_time_ms: float
    engines_used: list[str]
    cache_hit: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedSearchAPI:
    """High-performance unified search API.

    Features:
    - Single entry point for all search types
    - Automatic engine selection based on query
    - Multi-tier caching for <5ms cached results
    - Parallel execution across engines
    - M4 Pro hardware optimization
    """

    def __init__(self):
        self.orchestrator: SearchOrchestrator | None = None
        self.cache_manager = None
        self.initialized = False

        # Performance tracking
        self.request_count = 0
        self.total_time_ms = 0.0
        self.cache_hits = 0

        # Engine registry
        self.available_engines: set[str] = set()

    async def initialize(self):
        """Initialize the unified search API."""
        if self.initialized:
            return

        logger.info("🚀 Initializing Unified Search API...")

        # Initialize cache manager
        self.cache_manager = await get_cache_manager()

        # Initialize orchestrator
        self.orchestrator = SearchOrchestrator()
        await self.orchestrator.initialize()

        # Discover available engines
        self.available_engines = await self.orchestrator.get_available_engines()

        self.initialized = True
        logger.info(
            f"✅ Unified Search API ready with engines: {self.available_engines}"
        )

    async def search(
        self, query: str | SearchRequest, **kwargs
    ) -> SearchResponse:
        """Execute unified search.

        Args:
            query: Search query string or SearchRequest object
            **kwargs: Additional parameters if query is a string

        Returns:
            SearchResponse with results from all applicable engines
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.perf_counter()

        # Normalize request
        if isinstance(query, str):
            request = SearchRequest(
                query=query,
                search_types=kwargs.get("search_types", [SearchType.HYBRID]),
                max_results=kwargs.get("max_results", 50),
                optimization_target=kwargs.get(
                    "optimization_target", OptimizationTarget.BALANCED
                ),
                file_patterns=kwargs.get("file_patterns"),
                exclude_patterns=kwargs.get("exclude_patterns"),
                context=kwargs.get("context"),
                timeout_ms=kwargs.get("timeout_ms"),
            )
        else:
            request = query

        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = await self.cache_manager.get(cache_key)

            if cached_response:
                self.cache_hits += 1
                cached_response.cache_hit = True
                cached_response.search_time_ms = (
                    time.perf_counter() - start_time
                ) * 1000
                return cached_response

            # Execute search through orchestrator
            matches, engines_used = await self.orchestrator.execute_search(request)

            # Build response
            search_time = (time.perf_counter() - start_time) * 1000

            response = SearchResponse(
                request=request,
                matches=matches[: request.max_results],
                total_matches=len(matches),
                search_time_ms=search_time,
                engines_used=engines_used,
                cache_hit=False,
                metadata={
                    "api_version": "2.0",
                    "hardware": "M4 Pro optimized",
                    "available_engines": list(self.available_engines),
                },
            )

            # Cache successful response
            if not response.error:
                await self.cache_manager.put(cache_key, response, ttl_seconds=300)

            # Update stats
            self.request_count += 1
            self.total_time_ms += search_time

            return response

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            search_time = (time.perf_counter() - start_time) * 1000

            return SearchResponse(
                request=request,
                matches=[],
                total_matches=0,
                search_time_ms=search_time,
                engines_used=[],
                cache_hit=False,
                error=str(e),
            )

    async def batch_search(
        self, requests: list[str | SearchRequest], max_concurrent: int = 12
    ) -> list[SearchResponse]:
        """Execute multiple searches concurrently.

        Optimized for M4 Pro with intelligent batching and parallelization.
        """
        if not self.initialized:
            await self.initialize()

        logger.info(f"🎯 Processing batch of {len(requests)} searches")

        # Normalize requests
        normalized_requests = []
        for req in requests:
            if isinstance(req, str):
                normalized_requests.append(SearchRequest(query=req))
            else:
                normalized_requests.append(req)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def search_with_semaphore(request: SearchRequest) -> SearchResponse:
            async with semaphore:
                return await self.search(request)

        # Execute all searches
        tasks = [search_with_semaphore(req) for req in normalized_requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = SearchResponse(
                    request=normalized_requests[i],
                    matches=[],
                    total_matches=0,
                    search_time_ms=0.0,
                    engines_used=[],
                    cache_hit=False,
                    error=str(response),
                )
                final_responses.append(error_response)
            else:
                final_responses.append(response)

        return final_responses

    async def stream_search(
        self, request: str | SearchRequest, chunk_size: int = 10
    ):
        """Stream search results as they become available.

        Yields results in chunks for responsive UI updates.
        """
        if not self.initialized:
            await self.initialize()

        # Normalize request
        if isinstance(request, str):
            request = SearchRequest(query=request)

        # Execute search with streaming
        async for chunk in self.orchestrator.stream_search(request, chunk_size):
            yield chunk

    def _generate_cache_key(self, request: SearchRequest) -> str:
        """Generate cache key for request."""
        components = [
            request.query,
            str([t.value for t in request.search_types]),
            str(request.max_results),
            request.optimization_target.value,
            str(request.file_patterns),
            str(request.exclude_patterns),
        ]
        return ":".join(components)

    async def get_stats(self) -> dict[str, Any]:
        """Get API performance statistics."""
        avg_time = self.total_time_ms / max(1, self.request_count)
        cache_hit_rate = self.cache_hits / max(1, self.request_count)

        stats = {
            "total_requests": self.request_count,
            "average_time_ms": avg_time,
            "cache_hit_rate": cache_hit_rate,
            "available_engines": list(self.available_engines),
            "initialized": self.initialized,
        }

        # Add cache stats
        if self.cache_manager:
            stats["cache"] = self.cache_manager.get_stats()

        # Add orchestrator stats
        if self.orchestrator:
            stats["orchestrator"] = await self.orchestrator.get_stats()

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on search system."""
        health = {"api_healthy": self.initialized, "timestamp": time.time()}

        if self.orchestrator:
            health["orchestrator"] = await self.orchestrator.health_check()

        if self.cache_manager:
            health["cache"] = {"healthy": True, "stats": self.cache_manager.get_stats()}

        return health

    async def optimize(self):
        """Optimize search performance based on usage patterns."""
        logger.info("🚀 Optimizing Unified Search API...")

        # Optimize cache
        if self.cache_manager:
            await self.cache_manager.optimize()

        # Optimize orchestrator
        if self.orchestrator:
            await self.orchestrator.optimize()

        logger.info("✅ Search optimization complete")

    async def cleanup(self):
        """Cleanup API resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()

        if self.cache_manager:
            await self.cache_manager.cleanup()

        self.initialized = False
        logger.info("🧹 Unified Search API cleaned up")


# Convenience functions for direct use

_api_instance: UnifiedSearchAPI | None = None


async def get_search_api() -> UnifiedSearchAPI:
    """Get global search API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = UnifiedSearchAPI()
        await _api_instance.initialize()
    return _api_instance


async def search(query: str, **kwargs) -> SearchResponse:
    """Quick search function."""
    api = await get_search_api()
    return await api.search(query, **kwargs)


async def search_code(query: str, max_results: int = 20) -> SearchResponse:
    """Search specifically for code."""
    api = await get_search_api()
    return await api.search(
        query,
        search_types=[SearchType.CODE, SearchType.SEMANTIC],
        max_results=max_results,
        optimization_target=OptimizationTarget.ACCURACY,
    )


async def search_text(query: str, max_results: int = 50) -> SearchResponse:
    """Fast text search."""
    api = await get_search_api()
    return await api.search(
        query,
        search_types=[SearchType.TEXT],
        max_results=max_results,
        optimization_target=OptimizationTarget.SPEED,
    )


async def search_semantic(query: str, max_results: int = 20) -> SearchResponse:
    """Semantic similarity search."""
    api = await get_search_api()
    return await api.search(
        query,
        search_types=[SearchType.SEMANTIC],
        max_results=max_results,
        optimization_target=OptimizationTarget.ACCURACY,
    )
