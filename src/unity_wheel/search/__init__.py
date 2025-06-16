"""Unified Search System - Consolidating Einstein, Bolt, and Jarvis2 search capabilities.

This module provides a single, high-performance search interface that combines:
- Einstein's intelligent multi-modal search orchestration
- Bolt's hardware acceleration and Metal GPU optimization  
- Jarvis2's code-specific vector search and embeddings
- Accelerated tools' 30x faster text search and analysis

Performance targets:
- <50ms for 95% of queries
- >100 searches/second sustained throughput
- >85% cache hit rate across all query types
- 11x faster than current best individual implementations
"""

from .engines import (
    AnalyticalSearchEngine,
    CodeAnalysisEngine,
    SemanticSearchEngine,
    TextSearchEngine,
)
from .hybrid_cache_system import HybridCacheSystem
from .search_orchestrator import SearchOrchestrator
from .unified_query_router import UnifiedQueryRouter
from .unified_search_system import UnifiedSearchSystem

__all__ = [
    "UnifiedSearchSystem",
    "SearchOrchestrator",
    "TextSearchEngine",
    "SemanticSearchEngine",
    "CodeAnalysisEngine",
    "AnalyticalSearchEngine",
    "HybridCacheSystem",
    "UnifiedQueryRouter",
]

# Global search instance for singleton access
_unified_search: UnifiedSearchSystem = None


async def get_unified_search() -> UnifiedSearchSystem:
    """Get global unified search instance."""
    global _unified_search
    if _unified_search is None:
        _unified_search = UnifiedSearchSystem()
        await _unified_search.initialize()
    return _unified_search


# Performance optimized search functions for direct use
async def search(
    query: str,
    search_types: list = None,
    max_results: int = 50,
    optimization_target: str = "balanced",
):
    """High-performance unified search - single entry point."""
    search_system = await get_unified_search()
    return await search_system.search(
        query=query,
        search_types=search_types,
        max_results=max_results,
        optimization_target=optimization_target,
    )


async def search_burst(queries: list, max_concurrent: int = 12):
    """Burst search processing for multiple queries."""
    search_system = await get_unified_search()
    return await search_system.search_burst(queries, max_concurrent)


# Specialized search shortcuts
async def search_code(query: str, max_results: int = 20):
    """Fast code-specific search."""
    return await search(
        query, search_types=["code", "semantic"], max_results=max_results
    )


async def search_text(query: str, max_results: int = 50):
    """Ultra-fast text search using RipgrepTurbo."""
    return await search(query, search_types=["text"], max_results=max_results)


async def search_semantic(query: str, max_results: int = 20):
    """GPU-accelerated semantic similarity search."""
    return await search(query, search_types=["semantic"], max_results=max_results)
