"""Einstein Unified Indexing System

Built using all available Jarvis accelerated tools with auto-detected hardware optimization.
"""

from .query_router import QueryRouter
from .result_merger import ResultMerger
from .unified_index import EinsteinIndexHub

# Backward compatibility alias
UnifiedIndex = EinsteinIndexHub


def get_unified_index():
    """Get unified index instance for backward compatibility."""
    return EinsteinIndexHub()


__all__ = [
    "EinsteinIndexHub",
    "UnifiedIndex",
    "QueryRouter",
    "ResultMerger",
    "get_unified_index",
]
