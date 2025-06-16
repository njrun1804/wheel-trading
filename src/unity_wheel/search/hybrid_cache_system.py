"""Hybrid Cache System - Alias for CacheManager with compatible interface."""

from .cache_manager import CacheManager, get_cache_manager

# Provide compatibility with existing code
HybridCacheSystem = CacheManager


async def get_hybrid_cache_system():
    """Get global hybrid cache system instance."""
    return await get_cache_manager()
