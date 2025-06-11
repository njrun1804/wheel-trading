"""Local storage utilities for the wheel trading bot.

The storage layer is backed solely by DuckDB and provides helpers for
interacting with the local cache.
"""

from .storage import Storage, StorageConfig

# Optional DuckDB imports - only if duckdb is installed
try:
    from .duckdb_cache import CacheConfig, DuckDBCache

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

    # Create dummy classes for type hints
    class DuckDBCache:  # type: ignore
        pass

    class CacheConfig:  # type: ignore
        pass


__all__ = [
    "Storage",
    "StorageConfig",
]

if _HAS_DUCKDB:
    __all__.extend(["DuckDBCache", "CacheConfig"])
