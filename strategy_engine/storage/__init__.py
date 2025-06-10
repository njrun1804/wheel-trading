"""
Unified storage layer for wheel trading bot.
Local-first with DuckDB, optional GCS backup.
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


from .gcs_adapter import GCSAdapter, GCSConfig

__all__ = [
    "Storage",
    "StorageConfig",
    "GCSAdapter",
    "GCSConfig",
]

if _HAS_DUCKDB:
    __all__.extend(["DuckDBCache", "CacheConfig"])
