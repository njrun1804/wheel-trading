"""
Unified storage layer for wheel trading bot.
Local-first with DuckDB, optional GCS backup.
"""

from .duckdb_cache import CacheConfig, DuckDBCache
from .gcs_adapter import GCSAdapter, GCSConfig
from .storage import Storage, StorageConfig

__all__ = [
    "Storage",
    "StorageConfig",
    "DuckDBCache",
    "CacheConfig",
    "GCSAdapter",
    "GCSConfig",
]
