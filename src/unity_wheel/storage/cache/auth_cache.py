"""
from __future__ import annotations

Response caching for graceful degradation during auth failures.
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

from unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class AuthCache:
    """Cache for API responses to enable offline operation."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_ttl: int = 3600,
        max_cache_size_mb: int = 100,
    ):
        """Initialize response cache.

        Args:
            cache_dir: Directory for cache storage
            default_ttl: Default TTL in seconds
            max_cache_size_mb: Maximum cache size in MB
        """
        self.cache_dir = cache_dir or Path.home() / ".wheel_trading" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size_bytes = max_cache_size_mb * 1024 * 1024

        # In-memory cache for hot data
        self._memory_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "__init__",
            cache_dir=str(self.cache_dir),
            default_ttl=default_ttl,
            max_size_mb=max_cache_size_mb,
        )

    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from endpoint and params."""
        key_data = {"endpoint": endpoint, "params": params or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        # Use subdirectories to avoid too many files in one directory
        return self.cache_dir / cache_key[:2] / cache_key[2:4] / f"{cache_key}.cache"

    def get(
        self, endpoint: str, params: Optional[Dict] = None, max_age: Optional[int] = None
    ) -> Optional[T]:
        """Get cached response if available and fresh.

        Args:
            endpoint: API endpoint
            params: Request parameters
            max_age: Maximum age in seconds (overrides TTL)

        Returns:
            Cached data or None
        """
        cache_key = self._get_cache_key(endpoint, params)

        # Check memory cache first
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            if self._is_fresh(cached, max_age):
                logger.debug("get", source="memory", endpoint=endpoint, hit=True)
                return cached["data"]

        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                # Use JSON instead of pickle for security
                with open(cache_path, "r") as f:
                    cached = json.load(f)

                if self._is_fresh(cached, max_age):
                    # Promote to memory cache
                    self._memory_cache[cache_key] = cached
                    logger.debug("get", source="disk", endpoint=endpoint, hit=True)
                    return cached["data"]
                else:
                    # Clean up stale cache
                    cache_path.unlink()

            except (ValueError, KeyError, AttributeError) as e:
                logger.error("get", endpoint=endpoint, error=str(e))
                cache_path.unlink()  # Remove corrupted cache

        logger.debug("get", endpoint=endpoint, hit=False)
        return None

    def set(
        self, endpoint: str, data: T, params: Optional[Dict] = None, ttl: Optional[int] = None
    ) -> None:
        """Cache response data.

        Args:
            endpoint: API endpoint
            data: Response data to cache
            params: Request parameters
            ttl: TTL in seconds (overrides default)
        """
        cache_key = self._get_cache_key(endpoint, params)
        ttl = ttl or self.default_ttl

        cached = {
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "ttl": ttl,
            "endpoint": endpoint,
            "params": params,
        }

        # Store in memory cache
        self._memory_cache[cache_key] = cached

        # Store on disk
        cache_path = self._get_cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use JSON instead of pickle for security
            with open(cache_path, "w") as f:
                json.dump(cached, f, default=str)

            logger.debug("set", endpoint=endpoint, ttl=ttl, size_bytes=cache_path.stat().st_size)

            # Check cache size
            self._enforce_size_limit()

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("set", endpoint=endpoint, error=str(e))

    def _is_fresh(self, cached: Dict[str, Any], max_age: Optional[int] = None) -> bool:
        """Check if cached data is still fresh."""
        timestamp = datetime.fromisoformat(cached["timestamp"])
        ttl = max_age or cached.get("ttl", self.default_ttl)

        age = (datetime.utcnow() - timestamp).total_seconds()
        return age < ttl

    def _enforce_size_limit(self) -> None:
        """Remove oldest cache entries if size limit exceeded."""
        total_size = 0
        cache_files = []

        # Collect all cache files with metadata
        for cache_file in self.cache_dir.rglob("*.cache"):
            stat = cache_file.stat()
            total_size += stat.st_size
            cache_files.append((cache_file, stat.st_mtime, stat.st_size))

        if total_size <= self.max_size_bytes:
            return

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])

        # Remove oldest files until under limit
        for cache_file, _, size in cache_files:
            if total_size <= self.max_size_bytes:
                break

            try:
                cache_file.unlink()
                total_size -= size
                logger.debug(
                    "_enforce_size_limit", action="removed", file=cache_file.name, size_bytes=size
                )
            except (ValueError, KeyError, AttributeError):
                pass  # File might already be deleted

    def get_fallback(self, endpoint: str, params: Optional[Dict] = None) -> Optional[T]:
        """Get cached data regardless of age (for offline mode).

        Args:
            endpoint: API endpoint
            params: Request parameters

        Returns:
            Cached data or None
        """
        cache_key = self._get_cache_key(endpoint, params)

        # Try memory cache
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            logger.warning(
                "get_fallback",
                source="memory",
                endpoint=endpoint,
                age_hours=self._get_age_hours(cached),
            )
            return cached["data"]

        # Try disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                # Use JSON instead of pickle for security
                with open(cache_path, "r") as f:
                    cached = json.load(f)

                logger.warning(
                    "get_fallback",
                    source="disk",
                    endpoint=endpoint,
                    age_hours=self._get_age_hours(cached),
                )
                return cached["data"]

            except (ValueError, KeyError, AttributeError) as e:
                logger.error("get_fallback", endpoint=endpoint, error=str(e))

        return None

    def _get_age_hours(self, cached: Dict[str, Any]) -> float:
        """Get age of cached data in hours."""
        timestamp = datetime.fromisoformat(cached["timestamp"])
        age = datetime.utcnow() - timestamp
        return age.total_seconds() / 3600

    def clear(self, endpoint: Optional[str] = None) -> None:
        """Clear cache entries.

        Args:
            endpoint: Clear only this endpoint, or all if None
        """
        if endpoint:
            # Clear specific endpoint
            for params_hash in list(self._memory_cache.keys()):
                if endpoint in params_hash:
                    del self._memory_cache[params_hash]

            # Clear from disk
            for cache_file in self.cache_dir.rglob("*.cache"):
                try:
                    # Use JSON instead of pickle for security
                    with open(cache_file, "r") as f:
                        cached = json.load(f)
                    if cached.get("endpoint") == endpoint:
                        cache_file.unlink()
                except (ValueError, KeyError, AttributeError):
                    import logging
                    logging.debug(f"Exception caught: {e}", exc_info=True)
                    pass
        else:
            # Clear all
            self._memory_cache.clear()
            for cache_file in self.cache_dir.rglob("*.cache"):
                try:
                    cache_file.unlink()
                except (ValueError, KeyError, AttributeError):
                    import logging
                    logging.debug(f"Exception caught: {e}", exc_info=True)
                    pass

        logger.info("clear", endpoint=endpoint or "all")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        file_count = 0
        endpoints = set()

        for cache_file in self.cache_dir.rglob("*.cache"):
            file_count += 1
            total_size += cache_file.stat().st_size

            try:
                # Use JSON instead of pickle for security
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                endpoints.add(cached.get("endpoint", "unknown"))
            except (ValueError, KeyError, AttributeError):
                import logging
                logging.debug(f"Exception caught: {e}", exc_info=True)
                pass

        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": file_count,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "unique_endpoints": len(endpoints),
            "endpoints": list(endpoints),
        }