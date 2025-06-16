"""Unified local storage layer backed by DuckDB.

All data is stored locally using a DuckDB cache.  The module provides a
``get_or_fetch`` pattern for convenient caching of API responses and model
results.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..utils import get_logger, with_recovery
from .duckdb_cache import CacheConfig, DuckDBCache

logger = get_logger(__name__)


@dataclass
class StorageConfig:
    """Configuration for unified storage."""

    cache_config: CacheConfig = field(default_factory=CacheConfig)


class Storage:
    """Unified storage with get_or_fetch pattern and concurrent access support."""

    def __init__(self, config: StorageConfig | None = None):
        self.config = config or StorageConfig()

        # Initialize local cache
        self.cache = DuckDBCache(self.config.cache_config)

        # Track if we need to cleanup on shutdown
        self._initialized = False

    async def initialize(self):
        """Initialize storage components."""
        await self.cache.initialize()
        self._initialized = True
        logger.info("Storage system initialized with concurrent access support")

    async def get_or_fetch_option_chain(
        self,
        symbol: str,
        expiration: datetime,
        fetch_func: Callable,
        max_age_minutes: int = 15,
    ) -> dict[str, Any]:
        """Get option chain from cache or fetch if stale."""
        # Check cache first
        cached = await self.cache.get_option_chain(symbol, expiration, max_age_minutes)
        if cached:
            logger.info(
                "option_chain_cache_hit", symbol=symbol, expiration=expiration.date()
            )
            return cached

        # Cache miss - fetch fresh data
        logger.info(
            "option_chain_cache_miss", symbol=symbol, expiration=expiration.date()
        )

        # Fetch with recovery wrapper
        @with_recovery(max_attempts=3, backoff_factor=2.0)
        async def fetch_with_retry():
            return await fetch_func(symbol, expiration)

        chain_data = await fetch_with_retry()

        # Store in cache
        timestamp = datetime.utcnow()
        await self.cache.store_option_chain(
            symbol=symbol,
            expiration=expiration,
            timestamp=timestamp,
            spot_price=chain_data.get("spot_price", 0),
            chain_data=chain_data,
        )

        return chain_data

    async def get_or_fetch_positions(
        self, account_id: str, fetch_func: Callable, max_age_minutes: int = 30
    ) -> dict[str, Any]:
        """Get positions from cache or fetch if stale."""
        # Check cache first
        cached = await self.cache.get_latest_positions(account_id, max_age_minutes)
        if cached:
            logger.info("positions_cache_hit", account_id=account_id)
            return cached

        # Cache miss - fetch fresh data
        logger.info("positions_cache_miss", account_id=account_id)

        # Fetch with recovery
        @with_recovery(max_attempts=3)
        async def fetch_with_retry():
            return await fetch_func(account_id)

        data = await fetch_with_retry()

        # Store in cache
        await self.cache.store_positions(
            account_id=account_id,
            positions=data["positions"],
            account_data=data["account_data"],
        )

        return data

    async def store_prediction(
        self,
        prediction_id: str,
        input_features: dict[str, Any],
        predictions: dict[str, Any],
        model_version: str,
    ):
        """Store model prediction in cache with concurrent access support."""
        # Use concurrent database for writes if available
        if hasattr(self.cache, "_concurrent_db") and self.cache._concurrent_db:
            try:
                self.cache._concurrent_db.execute(
                    """
                    INSERT INTO predictions_cache
                    (prediction_id, timestamp, input_features, predictions, model_version)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        prediction_id,
                        datetime.utcnow(),
                        input_features,
                        predictions,
                        model_version,
                    ),
                )
                await self.cache._check_vacuum()
                return
            except Exception as e:
                logger.warning(
                    f"Concurrent prediction store failed, using fallback: {e}"
                )

        # Fallback to regular connection
        async with self.cache.connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions_cache
                (prediction_id, timestamp, input_features, predictions, model_version)
                VALUES (?, ?, ?, ?, ?)
            """,
                [
                    prediction_id,
                    datetime.utcnow(),
                    input_features,
                    predictions,
                    model_version,
                ],
            )

        await self.cache._check_vacuum()

    async def store_greeks(
        self,
        option_symbol: str,
        spot_price: float,
        risk_free_rate: float,
        greeks: dict[str, float],
    ):
        """Store Greeks calculation in cache."""
        async with self.cache.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO greeks_cache
                (option_symbol, timestamp, spot_price, risk_free_rate,
                 delta, gamma, theta, vega, rho, iv)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    option_symbol,
                    datetime.utcnow(),
                    spot_price,
                    risk_free_rate,
                    greeks.get("delta", 0),
                    greeks.get("gamma", 0),
                    greeks.get("theta", 0),
                    greeks.get("vega", 0),
                    greeks.get("rho", 0),
                    greeks.get("iv", 0),
                ],
            )

        await self.cache._check_vacuum()

    async def get_historical_data(
        self,
        dataset: str,
        start_date: datetime,
        end_date: datetime,
        symbols: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get historical data from the local cache."""
        results = []

        # First check local cache
        async with self.cache.connection() as conn:
            # Validate dataset name to prevent SQL injection
            valid_datasets = {"price_history", "option_chains", "fred_data"}
            if dataset not in valid_datasets:
                raise ValueError(f"Invalid dataset: {dataset}")

            query = f"""
                SELECT * FROM {dataset}
                WHERE timestamp >= ? AND timestamp <= ?
            """  # nosec B608 - dataset validated against whitelist
            params = [start_date, end_date]

            if symbols and dataset == "option_chains":
                query += " AND symbol IN (" + ",".join(["?"] * len(symbols)) + ")"
                params.extend(symbols)

            df = conn.execute(query, params).df()

            if not df.empty:
                results.extend(df.to_dict("records"))

        logger.info(
            "historical_data_retrieved",
            dataset=dataset,
            records=len(results),
            start=start_date.date(),
            end=end_date.date(),
        )

        return results

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        return await self.cache.get_storage_stats()

    async def cleanup_old_data(self):
        """Run maintenance to clean up old data."""
        await self.cache._vacuum()

    async def close(self):
        """Close storage system and cleanup resources."""
        if self._initialized:
            try:
                self.cache.close()
                logger.info("Storage system closed")
            except Exception as e:
                logger.error(f"Error closing storage system: {e}")
            finally:
                self._initialized = False

    def get_cache_lock_info(self) -> dict[str, Any] | None:
        """Get information about cache database locks."""
        return self.cache.get_lock_info()

    def force_unlock_cache(self) -> bool:
        """Force unlock cache database (use with caution)."""
        return self.cache.force_unlock()
