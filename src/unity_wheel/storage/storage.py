"""
Unified storage layer with local-first DuckDB cache and GCS backup.
Implements get_or_fetch pattern for all data types.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ..utils import get_logger, with_recovery
from .duckdb_cache import CacheConfig, DuckDBCache
from .gcs_adapter import GCSAdapter, GCSConfig

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class StorageConfig:
    """Configuration for unified storage."""

    cache_config: CacheConfig = field(default_factory=CacheConfig)
    gcs_config: Optional[GCSConfig] = None
    enable_gcs_backup: bool = True
    backup_interval_hours: int = 24


class Storage:
    """Unified storage with get_or_fetch pattern."""

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()

        # Initialize components
        self.cache = DuckDBCache(self.config.cache_config)
        self.gcs = GCSAdapter(self.config.gcs_config) if self.config.enable_gcs_backup else None

        # Track last backup time
        self._last_backup: Dict[str, datetime] = {}

    async def initialize(self):
        """Initialize storage components."""
        await self.cache.initialize()

        if self.gcs and self.gcs.enabled:
            self.gcs.set_lifecycle_policy()

    async def get_or_fetch_option_chain(
        self, symbol: str, expiration: datetime, fetch_func: Callable, max_age_minutes: int = 15
    ) -> Dict[str, Any]:
        """Get option chain from cache or fetch if stale."""
        # Check cache first
        cached = await self.cache.get_option_chain(symbol, expiration, max_age_minutes)
        if cached:
            logger.info("option_chain_cache_hit", symbol=symbol, expiration=expiration.date())
            return cached

        # Cache miss - fetch fresh data
        logger.info("option_chain_cache_miss", symbol=symbol, expiration=expiration.date())

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

        # Backup raw response to GCS if enabled
        if self.gcs and self.gcs.enabled:
            await self.gcs.upload_raw_response(
                source=f"options_{symbol}", timestamp=timestamp, data=chain_data
            )

        # Check if we need to backup to Parquet
        await self._check_backup("option_chains")

        return chain_data

    async def get_or_fetch_positions(
        self, account_id: str, fetch_func: Callable, max_age_minutes: int = 30
    ) -> Dict[str, Any]:
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
            account_id=account_id, positions=data["positions"], account_data=data["account_data"]
        )

        # Backup raw response to GCS
        if self.gcs and self.gcs.enabled:
            await self.gcs.upload_raw_response(
                source=f"positions_{account_id}", timestamp=datetime.utcnow(), data=data
            )

        # Check if we need to backup
        await self._check_backup("position_snapshots")

        return data

    async def store_prediction(
        self,
        prediction_id: str,
        input_features: Dict[str, Any],
        predictions: Dict[str, Any],
        model_version: str,
    ):
        """Store model prediction in cache."""
        async with self.cache.connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions_cache
                (prediction_id, timestamp, input_features, predictions, model_version)
                VALUES (?, ?, ?, ?, ?)
            """,
                [prediction_id, datetime.utcnow(), input_features, predictions, model_version],
            )

        await self.cache._check_vacuum()

    async def store_greeks(
        self, option_symbol: str, spot_price: float, risk_free_rate: float, greeks: Dict[str, float]
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
        symbols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get historical data from cache or GCS."""
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

        # If we need more data and GCS is enabled
        if self.gcs and self.gcs.enabled and len(results) == 0:
            # List available files in GCS
            files = await self.gcs.list_parquet_files(
                dataset=dataset, start_date=start_date, end_date=end_date
            )

            # Download and import relevant files
            for file_path in files[:5]:  # Limit to avoid huge downloads
                local_file = await self.gcs.download_parquet(
                    file_path, self.config.cache_config.cache_dir / "downloads"
                )

                if local_file:
                    await self.cache.import_from_parquet(local_file, dataset)

            # Query again after import
            async with self.cache.connection() as conn:
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

    async def _check_backup(self, table: str):
        """Check if we need to backup table to GCS."""
        if not self.gcs or not self.gcs.enabled:
            return

        last_backup = self._last_backup.get(table, datetime.min)
        hours_since = (datetime.utcnow() - last_backup).total_seconds() / 3600

        if hours_since >= self.config.backup_interval_hours:
            await self._backup_table(table)

    async def _backup_table(self, table: str):
        """Backup table to GCS as Parquet."""
        try:
            # Export to local Parquet
            export_dir = self.config.cache_config.cache_dir / "exports"
            export_dir.mkdir(exist_ok=True)

            parquet_file = await self.cache.export_to_parquet(table, export_dir)

            # Upload to GCS
            await self.gcs.upload_parquet(parquet_file, table)

            # Clean up local file
            parquet_file.unlink()

            # Update last backup time
            self._last_backup[table] = datetime.utcnow()

            logger.info("table_backed_up", table=table)

        except Exception as e:
            logger.error("backup_failed", table=table, error=str(e))

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = await self.cache.get_storage_stats()

        # Add backup status
        stats["last_backups"] = {
            table: last.isoformat() if last != datetime.min else "never"
            for table, last in self._last_backup.items()
        }

        # Add GCS status
        stats["gcs_enabled"] = self.gcs.enabled if self.gcs else False

        return stats

    async def cleanup_old_data(self):
        """Run maintenance to clean up old data."""
        await self.cache._vacuum()

        # Also clean up old export files
        export_dir = self.config.cache_config.cache_dir / "exports"
        if export_dir.exists():
            for file in export_dir.glob("*.parquet"):
                if file.stat().st_mtime < (datetime.utcnow() - timedelta(days=7)).timestamp():
                    file.unlink()
                    logger.info("cleaned_old_export", file=file.name)
