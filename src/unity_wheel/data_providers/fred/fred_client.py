"""FRED API client with rate limiting and autonomous error recovery."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union

import aiohttp
from aiohttp import ClientError, ClientSession, ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential

from src.unity_wheel.secrets.integration import get_fred_api_key
from src.unity_wheel.storage.cache.general_cache import cached
from src.unity_wheel.utils import RecoveryStrategy, get_logger, timed_operation, with_recovery
from src.unity_wheel.utils.data_validator import die

from .fred_models import (
    FREDDataset,
    FREDObservation,
    FREDSeries,
    UpdateFrequency,
    WheelStrategyFREDSeries,
)

logger = get_logger(__name__)


class FREDRateLimiter:
    """Rate limiter for FRED API (120 rpm)."""

    def __init__(self, rpm: int = 90):
        """Initialize with requests per minute limit."""
        self.rpm = rpm
        self.min_interval = 60.0 / rpm
        self.last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_request

            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

            self.last_request = time.time()


class FREDClient:
    """FRED API client with autonomous operation features."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rpm_limit: int = 90,
        timeout: int = 30,
    ):
        """
        Initialize FRED client.

        Parameters
        ----------
        api_key : str, optional
            FRED API key (optional, will use SecretManager if not provided)
        rpm_limit : int
            Requests per minute limit (default 90, max 120)
        timeout : int
            Request timeout in seconds
        """
        # Use provided API key or fall back to SecretManager
        if not api_key:
            logger.info("No API key provided, retrieving from SecretManager")
            api_key = get_ofred_api_key()

        self.api_key = api_key
        if not self.api_key:
            raise ValueError("FRED API key required")

        self.rate_limiter = FREDRateLimiter(min(rpm_limit, 115))
        self.timeout = ClientTimeout(total=timeout)
        self._session: Optional[ClientSession] = None

        logger.info("FRED client initialized", extra={"rpm_limit": rpm_limit, "timeout": timeout})

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    @property
    def session(self) -> ClientSession:
        """Get active session."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use 'async with FREDClient() as client:'")
        return self._session

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    async def _request(
        self,
        endpoint: str,
        params: Dict[str, Union[str, int]],
    ) -> Dict:
        """Make rate-limited request to FRED API."""
        await self.rate_limiter.acquire()

        params["api_key"] = self.api_key
        params["file_type"] = "json"

        url = f"{self.BASE_URL}/{endpoint}"

        logger.debug(f"FRED request: {endpoint}", extra={"params": params})

        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                # Check for API errors
                if "error_code" in data:
                    raise ValueError(f"FRED API error: {data.get('error_message', 'Unknown')}")

                return data

        except Exception as e:
            logger.error(f"FRED request failed: {e}", exc_info=True)
            raise

    @timed_operation(threshold_ms=100)
    @cached(ttl=timedelta(hours=24))
    async def get_series_metadata(self, series_id: str) -> FREDSeries:
        """
        Get series metadata.

        Parameters
        ----------
        series_id : str
            FRED series ID

        Returns
        -------
        FREDSeries
            Series metadata
        """
        data = await self._request("series", {"series_id": series_id})

        if "seriess" not in data or not data["seriess"]:
            raise ValueError(f"Series {series_id} not found")

        series_data = data["seriess"][0]

        # Parse dates and create model
        series = FREDSeries(
            series_id=series_data["id"],
            title=series_data["title"],
            observation_start=series_data["observation_start"],
            observation_end=series_data["observation_end"],
            frequency=series_data["frequency_short"],
            units=series_data["units"],
            seasonal_adjustment=series_data.get("seasonal_adjustment", "Not Applicable"),
            last_updated=series_data["last_updated"],
            popularity=series_data.get("popularity", 0),
            notes=series_data.get("notes"),
        )

        logger.info(
            f"Retrieved metadata for {series_id}",
            extra={
                "title": series.title,
                "frequency": series.frequency.value,
                "last_updated": series.last_updated.isoformat(),
            },
        )

        return series

    @timed_operation(threshold_ms=500)
    @with_recovery(strategy=RecoveryStrategy.RETRY)
    async def get_observations(
        self,
        series_id: str,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        limit: int = 100000,
        include_vintage: bool = False,
    ) -> List[FREDObservation]:
        """
        Get series observations.

        Parameters
        ----------
        series_id : str
            FRED series ID
        start_date : date, optional
            Start date for observations
        end_date : date, optional
            End date for observations
        limit : int
            Max observations per request
        include_vintage : bool
            Include vintage/revised data

        Returns
        -------
        List[FREDObservation]
            Series observations
        """
        params = {
            "series_id": series_id,
            "limit": min(limit, 100000),
        }

        if start_date:
            params["observation_start"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["observation_end"] = end_date.strftime("%Y-%m-%d")

        if include_vintage:
            params["output_type"] = "2"  # All vintage data

        observations = []
        offset = 0

        while True:
            params["offset"] = offset
            data = await self._request("series/observations", params)

            obs_data = data.get("observations", [])
            if not obs_data:
                break

            for obs in obs_data:
                # Validate observation has required fields
                if "date" not in obs:
                    die(f"Missing 'date' in FRED observation for series {series_id}")
                if "value" not in obs:
                    die(f"Missing 'value' in FRED observation for series {series_id}")

                observations.append(
                    FREDObservation(
                        date=obs["date"],
                        value=obs["value"],
                    )
                )

            # Check if more data available
            if len(obs_data) < params["limit"]:
                break

            offset += params["limit"]
            logger.debug(f"Fetching more observations, offset={offset}")

        logger.info(
            f"Retrieved {len(observations)} observations for {series_id}",
            extra={
                "date_range": (
                    f"{observations[0].date} to {observations[-1].date}"
                    if observations
                    else "empty"
                ),
            },
        )

        return observations

    async def get_dataset(
        self,
        series_id: str,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
    ) -> FREDDataset:
        """
        Get complete dataset with metadata and observations.

        Returns
        -------
        FREDDataset
            Complete dataset
        """
        # Fetch metadata and observations concurrently
        metadata_task = self.get_series_metadata(series_id)
        obs_task = self.get_observations(series_id, start_date, end_date)

        metadata, observations = await asyncio.gather(metadata_task, obs_task)

        return FREDDataset(
            series=metadata,
            observations=observations,
        )

    async def get_wheel_strategy_data(
        self,
        lookback_days: int = 1825,  # 5 years
    ) -> Dict[str, FREDDataset]:
        """
        Get all data series relevant to wheel strategy.

        Parameters
        ----------
        lookback_days : int
            Days of historical data to fetch

        Returns
        -------
        Dict[str, FREDDataset]
            Datasets keyed by series ID
        """
        start_date = Date.today() - timedelta(days=lookback_days)
        series_ids = [s.value for s in WheelStrategyFREDSeries]

        logger.info(
            f"Fetching {len(series_ids)} series for wheel strategy",
            extra={"series": series_ids, "start_date": start_date},
        )

        # Create tasks for concurrent fetching
        tasks = {series_id: self.get_dataset(series_id, start_date) for series_id in series_ids}

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(8)

        async def fetch_with_semaphore(series_id: str, task):
            async with semaphore:
                try:
                    return series_id, await task
                except Exception as e:
                    logger.error(f"Failed to fetch {series_id}: {e}")
                    return series_id, None

        # Execute all tasks
        results = await asyncio.gather(
            *[fetch_with_semaphore(sid, task) for sid, task in tasks.items()]
        )

        # Build results dictionary, filtering out failures
        datasets = {series_id: dataset for series_id, dataset in results if dataset is not None}

        logger.info(
            f"Successfully fetched {len(datasets)}/{len(series_ids)} series",
            extra={
                "fetched": list(datasets.keys()),
                "failed": [s for s in series_ids if s not in datasets],
            },
        )

        return datasets

    async def check_for_updates(
        self,
        series_id: str,
        last_known_date: Date,
    ) -> bool:
        """
        Check if series has new data since last known date.

        Returns
        -------
        bool
            True if new data available
        """
        metadata = await self.get_series_metadata(series_id)
        return metadata.observation_end > last_known_date
