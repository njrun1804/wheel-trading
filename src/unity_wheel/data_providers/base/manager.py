"""Data manager for coordinating FRED data fetching and storage."""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta
from typing import Dict, Optional, Tuple

import numpy as np

from src.unity_wheel.storage.storage import Storage
from src.unity_wheel.utils import get_logger, timed_operation, with_recovery

from ..fred.fred_client import FREDClient
from ..fred.fred_models import FREDDataset, WheelStrategyFREDSeries
from ..fred.fred_storage import FREDStorage
from .validation import get_anomaly_detector

logger = get_logger(__name__)


class FREDDataManager:
    """Manages FRED data fetching, storage, and feature calculation."""

    async def get_or_fetch_risk_free_rate(
        self,
        tenor_months: int = 3,
        fetch_if_stale_days: int = 1,
    ) -> Tuple[float, float]:
        """
        Get risk-free rate using get_or_fetch pattern.

        Returns
        -------
        Tuple[float, float]
            (rate as decimal, confidence)
        """
        series_map = {3: "DGS3", 12: "DGS1"}
        series_id = series_map.get(tenor_months, "DGS3")

        # Check if we have recent data
        latest = await self.fred_storage.get_latest_observation(series_id)

        if latest:
            value, obs_date = latest
            days_old = (Date.today() - obs_date).days

            if days_old <= fetch_if_stale_days:
                # Data is fresh enough
                rate = value / 100.0
                confidence = 1.0
                return rate, confidence

        # Data is stale or missing, fetch new
        logger.info(f"Fetching fresh data for {series_id}")

        @with_recovery(max_attempts=3)
        async def fetch_latest():
            async with FREDClient(self.api_key) as client:
                # Fetch last 7 days to ensure we get latest
                start_date = Date.today() - timedelta(days=7)
                observations = await client.get_observations(series_id, start_date=start_date)

                if observations:
                    # Save the new observations
                    await self.fred_storage.save_observations(series_id, observations)

                    # Return the latest non-null value
                    for obs in reversed(observations):
                        if obs.value is not None:
                            return obs.value / 100.0, 1.0

            return None

        result = await fetch_latest()
        if result:
            return result

        # Fallback
        logger.warning(f"Failed to fetch {series_id}, using default")
        return 0.05, 0.5

    def __init__(
        self,
        storage: Optional[Storage] = None,
        api_key: Optional[str] = None,
        auto_update: bool = True,
    ):
        """
        Initialize data manager.

        Parameters
        ----------
        storage : Storage, optional
            Unified storage instance
        api_key : str, optional
            FRED API key (will use SecretManager if not provided)
        auto_update : bool
            Enable automatic data updates
        """
        self.storage = storage or Storage()
        self.fred_storage = FREDStorage(self.storage)
        self.api_key = api_key
        self.auto_update = auto_update
        self.anomaly_detector = get_anomaly_detector()
        self._client: Optional[FREDClient] = None

        logger.info("FRED data manager initialized", extra={"auto_update": auto_update})

    async def initialize_data(self, lookback_days: int = 1825) -> Dict[str, int]:
        """
        Initialize database with historical data.

        Returns
        -------
        Dict[str, int]
            Series ID -> observation count
        """
        logger.info(f"Initializing FRED data with {lookback_days} days lookback")

        async with FREDClient(self.api_key) as client:
            datasets = await client.get_wheel_strategy_data(lookback_days)

        # Initialize storage
        await self.storage.initialize()
        await self.fred_storage.initialize()

        # Save all datasets
        counts = {}
        for series_id, dataset in datasets.items():
            if dataset:
                await self.fred_storage.save_dataset(dataset)
                counts[series_id] = len(dataset.observations)

                # Calculate and store features
                await self._calculate_features(series_id, dataset)

        logger.info(
            "Data initialization complete",
            extra={"series_count": len(counts), "total_observations": sum(counts.values())},
        )

        return counts

    async def update_data(self) -> Dict[str, int]:
        """
        Update data with latest observations.

        Returns
        -------
        Dict[str, int]
            Series ID -> new observation count
        """
        updates = {}

        async with FREDClient(self.api_key) as client:
            for series_enum in WheelStrategyFREDSeries:
                series_id = series_enum.value

                # Get last observation date
                last_obs = await self.fred_storage.get_latest_observation(series_id)
                last_date = last_obs[1] if last_obs else None

                if last_date:
                    # Check if updates available
                    has_updates = await client.check_for_updates(series_id, last_date)

                    if has_updates:
                        # Fetch only new data
                        start_date = last_date + timedelta(days=1)
                        dataset = await client.get_dataset(series_id, start_date)

                        if dataset.observations:
                            # Validate new data
                            issues = self.anomaly_detector.check_observations(
                                series_id,
                                [obs.value for obs in dataset.observations if obs.value],
                            )

                            if not issues:
                                count = await self.fred_storage.save_observations(
                                    series_id, dataset.observations
                                )
                                updates[series_id] = count

                                # Recalculate features
                                await self._calculate_features(series_id, dataset)
                            else:
                                logger.warning(
                                    f"Data quality issues detected for {series_id}",
                                    extra={"issues": issues},
                                )
                else:
                    # No data exists, fetch all
                    dataset = await client.get_dataset(series_id)
                    await self.fred_storage.save_dataset(dataset)
                    updates[series_id] = len(dataset.observations)

        if updates:
            logger.info(
                "Data update complete",
                extra={
                    "updated_series": list(updates.keys()),
                    "new_observations": sum(updates.values()),
                },
            )
        else:
            logger.debug("No new data available")

        return updates

    @timed_operation(threshold_ms=50)
    async def get_latest_values(self) -> Dict[str, Tuple[float, Date]]:
        """
        Get latest value for each series.

        Returns
        -------
        Dict[str, Tuple[float, Date]]
            Series ID -> (value, date)
        """
        values = {}

        for series_enum in WheelStrategyFREDSeries:
            series_id = series_enum.value
            latest = await self.fred_storage.get_latest_observation(series_id)

            if latest:
                values[series_id] = latest

        return values

    async def get_risk_free_rate(self, tenor_months: int = 3) -> Tuple[float, float]:
        """
        Get risk-free rate for given tenor.

        Returns
        -------
        Tuple[float, float]
            (rate as decimal, confidence)
        """
        # Map tenor to series
        series_map = {
            3: "DGS3",
            12: "DGS1",
        }

        series_id = series_map.get(tenor_months, "DGS3")

        # Check cached risk metrics first
        today = Date.today()
        risk_metrics = await self.fred_storage.get_risk_metrics(today)

        if risk_metrics:
            rate_key = f"risk_free_rate_{tenor_months}m"
            if rate_key in risk_metrics and risk_metrics[rate_key] is not None:
                return risk_metrics[rate_key], 1.0

        # Fallback to latest observation
        latest = await self.fred_storage.get_latest_observation(series_id)

        if latest:
            value, obs_date = latest
            # Convert from percentage to decimal
            rate = value / 100.0

            # Check data freshness for confidence
            days_old = (today - obs_date).days
            confidence = 1.0 if days_old <= 1 else max(0.8, 1.0 - days_old * 0.05)

            # Cache the result
            await self.fred_storage.save_risk_metrics(
                today, {f"risk_free_rate_{tenor_months}m": rate}
            )

            return rate, confidence

        # Fallback to reasonable default
        logger.warning(f"No risk-free rate data for {tenor_months}M, using default")
        return 0.05, 0.5  # 5% with low confidence

    async def get_volatility_regime(self) -> Tuple[str, float]:
        """
        Get current volatility regime.

        Returns
        -------
        Tuple[str, float]
            (regime name, VIX value)
        """
        # Check cached risk metrics first
        today = Date.today()
        risk_metrics = await self.fred_storage.get_risk_metrics(today)

        if risk_metrics and risk_metrics.get("vix") is not None:
            vix = risk_metrics["vix"]
            regime = risk_metrics.get("volatility_regime", self._classify_volatility_regime(vix))
            return regime, vix

        # Fallback to latest observation
        latest = await self.fred_storage.get_latest_observation("VIXCLS")

        if latest:
            vix, _ = latest
            regime = self._classify_volatility_regime(vix)

            # Cache the result
            await self.fred_storage.save_risk_metrics(
                today, {"vix": vix, "volatility_regime": regime}
            )

            return regime, vix

        return "unknown", 0.0

    def _classify_volatility_regime(self, vix: float) -> str:
        """Classify VIX value into regime."""
        if vix < 12:
            return "low"
        elif vix < 20:
            return "normal"
        elif vix < 30:
            return "elevated"
        elif vix < 40:
            return "high"
        else:
            return "extreme"

    async def calculate_iv_rank(
        self, current_iv: float, lookback_days: int = 252
    ) -> Tuple[float, float]:
        """
        Calculate IV rank based on VIX history.

        Returns
        -------
        Tuple[float, float]
            (IV rank 0-100, confidence)
        """
        # Get VIX history
        start_date = Date.today() - timedelta(days=lookback_days)
        vix_obs = await self.fred_storage.get_observations("VIXCLS", start_date)

        if len(vix_obs) < 20:
            logger.warning("Insufficient VIX data for IV rank")
            return 50.0, 0.5

        vix_values = [obs.value for obs in vix_obs if obs.value is not None]

        if not vix_values:
            return 50.0, 0.5

        # Scale current IV to VIX equivalent (rough approximation)
        vix_equivalent = current_iv * 100

        # Calculate percentile rank
        rank = np.percentile(
            vix_values + [vix_equivalent],
            [100 * i / len(vix_values) for i in range(len(vix_values))],
        )
        iv_rank = np.interp(vix_equivalent, rank, range(len(vix_values))) / len(vix_values) * 100

        # Confidence based on data completeness
        confidence = min(1.0, len(vix_values) / lookback_days)

        return iv_rank, confidence

    async def _calculate_features(self, series_id: str, dataset: FREDDataset) -> None:
        """Calculate and store derived features."""
        if not dataset.observations:
            return

        df = dataset.to_dataframe()

        # Calculate rolling statistics
        for window in [20, 50, 200]:
            if len(df) >= window:
                # Rolling mean
                feature_name = f"sma_{window}"
                rolling_mean = df["value"].rolling(window).mean()

                for date_val, value in rolling_mean.dropna().items():
                    await self.fred_storage.save_feature(
                        series_id,
                        feature_name,
                        date_val.date(),
                        value,
                        confidence=0.95,
                        parameters={"window": window},
                    )

                # Rolling volatility
                feature_name = f"volatility_{window}"
                rolling_vol = df["value"].pct_change(fill_method=None).rolling(
                    window
                ).std() * np.sqrt(252)

                for date_val, value in rolling_vol.dropna().items():
                    await self.fred_storage.save_feature(
                        series_id,
                        feature_name,
                        date_val.date(),
                        value,
                        confidence=0.90,
                        parameters={"window": window, "annualized": True},
                    )

        logger.debug(f"Calculated features for {series_id}")

    async def get_data_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive data health report."""
        summary = await self.fred_storage.get_data_summary()
        latest_values = await self.get_latest_values()

        # Check data freshness
        freshness_issues = []
        for series_enum in WheelStrategyFREDSeries:
            series_id = series_enum.value
            if series_id in latest_values:
                _, last_date = latest_values[series_id]
                days_old = (Date.today() - last_date).days
                expected_days = series_enum.update_frequency.days * 2

                if days_old > expected_days:
                    freshness_issues.append(
                        {
                            "series_id": series_id,
                            "last_date": last_date,
                            "days_old": days_old,
                            "expected_max_days": expected_days,
                        }
                    )

        volatility_regime = await self.get_volatility_regime()

        return {
            "summary": summary,
            "latest_values": {
                k: {"value": v[0], "date": v[1].isoformat()} for k, v in latest_values.items()
            },
            "freshness_issues": freshness_issues,
            "volatility_regime": volatility_regime,
            "health_score": 100 - len(freshness_issues) * 10,
            "storage_stats": await self.storage.get_storage_stats(),
        }
