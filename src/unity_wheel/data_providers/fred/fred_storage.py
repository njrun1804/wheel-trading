"""FRED data storage adapter using unified storage pattern."""

from __future__ import annotations

import json
from datetime import date as Date
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from unity_wheel.storage.storage import Storage
from unity_wheel.utils import get_logger, timed_operation

from .fred_models import FREDDataset, FREDObservation, FREDSeries, WheelStrategyFREDSeries

logger = get_logger(__name__)


class FREDStorage:
    """Storage adapter for FRED data using DuckDB cache."""

    def __init__(self, storage: Storage):
        """Initialize FRED storage adapter."""
        self.storage = storage
        self._initialized = False

    async def initialize(self):
        """Create FRED-specific tables in DuckDB."""
        if self._initialized:
            return

        async with self.storage.cache.connection() as conn:
            # FRED series metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fred_series (
                    series_id VARCHAR PRIMARY KEY,
                    title VARCHAR,
                    observation_start DATE,
                    observation_end DATE,
                    frequency VARCHAR,
                    units VARCHAR,
                    seasonal_adjustment VARCHAR,
                    last_updated TIMESTAMP,
                    popularity INTEGER,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # FRED observations table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fred_observations (
                    series_id VARCHAR NOT NULL,
                    observation_date DATE NOT NULL,
                    value DECIMAL(18,6),
                    is_revised BOOLEAN DEFAULT FALSE,
                    revision_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (series_id, observation_date)
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fred_obs_series_date
                ON fred_observations(series_id, observation_date DESC)
            """
            )

            # Calculated features cache
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fred_features (
                    series_id VARCHAR NOT NULL,
                    feature_name VARCHAR NOT NULL,
                    calculation_date DATE NOT NULL,
                    value DECIMAL(18,6),
                    confidence DECIMAL(4,3),
                    parameters JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (series_id, feature_name, calculation_date)
                )
            """
            )

            # Risk metrics cache
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    metric_date DATE PRIMARY KEY,
                    risk_free_rate_3m DECIMAL(6,4),
                    risk_free_rate_1y DECIMAL(6,4),
                    vix DECIMAL(6,2),
                    volatility_regime VARCHAR,
                    ted_spread DECIMAL(6,4),
                    high_yield_spread DECIMAL(6,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

        self._initialized = True
        logger.info("FRED storage tables initialized")

    @timed_operation(threshold_ms=50)
    async def save_series_metadata(self, series: FREDSeries) -> None:
        """Save or update series metadata."""
        await self.initialize()

        async with self.storage.cache.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fred_series
                (series_id, title, observation_start, observation_end,
                 frequency, units, seasonal_adjustment, last_updated,
                 popularity, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                [
                    series.series_id,
                    series.title,
                    series.observation_start,
                    series.observation_end,
                    series.frequency.value,
                    series.units,
                    series.seasonal_adjustment,
                    series.last_updated,
                    series.popularity,
                    series.notes,
                ],
            )

    @timed_operation(threshold_ms=100)
    async def save_observations(
        self,
        series_id: str,
        observations: List[FREDObservation],
        mark_revised: bool = False,
    ) -> int:
        """Save observations to storage."""
        if not observations:
            return 0

        await self.initialize()

        async with self.storage.cache.connection() as conn:
            # Prepare data for bulk insert
            data = []
            for obs in observations:
                if obs.value is not None:  # Skip null values
                    data.append(
                        [
                            series_id,
                            obs.date,
                            float(obs.value),
                            mark_revised,
                            datetime.utcnow() if mark_revised else None,
                        ]
                    )

            if data:
                # Use batch insert for performance
                # Convert to DataFrame for faster bulk insert
                import pandas as pd

                df = pd.DataFrame(
                    data,
                    columns=[
                        "series_id",
                        "observation_date",
                        "value",
                        "is_revised",
                        "revision_date",
                    ],
                )

                # Use DuckDB's fast import via DataFrame
                # Need to explicitly map columns since table has created_at
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fred_observations
                    (series_id, observation_date, value, is_revised, revision_date)
                    SELECT series_id, observation_date, value, is_revised, revision_date FROM df
                """
                )

            logger.info(f"Saved {len(data)} observations for {series_id}")
            return len(data)

    async def save_dataset(self, dataset: FREDDataset) -> None:
        """Save complete dataset."""
        await self.save_series_metadata(dataset.series)
        await self.save_observations(dataset.series.series_id, dataset.observations)

    async def get_latest_observation(self, series_id: str) -> Optional[tuple[float, Date]]:
        """Get the most recent observation for a series."""
        await self.initialize()

        async with self.storage.cache.connection() as conn:
            result = conn.execute(
                """
                SELECT value, observation_date
                FROM fred_observations
                WHERE series_id = ?
                ORDER BY observation_date DESC
                LIMIT 1
            """,
                [series_id],
            ).fetchone()

            if result:
                return float(result[0]), result[1]
            return None

    async def get_observations(
        self,
        series_id: str,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        limit: Optional[int] = None,
    ) -> List[FREDObservation]:
        """Get observations from storage."""
        await self.initialize()

        async with self.storage.cache.connection() as conn:
            query = """
                SELECT observation_date, value
                FROM fred_observations
                WHERE series_id = ?
            """
            params = [series_id]

            if start_date:
                query += " AND observation_date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND observation_date <= ?"
                params.append(end_date)

            query += " ORDER BY observation_date DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            df = conn.execute(query, params).df()

            observations = []
            for _, row in df.iterrows():
                observations.append(
                    FREDObservation(
                        date=row["observation_date"],
                        value=float(row["value"]) if row["value"] is not None else None,
                    )
                )

            # Return in chronological order
            return list(reversed(observations))

    async def save_risk_metrics(self, metric_date: Date, metrics: Dict[str, Any]) -> None:
        """Save calculated risk metrics."""
        await self.initialize()

        async with self.storage.cache.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO risk_metrics
                (metric_date, risk_free_rate_3m, risk_free_rate_1y,
                 vix, volatility_regime, ted_spread, high_yield_spread)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    metric_date,
                    metrics.get("risk_free_rate_3m"),
                    metrics.get("risk_free_rate_1y"),
                    metrics.get("vix"),
                    metrics.get("volatility_regime"),
                    metrics.get("ted_spread"),
                    metrics.get("high_yield_spread"),
                ],
            )

    async def get_risk_metrics(self, metric_date: Date) -> Optional[Dict[str, Any]]:
        """Get risk metrics for a specific date."""
        await self.initialize()

        async with self.storage.cache.connection() as conn:
            result = conn.execute(
                """
                SELECT * FROM risk_metrics
                WHERE metric_date = ?
            """,
                [metric_date],
            ).fetchone()

            if result:
                # Convert DuckDB row to dict - result is a tuple-like object
                columns = [
                    "metric_date",
                    "risk_free_rate_3m",
                    "risk_free_rate_1y",
                    "vix",
                    "volatility_regime",
                    "ted_spread",
                    "high_yield_spread",
                    "created_at",
                ]
                return {col: result[i] for i, col in enumerate(columns) if i < len(result)}
            return None

    async def save_feature(
        self,
        series_id: str,
        feature_name: str,
        calculation_date: Date,
        value: float,
        confidence: float = 1.0,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save calculated feature value."""
        await self.initialize()

        async with self.storage.cache.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fred_features
                (series_id, feature_name, calculation_date, value, confidence, parameters)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                [
                    series_id,
                    feature_name,
                    calculation_date,
                    value,
                    confidence,
                    json.dumps(parameters or {}),
                ],
            )

    async def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of stored FRED data."""
        await self.initialize()

        async with self.storage.cache.connection() as conn:
            # Series count and date ranges
            series_info = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT series_id) as series_count,
                    MIN(observation_start) as earliest_date,
                    MAX(observation_end) as latest_date
                FROM fred_series
            """
            ).fetchone()

            # Observation count
            obs_count = conn.execute(
                """
                SELECT COUNT(*) as count FROM fred_observations
            """
            ).fetchone()[0]

            # Series details
            series_df = conn.execute(
                """
                SELECT
                    s.series_id,
                    ANY_VALUE(s.title) as title,
                    ANY_VALUE(s.frequency) as frequency,
                    ANY_VALUE(s.last_updated) as last_updated,
                    COUNT(o.value) as observation_count,
                    MAX(o.observation_date) as latest_observation
                FROM fred_series s
                LEFT JOIN fred_observations o ON s.series_id = o.series_id
                GROUP BY s.series_id
                ORDER BY s.series_id
            """
            ).df()

            return {
                "series_count": series_info[0] if series_info else 0,
                "total_observations": obs_count,
                "date_range": {
                    "earliest": series_info[1] if series_info else None,
                    "latest": series_info[2] if series_info else None,
                },
                "series": series_df.to_dict("records") if not series_df.empty else [],
            }

    async def cleanup_old_data(self, days_to_keep: int = 1825) -> int:
        """Remove observations older than specified days."""
        cutoff_date = Date.today() - timedelta(days=days_to_keep)

        async with self.storage.cache.connection() as conn:
            result = conn.execute(
                """
                DELETE FROM fred_observations
                WHERE observation_date < ?
            """,
                [cutoff_date],
            )

            deleted = result.rowcount

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old FRED observations")

            return deleted
