"""Local storage for FRED and market data with SQLite backend."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from unity_wheel.utils import get_logger, timed_operation

from ..fred.fred_models import (
    FREDDataset,
    FREDObservation,
    FREDSeries,
)

logger = get_logger(__name__)


class DataStorage:
    """SQLite-based storage for time series data."""

    def __init__(self, db_path: Path | None = None):
        """
        Initialize storage.

        Parameters
        ----------
        db_path : Path, optional
            Database file path (defaults to ./data/market_data.db)
        """
        self.db_path = db_path or Path("./data/market_data.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"Data storage initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except (ValueError, KeyError, AttributeError):
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Series metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fred_series (
                    series_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    observation_start DATE,
                    observation_end DATE,
                    frequency TEXT,
                    units TEXT,
                    seasonal_adjustment TEXT,
                    last_updated TIMESTAMP,
                    popularity INTEGER,
                    notes TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Observations table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fred_observations (
                    series_id TEXT,
                    observation_date DATE,
                    value REAL,
                    is_revised BOOLEAN DEFAULT 0,
                    revision_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (series_id, observation_date),
                    FOREIGN KEY (series_id) REFERENCES fred_series(series_id)
                )
            """
            )

            # Create indexes
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fred_obs_series_date
                ON fred_observations(series_id, observation_date DESC)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fred_obs_date
                ON fred_observations(observation_date DESC)
            """
            )

            # Data quality tracking table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_id TEXT,
                    check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    quality_score REAL,
                    issues TEXT,
                    metadata TEXT
                )
            """
            )

            # Feature calculations cache
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calculated_features (
                    series_id TEXT,
                    feature_name TEXT,
                    calculation_date DATE,
                    value REAL,
                    confidence REAL,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (series_id, feature_name, calculation_date)
                )
            """
            )

            logger.debug("Database schema initialized")

    @timed_operation(threshold_ms=50)
    def save_series_metadata(self, series: FREDSeries) -> None:
        """Save or update series metadata."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fred_series
                (series_id, title, observation_start, observation_end,
                 frequency, units, seasonal_adjustment, last_updated,
                 popularity, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
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
                ),
            )

            logger.debug(f"Saved metadata for {series.series_id}")

    @timed_operation(threshold_ms=100)
    def save_observations(
        self,
        series_id: str,
        observations: list[FREDObservation],
        mark_revised: bool = False,
    ) -> int:
        """
        Save observations to database.

        Returns
        -------
        int
            Number of observations saved
        """
        if not observations:
            return 0

        with self._get_connection() as conn:
            # Prepare data
            data = [
                (
                    series_id,
                    obs.date,
                    obs.value,
                    mark_revised,
                    datetime.now(UTC) if mark_revised else None,
                )
                for obs in observations
            ]

            # Bulk insert
            conn.executemany(
                """
                INSERT OR REPLACE INTO fred_observations
                (series_id, observation_date, value, is_revised, revision_date)
                VALUES (?, ?, ?, ?, ?)
            """,
                data,
            )

            count = len(data)
            logger.info(f"Saved {count} observations for {series_id}")
            return count

    def save_dataset(self, dataset: FREDDataset) -> None:
        """Save complete dataset."""
        self.save_series_metadata(dataset.series)
        self.save_observations(dataset.series.series_id, dataset.observations)

    def get_latest_observation_date(self, series_id: str) -> date | None:
        """Get the date of the most recent observation."""
        with self._get_connection() as conn:
            result = conn.execute(
                """
                SELECT MAX(observation_date) as latest_date
                FROM fred_observations
                WHERE series_id = ? AND value IS NOT NULL
            """,
                (series_id,),
            ).fetchone()

            return result["latest_date"] if result else None

    @timed_operation(threshold_ms=50)
    def get_observations(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
    ) -> list[FREDObservation]:
        """Get observations from storage."""
        with self._get_connection() as conn:
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

            rows = conn.execute(query, params).fetchall()

            return [
                FREDObservation(date=row["observation_date"], value=row["value"])
                for row in reversed(rows)  # Return in chronological order
            ]

    def get_series_metadata(self, series_id: str) -> FREDSeries | None:
        """Get series metadata from storage."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM fred_series WHERE series_id = ?
            """,
                (series_id,),
            ).fetchone()

            if not row:
                return None

            return FREDSeries(
                series_id=row["series_id"],
                title=row["title"],
                observation_start=row["observation_start"],
                observation_end=row["observation_end"],
                frequency=row["frequency"],
                units=row["units"],
                seasonal_adjustment=row["seasonal_adjustment"],
                last_updated=row["last_updated"],
                popularity=row["popularity"],
                notes=row["notes"],
            )

    def get_dataset(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> FREDDataset | None:
        """Get complete dataset from storage."""
        metadata = self.get_series_metadata(series_id)
        if not metadata:
            return None

        observations = self.get_observations(series_id, start_date, end_date)

        return FREDDataset(
            series=metadata,
            observations=observations,
        )

    def save_calculated_feature(
        self,
        series_id: str,
        feature_name: str,
        calculation_date: date,
        value: float,
        confidence: float = 1.0,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Save calculated feature value."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO calculated_features
                (series_id, feature_name, calculation_date, value, confidence, parameters)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    series_id,
                    feature_name,
                    calculation_date,
                    value,
                    confidence,
                    json.dumps(parameters or {}),
                ),
            )

    def get_calculated_feature(
        self,
        series_id: str,
        feature_name: str,
        calculation_date: date,
    ) -> tuple[float, float] | None:
        """
        Get calculated feature value.

        Returns
        -------
        Optional[Tuple[float, float]]
            (value, confidence) if found
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT value, confidence
                FROM calculated_features
                WHERE series_id = ? AND feature_name = ? AND calculation_date = ?
            """,
                (series_id, feature_name, calculation_date),
            ).fetchone()

            return (row["value"], row["confidence"]) if row else None

    def get_data_summary(self) -> dict[str, Any]:
        """Get summary of stored data."""
        with self._get_connection() as conn:
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
            ).fetchone()["count"]

            # Storage size
            db_size = self.db_path.stat().st_size / 1024 / 1024  # MB

            # Series details
            series_details = conn.execute(
                """
                SELECT
                    s.series_id,
                    s.title,
                    s.frequency,
                    s.last_updated,
                    COUNT(o.value) as observation_count,
                    MAX(o.observation_date) as latest_observation
                FROM fred_series s
                LEFT JOIN fred_observations o ON s.series_id = o.series_id
                GROUP BY s.series_id
                ORDER BY s.series_id
            """
            ).fetchall()

            return {
                "database_path": str(self.db_path),
                "database_size_mb": round(db_size, 2),
                "series_count": series_info["series_count"],
                "total_observations": obs_count,
                "date_range": {
                    "earliest": series_info["earliest_date"],
                    "latest": series_info["latest_date"],
                },
                "series": [
                    {
                        "series_id": row["series_id"],
                        "title": row["title"],
                        "frequency": row["frequency"],
                        "last_updated": row["last_updated"],
                        "observation_count": row["observation_count"],
                        "latest_observation": row["latest_observation"],
                    }
                    for row in series_details
                ],
            }

    def cleanup_old_data(self, days_to_keep: int = 1825) -> int:
        """
        Remove observations older than specified days.

        Returns
        -------
        int
            Number of observations deleted
        """
        cutoff_date = date.today() - pd.Timedelta(days=days_to_keep)

        with self._get_connection() as conn:
            result = conn.execute(
                """
                DELETE FROM fred_observations
                WHERE observation_date < ?
            """,
                (cutoff_date,),
            )

            deleted = result.rowcount

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old observations")

                # VACUUM to reclaim space
                conn.execute("VACUUM")

            return deleted
