"""Comprehensive tests for FRED data integration."""

import asyncio
import json
import os
import sqlite3
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientError

from unity_wheel.data_providers.base import (
    FREDClient,
    FREDDataManager,
    FREDDataset,
    FREDObservation,
    FREDSeries,
    FREDStorage,
    UpdateFrequency,
    WheelStrategyFREDSeries,
)


class TestFREDModels:
    """Test FRED data models."""

    def test_update_frequency_days(self):
        """Test frequency to days conversion."""
        assert UpdateFrequency.DAILY.days == 1
        assert UpdateFrequency.WEEKLY.days == 7
        assert UpdateFrequency.MONTHLY.days == 30
        assert UpdateFrequency.QUARTERLY.days == 90
        assert UpdateFrequency.ANNUAL.days == 365

    def test_fred_series_validation(self):
        """Test FREDSeries model validation."""
        series = FREDSeries(
            series_id="DGS3",
            title="3-Month Treasury",
            observation_start=Date(2020, 1, 1),
            observation_end=Date(2024, 12, 31),
            frequency="D",
            units="Percent",
            seasonal_adjustment="Not Applicable",
            last_updated=datetime.now(timezone.utc),
            popularity=80,
            notes="Test series",
        )

        assert series.series_id == "DGS3"
        assert series.frequency == UpdateFrequency.DAILY
        assert series.days_since_update >= 0
        assert not series.is_discontinued

    def test_fred_observation_value_parsing(self):
        """Test observation value parsing."""
        # Valid value
        obs1 = FREDObservation(date="2024-01-01", value="3.25")
        assert obs1.value == 3.25

        # Missing value representations
        for missing in [".", "nan", "NaN", None]:
            obs = FREDObservation(date="2024-01-01", value=missing)
            assert obs.value is None

        # Invalid value
        obs_invalid = FREDObservation(date="2024-01-01", value="invalid")
        assert obs_invalid.value is None

    def test_fred_dataset_methods(self):
        """Test FREDDataset convenience methods."""
        series = FREDSeries(
            series_id="TEST",
            title="Test Series",
            observation_start=Date(2024, 1, 1),
            observation_end=Date(2024, 1, 3),
            frequency="D",
            units="Units",
            seasonal_adjustment="NA",
            last_updated=datetime.now(timezone.utc),
        )

        observations = [
            FREDObservation(date=Date(2024, 1, 1), value=1.0),
            FREDObservation(date=Date(2024, 1, 2), value=None),
            FREDObservation(date=Date(2024, 1, 3), value=3.0),
        ]

        dataset = FREDDataset(series=series, observations=observations)

        assert dataset.latest_value == 3.0
        assert dataset.get_value(Date(2024, 1, 1)) == 1.0
        assert dataset.get_value(Date(2024, 1, 2)) is None
        assert dataset.date_range == (Date(2024, 1, 1), Date(2024, 1, 3))

    def test_wheel_strategy_series_metadata(self):
        """Test wheel strategy series enum."""
        series = WheelStrategyFREDSeries.DGS3
        assert series.value == "DGS3"
        assert series.description == "3-Month Treasury Constant Maturity Rate"
        assert series.update_frequency == UpdateFrequency.DAILY


class TestFREDClient:
    """Test FRED API client."""

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = MagicMock()
        response = MagicMock()
        response.json = AsyncMock()
        response.raise_for_status = MagicMock()
        session.get.return_value.__aenter__.return_value = response
        return session, response

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization."""
        # Without API key
        with pytest.raises(ValueError):
            FREDClient(api_key=None)

        # With API key
        client = FREDClient(api_key="test_key", rpm_limit=60)
        assert client.api_key == "test_key"
        assert client.rate_limiter.rpm == 60

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting behavior."""
        client = FREDClient(api_key="test", rpm_limit=120)

        # Measure time for multiple requests
        start = asyncio.get_event_loop().time()
        for _ in range(3):
            await client.rate_limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        # Should take at least 2 * min_interval
        expected_min = 2 * (60.0 / 120)
        assert elapsed >= expected_min * 0.9  # Allow 10% tolerance

    @pytest.mark.asyncio
    async def test_get_series_metadata(self, mock_session):
        """Test fetching series metadata."""
        session_mock, response_mock = mock_session
        response_mock.json.return_value = {
            "seriess": [
                {
                    "id": "DGS3",
                    "title": "3-Month Treasury",
                    "observation_start": "2000-01-01",
                    "observation_end": "2024-12-31",
                    "frequency_short": "D",
                    "units": "Percent",
                    "seasonal_adjustment": "Not Applicable",
                    "last_updated": "2024-12-31 08:00:00-06:00",
                    "popularity": 85,
                    "notes": "Test notes",
                }
            ]
        }

        client = FREDClient(api_key="test")
        client._session = session_mock

        series = await client.get_series_metadata("DGS3")

        assert series.series_id == "DGS3"
        assert series.title == "3-Month Treasury"
        assert series.frequency == UpdateFrequency.DAILY
        assert series.units == "Percent"

    @pytest.mark.asyncio
    async def test_get_observations(self, mock_session):
        """Test fetching observations."""
        session_mock, response_mock = mock_session
        response_mock.json.return_value = {
            "observations": [
                {"date": "2024-01-01", "value": "3.25"},
                {"date": "2024-01-02", "value": "3.30"},
                {"date": "2024-01-03", "value": "."},  # Missing value
            ]
        }

        client = FREDClient(api_key="test")
        client._session = session_mock

        observations = await client.get_observations("DGS3")

        assert len(observations) == 3
        assert observations[0].date == Date(2024, 1, 1)
        assert observations[0].value == 3.25
        assert observations[2].value is None

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_session):
        """Test error handling and retries."""
        session_mock, response_mock = mock_session

        # Simulate API error
        response_mock.json.return_value = {"error_code": 400, "error_message": "Invalid request"}

        client = FREDClient(api_key="test")
        client._session = session_mock

        with pytest.raises(ValueError, match="FRED API error"):
            await client.get_series_metadata("INVALID")

    @pytest.mark.asyncio
    async def test_get_wheel_strategy_data(self, mock_session):
        """Test fetching all wheel strategy data."""
        session_mock, response_mock = mock_session

        # Mock metadata response
        def mock_json_metadata():
            return {
                "seriess": [
                    {
                        "id": "DGS3",
                        "title": "Test",
                        "observation_start": "2020-01-01",
                        "observation_end": "2024-01-01",
                        "frequency_short": "D",
                        "units": "Percent",
                        "seasonal_adjustment": "NA",
                        "last_updated": "2024-01-01 00:00:00+00:00",
                    }
                ]
            }

        # Mock observations response
        def mock_json_observations():
            return {"observations": [{"date": "2024-01-01", "value": "3.0"}]}

        response_mock.json.side_effect = [
            mock_json_metadata() if i % 2 == 0 else mock_json_observations()
            for i in range(20)  # Enough for all series
        ]

        client = FREDClient(api_key="test")
        client._session = session_mock

        datasets = await client.get_wheel_strategy_data(lookback_days=30)

        # Should have data for multiple series
        assert len(datasets) > 0
        assert all(isinstance(d, FREDDataset) for d in datasets.values())


class TestFREDStorage:
    """Test data storage functionality."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database."""
        db_path = tmp_path / "test.db"
        storage = FREDStorage(db_path)
        yield storage
        # Cleanup happens automatically with tmp_path

    def test_database_initialization(self, temp_db):
        """Test database schema creation."""
        with temp_db._get_connection() as conn:
            # Check tables exist
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table'
            """
            )
            tables = {row[0] for row in cursor.fetchall()}

            expected_tables = {
                "fred_series",
                "fred_observations",
                "data_quality_log",
                "calculated_features",
            }
            assert expected_tables.issubset(tables)

    def test_save_and_retrieve_series(self, temp_db):
        """Test saving and retrieving series metadata."""
        series = FREDSeries(
            series_id="TEST123",
            title="Test Series",
            observation_start=Date(2020, 1, 1),
            observation_end=Date(2024, 1, 1),
            frequency="D",
            units="Units",
            seasonal_adjustment="NA",
            last_updated=datetime.now(timezone.utc),
            popularity=50,
        )

        temp_db.save_series_metadata(series)

        # Retrieve
        retrieved = temp_db.get_series_metadata("TEST123")
        assert retrieved is not None
        assert retrieved.series_id == "TEST123"
        assert retrieved.title == "Test Series"
        assert retrieved.frequency == UpdateFrequency.DAILY

    def test_save_and_retrieve_observations(self, temp_db):
        """Test saving and retrieving observations."""
        observations = [
            FREDObservation(date=Date(2024, 1, 1), value=1.0),
            FREDObservation(date=Date(2024, 1, 2), value=2.0),
            FREDObservation(date=Date(2024, 1, 3), value=None),
            FREDObservation(date=Date(2024, 1, 4), value=4.0),
        ]

        count = temp_db.save_observations("TEST", observations)
        assert count == 4

        # Retrieve all
        retrieved = temp_db.get_observations("TEST")
        assert len(retrieved) == 4
        assert retrieved[0].value == 1.0
        assert retrieved[2].value is None

        # Retrieve with date range
        subset = temp_db.get_observations(
            "TEST", start_date=Date(2024, 1, 2), end_date=Date(2024, 1, 3)
        )
        assert len(subset) == 2

        # Test latest observation date
        latest = temp_db.get_latest_observation_Date("TEST")
        assert latest == Date(2024, 1, 4)

    def test_calculated_features(self, temp_db):
        """Test saving and retrieving calculated features."""
        temp_db.save_calculated_feature(
            "DGS3", "sma_20", Date(2024, 1, 1), 3.5, confidence=0.95, parameters={"window": 20}
        )

        # Retrieve
        result = temp_db.get_calculated_feature("DGS3", "sma_20", Date(2024, 1, 1))
        assert result is not None
        value, confidence = result
        assert value == 3.5
        assert confidence == 0.95

        # Non-existent feature
        result = temp_db.get_calculated_feature("DGS3", "unknown", Date(2024, 1, 1))
        assert result is None

    def test_data_summary(self, temp_db):
        """Test data summary generation."""
        # Add some data
        series = FREDSeries(
            series_id="TEST",
            title="Test",
            observation_start=Date(2024, 1, 1),
            observation_end=Date(2024, 1, 5),
            frequency="D",
            units="Units",
            seasonal_adjustment="NA",
            last_updated=datetime.now(timezone.utc),
        )
        temp_db.save_series_metadata(series)

        observations = [FREDObservation(date=Date(2024, 1, i), value=float(i)) for i in range(1, 6)]
        temp_db.save_observations("TEST", observations)

        summary = temp_db.get_data_summary()

        assert summary["series_count"] == 1
        assert summary["total_observations"] == 5
        assert len(summary["series"]) == 1
        assert summary["series"][0]["series_id"] == "TEST"
        assert summary["series"][0]["observation_count"] == 5


class TestFREDDataManager:
    """Test data manager functionality."""

    @pytest.fixture
    def temp_manager(self, tmp_path):
        """Create temporary data manager."""
        db_path = tmp_path / "test.db"
        manager = FREDDataManager(api_key="test", db_path=db_path)
        return manager

    def test_get_risk_free_rate(self, temp_manager):
        """Test risk-free rate retrieval."""
        # Add test data
        obs = [
            FREDObservation(date=Date.today(), value=3.5),
        ]
        temp_manager.storage.save_observations("DGS3", obs)

        rate, confidence = temp_manager.get_risk_free_rate(3)
        assert rate == 0.035  # 3.5% as decimal
        assert confidence >= 0.8

        # Test fallback
        rate, confidence = temp_manager.get_risk_free_rate(6)
        assert rate == 0.05  # Default fallback
        assert confidence == 0.5

    def test_get_volatility_regime(self, temp_manager):
        """Test volatility regime detection."""
        test_cases = [
            (10.0, "low"),
            (15.0, "normal"),
            (25.0, "elevated"),
            (35.0, "high"),
            (50.0, "extreme"),
        ]

        for vix_value, expected_regime in test_cases:
            obs = [FREDObservation(date=Date.today(), value=vix_value)]
            temp_manager.storage.save_observations("VIXCLS", obs)

            regime, vix = temp_manager.get_volatility_regime()
            assert regime == expected_regime
            assert vix == vix_value

    def test_calculate_iv_rank(self, temp_manager):
        """Test IV rank calculation."""
        # Create VIX history
        vix_history = []
        base_date = Date.today() - timedelta(days=252)

        # Create range from 10 to 30
        for i in range(252):
            vix_value = 10 + (i / 252) * 20  # Linear from 10 to 30
            vix_history.append(FREDObservation(date=base_date + timedelta(days=i), value=vix_value))

        temp_manager.storage.save_observations("VIXCLS", vix_history)

        # Test IV rank calculation
        # IV of 0.20 (20%) should be around 50th percentile
        rank, confidence = temp_manager.calculate_iv_rank(0.20)
        assert 40 <= rank <= 60  # Should be near middle
        assert confidence > 0.9  # Good data coverage

    def test_data_health_report(self, temp_manager):
        """Test health report generation."""
        # Add some test data
        series = FREDSeries(
            series_id="DGS3",
            title="Test",
            observation_start=Date(2024, 1, 1),
            observation_end=Date.today(),
            frequency="D",
            units="Percent",
            seasonal_adjustment="NA",
            last_updated=datetime.now(timezone.utc),
        )
        temp_manager.storage.save_series_metadata(series)

        obs = [FREDObservation(date=Date.today() - timedelta(days=1), value=3.0)]
        temp_manager.storage.save_observations("DGS3", obs)

        report = temp_manager.get_data_health_report()

        assert "summary" in report
        assert "latest_values" in report
        assert "freshness_issues" in report
        assert "volatility_regime" in report
        assert "health_score" in report
        assert report["health_score"] >= 0


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual FRED API access."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.environ.get("FRED_API_KEY"), reason="FRED_API_KEY not set")
    async def test_real_api_call(self):
        """Test actual API call to FRED."""
        async with FREDClient() as client:
            # Test with a known series
            series = await client.get_series_metadata("DGS3")
            assert series.series_id == "DGS3"
            assert "Treasury" in series.title

            # Get recent observations
            observations = await client.get_observations(
                "DGS3", start_date=Date.today() - timedelta(days=30)
            )
            assert len(observations) > 0
            assert all(obs.date <= Date.today() for obs in observations)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.environ.get("FRED_API_KEY"), reason="FRED_API_KEY not set")
    async def test_full_workflow(self, tmp_path):
        """Test complete workflow from fetch to analysis."""
        db_path = tmp_path / "test.db"
        manager = FREDDataManager(db_path=db_path)

        # Initialize with just 30 days of data for speed
        counts = await manager.initialize_data(lookback_days=30)
        assert len(counts) > 0

        # Check we can get values
        rf_rate, _ = manager.get_risk_free_rate(3)
        assert 0 <= rf_rate <= 0.10  # Reasonable range

        regime, vix = manager.get_volatility_regime()
        assert regime in ["low", "normal", "elevated", "high", "extreme", "unknown"]

        # Generate health report
        report = manager.get_data_health_report()
        assert report["health_score"] > 0
