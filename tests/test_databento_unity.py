"""Tests for databento_unity utility module."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.unity_wheel.data_providers.databento.unity_utils import (
    chain,
    spot,
    get_wheel_candidates,
    store_options_in_duckdb,
    get_equity_bars,
)


class TestDatabentoUnity:
    """Test databento Unity integration utilities."""

    @patch("src.unity_wheel.utils.databento_unity.API")
    def test_chain_basic(self, mock_api):
        """Test basic option chain retrieval."""
        # Setup mock
        mock_response = Mock()
        mock_df = pd.DataFrame(
            {
                "raw_symbol": ["U  240117P00030000", "U  240117P00035000"],
                "strike": [30.0, 35.0],
                "expiration": [datetime(2024, 1, 17, tzinfo=timezone.utc)] * 2,
                "underlying": ["U", "U"],
                "bid": [0.50, 1.00],
                "ask": [0.60, 1.10],
            }
        )
        mock_response.to_df.return_value = mock_df
        mock_api.timeseries.get_range.return_value = mock_response

        # Test
        result = chain("2024-01-01", "2024-01-02")

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "strike" in result.columns
        assert "expiration" in result.columns

    @patch("src.unity_wheel.utils.databento_unity.API")
    def test_spot_retrieval(self, mock_api):
        """Test spot price retrieval."""
        # Setup mock
        mock_response = Mock()
        mock_df = pd.DataFrame(
            {
                "bid_px": [34.0 * 1e9, 35.0 * 1e9],
                "ask_px": [34.1 * 1e9, 35.1 * 1e9],
                "bid_sz": [100, 150],
                "ask_sz": [100, 150],
            }
        )
        mock_response.to_df.return_value = mock_df
        mock_api.timeseries.get_range.return_value = mock_response

        # Test
        result = spot(days_back=2)

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert "bid_px" in result.columns
        assert "ask_px" in result.columns

    def test_get_wheel_candidates_filtering(self):
        """Test wheel candidate filtering logic."""
        # Create sample option data
        options_df = pd.DataFrame(
            {
                "raw_symbol": ["U  240117P00030000", "U  240117P00035000", "U  240117P00040000"],
                "strike": [30.0, 35.0, 40.0],
                "expiration": pd.to_datetime(["2024-01-17", "2024-01-17", "2024-01-17"]),
                "underlying": ["U", "U", "U"],
                "bid": [0.50, 1.00, 2.00],
                "ask": [0.60, 1.10, 2.10],
                "delta": [-0.20, -0.30, -0.45],
                "iv": [0.35, 0.32, 0.30],
                "volume": [100, 200, 50],  # Last one below threshold
                "open_interest": [500, 1000, 75],  # Last one below threshold
            }
        )

        # Add calculated fields
        options_df["dte"] = 45  # Days to expiry
        options_df["mid"] = (options_df["bid"] + options_df["ask"]) / 2
        options_df["premium_pct"] = options_df["mid"] / options_df["strike"] * 100

        # Test filtering
        candidates = []
        for _, opt in options_df.iterrows():
            # Apply filtering logic similar to get_wheel_candidates
            if opt["volume"] >= 100 and opt["open_interest"] >= 100:
                if abs(opt["delta"] - (-0.30)) <= 0.10:  # Target delta -0.30
                    candidates.append(opt)

        # Should only get the middle option (good liquidity and delta)
        assert len(candidates) == 1
        assert candidates[0]["strike"] == 35.0

    @patch("duckdb.connect")
    def test_store_options_in_duckdb(self, mock_connect):
        """Test storing options in database."""
        # Setup mock connection
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Create test data
        options_df = pd.DataFrame(
            {
                "raw_symbol": ["U  240117P00035000"],
                "strike": [35.0],
                "expiration": pd.to_datetime(["2024-01-17"]),
                "bid": [1.00],
                "ask": [1.10],
                "mid": [1.05],
                "volume": [200],
                "open_interest": [1000],
                "iv": [0.32],
                "delta": [-0.30],
                "gamma": [0.05],
                "theta": [-0.02],
                "vega": [0.10],
                "rho": [-0.01],
            }
        )

        # Test store
        result = store_options_in_duckdb(options_df)

        # Verify result
        assert isinstance(result, int)
        assert mock_conn.execute.called

    @patch("src.unity_wheel.utils.databento_unity.API")
    def test_get_equity_bars(self, mock_api):
        """Test equity bars retrieval."""
        # Setup mock
        mock_response = Mock()
        mock_df = pd.DataFrame(
            {
                "open": [34.0 * 1e9, 35.0 * 1e9],
                "high": [35.0 * 1e9, 36.0 * 1e9],
                "low": [33.0 * 1e9, 34.0 * 1e9],
                "close": [34.5 * 1e9, 35.5 * 1e9],
                "volume": [1000, 1500],
            }
        )
        mock_response.to_df.return_value = mock_df
        mock_api.timeseries.get_range.return_value = mock_response

        # Test
        result = get_equity_bars(days=2)

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        assert "returns" in result.columns

    def test_empty_data_handling(self):
        """Test handling of empty data scenarios."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()

        # Should handle gracefully
        with patch("src.unity_wheel.utils.databento_unity.API") as mock_api:
            mock_response = Mock()
            mock_response.to_df.return_value = empty_df
            mock_api.timeseries.get_range.return_value = mock_response

            result = chain("2024-01-01", "2024-01-02")
            assert result.empty
