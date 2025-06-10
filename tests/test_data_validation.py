"""Test data validation with hard failures."""

from datetime import datetime

import pytest

from src.unity_wheel.utils.data_validator import DataValidator, die


class TestDataValidator:
    """Test data validation that dies on missing data."""

    def test_die_function(self):
        """Test that die() exits with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            die("Test failure")
        assert exc_info.value.code == 1

    def test_validate_market_snapshot_missing_timestamp(self):
        """Test dies when timestamp is missing."""
        snapshot = {
            "ticker": "U",
            "current_price": 35.0,
            "buying_power": 100000,
            "positions": [],
            "option_chain": {"test": {}},
            "implied_volatility": 0.45,
            "risk_free_rate": 0.05,
        }
        with pytest.raises(SystemExit):
            DataValidator.validate_market_snapshot(snapshot)

    def test_validate_market_snapshot_missing_ticker(self):
        """Test dies when ticker is missing."""
        snapshot = {
            "timestamp": datetime.now(),
            "current_price": 35.0,
            "buying_power": 100000,
            "positions": [],
            "option_chain": {"test": {}},
            "implied_volatility": 0.45,
            "risk_free_rate": 0.05,
        }
        with pytest.raises(SystemExit):
            DataValidator.validate_market_snapshot(snapshot)

    def test_validate_market_snapshot_invalid_price(self):
        """Test dies when price is invalid."""
        snapshot = {
            "timestamp": datetime.now(),
            "ticker": "U",
            "current_price": -10.0,  # Invalid
            "buying_power": 100000,
            "positions": [],
            "option_chain": {"test": {}},
            "implied_volatility": 0.45,
            "risk_free_rate": 0.05,
        }
        with pytest.raises(SystemExit):
            DataValidator.validate_market_snapshot(snapshot)

    def test_validate_market_snapshot_empty_option_chain(self):
        """Test dies when option chain is empty."""
        snapshot = {
            "timestamp": datetime.now(),
            "ticker": "U",
            "current_price": 35.0,
            "buying_power": 100000,
            "positions": [],
            "option_chain": {},  # Empty
            "implied_volatility": 0.45,
            "risk_free_rate": 0.05,
        }
        with pytest.raises(SystemExit):
            DataValidator.validate_market_snapshot(snapshot)

    def test_validate_option_data_missing_strike(self):
        """Test dies when option missing strike."""
        option = {
            "bid": 2.00,
            "ask": 2.10,
            "volume": 100,
            "open_interest": 500,
            "delta": -0.30,
            "implied_volatility": 0.45,
        }
        with pytest.raises(SystemExit):
            DataValidator.validate_option_data(option, "test_option")

    def test_validate_option_data_invalid_bid_ask(self):
        """Test dies when bid > ask."""
        option = {
            "strike": 35.0,
            "bid": 2.10,  # Greater than ask
            "ask": 2.00,
            "volume": 100,
            "open_interest": 500,
            "delta": -0.30,
            "implied_volatility": 0.45,
        }
        with pytest.raises(SystemExit):
            DataValidator.validate_option_data(option, "test_option")

    def test_validate_option_data_invalid_delta(self):
        """Test dies when delta is invalid."""
        option = {
            "strike": 35.0,
            "bid": 2.00,
            "ask": 2.10,
            "volume": 100,
            "open_interest": 500,
            "delta": -1.5,  # Invalid (outside -1 to 1)
            "implied_volatility": 0.45,
        }
        with pytest.raises(SystemExit):
            DataValidator.validate_option_data(option, "test_option")

    def test_validate_api_response_none(self):
        """Test dies when API response is None."""
        with pytest.raises(SystemExit):
            DataValidator.validate_api_response(None, "test_endpoint")

    def test_validate_api_response_error(self):
        """Test dies when API response contains error."""
        response = {"error": "API failed"}
        with pytest.raises(SystemExit):
            DataValidator.validate_api_response(response, "test_endpoint")

    def test_validate_api_response_failure_status(self):
        """Test dies when API response has failure status."""
        response = {"status": "failure", "message": "Something went wrong"}
        with pytest.raises(SystemExit):
            DataValidator.validate_api_response(response, "test_endpoint")

    def test_validate_config_value_none(self):
        """Test dies when config value is None."""
        with pytest.raises(SystemExit):
            DataValidator.validate_config_value(None, "test_config")

    def test_validate_config_value_wrong_type(self):
        """Test dies when config value has wrong type."""
        with pytest.raises(SystemExit):
            DataValidator.validate_config_value("string", "test_config", expected_type=int)

    def test_validate_historical_data_empty(self):
        """Test dies when historical data is empty."""
        with pytest.raises(SystemExit):
            DataValidator.validate_historical_data([], "U")

    def test_validate_historical_data_insufficient(self):
        """Test dies when historical data has too few points."""
        data = [1, 2, 3, 4, 5]  # Only 5 points
        with pytest.raises(SystemExit):
            DataValidator.validate_historical_data(data, "U", min_points=20)

    def test_validate_calculation_result_none(self):
        """Test dies when calculation returns None."""
        with pytest.raises(SystemExit):
            DataValidator.validate_calculation_result(None, "test_calculation")

    def test_validate_calculation_result_nan(self):
        """Test dies when calculation returns NaN."""
        with pytest.raises(SystemExit):
            DataValidator.validate_calculation_result(float("nan"), "test_calculation")

    def test_validate_schwab_order_response_missing_order_id(self):
        """Test dies when order response missing orderId."""
        response = {"status": "FILLED"}
        with pytest.raises(SystemExit):
            DataValidator.validate_schwab_order_response(response)

    def test_validate_schwab_order_response_rejected(self):
        """Test dies when order is rejected."""
        response = {
            "orderId": "12345",
            "status": "REJECTED",
            "rejectReason": "Insufficient buying power",
        }
        with pytest.raises(SystemExit):
            DataValidator.validate_schwab_order_response(response)

    def test_validate_risk_metrics_missing_field(self):
        """Test dies when risk metrics missing required field."""
        metrics = {
            "max_loss": 1000,
            "probability_assignment": 0.3,
            "expected_return": 0.05,
            # Missing var_95 and margin_required
        }
        with pytest.raises(SystemExit):
            DataValidator.validate_risk_metrics(metrics)

    def test_valid_market_snapshot(self):
        """Test that valid snapshot passes validation."""
        snapshot = {
            "timestamp": datetime.now(),
            "ticker": "U",
            "current_price": 35.0,
            "buying_power": 100000,
            "positions": [],
            "option_chain": {
                "test": {
                    "strike": 35.0,
                    "bid": 2.00,
                    "ask": 2.10,
                    "volume": 100,
                    "open_interest": 500,
                    "delta": -0.30,
                    "implied_volatility": 0.45,
                }
            },
            "implied_volatility": 0.45,
            "risk_free_rate": 0.05,
        }
        # Should not raise
        DataValidator.validate_market_snapshot(snapshot)
