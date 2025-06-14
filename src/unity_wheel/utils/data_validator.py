"""Data validation with hard failures - no missing data allowed."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)



import sys
from datetime import datetime
from typing import Any, Dict, List, NoReturn, Optional

from .logging import get_logger

logger = get_logger(__name__)


def die(message: str) -> NoReturn:
    """Exit program with error message."""
    logger.error(f"FATAL: {message}")
    print(f"\n❌ FATAL ERROR: {message}\n", file=sys.stderr)
    sys.exit(1)


class DataValidator:
    """Validates all data with hard exits on ANY missing fields."""

    @staticmethod
    def validate_market_snapshot(snapshot: Dict[str, Any]) -> None:
        """
        Validate market snapshot data. Dies if ANY required field is missing.

        Required fields:
        - timestamp
        - ticker
        - current_price
        - buying_power
        - positions (can be empty list)
        - option_chain (must have entries)
        - implied_volatility
        - risk_free_rate
        """
        if not snapshot:
            die("Market snapshot is None or empty")

        # Core fields
        if "timestamp" not in snapshot:
            die("Missing 'timestamp' in market snapshot")
        if not isinstance(snapshot["timestamp"], datetime):
            die(f"Invalid timestamp type: {type(snapshot['timestamp'])}")

        if "ticker" not in snapshot:
            die("Missing 'ticker' in market snapshot")
        if not snapshot["ticker"]:
            die("Empty ticker symbol")

        if "current_price" not in snapshot:
            die("Missing 'current_price' in market snapshot")
        if snapshot["current_price"] <= 0:
            die(f"Invalid current price: {snapshot['current_price']}")

        if "buying_power" not in snapshot:
            die("Missing 'buying_power' in market snapshot")
        if snapshot["buying_power"] < 0:
            die(f"Invalid buying power: {snapshot['buying_power']}")

        if "positions" not in snapshot:
            die("Missing 'positions' in market snapshot")
        if not isinstance(snapshot["positions"], list):
            die(f"Positions must be a list, got: {type(snapshot['positions'])}")

        if "option_chain" not in snapshot:
            die("Missing 'option_chain' in market snapshot")
        if not snapshot["option_chain"]:
            die("Empty option chain - no strikes available")

        if "implied_volatility" not in snapshot:
            die("Missing 'implied_volatility' in market snapshot")
        if snapshot["implied_volatility"] <= 0:
            die(f"Invalid implied volatility: {snapshot['implied_volatility']}")

        if "risk_free_rate" not in snapshot:
            die("Missing 'risk_free_rate' in market snapshot")

    @staticmethod
    def validate_option_data(option: Dict[str, Any], key: str) -> None:
        """
        Validate single option data. Dies if ANY required field is missing.

        Required fields:
        - strike
        - bid
        - ask
        - volume
        - open_interest
        - delta
        - implied_volatility
        """
        if not option:
            die(f"Option data for {key} is None or empty")

        if "strike" not in option:
            die(f"Missing 'strike' in option {key}")
        if option["strike"] <= 0:
            die(f"Invalid strike price for {key}: {option['strike']}")

        if "bid" not in option:
            die(f"Missing 'bid' in option {key}")
        if option["bid"] < 0:
            die(f"Invalid bid price for {key}: {option['bid']}")

        if "ask" not in option:
            die(f"Missing 'ask' in option {key}")
        if option["ask"] < 0:
            die(f"Invalid ask price for {key}: {option['ask']}")

        if option["bid"] > option["ask"]:
            die(f"Invalid bid/ask for {key}: bid ({option['bid']}) > ask ({option['ask']})")

        if "volume" not in option:
            die(f"Missing 'volume' in option {key}")
        if option["volume"] < 0:
            die(f"Invalid volume for {key}: {option['volume']}")

        if "open_interest" not in option:
            die(f"Missing 'open_interest' in option {key}")
        if option["open_interest"] < 0:
            die(f"Invalid open interest for {key}: {option['open_interest']}")

        if "delta" not in option:
            die(f"Missing 'delta' in option {key}")
        if not (-1 <= option["delta"] <= 1):
            die(f"Invalid delta for {key}: {option['delta']}")

        if "implied_volatility" not in option:
            die(f"Missing 'implied_volatility' in option {key}")
        if option["implied_volatility"] <= 0:
            die(f"Invalid IV for {key}: {option['implied_volatility']}")

    @staticmethod
    def validate_option_chain(chain: Dict[str, Dict[str, Any]]) -> None:
        """Validate entire option chain. Dies if ANY option has missing data."""
        if not chain:
            die("Option chain is empty")

        for key, option_data in chain.items():
            DataValidator.validate_option_data(option_data, key)

    @staticmethod
    def validate_position(position: Dict[str, Any]) -> None:
        """
        Validate position data. Dies if required fields missing.

        Required:
        - symbol
        - quantity
        """
        if not position:
            die("Position data is None")

        if "symbol" not in position:
            die("Missing 'symbol' in position")
        if not position["symbol"]:
            die("Empty symbol in position")

        if "quantity" not in position:
            die("Missing 'quantity' in position")

    @staticmethod
    def validate_api_response(response: Any, endpoint: str) -> None:
        """Validate API response. Dies if None or error."""
        if response is None:
            die(f"API response from {endpoint} is None")

        if isinstance(response, dict):
            if "error" in response:
                die(f"API error from {endpoint}: {response['error']}")
            if "status" in response and response["status"] != "success":
                die(f"API failure from {endpoint}: {response.get('message', 'Unknown error')}")

    @staticmethod
    def validate_config_value(value: Any, key: str, expected_type: type = None) -> None:
        """Validate configuration value. Dies if missing or wrong type."""
        if value is None:
            die(f"Configuration value '{key}' is None")

        if expected_type and not isinstance(value, expected_type):
            die(
                f"Configuration '{key}' has wrong type: expected {expected_type}, got {type(value)}"
            )

    @staticmethod
    def validate_schwab_order_response(response: Dict[str, Any]) -> None:
        """Validate Schwab order response. Dies if missing required fields."""
        if not response:
            die("Schwab order response is None or empty")

        if "orderId" not in response:
            die("Missing 'orderId' in Schwab order response")

        if "status" not in response:
            die("Missing 'status' in Schwab order response")

        if response["status"] == "REJECTED":
            die(f"Order rejected: {response.get('rejectReason', 'Unknown reason')}")

    @staticmethod
    def validate_databento_response(data: Any, symbol: str) -> None:
        """Validate Databento market data response. Dies if no data."""
        if data is None:
            die(f"Databento returned None for {symbol}")

        if hasattr(data, "__iter__"):
            # Check if iterator is empty
            first_item = next(data, None)
            if first_item is None:
                die(f"Databento returned no data for {symbol}")
            # Put item back if we got one
            import itertools

            return itertools.chain([first_item], data)

        return data

    @staticmethod
    def validate_risk_metrics(metrics: Dict[str, Any]) -> None:
        """Validate risk metrics. Dies if critical values missing."""
        if not metrics:
            die("Risk metrics are None or empty")

        required_fields = [
            "max_loss",
            "probability_assignment",
            "expected_return",
            "var_95",
            "margin_required",
        ]

        for field in required_fields:
            if field not in metrics:
                die(f"Missing required risk metric: {field}")
            if metrics[field] is None:
                die(f"Risk metric '{field}' is None")

    @staticmethod
    def validate_historical_data(data: List[Any], symbol: str, min_points: int = 20) -> None:
        """Validate historical data. Dies if insufficient data points."""
        if not data:
            die(f"No historical data for {symbol}")

        if len(data) < min_points:
            die(
                f"Insufficient historical data for {symbol}: {len(data)} points < {min_points} required"
            )

    @staticmethod
    def validate_calculation_result(result: Any, calculation: str) -> None:
        """Validate calculation result. Dies if NaN or invalid."""
        if result is None:
            die(f"{calculation} returned None")

        if hasattr(result, "value"):
            if result.value is None or (hasattr(result.value, "isnan") and result.value.isnan()):
                die(f"{calculation} returned NaN")
            if hasattr(result, "confidence") and result.confidence < 0.5:
                die(f"{calculation} has low confidence: {result.confidence}")
        else:
            import math

            if isinstance(result, float) and math.isnan(result):
                die(f"{calculation} returned NaN")


# Convenience functions for common validations
def validate_market_data(snapshot: Dict[str, Any]) -> None:
    """Validate complete market data including option chain."""
    DataValidator.validate_market_snapshot(snapshot)
    DataValidator.validate_option_chain(snapshot["option_chain"])


def validate_for_trading(snapshot: Dict[str, Any], account_data: Dict[str, Any]) -> None:
    """Validate all data needed for trading decision."""
    # Market data
    validate_market_data(snapshot)

    # Account data (handled by SingleAccountManager)
    from unity_wheel.portfolio import SingleAccountManager

    manager = SingleAccountManager()
    manager.parse_account(account_data)  # Dies if invalid


def validate_api_call(func_name: str, *args, **kwargs) -> None:
    """Decorator to validate API responses."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            DataValidator.validate_api_response(result, func_name)
            return result

        return wrapper

    return decorator