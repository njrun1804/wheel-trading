"""Live data validation to ensure no mock/dummy data is ever used.

This module provides strict validation to ensure all financial decisions
are based on real market data from Databento, never mock or placeholder values.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.unity_wheel.utils import get_logger

logger = get_logger(__name__)


class LiveDataValidator:
    """Validates that all data is from live market sources."""

    # Patterns that indicate mock/fake data
    MOCK_PATTERNS = [
        "mock",
        "dummy",
        "fake",
        "test",
        "demo",
        "placeholder",
        "example",
        "sample",
        "hardcoded",
    ]

    # Suspicious round numbers that might indicate fake data
    SUSPICIOUS_PRICES = [
        1.0,
        5.0,
        10.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        50.0,
        100.0,
        0.25,
        0.50,
        0.75,
        1.25,
        1.50,
        1.75,
        2.0,
        2.50,
        3.0,
    ]

    # Suspicious volatilities (too round)
    SUSPICIOUS_VOLS = [
        0.10,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.65,
        0.70,
        0.75,
        0.77,
        0.80,
        0.90,
        1.0,
    ]

    @classmethod
    def validate_environment(cls) -> None:
        """Ensure environment is configured for live data only."""
        # Check for skip validation flag
        if os.getenv("DATABENTO_SKIP_VALIDATION", "").lower() == "true":
            raise ValueError(
                "CRITICAL: DATABENTO_SKIP_VALIDATION is enabled! "
                "This allows the system to run without real data. "
                "Unset this environment variable immediately."
            )

        # Check for API key using SecretManager
        try:
            from src.unity_wheel.secrets.integration import get_databento_api_key

            get_databento_api_key()  # Will raise if not configured
        except Exception:
            raise ValueError(
                "CRITICAL: Databento API key not configured! "
                "Cannot fetch real market data without API credentials. "
                "Run: python scripts/setup-secrets.py"
            )

        # Check for any test/mock environment variables
        for key, value in os.environ.items():
            if any(pattern in key.lower() for pattern in ["mock", "test", "fake", "dummy"]):
                if key not in ["PATH", "PYTEST_CURRENT_TEST"]:
                    logger.warning(f"Suspicious environment variable: {key}={value}")

    @classmethod
    def validate_price(cls, price: float, symbol: str = "U") -> None:
        """Validate that a price looks like real market data."""
        if price <= 0:
            raise ValueError(f"Invalid price {price} for {symbol} - must be positive")

        # Check for suspicious round numbers
        if price in cls.SUSPICIOUS_PRICES:
            logger.warning(
                f"Suspicious round price {price} for {symbol} - verify this is real market data"
            )

        # Unity-specific checks
        if symbol == "U":
            if price < 15.0 or price > 60.0:
                raise ValueError(
                    f"Unity price {price} outside historical range [15-60]. "
                    "Possible mock data or data error."
                )

    @classmethod
    def validate_volatility(cls, vol: float, symbol: str = "U") -> None:
        """Validate that volatility looks realistic."""
        if vol <= 0 or vol > 5.0:
            raise ValueError(f"Invalid volatility {vol} - must be between 0 and 5.0")

        # Check for suspicious round numbers
        if vol in cls.SUSPICIOUS_VOLS:
            logger.warning(
                f"Suspicious round volatility {vol} for {symbol} - verify this is calculated from real data"
            )

        # Unity-specific checks
        if symbol == "U":
            if vol < 0.30 or vol > 2.0:
                logger.warning(f"Unity volatility {vol} outside typical range [0.30-2.0]")

    @classmethod
    def validate_option_chain(cls, chain: Dict[str, Any]) -> None:
        """Validate that option chain data is real."""
        if not chain:
            raise ValueError("Empty option chain - no market data available")

        # Check for too-perfect spreads or prices
        for strike, data in chain.items():
            bid = data.get("bid", 0)
            ask = data.get("ask", 0)

            if bid <= 0 or ask <= 0:
                raise ValueError(f"Invalid bid/ask for strike {strike}: {bid}/{ask}")

            spread = ask - bid
            if spread < 0.01:
                raise ValueError(f"Impossibly tight spread for strike {strike}: {spread}")

            # Check for round number premiums
            if bid in [0.05, 0.10, 0.25, 0.50, 1.00] and ask in [0.05, 0.10, 0.25, 0.50, 1.00]:
                logger.warning(f"Suspicious round bid/ask for strike {strike}: {bid}/{ask}")

    @classmethod
    def validate_timestamp(cls, timestamp: datetime) -> None:
        """Ensure timestamp is recent and during market hours."""
        now = datetime.now(timezone.utc)
        age = (now - timestamp).total_seconds()

        # Data should be very recent for live trading
        if age > 300:  # 5 minutes
            raise ValueError(
                f"Data is {age:.0f} seconds old - too stale for live trading. "
                "Ensure you're fetching real-time data."
            )

        # Check if it's a trading day/hour (basic check)
        if timestamp.weekday() >= 5:  # Weekend
            logger.warning("Data timestamp is on a weekend - market is closed")

        hour = timestamp.hour
        if hour < 13 or hour > 21:  # Rough market hours in UTC
            logger.warning(f"Data timestamp {hour}:00 UTC may be outside market hours")

    @classmethod
    def validate_market_snapshot(cls, snapshot: Dict[str, Any]) -> None:
        """Comprehensive validation of a market snapshot."""
        # Check required fields
        required = ["ticker", "current_price", "option_chain", "timestamp"]
        for field in required:
            if field not in snapshot:
                raise ValueError(f"Missing required field '{field}' in market snapshot")

        # Validate each component
        ticker = snapshot["ticker"]
        cls.validate_price(snapshot["current_price"], ticker)
        cls.validate_option_chain(snapshot["option_chain"])
        cls.validate_timestamp(snapshot["timestamp"])

        # Check for mock data indicators
        for key, value in snapshot.items():
            if isinstance(value, str):
                for pattern in cls.MOCK_PATTERNS:
                    if pattern in value.lower():
                        raise ValueError(
                            f"Found mock data indicator '{pattern}' in field '{key}': {value}"
                        )

        logger.info(
            f"Market snapshot validated for {ticker}",
            extra={
                "price": snapshot["current_price"],
                "option_count": len(snapshot["option_chain"]),
                "timestamp": snapshot["timestamp"].isoformat(),
            },
        )


def validate_market_data(data: Dict[str, Any]) -> None:
    """Convenience function to validate market data.

    Raises ValueError if data appears to be mock/fake.
    """
    LiveDataValidator.validate_environment()
    LiveDataValidator.validate_market_snapshot(data)
