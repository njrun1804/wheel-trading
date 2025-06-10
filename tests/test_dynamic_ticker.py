import inspect

import pytest

from src.config.loader import get_config
from src.unity_wheel.cli.databento_integration import (
    create_databento_market_snapshot,
    get_market_data_sync,
)
from src.unity_wheel.data_providers.validation.live_data_validator import LiveDataValidator


def test_ticker_changes_propagate() -> None:
    config = get_config()
    original = config.unity.ticker
    config.unity.ticker = "XYZ"
    try:
        with pytest.raises(ValueError):
            LiveDataValidator.validate_price(10.0)
        sig = inspect.signature(create_databento_market_snapshot)
        assert sig.parameters["ticker"].default is None
        sig2 = inspect.signature(get_market_data_sync)
        assert sig2.parameters["ticker"].default is None
    finally:
        config.unity.ticker = original
