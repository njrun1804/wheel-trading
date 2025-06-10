import sys
from types import ModuleType

import pytest

# Provide stub for sklearn if not installed
if "sklearn" not in sys.modules:
    sys.modules["sklearn"] = ModuleType("sklearn")
    sys.modules["sklearn.ensemble"] = ModuleType("sklearn.ensemble")
    sys.modules["sklearn.ensemble"].IsolationForest = object
    sys.modules["sklearn.mixture"] = ModuleType("sklearn.mixture")
    sys.modules["sklearn.mixture"].GaussianMixture = object

if "dotenv" not in sys.modules:
    dotenv_stub = ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "pydantic_settings" not in sys.modules:
    pydantic_stub = ModuleType("pydantic_settings")

    class BaseSettings:  # minimal stub
        pass

    pydantic_stub.BaseSettings = BaseSettings
    pydantic_stub.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = pydantic_stub

if "yaml" not in sys.modules:
    yaml_stub = ModuleType("yaml")
    yaml_stub.safe_load = lambda *args, **kwargs: {}
    sys.modules["yaml"] = yaml_stub

if "google" not in sys.modules:
    google_stub = ModuleType("google")
    sys.modules["google"] = google_stub
    cloud_stub = ModuleType("google.cloud")
    cloud_stub.storage = object
    cloud_stub.exceptions = ModuleType("google.cloud.exceptions")
    cloud_stub.exceptions.NotFound = Exception
    sys.modules["google.cloud"] = cloud_stub
    sys.modules["google.cloud.exceptions"] = cloud_stub.exceptions

from src.unity_wheel.api.advisor import WheelAdvisor


def test_validate_option_liquidity_uses_constraints():
    advisor = WheelAdvisor()
    option = {"bid": 1.0, "ask": 1.1, "volume": 150, "open_interest": 200}
    assert advisor._validate_option_liquidity(option)

    option_bad_spread = {"bid": 1.0, "ask": 10.0, "volume": 150, "open_interest": 200}
    assert not advisor._validate_option_liquidity(option_bad_spread)

    option_low_volume = {
        "bid": 1.0,
        "ask": 1.05,
        "volume": advisor.constraints.MIN_VOLUME - 1,
        "open_interest": 200,
    }
    assert not advisor._validate_option_liquidity(option_low_volume)

    option_low_oi = {
        "bid": 1.0,
        "ask": 1.05,
        "volume": 150,
        "open_interest": advisor.constraints.MIN_OPEN_INTEREST - 1,
    }
    assert not advisor._validate_option_liquidity(option_low_oi)
