import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from enum import Enum
import sys
import types
import pytest

# Mock google.cloud to avoid optional dependency issues
fake_google = types.ModuleType("google")
fake_cloud = types.ModuleType("cloud")
fake_cloud.storage = None
fake_exceptions = types.ModuleType("exceptions")
fake_exceptions.NotFound = type("NotFound", (), {})
fake_cloud.exceptions = fake_exceptions
sys.modules.setdefault("google", fake_google)
sys.modules.setdefault("google.cloud", fake_cloud)
sys.modules.setdefault("google.cloud.exceptions", fake_exceptions)
sys.modules.setdefault("databento", types.ModuleType("databento"))
fake_dbn = types.ModuleType("databento_dbn")
class DummyEnum:
    pass
fake_dbn.Schema = DummyEnum
fake_dbn.SType = DummyEnum
sys.modules.setdefault("databento_dbn", fake_dbn)
fake_tenacity = types.ModuleType("tenacity")
def dummy_decorator(*args, **kwargs):
    def wrapper(fn):
        async def inner(*i, **k):
            return await fn(*i, **k)
        return inner
    return wrapper
fake_tenacity.retry = dummy_decorator
fake_tenacity.retry_if_exception_type = lambda *a, **k: None
fake_tenacity.stop_after_attempt = lambda *a, **k: None
fake_tenacity.wait_exponential = lambda *a, **k: None
sys.modules.setdefault("tenacity", fake_tenacity)

from src.unity_wheel.data_providers.databento.integration import DatabentoIntegration
from src.unity_wheel.data_providers.base.manager import FREDDataManager

class DummyDBClient:
    async def _get_underlying_price(self, underlying, timestamp=None):
        return SimpleNamespace(last_price=30.0)

def test_get_wheel_candidates_concurrent(monkeypatch):
    client = DummyDBClient()
    integration = DatabentoIntegration(client)

    expirations = [
        datetime(2025, 6, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime(2025, 8, 1, tzinfo=timezone.utc),
    ]

    monkeypatch.setattr(integration, "_get_monthly_expirations", lambda s, e: expirations)

    async def fake_analyze(exp, *args, **kwargs):
        await asyncio.sleep(0.1)
        return [{"expiration": exp, "expected_return": 1.0}]

    monkeypatch.setattr(integration, "_analyze_expiration", fake_analyze)

    async def run_test():
        start = asyncio.get_event_loop().time()
        res = await integration.get_wheel_candidates(underlying="U")
        elapsed = asyncio.get_event_loop().time() - start
        return res, elapsed

    res, elapsed = asyncio.run(run_test())

    assert len(res) == len(expirations)
    assert elapsed < 0.25

class DummyClientCtx:
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass

def test_update_data_concurrent(monkeypatch):
    class Series(Enum):
        A = "A"
        B = "B"
        C = "C"

    monkeypatch.setattr(
        "src.unity_wheel.data_providers.base.manager.WheelStrategyFREDSeries", Series
    )
    monkeypatch.setattr(
        "src.unity_wheel.data_providers.base.manager.FREDClient", lambda *a, **k: DummyClientCtx()
    )

    manager = FREDDataManager(auto_update=True)

    async def fake_update(self, client, series_enum):
        await asyncio.sleep(0.1)
        return series_enum.value, 1

    monkeypatch.setattr(FREDDataManager, "_update_series", fake_update, raising=False)

    async def run_test():
        start = asyncio.get_event_loop().time()
        updates = await manager.update_data()
        elapsed = asyncio.get_event_loop().time() - start
        return updates, elapsed

    updates, elapsed = asyncio.run(run_test())

    assert len(updates) == 3
    assert elapsed < 0.25
