import logging
import sys
import types

# Stub optional heavy dependencies so imports succeed
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))

config_loader = types.ModuleType("loader")
def get_config():
    class Config:
        unity = types.SimpleNamespace(ticker="U")
    return Config()
config_loader.get_config = get_config
sys.modules.setdefault("src.config.loader", config_loader)
sys.modules.setdefault("src.config", types.ModuleType("config"))

import pytest
scipy_mod = types.ModuleType("scipy")
opt_mod = types.ModuleType("optimize")
stats_mod = types.ModuleType("stats")

def _brentq(*args, **kwargs):
    raise ValueError("stub")

opt_mod.brentq = _brentq
scipy_mod.optimize = opt_mod
scipy_mod.stats = stats_mod
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.optimize", opt_mod)
sys.modules.setdefault("scipy.stats", stats_mod)

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Prepare package hierarchy for relative imports
for mod in [
    ("src", "src"),
    ("src.unity_wheel", "unity_wheel"),
    ("src.unity_wheel.risk", "risk"),
    ("src.unity_wheel.data_providers", "data_providers"),
    ("src.unity_wheel.utils", "utils"),
    ("src.unity_wheel.monitoring.scripts", "scripts"),
]:
    m = types.ModuleType(mod[1])
    m.__path__ = []
    sys.modules.setdefault(mod[0], m)

pb_spec = importlib.util.spec_from_file_location(
    "src.unity_wheel.risk.pure_borrowing_analyzer",
    ROOT / "src/unity_wheel/risk/pure_borrowing_analyzer.py",
)
pb_module = importlib.util.module_from_spec(pb_spec)
assert pb_spec.loader
pb_spec.loader.exec_module(pb_module)
PureBorrowingAnalyzer = pb_module.PureBorrowingAnalyzer

val_spec = importlib.util.spec_from_file_location(
    "src.unity_wheel.data_providers.base.validation",
    ROOT / "src/unity_wheel/data_providers/base/validation.py",
)
val_module = importlib.util.module_from_spec(val_spec)
assert val_spec.loader
val_spec.loader.exec_module(val_module)
MarketDataValidator = val_module.MarketDataValidator

dqm_spec = importlib.util.spec_from_file_location(
    "src.unity_wheel.monitoring.scripts.data_quality_monitor",
    ROOT / "src/unity_wheel/monitoring/scripts/data_quality_monitor.py",
)
dqm = importlib.util.module_from_spec(dqm_spec)
assert dqm_spec.loader
dqm_spec.loader.exec_module(dqm)


def test_calculate_irr_failure_logs(caplog):
    analyzer = PureBorrowingAnalyzer()
    cash_flows = [(0, -1000), (30, -100)]
    with caplog.at_level(logging.WARNING):
        result = analyzer.calculate_irr(cash_flows)
    assert result is None
    assert any("Failed to compute IRR" in r.getMessage() for r in caplog.records)


def test_check_freshness_invalid_logs(caplog):
    validator = MarketDataValidator()
    with caplog.at_level(logging.WARNING):
        assert not validator._check_freshness("bad", minutes=5)
    assert any("Invalid timestamp format" in r.getMessage() for r in caplog.records)


def test_check_data_freshness_db_error(monkeypatch, caplog):
    class DummyError(Exception):
        pass

    dummy_duckdb = types.SimpleNamespace(Error=DummyError)
    monkeypatch.setattr(dqm, "duckdb", dummy_duckdb)

    class DummyConn:
        def execute(self, *args, **kwargs):
            raise DummyError("fail")

    with caplog.at_level(logging.WARNING):
        result = dqm.check_data_freshness(DummyConn())

    assert result["unity_prices"]["status"] == "error"
    assert any(
        "Failed to fetch unity price freshness" in r.getMessage() for r in caplog.records
    )
