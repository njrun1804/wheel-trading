import importlib
import logging
import importlib
import logging
import sys
from types import SimpleNamespace

import pytest


def _setup_fake_duckdb(monkeypatch):
    class FakeError(Exception):
        pass

    fake_module = SimpleNamespace(Error=FakeError)
    monkeypatch.setitem(sys.modules, "duckdb", fake_module)
    return fake_module


def test_check_freshness_invalid_timestamp(monkeypatch, caplog):
    _setup_fake_duckdb(monkeypatch)
    from src.unity_wheel.data_providers.base.validation import MarketDataValidator

    validator = MarketDataValidator()
    with caplog.at_level(logging.WARNING):
        assert not validator._check_freshness("bad-date", 5)
    assert "Invalid timestamp format" in caplog.text


def test_data_quality_monitor_db_error(monkeypatch, caplog):
    fake = _setup_fake_duckdb(monkeypatch)
    dqm = importlib.reload(
        importlib.import_module("src.unity_wheel.monitoring.scripts.data_quality_monitor")
    )

    class BadConn:
        def execute(self, *args, **kwargs):
            raise fake.Error("boom")

    with caplog.at_level(logging.ERROR):
        result = dqm.check_data_freshness(BadConn())
    assert result["unity_prices"]["status"] == "error"
    assert "price freshness" in caplog.text


def test_calculate_irr_failure_logs(caplog):
    from src.unity_wheel.risk.pure_borrowing_analyzer import PureBorrowingAnalyzer

    analyzer = PureBorrowingAnalyzer()
    cash_flows = [(0, -100), (10, -50)]

    with caplog.at_level(logging.ERROR):
        result = analyzer.calculate_irr(cash_flows)
    assert result is None
    assert "IRR calculation failed" in caplog.text
