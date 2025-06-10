import importlib
import sys
import types
from unittest.mock import patch

import pytest

from src.unity_wheel.data_providers.base.validation import MarketDataValidator
from src.unity_wheel.risk.pure_borrowing_analyzer import PureBorrowingAnalyzer


def setup_dummy_duckdb(monkeypatch):
    fake = types.ModuleType("duckdb")

    class Error(Exception):
        pass

    fake.Error = Error
    monkeypatch.setitem(sys.modules, "duckdb", fake)
    return Error, importlib.import_module("src.unity_wheel.monitoring.scripts.data_quality_monitor")


def test_check_freshness_logs(monkeypatch):
    validator = MarketDataValidator()
    with patch("src.unity_wheel.data_providers.base.validation.logger") as logger:
        assert not validator._check_freshness("bad", 5)
        logger.warning.assert_called()


def test_calculate_irr_logs(monkeypatch):
    analyzer = PureBorrowingAnalyzer()
    with patch(
        "src.unity_wheel.risk.pure_borrowing_analyzer.brentq", side_effect=ValueError("fail")
    ):
        with patch("src.unity_wheel.risk.pure_borrowing_analyzer.logger") as logger:
            result = analyzer.calculate_irr([(0, -1000), (30, 1100)])
            assert result is None
            logger.warning.assert_called()


def test_data_quality_fallback(monkeypatch):
    Error, dq = setup_dummy_duckdb(monkeypatch)

    class BadConn:
        def execute(self, *a, **k):
            raise Error("db fail")

    with patch.object(dq, "logger") as logger:
        freshness = dq.check_data_freshness(BadConn())
        quality = dq.check_data_quality(BadConn())

        assert freshness["unity_prices"]["status"] == "error"
        assert freshness["options"]["status"] == "error"
        assert freshness["fred"]["status"] == "error"
        assert quality["price_gaps"]["status"] == "error"
        assert quality["volatility"]["status"] == "error"
        assert quality["spreads"]["status"] == "none"
        assert logger.error.call_count == 6
