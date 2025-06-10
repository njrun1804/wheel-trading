import types
import duckdb
import pytest

from src.unity_wheel.risk.pure_borrowing_analyzer import PureBorrowingAnalyzer
from src.unity_wheel.monitoring.scripts import data_quality_monitor as dqm
from src.unity_wheel.data_providers.base.validation import MarketDataValidator


class FailingConn:
    def execute(self, *args, **kwargs):
        raise duckdb.Error("fail")


def test_calculate_irr_logs_error(caplog, monkeypatch):
    monkeypatch.setattr(
        "src.unity_wheel.risk.pure_borrowing_analyzer.get_config",
        lambda: types.SimpleNamespace(),
    )
    analyzer = PureBorrowingAnalyzer()
    cash_flows = [(0, 100), (30, 100)]
    with caplog.at_level("ERROR"):
        irr = analyzer.calculate_irr(cash_flows)
    assert irr is None
    assert any("Failed to calculate IRR" in r.message for r in caplog.records)


def test_check_data_freshness_error(caplog):
    with caplog.at_level("ERROR"):
        result = dqm.check_data_freshness(FailingConn())
    assert result["unity_prices"]["status"] == "error"
    assert any("Failed to check unity_prices freshness" in r.message for r in caplog.records)


def test_validation_check_freshness_error(caplog):
    validator = MarketDataValidator()
    with caplog.at_level("ERROR"):
        ok = validator._check_freshness("bad", 5)
    assert ok is False
    assert any("Invalid timestamp format" in r.message for r in caplog.records)
