import logging

import duckdb
import pytest
from unity_wheel.data_providers.base.validation import MarketDataValidator
from unity_wheel.monitoring.scripts import data_quality_monitor as dq
from unity_wheel.risk.pure_borrowing_analyzer import PureBorrowingAnalyzer


def test_check_freshness_invalid_timestamp_logs(caplog):
    validator = MarketDataValidator()
    caplog.set_level(logging.WARNING)
    assert not validator._check_freshness("bad", 5)
    assert any("Invalid timestamp format" in r.getMessage() for r in caplog.records)


class BadConn:
    def execute(self, *args, **kwargs):
        raise duckdb.Error("boom")


def test_data_quality_monitor_fallback_on_error(caplog):
    caplog.set_level(logging.ERROR)
    result = dq.check_data_freshness(BadConn())
    assert result["unity_prices"]["status"] == "error"
    assert any("Failed checking unity_prices" in r.getMessage() for r in caplog.records)


def test_calculate_irr_failure_logs(caplog):
    analyzer = PureBorrowingAnalyzer()
    caplog.set_level(logging.WARNING)
    cash_flows = [(0, -1000), (1, -50)]
    assert analyzer.calculate_irr(cash_flows) is None
    assert any("IRR calculation failed" in r.getMessage() for r in caplog.records)
