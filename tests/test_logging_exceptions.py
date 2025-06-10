import logging

import duckdb
import pytest

from unity_wheel.data_providers.base.validation import DataValidator
from unity_wheel.monitoring.scripts import data_quality_monitor as dq
from unity_wheel.risk.pure_borrowing_analyzer import PureBorrowingAnalyzer


def test_check_freshness_invalid_timestamp(caplog):
    dv = DataValidator()
    caplog.set_level(logging.WARNING)
    assert not dv._check_freshness('bad timestamp', 5)
    assert any('Failed to parse timestamp' in rec.message for rec in caplog.records)


def test_data_quality_monitor_handles_db_error(caplog):
    class BadConn:
        def execute(self, *args, **kwargs):
            raise duckdb.Error('boom')

    caplog.set_level(logging.WARNING)
    res = dq.check_data_freshness(BadConn())
    assert res["unity_prices"]["status"] == "error"
    assert any('Failed to fetch unity price freshness' in r.message for r in caplog.records)


def test_calculate_irr_failure_logged(caplog):
    analyzer = PureBorrowingAnalyzer()
    caplog.set_level(logging.WARNING)
    irr = analyzer.calculate_irr([(0, -1000), (10, -900)])
    assert irr is None
    assert any('Failed to calculate IRR' in rec.message for rec in caplog.records)
