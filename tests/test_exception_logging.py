import sys
import types
from types import SimpleNamespace

import pytest


def test_validation_invalid_timestamp_logs_error(caplog):
    sys.modules.setdefault('numpy', types.ModuleType('numpy'))
    from src.unity_wheel.data_providers.base.validation import MarketDataValidator

    validator = MarketDataValidator()

    with caplog.at_level("ERROR"):
        assert validator._check_freshness("bad", minutes=5) is False

    assert any(
        "invalid_timestamp_format" in r.message for r in caplog.records
    )


def test_data_quality_monitor_handles_db_error(caplog):
    config_mod = types.ModuleType('src.config.loader')
    config_mod.get_config = lambda: SimpleNamespace(unity=SimpleNamespace(ticker='U'))
    sys.modules['src.config.loader'] = config_mod

    duckdb_mod = types.ModuleType('duckdb')
    class Error(Exception):
        pass
    duckdb_mod.Error = Error
    sys.modules['duckdb'] = duckdb_mod

    from src.unity_wheel.monitoring.scripts import data_quality_monitor as dqm

    class Conn:
        def execute(self, *_, **__):
            raise Error('db fail')

    with caplog.at_level("ERROR"):
        result = dqm.check_data_freshness(Conn())

    assert result['unity_prices']['status'] == 'error'
    assert any(
        'freshness_unity_prices_failed' in r.message for r in caplog.records
    )
