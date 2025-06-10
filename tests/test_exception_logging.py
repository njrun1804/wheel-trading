import logging
import duckdb
from src.unity_wheel.data_providers.base.validation import MarketDataValidator
from src.unity_wheel.monitoring.scripts.data_quality_monitor import check_data_freshness
from src.unity_wheel.risk.pure_borrowing_analyzer import PureBorrowingAnalyzer


def test_check_freshness_invalid_timestamp(caplog):
    validator = MarketDataValidator()
    with caplog.at_level(logging.ERROR):
        assert validator._check_freshness("bad timestamp", 5) is False
    assert "Invalid timestamp format" in caplog.text


def test_calculate_irr_logging(caplog):
    analyzer = PureBorrowingAnalyzer()
    flows = [(0, -100), (1, -150)]
    with caplog.at_level(logging.ERROR):
        irr = analyzer.calculate_irr(flows)
    assert irr is None
    assert "Failed to compute IRR" in caplog.text


def test_check_data_freshness_missing_tables(caplog):
    conn = duckdb.connect(":memory:")
    with caplog.at_level(logging.ERROR):
        result = check_data_freshness(conn)
    assert result["unity_prices"]["status"] == "error"
    assert result["options"]["status"] == "error"
    assert result["fred"]["status"] == "error"
    assert "Failed to fetch unity price freshness" in caplog.text
