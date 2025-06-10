import logging
import duckdb

from src.unity_wheel.data_providers.base.validation import MarketDataValidator
from src.unity_wheel.monitoring.scripts.data_quality_monitor import check_data_freshness
from src.unity_wheel.risk.pure_borrowing_analyzer import PureBorrowingAnalyzer


def test_check_freshness_invalid_timestamp(caplog):
    validator = MarketDataValidator()
    with caplog.at_level(logging.WARNING):
        assert validator._check_freshness("bad timestamp", 5) is False
        assert "Invalid timestamp format" in caplog.text


def test_check_data_freshness_db_error(caplog):
    conn = duckdb.connect(":memory:")
    with caplog.at_level(logging.WARNING):
        result = check_data_freshness(conn)
        assert result["unity_prices"]["status"] == "error"
        assert "Failed to fetch unity price freshness" in caplog.text


def test_calculate_irr_failure(caplog):
    analyzer = PureBorrowingAnalyzer()
    cash_flows = [(0, -100), (10, -10)]  # no sign change -> brentq fails
    with caplog.at_level(logging.WARNING):
        irr = analyzer.calculate_irr(cash_flows)
        assert irr is None
        assert "IRR calculation failed" in caplog.text
