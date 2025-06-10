import sys
import types
import numpy as np

# Stub google.cloud to avoid optional dependency errors during import
google = types.ModuleType("google")
cloud = types.ModuleType("google.cloud")
cloud.storage = types.ModuleType("storage")
cloud.storage.Client = object  # type: ignore
exceptions = types.ModuleType("exceptions")
exceptions.NotFound = Exception  # type: ignore
cloud.exceptions = exceptions  # type: ignore
google.cloud = cloud  # type: ignore
sys.modules.setdefault("google.cloud.exceptions", exceptions)
sys.modules.setdefault("google", google)
sys.modules.setdefault("google.cloud", cloud)

from src.unity_wheel.risk.analytics import RiskAnalyzer
from src.unity_wheel.models.position import Position
from src.unity_wheel.models.greeks import Greeks


def test_aggregate_greeks_default_correlation() -> None:
    analyzer = RiskAnalyzer()
    positions = [
        (Position("AAPL", 100), Greeks(delta=1.0), 150.0),
        (Position("MSFT", 100), Greeks(delta=1.0), 300.0),
    ]
    aggregated, _ = analyzer.aggregate_portfolio_greeks(positions)
    assert np.isclose(aggregated["delta"], np.sqrt(100**2 + 100**2))
    assert aggregated["delta"] < 200


def test_aggregate_greeks_with_correlation() -> None:
    analyzer = RiskAnalyzer()
    positions = [
        (Position("AAPL", 100), Greeks(delta=1.0), 150.0),
        (Position("MSFT", 100), Greeks(delta=1.0), 300.0),
    ]
    correlations = {("AAPL", "MSFT"): 0.8}
    aggregated, _ = analyzer.aggregate_portfolio_greeks(positions, correlations)
    expected = np.sqrt(100**2 + 100**2 + 2 * 0.8 * 100 * 100)
    assert np.isclose(aggregated["delta"], expected)
