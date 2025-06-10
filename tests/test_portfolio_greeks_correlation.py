import math
import sys
import types

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))

from src.config.loader import get_config
from src.unity_wheel.risk.analytics import RiskAnalyzer
from src.unity_wheel.models.position import Position
from src.unity_wheel.models.greeks import Greeks


def test_portfolio_greek_aggregation_with_correlation(monkeypatch) -> None:
    config = get_config()
    config_dict = config.model_dump()
    config_dict["risk"]["correlation_matrix"] = {
        "matrix": {
            "U": {"U": 1.0, "SPY": 0.5},
            "SPY": {"U": 0.5, "SPY": 1.0},
        }
    }
    from src.config.schema import WheelConfig

    custom_config = WheelConfig(**config_dict)
    monkeypatch.setattr("src.config.loader.get_config", lambda: custom_config)

    analyzer = RiskAnalyzer()
    positions = [
        (Position("U241220C00080000", 1), Greeks(delta=0.5, gamma=0.1, vega=0.2, theta=-0.01, rho=0.05), 50.0),
        (Position("SPY241220C00400000", 1), Greeks(delta=0.4, gamma=0.2, vega=0.1, theta=-0.02, rho=0.04), 400.0),
    ]

    aggregated, _ = analyzer.aggregate_portfolio_greeks(positions)

    g1 = 100 * 0.1
    g2 = 100 * 0.2
    expected_gamma = math.sqrt(g1 * g1 + g2 * g2 + 2 * g1 * g2 * 0.5)
    assert math.isclose(aggregated["gamma"], expected_gamma, rel_tol=1e-9)
    assert aggregated["delta"] == 90.0

