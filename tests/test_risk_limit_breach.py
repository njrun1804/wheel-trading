import sys
from unittest.mock import Mock

sys.modules["google"] = Mock()
sys.modules["google.cloud"] = Mock()
sys.modules["google.cloud.exceptions"] = Mock()

from src.unity_wheel.risk.analytics import RiskAnalyzer, RiskMetrics, RiskLimits


def test_risk_limit_breach_report():
    limits = RiskLimits(max_var_95=0.01, max_cvar_95=0.02, max_margin_utilization=0.1)
    analyzer = RiskAnalyzer(limits=limits)

    metrics = RiskMetrics(
        var_95=3000,
        var_99=4000,
        cvar_95=3500,
        cvar_99=4500,
        kelly_fraction=0.2,
        portfolio_delta=150,
        portfolio_gamma=20,
        portfolio_vega=6000,
        portfolio_theta=-500,
        margin_requirement=50000,
        margin_utilization=0.6,
    )

    breaches = analyzer.check_limits(metrics, 100000)
    report = analyzer.generate_risk_report(metrics, breaches, 100000)

    assert breaches
    assert report["breaches"]
