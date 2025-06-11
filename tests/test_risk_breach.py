import sys
import types

sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules.setdefault("google.cloud", types.ModuleType("google.cloud"))
sys.modules.setdefault("google.cloud.storage", types.ModuleType("google.cloud.storage"))
exc_mod = types.ModuleType("google.cloud.exceptions")
setattr(exc_mod, "NotFound", type("NotFound", (), {}))
sys.modules.setdefault("google.cloud.exceptions", exc_mod)

from src.unity_wheel.risk.analytics import RiskAnalyzer, RiskLimits, RiskMetrics


def test_risk_limit_breach_detection():
    limits = RiskLimits(max_var_95=0.05, max_cvar_95=0.07)
    analyzer = RiskAnalyzer(limits=limits)

    metrics = RiskMetrics(
        var_95=100.0,
        var_99=120.0,
        cvar_95=150.0,
        cvar_99=170.0,
        kelly_fraction=0.2,
        portfolio_delta=0.0,
        portfolio_gamma=0.0,
        portfolio_vega=0.0,
        portfolio_theta=0.0,
        margin_requirement=0.0,
        margin_utilization=0.0,
    )

    breaches = analyzer.check_limits(metrics, portfolio_value=1000.0)
    report = analyzer.generate_risk_report(metrics, breaches, 1000.0)

    assert breaches
    assert any(b.metric == "var_95" for b in breaches)
    assert report["breaches"]
