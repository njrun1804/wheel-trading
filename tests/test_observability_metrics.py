import sys
sys.modules.setdefault("google", type(sys)("google"))
sys.modules.setdefault("google.cloud", type(sys)("google.cloud"))
sys.modules.setdefault("google.cloud.storage", type(sys)("google.cloud.storage"))
sys.modules.setdefault("google.cloud.exceptions", type(sys)("google.cloud.exceptions"))
sys.modules["google.cloud.exceptions"].NotFound = Exception
sys.modules.setdefault("databento", type(sys)("databento"))
import types
dbn = types.ModuleType("databento_dbn")
dbn.Schema = object
dbn.SType = object
sys.modules.setdefault("databento_dbn", dbn)
from src.unity_wheel.monitoring import performance_monitored, get_performance_monitor
from src.unity_wheel.storage.cache.general_cache import cached, invalidate_cache
from src.unity_wheel.metrics import metrics_collector
from src.unity_wheel.risk.analytics import RiskMetrics

@performance_monitored("sample_func")
@cached(ttl=1)
def _sample(x: int) -> int:
    return x * 2


def test_function_timing_and_cache() -> None:
    invalidate_cache()
    metrics_collector.function_timings.clear()
    metrics_collector.cache_hits = 0
    metrics_collector.cache_misses = 0
    _sample(2)
    _sample(2)
    stats = metrics_collector.get_function_stats()
    cache = metrics_collector.get_cache_summary()
    assert "sample_func" in stats
    assert stats["sample_func"]["count"] >= 2
    assert cache["hits"] >= 1


def test_risk_metrics_recording() -> None:
    metrics_collector.risk_history.clear()
    rm = RiskMetrics(
        var_95=1.0,
        var_99=1.5,
        cvar_95=2.0,
        cvar_99=2.5,
        kelly_fraction=0.1,
        portfolio_delta=10.0,
        portfolio_gamma=1.0,
        portfolio_vega=2.0,
        portfolio_theta=-0.1,
        margin_requirement=100.0,
        margin_utilization=0.2,
    )
    metrics_collector.record_risk_metrics(rm)
    dist = metrics_collector.get_risk_distribution()
    assert dist["var_95"]["avg"] == 1.0
    assert "margin_requirement" in dist


def test_performance_report_generation() -> None:
    monitor = get_performance_monitor()
    text_report = monitor.generate_report(format="text")
    json_report = monitor.generate_report(format="json")
    assert "PERFORMANCE MONITORING REPORT" in text_report
    assert "operations_tracked" in json_report
    metrics_output = metrics_collector.generate_report()
    assert "DECISION METRICS REPORT" in metrics_output


