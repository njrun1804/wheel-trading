

from .borrowing_cost_analyzer import (
    BorrowingCostAnalyzer,
    BorrowingSource,
    CapitalAllocationResult,
    analyze_borrowing_decision,
)
from .ev_analytics import EVRiskAnalyzer
from .stress_testing import StressTestScenarios
from .unity_margin import MarginResult, UnityMarginCalculator, calculate_unity_margin_requirement

# Import from analytics.py for compatibility
from .analytics import (
    RiskAnalyzer,
    RiskLevel,
    RiskLimitBreach,
    RiskLimits,
    RiskMetrics,
)
from .advanced_financial_modeling import AdvancedFinancialModeling

# GPU-accelerated risk analytics
try:
    from .analytics_gpu import RiskMonitorGPU, get_risk_monitor_gpu
    GPU_RISK_AVAILABLE = True
except ImportError:
    RiskMonitorGPU = None
    get_risk_monitor_gpu = None
    GPU_RISK_AVAILABLE = False

# Legacy GPU import for compatibility
try:
    from .risk_analytics_gpu import RiskAnalyticsGPU
except ImportError:
    RiskAnalyticsGPU = None

__all__ = [
    "BorrowingCostAnalyzer",
    "BorrowingSource",
    "CapitalAllocationResult",
    "analyze_borrowing_decision",
    "UnityMarginCalculator",
    "MarginResult",
    "calculate_unity_margin_requirement",
    "EVRiskAnalyzer",
    "StressTestScenarios",
    # Deprecated but still exported for compatibility
    "RiskAnalyzer",
    "RiskLevel",
    "RiskLimitBreach",
    "RiskLimits",
    "RiskMetrics",
    "AdvancedFinancialModeling",
    # GPU versions
    "RiskMonitorGPU",
    "get_risk_monitor_gpu",
    "RiskAnalyticsGPU",
    "GPU_RISK_AVAILABLE",
]
