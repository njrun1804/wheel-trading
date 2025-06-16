"""API for wheel trading advisor."""

from .advisor import TradingConstraints, WheelAdvisor
from .advisor_simple import SimpleWheelAdvisor
from .types import (
    Action,
    MarketSnapshot,
    OptionData,
    PositionData,
    Recommendation,
    RiskMetrics,
)

# Alias for backward compatibility and Agent 8 system analysis
TradingAdvisor = WheelAdvisor

__all__ = [
    # Advisor
    "WheelAdvisor",
    "TradingAdvisor",  # Alias for WheelAdvisor
    "SimpleWheelAdvisor",
    "TradingConstraints",
    # Types
    "Action",
    "MarketSnapshot",
    "OptionData",
    "PositionData",
    "Recommendation",
    "RiskMetrics",
]
