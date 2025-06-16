"""Integration module for component wiring and optimization."""

from .component_wiring import (
    ComponentRegistry,
    IntegratedDecisionTracker,
    IntegratedRiskAnalyzer,
    IntegratedStatsAnalyzer,
    IntegratedWheelStrategy,
    get_component_registry,
)

__all__ = [
    "ComponentRegistry",
    "IntegratedDecisionTracker",
    "IntegratedRiskAnalyzer",
    "IntegratedStatsAnalyzer",
    "IntegratedWheelStrategy",
    "get_component_registry",
]
