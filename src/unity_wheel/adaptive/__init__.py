"""
Adaptive trading system for Unity wheel strategy.

This module contains all adaptive logic for dynamically adjusting
trading parameters based on market conditions.

Core Components:
- AdaptiveBase: Base class for adaptive strategies
- AdaptiveWheel: Unity-specific adaptive wheel implementation
- RegimeDetector: Volatility regime detection
- DynamicOptimizer: Continuous parameter optimization
"""

from .adaptive_base import (
    MarketRegime,
    OutcomeTracker,
    UnityAdaptiveSystem,
    UnityConditions,
    WheelOutcome,
    WheelRecommendation,
)
from .adaptive_wheel import AdaptiveWheelStrategy, create_adaptive_wheel_strategy
from .dynamic_optimizer import DynamicOptimizer, MarketState, OptimizationResult
from .regime_detector import RegimeDetector, RegimeInfo


# Convenience functions for common adaptive operations
def get_volatility_tier(volatility: float) -> str:
    """
    Get volatility tier for current market conditions.

    Parameters
    ----------
    volatility : float
        Current implied volatility (0.0 to 2.0)

    Returns
    -------
    str
        Volatility tier: 'low', 'normal', 'elevated', 'high', 'extreme'
    """
    if volatility < 0.40:
        return "low"
    elif volatility < 0.60:
        return "normal"
    elif volatility < 0.80:
        return "elevated"
    elif volatility < 1.00:
        return "high"
    else:
        return "extreme"


def get_position_size_multiplier(volatility: float, drawdown: float) -> float:
    """
    Get position size multiplier based on market conditions.

    Parameters
    ----------
    volatility : float
        Current implied volatility
    drawdown : float
        Current portfolio drawdown (negative value)

    Returns
    -------
    float
        Position size multiplier (0.0 to 1.2)
    """
    # Volatility-based sizing
    if volatility < 0.40:
        vol_mult = 1.2  # Opportunity
    elif volatility < 0.60:
        vol_mult = 1.0  # Normal
    elif volatility < 0.80:
        vol_mult = 0.7  # Caution
    elif volatility < 1.00:
        vol_mult = 0.5  # Defensive
    else:
        vol_mult = 0.3  # Extreme

    # Drawdown adjustment
    drawdown_mult = max(0, 1 + drawdown / 0.20)  # Linear to -20%

    return vol_mult * drawdown_mult


def should_trade_unity(
    volatility: float, drawdown: float, days_to_earnings: int, max_volatility: float = 1.50
) -> tuple[bool, str]:
    """
    Determine if trading conditions are suitable for Unity.

    Parameters
    ----------
    volatility : float
        Current implied volatility
    drawdown : float
        Current portfolio drawdown
    days_to_earnings : int
        Days until next earnings
    max_volatility : float
        Maximum volatility threshold

    Returns
    -------
    tuple[bool, str]
        (should_trade, reason)
    """
    if volatility > max_volatility:
        return False, f"Volatility {volatility:.0%} exceeds {max_volatility:.0%} limit"

    if drawdown < -0.20:
        return False, f"Drawdown {drawdown:.0%} exceeds -20% limit"

    if days_to_earnings < 7:
        return False, f"Earnings in {days_to_earnings} days"

    return True, "Conditions acceptable"


__all__ = [
    # Base classes
    "MarketRegime",
    "UnityConditions",
    "WheelRecommendation",
    "UnityAdaptiveSystem",
    "WheelOutcome",
    "OutcomeTracker",
    # Wheel strategy
    "AdaptiveWheelStrategy",
    "create_adaptive_wheel_strategy",
    # Regime detection
    "RegimeDetector",
    "RegimeInfo",
    # Dynamic optimization
    "DynamicOptimizer",
    "MarketState",
    "OptimizationResult",
    # Convenience functions
    "get_volatility_tier",
    "get_position_size_multiplier",
    "should_trade_unity",
]
