"""
from __future__ import annotations

Dynamic parameter optimizer using continuous functions.
Directly optimizes: CAGR - 0.20 × |CVaR₉₅| with autonomous operation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from ..config.loader import get_config
from ..utils import get_logger, timed_operation

logger = get_logger(__name__)


class OptimizationResult(NamedTuple):
    """Result of dynamic optimization with confidence scoring."""

    delta_target: float
    dte_target: int
    kelly_fraction: float
    expected_cagr: float
    expected_cvar: float
    objective_value: float
    confidence_score: float
    diagnostics: Dict[str, float]


@dataclass
class MarketState:
    """Current market conditions as continuous variables."""

    realized_volatility: float  # 20-day realized vol (annualized)
    volatility_percentile: float  # 0-1, where in historical distribution
    price_momentum: float  # 20-day price return
    volume_ratio: float  # Current vs 20-day average
    iv_rank: Optional[float] = None  # 0-100 if available
    days_to_earnings: Optional[int] = None


class DynamicOptimizer:
    """
    Continuous parameter optimization for wheel strategy.
    No discrete regimes - smooth transitions based on market state.
    """

    # Constants from objective function
    CVAR_PENALTY = 0.20  # Penalty weight for CVaR
    BASE_KELLY = 0.50  # Half-Kelly as specified

    def __init__(self, symbol: str = None, config: Optional[Dict] = None):
        if symbol is None:
            app_config = get_config()
            symbol = app_config.unity.ticker
        self.symbol = symbol
        self.config = config or {}
        self.vol_history: Optional[pd.Series] = None
        self.optimization_history: list = []

    def _get_config_bounds(self, param_type: str) -> Tuple[float, float]:
        """Get min/max bounds from config or defaults."""
        defaults = {"delta": (0.10, 0.40), "dte": (21, 49), "kelly": (0.10, 0.50)}
        if self.config:
            bounds = self.config.get("optimization", {}).get("bounds", {})
            return (
                bounds.get(f"{param_type}_min", defaults[param_type][0]),
                bounds.get(f"{param_type}_max", defaults[param_type][1]),
            )
        return defaults[param_type]

    @timed_operation(threshold_ms=50)
    def optimize_parameters(
        self, market_state: MarketState, historical_returns: np.ndarray
    ) -> OptimizationResult:
        """
        Dynamically optimize parameters to maximize objective function.

        Objective: maximize CAGR - 0.20 × |CVaR₉₅|
        """
        logger.info(
            "Starting dynamic optimization",
            volatility=f"{market_state.realized_volatility:.1%}",
            vol_percentile=f"{market_state.volatility_percentile:.1%}",
        )

        # 1. Calculate dynamic delta target (continuous function)
        delta_target = self._calculate_dynamic_delta(market_state)

        # 2. Calculate dynamic DTE (continuous function)
        dte_target = self._calculate_dynamic_dte(market_state)

        # 3. Calculate dynamic Kelly fraction
        kelly_fraction = self._calculate_dynamic_kelly(market_state, historical_returns)

        # 4. Estimate expected outcomes
        expected_cagr = self._estimate_cagr(delta_target, dte_target, kelly_fraction, market_state)
        expected_cvar = self._estimate_cvar(delta_target, kelly_fraction, historical_returns)

        # 5. Calculate objective value
        objective_value = expected_cagr - self.CVAR_PENALTY * abs(expected_cvar)

        # 6. Calculate confidence in optimization
        confidence = self._calculate_confidence(market_state, len(historical_returns))

        # 7. Create diagnostics for autonomous monitoring
        diagnostics = {
            "vol_impact": (market_state.realized_volatility - 0.50) / 0.50,
            "momentum_impact": market_state.price_momentum,
            "data_sufficiency": min(1.0, len(historical_returns) / 500),
            "parameter_stability": self._check_parameter_stability(delta_target, dte_target),
            "objective_improvement": self._calculate_improvement(objective_value),
        }

        result = OptimizationResult(
            delta_target=delta_target,
            dte_target=dte_target,
            kelly_fraction=kelly_fraction,
            expected_cagr=expected_cagr,
            expected_cvar=expected_cvar,
            objective_value=objective_value,
            confidence_score=confidence,
            diagnostics=diagnostics,
        )

        # Store for stability checking
        self.optimization_history.append(
            {"timestamp": datetime.now(), "result": result, "market_state": market_state}
        )

        # Log for autonomous monitoring
        logger.info(
            "Optimization complete",
            delta=f"{delta_target:.3f}",
            dte=dte_target,
            kelly=f"{kelly_fraction:.3f}",
            objective=f"{objective_value:.4f}",
            confidence=f"{confidence:.1%}",
        )

        return result

    def _calculate_dynamic_delta(self, state: MarketState) -> float:
        """
        Calculate delta as continuous function of market state.
        Higher volatility = lower delta (further OTM).
        """
        # Base delta at 50% volatility percentile
        base_delta = 0.25

        # Volatility adjustment (sigmoid for smooth transition)
        vol_adjustment = -0.15 * self._sigmoid((state.volatility_percentile - 0.5) * 4)

        # Momentum adjustment (sell into strength)
        momentum_adjustment = 0.05 * np.tanh(state.price_momentum * 10)

        # IV rank adjustment if available
        iv_adjustment = 0.0
        if state.iv_rank is not None:
            # Higher IV = can sell closer to money
            iv_adjustment = 0.05 * (state.iv_rank - 50) / 50

        # Earnings adjustment if available
        earnings_adjustment = 0.0
        if state.days_to_earnings is not None and state.days_to_earnings < 30:
            # Reduce delta near earnings
            earnings_adjustment = -0.05 * (1 - state.days_to_earnings / 30)

        delta = (
            base_delta + vol_adjustment + momentum_adjustment + iv_adjustment + earnings_adjustment
        )

        # Get bounds and apply
        delta_min, delta_max = self._get_config_bounds("delta")
        return np.clip(delta, delta_min, delta_max)

    def _calculate_dynamic_dte(self, state: MarketState) -> int:
        """
        Calculate DTE as continuous function of market state.
        Higher volatility = shorter DTE.
        """
        # Base DTE at 50% volatility percentile
        base_dte = 35.0

        # Volatility adjustment (exponential decay)
        vol_factor = np.exp(-2 * (state.volatility_percentile - 0.5))
        dte_continuous = base_dte * vol_factor

        # Volume adjustment (high volume = more liquid = can go shorter)
        if state.volume_ratio > 1.5:
            dte_continuous *= 0.9

        # Round to nearest weekly expiration
        dte_target = int(round(dte_continuous / 7) * 7)

        # Get bounds and apply
        dte_min, dte_max = self._get_config_bounds("dte")
        return int(np.clip(dte_target, dte_min, dte_max))

    def _calculate_dynamic_kelly(self, state: MarketState, historical_returns: np.ndarray) -> float:
        """
        Calculate Kelly fraction dynamically based on edge and uncertainty.
        """
        # Start with half-Kelly as specified
        base_kelly = self.BASE_KELLY

        # Calculate current Sharpe ratio
        if len(historical_returns) > 60:
            recent_returns = historical_returns[-60:]
            sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe = 0.5  # Conservative default

        # Sharpe adjustment (better risk-adjusted returns = can size up)
        sharpe_factor = np.clip(sharpe / 1.0, 0.5, 1.5)

        # Volatility adjustment (reduce in extreme volatility)
        vol_factor = 1.0
        if state.realized_volatility > 0.80:  # 80% annual vol
            vol_factor = 0.80 / state.realized_volatility

        # Confidence adjustment based on data
        data_factor = min(1.0, len(historical_returns) / 500)

        kelly = base_kelly * sharpe_factor * vol_factor * data_factor

        # Get bounds and apply
        kelly_min, kelly_max = self._get_config_bounds("kelly")
        return np.clip(kelly, kelly_min, kelly_max)

    def _estimate_cagr(self, delta: float, dte: int, kelly: float, state: MarketState) -> float:
        """Estimate expected CAGR for given parameters."""
        # Simplified model - would use ML in production

        # Base return from theta decay
        annual_trades = 365 / dte
        win_rate = 1 - delta  # Approximation

        # Premium as function of volatility and delta
        premium_pct = state.realized_volatility * np.sqrt(dte / 365) * delta * 0.4

        # Expected return per trade
        expected_return = win_rate * premium_pct - (1 - win_rate) * 0.10

        # Annual return with compounding
        cagr = (1 + expected_return) ** annual_trades - 1

        # Scale by Kelly fraction
        return cagr * kelly

    def _estimate_cvar(self, delta: float, kelly: float, historical_returns: np.ndarray) -> float:
        """Estimate CVaR (Conditional Value at Risk) at 95% level."""
        if len(historical_returns) < 100:
            # Fallback for insufficient data
            return -0.10 * kelly

        # Get worst 5% of returns
        var_95 = np.percentile(historical_returns, 5)
        cvar_95 = np.mean(historical_returns[historical_returns <= var_95])

        # Adjust for position sizing and leverage from selling options
        # Delta affects potential drawdown
        leverage_factor = 1 / (1 - delta)  # Approximation

        return cvar_95 * kelly * leverage_factor

    def _calculate_confidence(self, state: MarketState, data_points: int) -> float:
        """Calculate confidence score for autonomous monitoring."""
        # Data sufficiency
        data_conf = min(1.0, data_points / 750)

        # Parameter stability
        stability_conf = 1.0
        if len(self.optimization_history) > 5:
            recent_deltas = [h["result"].delta_target for h in self.optimization_history[-5:]]
            stability_conf = 1.0 - np.std(recent_deltas) / 0.1

        # Market normalcy (not in extreme conditions)
        normalcy_conf = 1.0 - abs(state.volatility_percentile - 0.5) * 2

        # Combine with weights
        confidence = 0.4 * data_conf + 0.3 * stability_conf + 0.3 * normalcy_conf

        return np.clip(confidence, 0.0, 1.0)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for smooth transitions."""
        return 1 / (1 + np.exp(-x))

    def _check_parameter_stability(self, delta: float, dte: int) -> float:
        """Check if parameters are stable vs recent history."""
        if len(self.optimization_history) < 1:
            return 1.0

        # Look at previous result if it exists
        if self.optimization_history and "result" in self.optimization_history[-1]:
            prev = self.optimization_history[-1]["result"]
            delta_change = abs(delta - prev.delta_target) / prev.delta_target
            dte_change = abs(dte - prev.dte_target) / prev.dte_target

            stability = 1.0 - (delta_change + dte_change) / 2
            return max(0.0, stability)
        else:
            return 1.0

    def _calculate_improvement(self, objective_value: float) -> float:
        """Calculate improvement vs previous optimization."""
        if not self.optimization_history:
            return 0.0

        if len(self.optimization_history) > 0 and "result" in self.optimization_history[-1]:
            prev_objective = self.optimization_history[-1]["result"].objective_value
            return (objective_value - prev_objective) / (abs(prev_objective) + 1e-6)
        else:
            return 0.0

    def validate_optimization(self, result: OptimizationResult) -> Dict[str, bool]:
        """Autonomous validation of optimization results."""
        checks = {
            "delta_in_range": 0.10 <= result.delta_target <= 0.40,
            "dte_in_range": 21 <= result.dte_target <= 49,
            "kelly_conservative": result.kelly_fraction <= self.BASE_KELLY,
            "objective_positive": result.objective_value > 0,
            "confidence_sufficient": result.confidence_score > 0.6,
            "cvar_reasonable": result.expected_cvar > -0.20,
        }

        # Log any failures for autonomous monitoring
        failures = [check for check, passed in checks.items() if not passed]
        if failures:
            logger.warning(
                f"Optimization validation failures: {failures}",
                extra={
                    "failures": failures,
                    "delta": result.delta_target,
                    "objective": result.objective_value,
                },
            )

        return checks