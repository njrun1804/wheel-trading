"""Enhanced wheel strategy implementation with self-validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from src.config.loader import get_config

# META INTEGRATION: Enable observation and evolution (SHARED SINGLETON)
def get_meta():
    """Get shared MetaPrime instance to prevent multiple spawns."""
    from ..meta import get_shared_meta
    meta = get_shared_meta()
    # Record wheel strategy usage
    meta.observe("wheel_strategy_meta_access", {"timestamp": datetime.now().isoformat()})
    return meta

from ..math import (
    CalculationResult,
    black_scholes_price_validated,
    calculate_all_greeks,
    probability_itm_validated,
)
from ..models import Position, PositionType
from ..risk import RiskAnalyzer
from ..storage.cache.general_cache import cached
from ..utils import RecoveryStrategy, StructuredLogger, get_logger, timed_operation, with_recovery
from ..utils.position_sizing import DynamicPositionSizer

logger = get_logger(__name__)
structured_logger = StructuredLogger(__name__)


class StrikeRecommendation(NamedTuple):
    """Strike selection recommendation with confidence."""

    strike: float
    delta: float
    probability_itm: float
    premium: float
    confidence: float
    reason: str


@dataclass
class WheelParameters:
    """Configuration for wheel strategy."""

    target_delta: float = field(default=None)
    target_dte: int = field(default=None)
    max_position_size: float = field(default=None)
    min_premium_yield: float = field(default=None)
    roll_dte_threshold: int = field(default=None)
    roll_delta_threshold: float = field(default=None)

    def __post_init__(self) -> None:
        """Initialize from config and validate parameters."""
        config = get_config()

        # Use config values as defaults if not explicitly set
        if self.target_delta is None:
            self.target_delta = config.strategy.delta_target
        if self.target_dte is None:
            self.target_dte = config.strategy.days_to_expiry_target
        if self.max_position_size is None:
            self.max_position_size = config.risk.circuit_breakers.max_position_pct
        if self.min_premium_yield is None:
            self.min_premium_yield = config.strategy.min_premium_yield
        if self.roll_dte_threshold is None:
            self.roll_dte_threshold = config.strategy.roll_triggers.dte_threshold
        if self.roll_delta_threshold is None:
            self.roll_delta_threshold = config.strategy.roll_triggers.delta_breach_threshold

        # Validate parameters
        if not 0 < self.target_delta < 1:
            raise ValueError(f"Target delta must be between 0 and 1, got {self.target_delta}")
        if self.target_dte < 1:
            raise ValueError(f"Target DTE must be positive, got {self.target_dte}")
        if not 0 < self.max_position_size <= 1:
            raise ValueError(
                f"Max position size must be between 0 and 1, got {self.max_position_size}"
            )


class WheelStrategy:
    """Enhanced wheel strategy with self-validation and risk awareness."""

    def __init__(
        self,
        parameters: Optional[WheelParameters] = None,
        risk_analyzer: Optional[RiskAnalyzer] = None,
    ):
        """Initialize wheel strategy."""
        self.params = parameters or WheelParameters()
        self.risk_analyzer = risk_analyzer or RiskAnalyzer()
        self.position_sizer = DynamicPositionSizer()

        logger.info(
            "Wheel strategy initialized",
            extra={
                "target_delta": self.params.target_delta,
                "target_dte": self.params.target_dte,
                "max_position_size": self.params.max_position_size,
            },
        )

    @timed_operation(threshold_ms=100.0)
    @with_recovery(strategy=RecoveryStrategy.FALLBACK)
    def find_optimal_put_strike(
        self,
        current_price: float,
        available_strikes: List[float],
        volatility: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
        portfolio_value: float = 100000.0,
    ) -> Optional[StrikeRecommendation]:
        """
        Find optimal put strike with validation and confidence scoring.

        Parameters
        ----------
        current_price : float
            Current stock price
        available_strikes : List[float]
            Available strike prices
        volatility : float
            Implied volatility
        days_to_expiry : int
            Days until expiration
        risk_free_rate : float
            Risk-free rate
        portfolio_value : float
            Total portfolio value for yield calculations

        Returns
        -------
        Optional[StrikeRecommendation]
            Recommendation with confidence score, or None if no suitable strike
        """
        # Use the vectorized implementation for better performance
        return self.find_optimal_put_strike_vectorized(
            current_price=current_price,
            available_strikes=available_strikes,
            volatility=volatility,
            days_to_expiry=days_to_expiry,
            risk_free_rate=risk_free_rate,
            target_delta=target_delta,
        )

    # === BEGIN vectorized_strike_selection ===
    @timed_operation(threshold_ms=20.0)  # Much faster!
    def find_optimal_put_strike_vectorized(
        self,
        current_price: float,
        available_strikes: List[float],
        volatility: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
        target_delta: Optional[float] = None,
    ) -> Optional[StrikeRecommendation]:
        """
        Vectorized version of find_optimal_put_strike for better performance.
        Processes all strikes at once using numpy arrays.
        """
        if not available_strikes:
            logger.warning("No strikes available")
            return None

        if target_delta is None:
            target_delta = self.params.target_delta

        time_to_expiry = days_to_expiry / 365.0

        # Convert to numpy array for vectorized operations
        strikes = np.array(available_strikes)

        # Filter strikes by moneyness
        config = get_config()
        min_moneyness = config.strategy.strike_range.min_moneyness
        valid_mask = strikes >= current_price * min_moneyness
        strikes = strikes[valid_mask]

        if len(strikes) == 0:
            logger.warning("No strikes within valid moneyness range")
            return None

        # Vectorized Greeks calculation - all strikes at once!
        greeks, greek_confidence = calculate_all_greeks(
            S=current_price,
            K=strikes,  # Array of strikes
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
            option_type="put",
        )

        # Vectorized price calculation
        price_result = black_scholes_price_validated(
            S=current_price,
            K=strikes,  # Array of strikes
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
            option_type="put",
        )

        # Vectorized probability ITM
        prob_result = probability_itm_validated(
            S=current_price,
            K=strikes,  # Array of strikes
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
            option_type="put",
        )

        # Extract arrays
        deltas = greeks["delta"]
        premiums = price_result.value
        prob_itms = prob_result.value

        # Calculate scores for all strikes at once
        # Using absolute error for delta distance (more intuitive than squared error)
        # CODEX-PATTERN: Lower scores are better (minimize delta distance, maximize premium)
        delta_scores = np.abs(deltas - target_delta)
        premium_ratios = premiums / strikes

        # Combined scoring
        # CODEX-NOTE: The 0.1 weight prioritizes delta match over premium
        # CODEX-EXAMPLE: For deeper analysis, see examples/architecture_examples/strike_scoring_tuning.py
        scores = delta_scores + 0.1 * (1 - premium_ratios)

        # Apply confidence filtering
        confidence_mask = (
            (greek_confidence > 0.5)
            & (price_result.confidence > 0.5)
            & (prob_result.confidence > 0.5)
        )

        valid_indices = np.where(confidence_mask)[0]
        if len(valid_indices) == 0:
            logger.warning("No strikes with sufficient confidence")
            return None

        # Find best among valid strikes
        valid_scores = scores[valid_indices]
        best_idx = valid_indices[np.argmin(valid_scores)]

        # Extract best strike data
        best_strike = strikes[best_idx]
        best_delta = deltas[best_idx]
        best_premium = premiums[best_idx]
        best_prob_itm = prob_itms[best_idx]

        # Calculate overall confidence
        confidence = np.mean(
            [
                greek_confidence,
                price_result.confidence,
                prob_result.confidence,
                0.9 if abs(best_delta - target_delta) < 0.05 else 0.7,
            ]
        )

        # Determine reason
        reason = f"Delta {best_delta:.2f} closest to target {target_delta:.2f}"
        if best_premium / best_strike >= 0.02:
            reason += " with attractive premium"

        logger.info(
            "Optimal put strike selected (vectorized)",
            extra={
                "strike": best_strike,
                "delta": best_delta,
                "premium": best_premium,
                "prob_itm": best_prob_itm,
                "confidence": confidence,
                "strikes_evaluated": len(strikes),
            },
        )

        return StrikeRecommendation(
            strike=float(best_strike),
            delta=float(best_delta),
            probability_itm=float(best_prob_itm),
            premium=float(best_premium),
            confidence=float(confidence),
            reason=reason,
        )

    # === END vectorized_strike_selection ===

    @timed_operation(threshold_ms=20.0)
    def find_optimal_call_strike_vectorized(
        self,
        current_price: float,
        cost_basis: float,
        available_strikes: List[float],
        volatility: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
        target_delta: Optional[float] = None,
    ) -> Optional[StrikeRecommendation]:
        """Vectorized version of find_optimal_call_strike."""

        if not available_strikes:
            logger.warning("No strikes available")
            return None

        if target_delta is None:
            target_delta = self.params.target_delta

        time_to_expiry = days_to_expiry / 365.0

        strikes = np.array(available_strikes, dtype=float)

        config = get_config()
        max_moneyness = config.strategy.strike_range.max_moneyness

        valid_mask = (strikes >= cost_basis) & (strikes <= current_price * max_moneyness)
        strikes = strikes[valid_mask]

        if len(strikes) == 0:
            logger.warning("No valid call strikes within range")
            return None

        greeks, greek_confidence = calculate_all_greeks(
            S=current_price,
            K=strikes,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
            option_type="call",
        )

        price_result = black_scholes_price_validated(
            S=current_price,
            K=strikes,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
            option_type="call",
        )

        prob_result = probability_itm_validated(
            S=current_price,
            K=strikes,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
            option_type="call",
        )

        deltas = greeks["delta"]
        premiums = price_result.value
        prob_itms = prob_result.value

        safety_margin = (strikes - cost_basis) / cost_basis
        delta_scores = np.abs(deltas - target_delta)
        delta_scores -= 0.05 * (safety_margin > 0.05)

        scores = delta_scores

        confidence_mask = (
            (greek_confidence > 0.5)
            & (price_result.confidence > 0.5)
            & (prob_result.confidence > 0.5)
        )

        valid_indices = np.where(confidence_mask)[0]
        if len(valid_indices) == 0:
            logger.warning("No call strikes with sufficient confidence")
            return None

        best_idx = valid_indices[np.argmin(scores[valid_indices])]

        best_strike = strikes[best_idx]
        best_delta = deltas[best_idx]
        best_premium = premiums[best_idx]
        best_prob_itm = prob_itms[best_idx]
        best_safety_margin = safety_margin[best_idx]

        confidence = np.mean(
            [
                greek_confidence,
                price_result.confidence,
                prob_result.confidence,
                0.9 if abs(best_delta - target_delta) < 0.05 else 0.7,
            ]
        )

        reason = f"Delta {best_delta:.2f} with {best_safety_margin:.1%} above cost basis"

        logger.info(
            "Optimal call strike selected (vectorized)",
            extra={
                "strike": best_strike,
                "delta": best_delta,
                "premium": best_premium,
                "prob_itm": best_prob_itm,
                "confidence": confidence,
                "strikes_evaluated": len(strikes),
            },
        )

        return StrikeRecommendation(
            strike=float(best_strike),
            delta=float(best_delta),
            probability_itm=float(best_prob_itm),
            premium=float(best_premium),
            confidence=float(confidence),
            reason=reason,
        )

    # === END vectorized_call_selection ===

    @timed_operation(threshold_ms=100.0)
    @with_recovery(strategy=RecoveryStrategy.FALLBACK)
    def find_optimal_call_strike(
        self,
        current_price: float,
        cost_basis: float,
        available_strikes: List[float],
        volatility: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
    ) -> Optional[StrikeRecommendation]:
        """Find optimal call strike with vectorized implementation."""

        return self.find_optimal_call_strike_vectorized(
            current_price=current_price,
            cost_basis=cost_basis,
            available_strikes=available_strikes,
            volatility=volatility,
            days_to_expiry=days_to_expiry,
            risk_free_rate=risk_free_rate,
            target_delta=self.params.target_delta,
        )

    @timed_operation(threshold_ms=10.0)
    @cached(ttl=timedelta(minutes=5))
    def calculate_position_size(
        self,
        strike_price: float,
        portfolio_value: float,
        current_margin_used: float = 0.0,
    ) -> Tuple[int, float]:
        """
        Calculate position size with Kelly criterion and risk limits.

        Parameters
        ----------
        strike_price : float
            Strike price of the option
        portfolio_value : float
            Total portfolio value
        current_margin_used : float
            Currently used margin

        Returns
        -------
        Tuple[int, float]
            (Number of contracts, confidence in sizing)
        """
        # Use DynamicPositionSizer for consistent position sizing across the system
        result = self.position_sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            option_price=1.0,  # Placeholder - actual premium will be calculated later
            strike_price=strike_price,
            buying_power=portfolio_value - current_margin_used,
            kelly_fraction=get_config().risk.kelly_fraction,
        )

        return result.contracts, result.confidence

    @timed_operation(threshold_ms=50.0)
    def should_roll_position(
        self,
        position: Position,
        current_price: float,
        days_to_expiry: int,
        current_greeks: dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Determine if position should be rolled with reason.

        Parameters
        ----------
        position : Position
            Current option position
        current_price : float
            Current stock price
        days_to_expiry : int
            Days until expiration
        current_greeks : dict[str, float]
            Current Greeks of the position

        Returns
        -------
        Tuple[bool, str]
            (Should roll, reason for decision)
        """
        reasons = []

        # Check expiry
        if days_to_expiry <= self.params.roll_dte_threshold:
            reasons.append(f"Approaching expiry ({days_to_expiry} days)")

        # Check delta
        current_delta = abs(current_greeks.get("delta", 0))
        if current_delta > self.params.roll_delta_threshold:
            reasons.append(f"Deep ITM (delta={current_delta:.2f})")

        # Check moneyness
        config = get_config()
        profit_threshold = config.strategy.roll_triggers.profit_threshold_2  # 95% threshold

        if position.strike:
            if position.position_type == PositionType.PUT:
                if current_price < position.strike * profit_threshold:
                    reasons.append("Put moving ITM")
            elif position.position_type == PositionType.CALL:
                if current_price > position.strike * profit_threshold:
                    reasons.append("Call moving ITM")

        # Check theta decay
        theta = current_greeks.get("theta", 0)
        if theta < -1.0:  # Losing more than $1/day
            reasons.append(f"High theta decay (${theta:.2f}/day)")

        should_roll = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "Position healthy"

        if should_roll:
            logger.info(
                "Position roll recommended",
                extra={
                    "position": str(position),
                    "days_to_expiry": days_to_expiry,
                    "current_delta": current_delta,
                    "reasons": reasons,
                },
            )

        return should_roll, reason

    def analyze_position_risk(
        self,
        position: Position,
        current_price: float,
        volatility: float,
        portfolio_value: float,
    ) -> dict[str, float]:
        """
        Analyze risk metrics for a wheel position.

        Returns
        -------
        dict[str, float]
            Risk metrics including VaR, margin usage, etc.
        """
        if not position.strike or not position.expiration:
            return {"error": "Invalid position data"}

        # Calculate time to expiry
        days_to_expiry = (position.expiration - datetime.now(timezone.utc).date()).days
        time_to_expiry = max(0, days_to_expiry / 365.0)

        # Calculate current Greeks
        greeks, confidence = calculate_all_greeks(
            S=current_price,
            K=position.strike,
            T=time_to_expiry,
            r=0.05,
            sigma=volatility,
            option_type="put" if position.position_type == PositionType.PUT else "call",
        )

        # Position value
        contracts = abs(position.quantity)
        notional = position.strike * 100 * contracts

        # Risk metrics
        metrics = {
            "notional_exposure": notional,
            "portfolio_percentage": notional / portfolio_value,
            "delta_dollars": greeks.get("delta", 0) * 100 * contracts * current_price,
            "gamma_dollars": greeks.get("gamma", 0) * 100 * contracts * current_price,
            "theta_daily": greeks.get("theta", 0) * contracts,
            "vega_dollars": greeks.get("vega", 0) * contracts,
            "margin_requirement": notional * 0.20,  # Simplified
            "confidence": confidence,
        }

        # Calculate position-specific VaR
        # Simplified: use delta approximation
        position_var = abs(metrics["delta_dollars"]) * volatility * 1.65  # 95% VaR
        metrics["position_var_95"] = position_var
        metrics["position_var_pct"] = position_var / portfolio_value

        logger.debug(
            "Position risk analyzed",
            extra={
                "position": str(position),
                "metrics": metrics,
            },
        )

        return metrics
# Test comment to trigger meta observation
