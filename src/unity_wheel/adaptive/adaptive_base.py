"""
Simplified adaptive system for Unity wheel strategy.
Focuses only on what affects P&L: volatility, drawdown, earnings, IV rank.
"""

import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from config.loader import get_config

from .utils import get_logger


class MarketRegime(str, Enum):
    """Market regime classification."""

    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    STRESSED = "STRESSED"


logger = get_logger(__name__)


@dataclass
class UnityConditions:
    """Simplified market conditions - only what matters for Unity wheel."""

    # Essential metrics
    unity_price: float
    unity_volatility: float  # Realized 20-day vol
    unity_iv_rank: float  # 0-100 percentile

    # Risk metrics
    portfolio_drawdown: float  # Current drawdown from peak

    # Unity-specific
    days_to_earnings: Optional[int] = None

    # Derived properties
    @property
    def is_high_vol(self) -> bool:
        """Unity vol >60% is high."""
        return self.unity_volatility > 0.60

    @property
    def is_extreme_vol(self) -> bool:
        """Unity vol >80% is extreme."""
        return self.unity_volatility > 0.80

    @property
    def near_earnings(self) -> bool:
        """Within 7 days of earnings."""
        return self.days_to_earnings is not None and self.days_to_earnings <= 7

    @property
    def regime(self) -> MarketRegime:
        """Simple regime classification based on volatility."""
        if self.unity_volatility > 0.80:
            return MarketRegime.STRESSED
        elif self.unity_volatility > 0.60:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.NORMAL


@dataclass
class WheelRecommendation:
    """Recommendation with all parameters and reasoning."""

    # Decision
    should_trade: bool
    skip_reason: Optional[str] = None

    # Position sizing
    position_size: float = 0.0
    position_pct: float = 0.0
    size_factors: Dict[str, float] = field(default_factory=dict)

    # Wheel parameters
    put_delta: float = 0.30
    target_dte: int = 35
    roll_profit_target: float = 0.50

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    conditions: Optional[UnityConditions] = None


class UnityAdaptiveSystem:
    """
    Simplified adaptive system for Unity wheel trading.
    Rules-based, transparent, and testable.
    """

    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.config = get_config()
        self.base_position_pct = self.config.adaptive.base_position_pct

        # Regime persistence tracking
        self._regime_history: List[Tuple[datetime, MarketRegime]] = []
        self._regime_persistence_days = self.config.adaptive.regime_persistence_days

    def get_recommendation(self, conditions: UnityConditions) -> WheelRecommendation:
        """Get complete wheel recommendation based on current conditions."""

        # Check if we should skip trading
        should_trade, skip_reason = self._should_trade(conditions)
        if not should_trade:
            return WheelRecommendation(
                should_trade=False, skip_reason=skip_reason, conditions=conditions
            )

        # Calculate position size
        position_size, size_factors, size_confidence = self._calculate_position_size(conditions)

        # Get wheel parameters
        params = self._get_wheel_parameters(conditions)

        return WheelRecommendation(
            should_trade=True,
            position_size=position_size,
            position_pct=position_size / self.portfolio_value,
            size_factors=size_factors,
            put_delta=params["put_delta"],
            target_dte=params["target_dte"],
            roll_profit_target=params["roll_profit_target"],
            conditions=conditions,
        )

    def _should_trade(self, conditions: UnityConditions) -> Tuple[bool, Optional[str]]:
        """Determine if we should trade at all."""

        stop_conditions = self.config.adaptive.stop_conditions

        # Hard stops
        if conditions.portfolio_drawdown <= -stop_conditions.max_drawdown:
            return False, f"Maximum drawdown reached ({-stop_conditions.max_drawdown:.0%})"

        if conditions.near_earnings:
            return False, f"Too close to earnings ({conditions.days_to_earnings} days)"

        if conditions.unity_volatility > stop_conditions.max_volatility:
            return False, f"Extreme volatility ({conditions.unity_volatility:.0%})"

        return True, None

    def _calculate_position_size(
        self, conditions: UnityConditions
    ) -> Tuple[float, Dict[str, float], float]:
        """Calculate position size with simple, transparent rules.

        Returns:
            Tuple[float, Dict[str, float], float]: (position_size, factors, confidence)
        """

        base_size = self.portfolio_value * self.base_position_pct
        factors = {}

        # 1. Volatility adjustment (most important)
        vol_thresholds = self.config.adaptive.volatility_thresholds
        vol_factors = self.config.adaptive.volatility_factors

        if conditions.unity_volatility < vol_thresholds.low:
            factors["volatility"] = vol_factors.low
        elif conditions.unity_volatility < vol_thresholds.normal:
            factors["volatility"] = vol_factors.normal
        elif conditions.unity_volatility < vol_thresholds.high:
            factors["volatility"] = vol_factors.high
        else:
            factors["volatility"] = vol_factors.extreme

        # 2. Drawdown adjustment (capital preservation)
        # Linear: 0% dd = 1.0x, max_drawdown = 0.0x
        max_dd = self.config.adaptive.stop_conditions.max_drawdown
        factors["drawdown"] = max(0, 1 + conditions.portfolio_drawdown / max_dd)

        # 3. IV rank adjustment (edge indicator)
        if conditions.unity_iv_rank > 80:
            factors["iv_rank"] = 1.2  # Great premiums
        elif conditions.unity_iv_rank > 50:
            factors["iv_rank"] = 1.0  # Normal
        else:
            factors["iv_rank"] = 0.8  # Poor premiums

        # 4. Earnings proximity (if not skipping)
        if conditions.days_to_earnings and conditions.days_to_earnings <= 14:
            factors["earnings"] = 0.7  # Reduce near earnings
        else:
            factors["earnings"] = 1.0

        # Calculate final size
        total_factor = np.prod(list(factors.values()))
        position_size = base_size * total_factor

        # Apply maximum limits
        max_size = self.portfolio_value * self.config.adaptive.max_position_pct
        position_size = min(position_size, max_size)

        # Calculate confidence based on data quality and factors
        confidence = 0.95  # Start with high confidence

        # Reduce confidence for extreme conditions
        if conditions.unity_volatility > vol_thresholds.high:
            confidence *= 0.85  # High volatility reduces confidence

        if conditions.days_to_earnings < 14:
            confidence *= 0.90  # Near earnings reduces confidence

        if conditions.portfolio_drawdown < -0.10:
            confidence *= 0.85  # Drawdown reduces confidence

        # Reduce confidence if position was significantly adjusted
        if total_factor < 0.7:
            confidence *= 0.90  # Large reduction indicates uncertainty

        return position_size, factors, confidence

    def _get_persistent_regime(self, conditions: UnityConditions) -> MarketRegime:
        """
        Get regime with persistence to prevent daily flip-flopping.
        Requires consistent readings for regime_persistence_days to change.
        """
        current_regime = conditions.regime

        # Add to history
        self._regime_history.append((datetime.now(), current_regime))

        # Keep only recent history (last 10 days)
        cutoff = datetime.now() - timedelta(days=10)
        self._regime_history = [(dt, r) for dt, r in self._regime_history if dt > cutoff]

        # If not enough history, return current
        if len(self._regime_history) < self._regime_persistence_days:
            return current_regime

        # Check if last N days all agree on a new regime
        recent_regimes = [r for dt, r in self._regime_history[-self._regime_persistence_days :]]
        if all(r == current_regime for r in recent_regimes):
            return current_regime

        # Otherwise return the previous stable regime
        # Find the most common regime in history
        from collections import Counter

        regime_counts = Counter(r for dt, r in self._regime_history)
        stable_regime = regime_counts.most_common(1)[0][0]

        logger.info(
            f"Regime persistence: current={current_regime.value}, "
            f"stable={stable_regime.value}, "
            f"history={[r.value for dt, r in self._regime_history[-5:]]}"
        )

        return stable_regime

    def _get_wheel_parameters(self, conditions: UnityConditions) -> Dict[str, float]:
        """Get wheel parameters based on conditions."""

        # Get persistent regime to avoid flip-flopping
        regime = self._get_persistent_regime(conditions)

        # Get regime-specific parameters from config
        regime_key = regime.value.lower()

        # Handle NORMAL vs normal case
        if regime_key == "normal":
            regime_config = self.config.adaptive.regime_params.get(
                "normal", self.config.adaptive.regime_params.get("NORMAL")
            )
        else:
            regime_config = self.config.adaptive.regime_params.get(regime_key)

        # If we can't find the regime config, use normal as default
        if not regime_config:
            regime_config = self.config.adaptive.regime_params["normal"]

        params = {
            "put_delta": regime_config.put_delta,
            "target_dte": regime_config.target_dte,
            "roll_profit_target": regime_config.roll_profit_target,
        }

        # Additional adjustments for low volatility
        if (
            regime == MarketRegime.NORMAL
            and conditions.unity_volatility < self.config.adaptive.volatility_thresholds.low
        ):
            # Use low_volatility regime params if available
            if "low_volatility" in self.config.adaptive.regime_params:
                low_vol_config = self.config.adaptive.regime_params["low_volatility"]
                params["put_delta"] = low_vol_config.put_delta
                params["target_dte"] = low_vol_config.target_dte
                params["roll_profit_target"] = low_vol_config.roll_profit_target

        # Earnings adjustment (if trading)
        if conditions.days_to_earnings and conditions.days_to_earnings <= 45:
            # Make sure we expire before earnings
            max_dte = max(7, conditions.days_to_earnings - 7)
            params["target_dte"] = min(params["target_dte"], max_dte)

        return params


@dataclass
class WheelOutcome:
    """Track outcome of wheel recommendations for learning."""

    recommendation_id: str
    recommendation_date: datetime
    conditions: UnityConditions
    recommendation: WheelRecommendation

    # Actual outcomes (filled in later)
    actual_pnl: Optional[float] = None
    actual_return_pct: Optional[float] = None
    was_assigned: Optional[bool] = None
    outcome_date: Optional[datetime] = None

    # Analysis
    pnl_vs_expected: Optional[float] = None
    helped_avoid_loss: Optional[bool] = None


class OutcomeTracker:
    """Simple outcome tracking for continuous improvement."""

    def __init__(self, db_path: Optional[str] = None):
        config = get_config()
        if db_path is None:
            db_path = config.adaptive.outcome_tracking.database_path
        # Expand user path
        self.db_path = os.path.expanduser(db_path)
        self.outcomes: List[WheelOutcome] = []

    def track_recommendation(
        self, conditions: UnityConditions, recommendation: WheelRecommendation
    ) -> str:
        """Record a recommendation for later outcome tracking."""

        recommendation_id = f"REC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        outcome = WheelOutcome(
            recommendation_id=recommendation_id,
            recommendation_date=datetime.now(),
            conditions=conditions,
            recommendation=recommendation,
        )

        self.outcomes.append(outcome)
        logger.info(
            "Tracked recommendation",
            id=recommendation_id,
            trade=recommendation.should_trade,
            size=recommendation.position_size,
        )

        return recommendation_id

    def record_outcome(self, recommendation_id: str, pnl: float, was_assigned: bool):
        """Record actual outcome of a recommendation."""

        for outcome in self.outcomes:
            if outcome.recommendation_id == recommendation_id:
                outcome.actual_pnl = pnl
                outcome.actual_return_pct = pnl / outcome.recommendation.position_size
                outcome.was_assigned = was_assigned
                outcome.outcome_date = datetime.now()

                # Analyze if skipping helped
                if not outcome.recommendation.should_trade:
                    # We skipped - would we have lost money?
                    outcome.helped_avoid_loss = pnl < 0 if pnl else None

                logger.info(
                    "Recorded outcome", id=recommendation_id, pnl=pnl, assigned=was_assigned
                )
                break

    def get_performance_summary(self) -> Dict:
        """Analyze performance of adaptive rules."""

        completed = [o for o in self.outcomes if o.actual_pnl is not None]

        if not completed:
            return {"message": "No completed trades yet"}

        # Overall performance
        total_pnl = sum(o.actual_pnl for o in completed)
        win_rate = sum(1 for o in completed if o.actual_pnl > 0) / len(completed)

        # Rule effectiveness
        skipped = [o for o in completed if not o.recommendation.should_trade]
        avoided_losses = sum(1 for o in skipped if o.helped_avoid_loss)

        # Volatility adaptation
        high_vol_trades = [
            o for o in completed if o.recommendation.should_trade and o.conditions.is_high_vol
        ]
        high_vol_win_rate = (
            (sum(1 for o in high_vol_trades if o.actual_pnl > 0) / len(high_vol_trades))
            if high_vol_trades
            else 0
        )

        return {
            "total_outcomes": len(completed),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "trades_skipped": len(skipped),
            "losses_avoided": avoided_losses,
            "high_vol_win_rate": high_vol_win_rate,
        }


def calculate_assignment_probability(
    spot_price: float, strike_price: float, dte: int, volatility: float
) -> float:
    """Simple probability of finishing ITM for a put."""

    # Log-normal distribution parameters
    time_to_expiry = dte / 365.0
    vol_sqrt_t = volatility * np.sqrt(time_to_expiry)

    # Calculate d2 (probability of finishing ITM)
    d2 = (np.log(spot_price / strike_price) - 0.5 * volatility**2 * time_to_expiry) / vol_sqrt_t

    # Probability that price < strike at expiry
    prob_itm = norm.cdf(-d2)

    return prob_itm
