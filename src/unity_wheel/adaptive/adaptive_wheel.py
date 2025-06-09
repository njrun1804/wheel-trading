"""
Adaptive wheel strategy that integrates the Unity adaptive system.
Replaces static parameters with dynamic ones based on market conditions.
"""

from datetime import datetime
from typing import Dict, List, Optional

from ..adaptive import UnityConditions, UnityAdaptiveSystem, OutcomeTracker
from ..data.market_data import MarketDataFetcher, UnityEarningsCalendar
from ..models import Greeks, Position
from ..strategy.wheel import WheelStrategy, WheelParameters, StrikeRecommendation
from ..utils import get_logger

logger = get_logger(__name__)


class AdaptiveWheelStrategy:
    """
    Wheel strategy with adaptive position sizing and parameters.
    Combines the base wheel strategy with Unity-specific adaptations.
    """

    def __init__(self, portfolio_value: float, track_outcomes: bool = True):
        """
        Initialize adaptive wheel strategy.

        Args:
            portfolio_value: Total portfolio value
            track_outcomes: Whether to track recommendation outcomes
        """
        self.portfolio_value = portfolio_value
        self.adaptive_system = UnityAdaptiveSystem(portfolio_value)
        self.data_fetcher = MarketDataFetcher()
        self.outcome_tracker = OutcomeTracker() if track_outcomes else None

        # Base wheel strategy (will use adaptive parameters)
        self.base_wheel = WheelStrategy()

    def get_current_conditions(
        self, unity_price: float, portfolio_drawdown: Optional[float] = None
    ) -> Optional[UnityConditions]:
        """
        Get current Unity market conditions.

        Args:
            unity_price: Current Unity stock price
            portfolio_drawdown: Current drawdown if known

        Returns:
            UnityConditions if data is available, None otherwise
        """
        # Fetch market data
        volatility = self.data_fetcher.get_unity_volatility()
        if volatility is None:
            logger.error("Cannot get Unity volatility - no market data available")
            return None

        # IV rank is optional, default to 50 if unavailable
        iv_rank = self.data_fetcher.get_unity_iv_rank()
        if iv_rank is None:
            logger.warning("IV rank unavailable, using default 50")
            iv_rank = 50.0

        # Earnings date is optional
        days_to_earnings = UnityEarningsCalendar.days_to_next_earnings()

        # Calculate drawdown if not provided
        if portfolio_drawdown is None:
            # Simplified - in production would track actual P&L
            portfolio_drawdown = 0.0

        return UnityConditions(
            unity_price=unity_price,
            unity_volatility=volatility,
            unity_iv_rank=iv_rank,
            portfolio_drawdown=portfolio_drawdown,
            days_to_earnings=days_to_earnings,
        )

    def get_recommendation(
        self,
        unity_price: float,
        available_strikes: List[float],
        available_expirations: List[int],  # Days to expiry
        portfolio_drawdown: Optional[float] = None,
        risk_free_rate: float = 0.05,
    ) -> Dict:
        """
        Get adaptive wheel recommendation.

        Returns dict with:
        - should_trade: bool
        - skip_reason: str (if not trading)
        - position_size: float
        - recommended_strike: float
        - target_dte: int
        - expected_premium: float
        - confidence: float
        - adaptive_factors: dict
        """
        # Get current conditions
        conditions = self.get_current_conditions(unity_price, portfolio_drawdown)
        if conditions is None:
            # Return error result if no market data
            return {
                "recommendation_id": None,
                "timestamp": datetime.now(),
                "should_trade": False,
                "skip_reason": "Market data unavailable",
                "error": "Cannot fetch Unity market data - check connectivity",
            }

        # Get adaptive recommendation
        recommendation = self.adaptive_system.get_recommendation(conditions)

        # Track recommendation if enabled
        if self.outcome_tracker:
            rec_id = self.outcome_tracker.track_recommendation(conditions, recommendation)
        else:
            rec_id = None

        # Build result
        result = {
            "recommendation_id": rec_id,
            "timestamp": datetime.now(),
            "conditions": {
                "unity_price": unity_price,
                "volatility": conditions.unity_volatility,
                "iv_rank": conditions.unity_iv_rank,
                "days_to_earnings": conditions.days_to_earnings,
                "portfolio_drawdown": conditions.portfolio_drawdown,
                "regime": conditions.regime.value,
            },
            "should_trade": recommendation.should_trade,
            "skip_reason": recommendation.skip_reason,
        }

        if not recommendation.should_trade:
            logger.info(f"Skipping trade: {recommendation.skip_reason}")
            return result

        # Find optimal strike using adaptive parameters
        # Create temporary wheel parameters
        adaptive_params = WheelParameters(
            target_delta=recommendation.put_delta,
            target_dte=recommendation.target_dte,
            max_position_size=recommendation.position_pct,
            roll_delta_threshold=0.70 if conditions.is_high_vol else 0.60,
            roll_dte_threshold=7,
        )

        # Use base wheel strategy with adaptive parameters
        temp_wheel = WheelStrategy(parameters=adaptive_params)

        # Find best expiration date
        best_dte = min(available_expirations, key=lambda x: abs(x - recommendation.target_dte))

        # Find optimal strike
        strike_rec = temp_wheel.find_optimal_put_strike(
            current_price=unity_price,
            available_strikes=available_strikes,
            volatility=conditions.unity_volatility,
            days_to_expiry=best_dte,
            risk_free_rate=risk_free_rate,
            portfolio_value=self.portfolio_value,
        )

        if strike_rec:
            result.update(
                {
                    "position_size": recommendation.position_size,
                    "position_pct": recommendation.position_pct,
                    "recommended_strike": strike_rec.strike,
                    "strike_delta": strike_rec.delta,
                    "target_dte": best_dte,
                    "expected_premium": strike_rec.premium,
                    "probability_itm": strike_rec.probability_itm,
                    "confidence": strike_rec.confidence,
                    "adaptive_factors": recommendation.size_factors,
                    "roll_profit_target": recommendation.roll_profit_target,
                }
            )

            # Log summary
            logger.info(
                f"Adaptive recommendation: "
                f"${recommendation.position_size:,.0f} position, "
                f"{strike_rec.strike} strike, "
                f"{best_dte} DTE, "
                f"${strike_rec.premium:.2f} premium"
            )
        else:
            result["error"] = "Could not find suitable strike"

        return result

    def should_roll_position(
        self, position: Position, current_price: float, current_volatility: float
    ) -> Dict:
        """
        Determine if position should be rolled using adaptive rules.
        """
        # Get current conditions
        conditions = self.get_current_conditions(current_price)
        if conditions is None:
            # Default conservative behavior if no data
            return {
                "should_roll": False,
                "roll_reason": "Market data unavailable",
                "error": "Cannot fetch Unity market data",
            }

        # Calculate current P&L
        days_held = (datetime.now() - position.entry_date).days
        dte_remaining = position.days_to_expiry - days_held

        # Get Greeks for current assessment
        greeks = Greeks(
            delta=-position.delta,  # Short put has negative delta
            gamma=position.gamma,
            vega=position.vega,
            theta=position.theta,
            rho=0.0,
        )

        # Calculate profit percentage
        current_value = position.mark_price
        entry_value = position.entry_price
        profit_pct = (entry_value - current_value) / entry_value

        # Adaptive roll rules
        roll_profit_target = 0.25 if conditions.is_high_vol else 0.50

        # Roll conditions
        should_roll = False
        roll_reason = None

        # 1. Profit target reached
        if profit_pct >= roll_profit_target:
            should_roll = True
            roll_reason = f"Profit target reached ({profit_pct:.1%})"

        # 2. Delta breach (risk increasing)
        elif abs(position.delta) > 0.60:
            should_roll = True
            roll_reason = f"Delta breach (delta={abs(position.delta):.2f})"

        # 3. Time decay (too close to expiry)
        elif dte_remaining <= 7:
            should_roll = True
            roll_reason = f"Near expiration ({dte_remaining} days)"

        # 4. Earnings approaching
        elif conditions.days_to_earnings and conditions.days_to_earnings <= 10:
            should_roll = True
            roll_reason = f"Earnings approaching ({conditions.days_to_earnings} days)"

        return {
            "should_roll": should_roll,
            "roll_reason": roll_reason,
            "current_profit_pct": profit_pct,
            "dte_remaining": dte_remaining,
            "current_delta": abs(position.delta),
            "market_conditions": conditions.regime.value,
        }

    def record_outcome(self, recommendation_id: str, actual_pnl: float, was_assigned: bool):
        """Record actual outcome for a recommendation."""
        if self.outcome_tracker:
            self.outcome_tracker.record_outcome(recommendation_id, actual_pnl, was_assigned)

    def get_performance_summary(self) -> Dict:
        """Get summary of adaptive system performance."""
        if self.outcome_tracker:
            return self.outcome_tracker.get_performance_summary()
        return {"message": "Outcome tracking not enabled"}


def create_adaptive_wheel_strategy(portfolio_value: float) -> AdaptiveWheelStrategy:
    """
    Factory function to create properly configured adaptive wheel strategy.
    """
    return AdaptiveWheelStrategy(portfolio_value=portfolio_value, track_outcomes=True)
