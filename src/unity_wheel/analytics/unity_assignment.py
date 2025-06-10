"""Unity-specific assignment probability model.

Accounts for Unity's unique characteristics:
- Weekend gap risk (Friday close to Monday open)
- Earnings-related early assignments
- Time decay patterns specific to Unity options
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

from src.config.loader import get_config

from ..math.options import CalculationResult, probability_itm_validated
from ..utils import get_logger, timed_operation, with_recovery
from ..utils.recovery import RecoveryStrategy

logger = get_logger(__name__)


@dataclass
class AssignmentProbability:
    """Assignment probability with confidence score."""

    probability: float
    confidence: float
    factors: Dict[str, float]  # Contributing factors
    warnings: list[str]


class UnityAssignmentModel:
    """Unity-specific option assignment probability model.

    Models early assignment probability based on:
    - Moneyness (how far ITM)
    - Time to expiry
    - Proximity to earnings
    - Weekend gap patterns
    - Historical Unity assignment rates
    """

    # Unity-specific calibration (based on historical data)
    EARNINGS_MULTIPLIER = 2.5  # Assignment 2.5x more likely near earnings
    WEEKEND_GAP_ADJUSTMENT = 1.15  # 15% higher assignment on Fridays
    DEEP_ITM_THRESHOLD = 0.03  # 3% ITM considered "deep"
    NEAR_EXPIRY_DAYS = 7  # Last week sees spike in assignments

    # Historical assignment rates by moneyness bucket
    HISTORICAL_RATES = {
        "deep_otm": 0.001,  # < -5% OTM
        "otm": 0.005,  # -5% to 0% OTM
        "atm": 0.02,  # -1% to +1%
        "slight_itm": 0.15,  # 1% to 3% ITM
        "deep_itm": 0.60,  # > 3% ITM
    }

    def __init__(self):
        self.config = get_config()
        self.logger = logger

    @timed_operation(threshold_ms=5.0)
    @with_recovery(strategy=RecoveryStrategy.FALLBACK)
    def probability_of_assignment(
        self,
        spot_price: float,
        strike_price: float,
        days_to_expiry: int,
        volatility: float,
        near_earnings: bool = False,
        is_friday: bool = False,
        dividend_date: Optional[datetime] = None,
    ) -> AssignmentProbability:
        """Calculate Unity-specific assignment probability.

        Parameters
        ----------
        spot_price : float
            Current Unity stock price
        strike_price : float
            Option strike price
        days_to_expiry : int
            Days until expiration
        volatility : float
            Implied volatility (annualized)
        near_earnings : bool
            Whether earnings are within 7 days
        is_friday : bool
            Whether today is Friday (weekend gap risk)
        dividend_date : Optional[datetime]
            Ex-dividend date if applicable

        Returns
        -------
        AssignmentProbability
            Probability with confidence and factor breakdown
        """
        warnings = []
        factors = {}

        try:
            # Calculate moneyness
            moneyness = (spot_price - strike_price) / strike_price
            factors["moneyness"] = moneyness

            # Base probability from historical rates
            base_prob = self._get_base_probability(moneyness)
            factors["base_probability"] = base_prob

            # Time decay adjustment
            if days_to_expiry <= self.NEAR_EXPIRY_DAYS:
                time_multiplier = 1.0 + (self.NEAR_EXPIRY_DAYS - days_to_expiry) * 0.1
                factors["time_multiplier"] = time_multiplier
            else:
                # Earlier expiries have lower assignment probability
                time_multiplier = 0.5 + 0.5 * (30 - min(days_to_expiry, 30)) / 30
                factors["time_multiplier"] = time_multiplier

            # Earnings adjustment
            earnings_multiplier = 1.0
            if near_earnings:
                earnings_multiplier = self.EARNINGS_MULTIPLIER
                if moneyness > -0.02:  # Near or ITM
                    earnings_multiplier *= 1.5  # Extra boost for ITM near earnings
                warnings.append("Elevated assignment risk due to upcoming earnings")
            factors["earnings_multiplier"] = earnings_multiplier

            # Weekend gap adjustment
            weekend_multiplier = 1.0
            if is_friday and moneyness > -0.03:  # Not deep OTM
                weekend_multiplier = self.WEEKEND_GAP_ADJUSTMENT
                warnings.append("Friday: increased weekend gap assignment risk")
            factors["weekend_multiplier"] = weekend_multiplier

            # Dividend adjustment
            dividend_multiplier = 1.0
            if dividend_date and days_to_expiry > 0:
                days_to_dividend = (dividend_date - datetime.now()).days
                if 0 <= days_to_dividend <= days_to_expiry:
                    # Assignment more likely before dividend
                    if moneyness > 0:  # ITM
                        dividend_multiplier = 2.0
                        warnings.append(
                            f"Ex-dividend in {days_to_dividend} days increases assignment risk"
                        )
                    factors["dividend_multiplier"] = dividend_multiplier

            # Calculate final probability
            probability = (
                base_prob
                * time_multiplier
                * earnings_multiplier
                * weekend_multiplier
                * dividend_multiplier
            )

            # Cap at reasonable maximum
            probability = min(probability, 0.95)

            # Special case: Deep ITM near expiry
            if moneyness > self.DEEP_ITM_THRESHOLD and days_to_expiry <= 3:
                probability = max(probability, 0.80)
                warnings.append("Deep ITM near expiry - high assignment probability")

            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(
                spot_price, strike_price, days_to_expiry, volatility
            )

            return AssignmentProbability(
                probability=probability, confidence=confidence, factors=factors, warnings=warnings
            )

        except Exception as e:
            self.logger.error(f"Assignment probability calculation failed: {e}")
            return AssignmentProbability(
                probability=0.05,  # Conservative fallback
                confidence=0.0,
                factors={"error": str(e)},
                warnings=[f"Calculation error: {str(e)}"],
            )

    def _get_base_probability(self, moneyness: float) -> float:
        """Get base assignment probability from historical rates."""
        if moneyness < -0.05:
            return self.HISTORICAL_RATES["deep_otm"]
        elif moneyness < 0:
            return self.HISTORICAL_RATES["otm"]
        elif moneyness < 0.01:
            return self.HISTORICAL_RATES["atm"]
        elif moneyness < self.DEEP_ITM_THRESHOLD:
            return self.HISTORICAL_RATES["slight_itm"]
        else:
            return self.HISTORICAL_RATES["deep_itm"]

    def _calculate_confidence(
        self, spot_price: float, strike_price: float, days_to_expiry: int, volatility: float
    ) -> float:
        """Calculate confidence in assignment probability estimate."""
        confidence = 0.9  # Base confidence

        # Reduce confidence for extreme inputs
        if spot_price <= 0 or strike_price <= 0:
            confidence *= 0.1
        if volatility <= 0 or volatility > 2.0:
            confidence *= 0.8
        if days_to_expiry < 0:
            confidence *= 0.5
        elif days_to_expiry > 60:
            confidence *= 0.9  # Slightly lower for far expiries

        # Higher confidence for near-expiry ITM
        moneyness = (spot_price - strike_price) / strike_price
        if days_to_expiry <= 3 and moneyness > 0.02:
            confidence = min(confidence * 1.1, 0.99)

        return confidence

    @timed_operation(threshold_ms=10.0)
    def get_assignment_curve(
        self,
        spot_price: float,
        strikes: list[float],
        days_to_expiry: int,
        volatility: float,
        near_earnings: bool = False,
    ) -> Dict[float, AssignmentProbability]:
        """Calculate assignment probabilities for multiple strikes.

        Useful for visualizing assignment risk across the option chain.
        """
        results = {}
        is_friday = datetime.now().weekday() == 4

        for strike in strikes:
            prob = self.probability_of_assignment(
                spot_price=spot_price,
                strike_price=strike,
                days_to_expiry=days_to_expiry,
                volatility=volatility,
                near_earnings=near_earnings,
                is_friday=is_friday,
            )
            results[strike] = prob

        return results

    def suggest_assignment_avoidance_strike(
        self,
        spot_price: float,
        available_strikes: list[float],
        days_to_expiry: int,
        volatility: float,
        max_acceptable_probability: float = 0.10,
        near_earnings: bool = False,
    ) -> Optional[float]:
        """Suggest strike with assignment probability below threshold.

        Returns None if no suitable strike found.
        """
        is_friday = datetime.now().weekday() == 4

        # Sort strikes from furthest OTM to closest
        otm_strikes = [s for s in available_strikes if s < spot_price]
        otm_strikes.sort()  # Ascending order for puts

        for strike in otm_strikes:
            prob = self.probability_of_assignment(
                spot_price=spot_price,
                strike_price=strike,
                days_to_expiry=days_to_expiry,
                volatility=volatility,
                near_earnings=near_earnings,
                is_friday=is_friday,
            )

            if prob.probability <= max_acceptable_probability and prob.confidence > 0.7:
                self.logger.info(
                    f"Found assignment-avoiding strike ${strike:.2f} "
                    f"with {prob.probability:.1%} assignment probability"
                )
                return strike

        return None
