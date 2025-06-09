"""Enhanced wheel strategy implementation with self-validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from ..math import (
    CalculationResult,
    black_scholes_price_validated,
    calculate_all_greeks,
    probability_itm_validated,
)
from ..models import Position, PositionType
from ..risk import RiskAnalyzer
from ..utils import (
    cached,
    get_logger,
    timed_operation,
    with_recovery,
    RecoveryStrategy,
    StructuredLogger,
)

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
    target_delta: float = 0.30
    target_dte: int = 45
    max_position_size: float = 0.20
    min_premium_yield: float = 0.01  # 1% minimum
    roll_dte_threshold: int = 7
    roll_delta_threshold: float = 0.70
    
    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0 < self.target_delta < 1:
            raise ValueError(f"Target delta must be between 0 and 1, got {self.target_delta}")
        if self.target_dte < 1:
            raise ValueError(f"Target DTE must be positive, got {self.target_dte}")
        if not 0 < self.max_position_size <= 1:
            raise ValueError(f"Max position size must be between 0 and 1, got {self.max_position_size}")


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
        if not available_strikes:
            logger.warning("No available strikes provided")
            return None
        
        # Validate inputs
        if current_price <= 0 or volatility <= 0:
            logger.error("Invalid inputs for put strike selection")
            return None
        
        time_to_expiry = days_to_expiry / 365.0
        target_delta = -self.params.target_delta  # Puts have negative delta
        
        best_strike = None
        best_score = float('inf')
        results = []
        
        for strike in available_strikes:
            # Skip strikes too far OTM (> 20% below current)
            if strike < current_price * 0.8:
                continue
            
            # Calculate Greeks with validation
            greeks, greek_confidence = calculate_all_greeks(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="put",
            )
            
            if greek_confidence < 0.5:
                logger.warning(f"Low confidence Greeks for strike {strike}")
                continue
            
            delta = greeks.get("delta", 0)
            
            # Calculate premium
            price_result = black_scholes_price_validated(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="put",
            )
            
            if price_result.confidence < 0.5:
                logger.warning(f"Low confidence price for strike {strike}")
                continue
            
            premium = price_result.value
            
            # Calculate probability ITM
            prob_result = probability_itm_validated(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="put",
            )
            
            # Calculate premium yield
            premium_yield = premium / strike
            
            # Score based on delta distance and premium yield
            delta_distance = abs(delta - target_delta)
            
            # Penalize low premium yield
            if premium_yield < self.params.min_premium_yield:
                delta_distance += 0.1  # Add penalty
            
            # Overall confidence
            confidence = min(greek_confidence, price_result.confidence, prob_result.confidence)
            
            results.append({
                "strike": strike,
                "delta": delta,
                "premium": premium,
                "prob_itm": prob_result.value,
                "yield": premium_yield,
                "score": delta_distance,
                "confidence": confidence,
            })
            
            if delta_distance < best_score:
                best_score = delta_distance
                best_strike = {
                    "strike": strike,
                    "delta": delta,
                    "premium": premium,
                    "prob_itm": prob_result.value,
                    "confidence": confidence,
                }
        
        if not best_strike:
            logger.warning("No suitable put strike found")
            return None
        
        # Determine reason for selection
        reason = f"Delta {best_strike['delta']:.2f} closest to target {target_delta:.2f}"
        if best_strike['premium'] / best_strike['strike'] >= 0.02:
            reason += " with attractive premium"
        
        structured_logger.log(
            level="INFO",
            message="Optimal put strike selected",
            context={
                "function": "find_optimal_put_strike",
                "inputs": {
                    "current_price": current_price,
                    "volatility": volatility,
                    "days_to_expiry": days_to_expiry,
                    "available_strikes": len(available_strikes),
                },
                "output": {
                    "strike": best_strike['strike'],
                    "delta": best_strike['delta'],
                    "premium": best_strike['premium'],
                    "prob_itm": best_strike['prob_itm'],
                    "confidence": best_strike['confidence'],
                },
                "performance": {
                    "candidates_evaluated": len(results),
                    "best_score": best_score,
                },
            },
        )
        
        return StrikeRecommendation(
            strike=best_strike['strike'],
            delta=best_strike['delta'],
            probability_itm=best_strike['prob_itm'],
            premium=best_strike['premium'],
            confidence=best_strike['confidence'],
            reason=reason,
        )
    
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
        """
        Find optimal call strike above cost basis with validation.
        
        Parameters
        ----------
        current_price : float
            Current stock price
        cost_basis : float
            Average cost basis of shares
        available_strikes : List[float]
            Available strike prices
        volatility : float
            Implied volatility
        days_to_expiry : int
            Days until expiration
        risk_free_rate : float
            Risk-free rate
        
        Returns
        -------
        Optional[StrikeRecommendation]
            Recommendation with confidence score
        """
        # Filter strikes above cost basis
        valid_strikes = [s for s in available_strikes if s >= cost_basis]
        if not valid_strikes:
            logger.warning(f"No strikes above cost basis {cost_basis}")
            return None
        
        time_to_expiry = days_to_expiry / 365.0
        target_delta = self.params.target_delta
        
        best_strike = None
        best_score = float('inf')
        
        for strike in valid_strikes:
            # Skip strikes too far OTM (> 10% above current)
            if strike > current_price * 1.1:
                continue
            
            # Calculate Greeks
            greeks, greek_confidence = calculate_all_greeks(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="call",
            )
            
            if greek_confidence < 0.5:
                continue
            
            delta = greeks.get("delta", 0)
            
            # Calculate premium
            price_result = black_scholes_price_validated(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="call",
            )
            
            premium = price_result.value
            
            # Calculate probability ITM
            prob_result = probability_itm_validated(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="call",
            )
            
            # Score based on delta distance
            delta_distance = abs(delta - target_delta)
            
            # Bonus for strikes further above cost basis
            safety_margin = (strike - cost_basis) / cost_basis
            if safety_margin > 0.05:  # 5% above cost basis
                delta_distance -= 0.05  # Improve score
            
            confidence = min(greek_confidence, price_result.confidence, prob_result.confidence)
            
            if delta_distance < best_score:
                best_score = delta_distance
                best_strike = {
                    "strike": strike,
                    "delta": delta,
                    "premium": premium,
                    "prob_itm": prob_result.value,
                    "confidence": confidence,
                    "safety_margin": safety_margin,
                }
        
        if not best_strike:
            return None
        
        reason = f"Delta {best_strike['delta']:.2f} with {best_strike['safety_margin']:.1%} above cost basis"
        
        logger.info(
            "Optimal call strike selected",
            extra={
                "current_price": current_price,
                "cost_basis": cost_basis,
                "strike": best_strike['strike'],
                "delta": best_strike['delta'],
                "safety_margin": best_strike['safety_margin'],
                "confidence": best_strike['confidence'],
            },
        )
        
        return StrikeRecommendation(
            strike=best_strike['strike'],
            delta=best_strike['delta'],
            probability_itm=best_strike['prob_itm'],
            premium=best_strike['premium'],
            confidence=best_strike['confidence'],
            reason=reason,
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
        # Maximum allocation based on position size limit
        max_allocation = portfolio_value * self.params.max_position_size
        
        # Calculate contracts based on strike (100 shares per contract)
        max_contracts = int(max_allocation / (strike_price * 100))
        
        # Apply Kelly criterion if we have win rate data
        # For now, use conservative sizing
        kelly_contracts = max(1, int(max_contracts * 0.5))  # Half-Kelly
        
        # Check margin constraints
        # Assume 20% margin requirement for naked puts
        margin_per_contract = strike_price * 100 * 0.20
        available_margin = portfolio_value * 0.5 - current_margin_used
        margin_contracts = int(available_margin / margin_per_contract)
        
        # Take minimum of all constraints
        final_contracts = max(1, min(kelly_contracts, margin_contracts))
        
        # Confidence based on how many constraints are binding
        constraints_ok = sum([
            final_contracts == kelly_contracts,
            final_contracts < max_contracts,
            margin_contracts > kelly_contracts,
        ])
        confidence = constraints_ok / 3.0
        
        logger.info(
            "Position size calculated",
            extra={
                "strike_price": strike_price,
                "portfolio_value": portfolio_value,
                "max_contracts": max_contracts,
                "kelly_contracts": kelly_contracts,
                "margin_contracts": margin_contracts,
                "final_contracts": final_contracts,
                "confidence": confidence,
            },
        )
        
        return final_contracts, confidence
    
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
        if position.strike:
            if position.position_type == PositionType.PUT:
                if current_price < position.strike * 0.95:
                    reasons.append("Put moving ITM")
            elif position.position_type == PositionType.CALL:
                if current_price > position.strike * 0.95:
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