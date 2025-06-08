"""Wheel strategy implementation."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from .config import Settings, get_settings
from .models import Position, WheelPosition
from .utils.math import calculate_delta, probability_itm

# Configure structured logging
logger = logging.getLogger(__name__)


class WheelStrategy:
    """Implements the wheel options strategy."""

    def __init__(self) -> None:
        """Initialize wheel strategy."""
        self.settings: Settings = get_settings()
        self.positions: dict[str, WheelPosition] = {}
        logger.info(
            "Wheel strategy initialized",
            extra={
                "delta_target": self.settings.wheel_delta_target,
                "dte_target": self.settings.days_to_expiry_target,
                "max_position_size": self.settings.max_position_size
            }
        )

    def find_optimal_put_strike(
        self,
        current_price: float,
        available_strikes: list[float],
        volatility: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
    ) -> float | None:
        """Find optimal put strike near target delta.

        Parameters
        ----------
        current_price : float
            Current stock price
        available_strikes : list[float]
            Available strike prices
        volatility : float
            Implied volatility
        days_to_expiry : int
            Days until expiration
        risk_free_rate : float
            Risk-free rate

        Returns
        -------
        Optional[float]
            Optimal strike price or None if no suitable strike
        """
        if not available_strikes:
            return None

        time_to_expiry = days_to_expiry / 365.0
        target_delta = -self.settings.wheel_delta_target  # Puts have negative delta

        # Calculate delta for each strike
        deltas_list = []
        for strike in available_strikes:
            delta = calculate_delta(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="put",
            )
            deltas_list.append(delta)

        # Find strike closest to target delta
        deltas = np.array(deltas_list)
        distances = np.abs(deltas - target_delta)
        optimal_idx = np.argmin(distances)

        optimal_strike = available_strikes[optimal_idx]
        optimal_delta = deltas[optimal_idx]

        logger.info(
            "Optimal put strike selected",
            extra={
                "function": "find_optimal_put_strike",
                "current_price": current_price,
                "optimal_strike": optimal_strike,
                "optimal_delta": float(optimal_delta),
                "target_delta": float(target_delta),
                "days_to_expiry": days_to_expiry
            }
        )

        return optimal_strike

    def find_optimal_call_strike(
        self,
        current_price: float,
        cost_basis: float,
        available_strikes: list[float],
        volatility: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
    ) -> float | None:
        """Find optimal call strike near target delta and above cost basis.

        Parameters
        ----------
        current_price : float
            Current stock price
        cost_basis : float
            Average cost basis of shares
        available_strikes : list[float]
            Available strike prices
        volatility : float
            Implied volatility
        days_to_expiry : int
            Days until expiration
        risk_free_rate : float
            Risk-free rate

        Returns
        -------
        Optional[float]
            Optimal strike price or None if no suitable strike
        """
        # Filter strikes above cost basis
        valid_strikes = [s for s in available_strikes if s >= cost_basis]
        if not valid_strikes:
            logger.warning(f"No strikes above cost basis {cost_basis}")
            return None

        time_to_expiry = days_to_expiry / 365.0
        target_delta = self.settings.wheel_delta_target

        # Calculate delta for each valid strike
        deltas_list = []
        for strike in valid_strikes:
            delta = calculate_delta(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="call",
            )
            deltas_list.append(delta)

        # Find strike closest to target delta
        deltas = np.array(deltas_list)
        distances = np.abs(deltas - target_delta)
        optimal_idx = np.argmin(distances)

        optimal_strike = valid_strikes[optimal_idx]
        optimal_delta = deltas[optimal_idx]

        # Calculate probability of assignment
        prob_itm = probability_itm(
            S=current_price,
            K=optimal_strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
            option_type="call",
        )

        logger.info(
            "Optimal call strike selected",
            extra={
                "function": "find_optimal_call_strike",
                "current_price": current_price,
                "cost_basis": cost_basis,
                "optimal_strike": optimal_strike,
                "optimal_delta": float(optimal_delta),
                "target_delta": float(target_delta),
                "probability_itm": float(prob_itm),
                "days_to_expiry": days_to_expiry
            }
        )

        return optimal_strike

    def calculate_position_size(
        self, symbol: str, current_price: float, portfolio_value: float
    ) -> int:
        """Calculate appropriate position size.

        Parameters
        ----------
        symbol : str
            Trading symbol
        current_price : float
            Current stock price
        portfolio_value : float
            Total portfolio value

        Returns
        -------
        int
            Number of contracts (each represents 100 shares)
        """
        max_allocation = portfolio_value * self.settings.max_position_size
        contracts = int(max_allocation / (current_price * 100))

        logger.info(
            "Position size calculated",
            extra={
                "function": "calculate_position_size",
                "symbol": symbol,
                "current_price": current_price,
                "portfolio_value": portfolio_value,
                "max_allocation": max_allocation,
                "contracts": contracts,
                "position_size_pct": self.settings.max_position_size
            }
        )

        return max(1, contracts)  # At least 1 contract

    def should_roll_position(
        self,
        position: Position,
        current_price: float,
        days_to_expiry: int,
        current_delta: float,
    ) -> bool:
        """Determine if a position should be rolled.

        Parameters
        ----------
        position : Position
            Current option position
        current_price : float
            Current stock price
        days_to_expiry : int
            Days until expiration
        current_delta : float
            Current delta of the position

        Returns
        -------
        bool
            True if position should be rolled
        """
        # Roll if too close to expiry
        if days_to_expiry <= 7:
            logger.info(
                "Position roll triggered - approaching expiry",
                extra={
                    "symbol": position.symbol,
                    "days_to_expiry": days_to_expiry,
                    "reason": "approaching_expiry"
                }
            )
            return True

        # Roll puts if too far ITM (delta too negative)
        if position.option_type == "put" and current_delta < -0.7:
            logger.info(
                "Position roll triggered - deep ITM put",
                extra={
                    "symbol": position.symbol,
                    "option_type": "put",
                    "current_delta": current_delta,
                    "reason": "deep_itm"
                }
            )
            return True

        # Roll calls if too far ITM (delta too high)
        if position.option_type == "call" and current_delta > 0.7:
            logger.info(
                "Position roll triggered - deep ITM call",
                extra={
                    "symbol": position.symbol,
                    "option_type": "call",
                    "current_delta": current_delta,
                    "reason": "deep_itm"
                }
            )
            return True

        return False

    def track_position(self, symbol: str) -> WheelPosition:
        """Get or create position tracking for a symbol.

        Parameters
        ----------
        symbol : str
            Trading symbol

        Returns
        -------
        WheelPosition
            Position tracking object
        """
        if symbol not in self.positions:
            self.positions[symbol] = WheelPosition(symbol=symbol)
        return self.positions[symbol]
