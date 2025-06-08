"""Wheel strategy implementation."""

import logging
from typing import Optional

import numpy as np

from .config import get_settings
from .models import Position, WheelPosition
from .utils.math import calculate_delta, probability_itm

logger = logging.getLogger(__name__)


class WheelStrategy:
    """Implements the wheel options strategy."""

    def __init__(self):
        """Initialize wheel strategy."""
        self.settings = get_settings()
        self.positions: dict[str, WheelPosition] = {}

    def find_optimal_put_strike(
        self,
        current_price: float,
        available_strikes: list[float],
        volatility: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
    ) -> Optional[float]:
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
        deltas = []
        for strike in available_strikes:
            delta = calculate_delta(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="put",
            )
            deltas.append(delta)

        # Find strike closest to target delta
        deltas = np.array(deltas)
        distances = np.abs(deltas - target_delta)
        optimal_idx = np.argmin(distances)

        optimal_strike = available_strikes[optimal_idx]
        optimal_delta = deltas[optimal_idx]

        logger.info(
            f"Selected put strike {optimal_strike} with delta {optimal_delta:.3f} "
            f"(target: {target_delta:.3f})"
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
    ) -> Optional[float]:
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
        deltas = []
        for strike in valid_strikes:
            delta = calculate_delta(
                S=current_price,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="call",
            )
            deltas.append(delta)

        # Find strike closest to target delta
        deltas = np.array(deltas)
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
            f"Selected call strike {optimal_strike} with delta {optimal_delta:.3f} "
            f"(target: {target_delta:.3f}, P(ITM): {prob_itm:.2%})"
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
            f"Position sizing for {symbol}: "
            f"max allocation ${max_allocation:,.2f}, "
            f"{contracts} contracts"
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
            logger.info(f"Rolling {position.symbol} - approaching expiry")
            return True

        # Roll puts if too far ITM (delta too negative)
        if position.option_type == "put" and current_delta < -0.7:
            logger.info(f"Rolling {position.symbol} put - deep ITM")
            return True

        # Roll calls if too far ITM (delta too high)
        if position.option_type == "call" and current_delta > 0.7:
            logger.info(f"Rolling {position.symbol} call - deep ITM")
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
