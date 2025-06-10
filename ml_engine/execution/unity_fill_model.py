"""Unity-specific fill price modeling with bid-ask spreads and size impact."""

import sys
from dataclasses import dataclass
from typing import NoReturn, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)


def die(message: str) -> NoReturn:
    """Exit program with error message."""
    logger.error(f"FATAL: {message}")
    print(f"\n❌ FATAL ERROR: {message}\n", file=sys.stderr)
    sys.exit(1)


@dataclass
class FillEstimate:
    """Estimated fill price and costs."""

    fill_price: float
    commission: float
    spread_cost: float
    size_impact: float
    total_cost: float
    confidence: float


class UnityFillModel:
    """Unity-specific fill price estimation with realistic bid-ask modeling."""

    # Unity-specific constants
    TYPICAL_SPREAD_RANGE = (0.05, 0.10)  # Unity typically has 0.05-0.10 spreads
    COMMISSION_PER_CONTRACT = 0.65  # Standard options commission
    ASSIGNMENT_FEE = 5.00  # Assignment/exercise fee
    SIZE_IMPACT_THRESHOLD = 10  # Contracts before size impact
    SIZE_IMPACT_PER_CONTRACT = 0.01  # $0.01 per contract over threshold

    # Fill positioning within spread (0=bid, 1=ask)
    OPENING_FILL_POSITION = 0.1  # Selling puts: 10% from bid
    CLOSING_FILL_POSITION = 0.9  # Buying back: 90% from bid (10% from ask)

    def __init__(self) -> None:
        self.logger = logger

    def estimate_fill_price(
        self, bid: float, ask: float, size: int, is_opening: bool, urgency: float = 0.5
    ) -> Tuple[FillEstimate, float]:
        """
        Estimate realistic fill price for Unity options.

        Args:
            bid: Current bid price
            ask: Current ask price
            size: Number of contracts
            is_opening: True if opening position (selling puts)
            urgency: 0-1, higher means more aggressive fills

        Returns:
            Tuple of (FillEstimate, confidence_score)
        """
        # Validate inputs - die on any error
        if bid <= 0:
            die(f"Invalid bid price: {bid}")
        if ask <= 0:
            die(f"Invalid ask price: {ask}")
        if bid > ask:
            die(f"Invalid bid/ask spread: bid ({bid}) > ask ({ask})")
        if size <= 0:
            die(f"Invalid position size: {size}")

        # Calculate spread
        spread = ask - bid

        # Determine fill position within spread based on trade type and urgency
        if is_opening:
            # Selling puts: expect to fill closer to bid
            base_position = self.OPENING_FILL_POSITION
            # Higher urgency means filling closer to mid
            fill_position = base_position + (0.4 * urgency)
        else:
            # Buying back: expect to fill closer to ask
            base_position = self.CLOSING_FILL_POSITION
            # Higher urgency means filling even closer to ask
            fill_position = base_position - (0.3 * (1 - urgency))

        # Calculate base fill price
        base_fill = bid + (spread * fill_position)

        # Calculate size impact
        size_impact = self._calculate_size_impact(size, spread)

        # Adjust fill price for size impact
        if is_opening:
            # Selling larger size pushes price down (worse fill)
            fill_price = base_fill - size_impact
        else:
            # Buying larger size pushes price up (worse fill)
            fill_price = base_fill + size_impact

        # Ensure fill price is within bid-ask
        fill_price = max(bid, min(ask, fill_price))

        # Calculate costs
        commission = self._calculate_commission(size)
        spread_cost = abs(fill_price - (bid + ask) / 2) * size * 100
        total_size_impact = size_impact * size * 100

        # Total cost includes commission and market impact
        total_cost = commission + spread_cost + total_size_impact

        # Calculate confidence based on spread width and size
        confidence = self._calculate_confidence(spread, size, bid, ask)

        result = FillEstimate(
            fill_price=round(fill_price, 2),
            commission=commission,
            spread_cost=round(spread_cost, 2),
            size_impact=round(size_impact, 4),
            total_cost=round(total_cost, 2),
            confidence=confidence,
        )

        self.logger.info(
            f"Fill estimate: {result}",
            extra={
                "bid": bid,
                "ask": ask,
                "size": size,
                "is_opening": is_opening,
                "urgency": urgency,
            },
        )

        return result, confidence

    def _calculate_size_impact(self, size: int, spread: float) -> float:
        """
        Calculate price impact based on order size.

        Unity is less liquid than SPY, so larger orders move the market.
        """
        if size <= self.SIZE_IMPACT_THRESHOLD:
            return 0.0

        # Each contract over threshold adds impact
        excess_contracts = size - self.SIZE_IMPACT_THRESHOLD

        # Impact increases with size, proportional to spread
        base_impact = excess_contracts * self.SIZE_IMPACT_PER_CONTRACT

        # Larger spreads indicate less liquidity, more impact
        spread_multiplier = max(1.0, spread / 0.05)  # Normalized to 0.05 spread

        return base_impact * spread_multiplier

    def _calculate_commission(self, size: int) -> float:
        """Calculate total commission for the trade."""
        return size * self.COMMISSION_PER_CONTRACT

    def _calculate_confidence(self, spread: float, size: int, bid: float, ask: float) -> float:
        """
        Calculate confidence in fill estimate.

        Lower confidence for:
        - Wide spreads (low liquidity)
        - Large sizes (market impact uncertainty)
        - Very low or high prices (less liquid strikes)
        """
        confidence = 0.95

        # Penalize wide spreads
        if spread > 0.10:
            confidence *= 0.9
        if spread > 0.20:
            confidence *= 0.8

        # Penalize large sizes
        if size > 20:
            confidence *= 0.9
        if size > 50:
            confidence *= 0.8

        # Penalize extreme prices (likely far OTM/ITM)
        mid_price = (bid + ask) / 2
        if mid_price < 0.10 or mid_price > 10.0:
            confidence *= 0.85

        return round(confidence, 2)

    def estimate_assignment_cost(self, size: int) -> float:
        """Estimate cost of assignment/exercise."""
        # Typically a flat fee per assignment event
        return self.ASSIGNMENT_FEE

    def estimate_round_trip_cost(
        self, open_bid: float, open_ask: float, close_bid: float, close_ask: float, size: int
    ) -> Tuple[float, float]:
        """
        Estimate total cost for round-trip trade (open and close).

        Returns:
            Tuple of (total_cost, confidence)
        """
        # Estimate opening trade
        open_estimate, open_conf = self.estimate_fill_price(
            open_bid, open_ask, size, is_opening=True
        )

        # Estimate closing trade
        close_estimate, close_conf = self.estimate_fill_price(
            close_bid, close_ask, size, is_opening=False
        )

        # Total round-trip cost
        total_cost = open_estimate.total_cost + close_estimate.total_cost

        # Combined confidence
        confidence = min(open_conf, close_conf) * 0.9  # Slight penalty for uncertainty

        return total_cost, confidence
