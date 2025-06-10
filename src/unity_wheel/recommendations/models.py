"""Data models for wheel strategy recommendations with full type safety."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

logger = logging.getLogger(__name__)


class RecommendationAction(str, Enum):
    """Possible recommendation actions."""

    HOLD = "hold"
    ROLL = "roll"
    CLOSE = "close"
    OPEN_CSP = "open_csp"
    OPEN_CC = "open_cc"
    ADJUST_SIZE = "adjust_size"


@dataclass(frozen=True)
class AccountState:
    """Immutable account state with validation."""

    total_value: float
    cash: float
    margin_available: float
    margin_used: float
    external_loans: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate account state consistency."""
        if self.cash < 0:
            logger.warning(f"Negative cash balance: {self.cash}")
        if self.margin_used > self.margin_available:
            logger.error(
                "Margin used (%s) exceeds available (%s)",
                self.margin_used,
                self.margin_available,
            )

        logger.debug(
            "Account state: value=$%.2f, cash=$%.2f",
            self.total_value,
            self.cash,
        )

    @property
    def buying_power(self) -> float:
        """Calculate total buying power."""
        return self.cash + (self.margin_available - self.margin_used)

    @property
    def cost_of_capital(self) -> float:
        """Calculate weighted average cost of capital."""
        if not self.external_loans:
            return 0.0

        # Example: {"AMEX": 0.07, "margin": 0.10}
        total_borrowed = sum(self.external_loans.values())
        if total_borrowed == 0:
            return 0.0

        # For now, assume loan amounts equal rates (simplified)
        # In production, would have separate loan_amounts dict
        weighted_sum = sum(rate * rate for rate in self.external_loans.values())
        return weighted_sum / len(self.external_loans)


@dataclass(frozen=True)
class RiskMetrics:
    """Risk metrics for position evaluation."""

    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)
    max_drawdown: float
    sharpe_ratio: float
    kelly_fraction: float
    half_kelly_size: float

    def __post_init__(self) -> None:
        """Validate risk metrics."""
        if not 0 <= self.kelly_fraction <= 1:
            logger.warning(f"Kelly fraction {self.kelly_fraction} outside [0, 1]")
        if self.cvar_95 > self.var_95:
            logger.error(f"CVaR ({self.cvar_95}) should be <= VaR ({self.var_95})")


@dataclass(frozen=True)
class Recommendation:
    """Immutable recommendation with full traceability."""

    action: RecommendationAction
    symbol: str
    confidence: float
    expected_return: float
    risk_adjusted_return: float  # CAGR - 0.20 × |CVaR₉₅|
    position_size: float  # ½-Kelly sizing
    reasoning: List[str]
    metadata: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate recommendation consistency."""
        if not 0 <= self.confidence <= 1:
            logger.error(f"Confidence {self.confidence} outside [0, 1]")
        if self.position_size < 0:
            logger.error(f"Negative position size: {self.position_size}")

        logger.info(
            f"Recommendation: {self.action} {self.symbol} "
            f"size={self.position_size:.1%} conf={self.confidence:.1%}"
        )
