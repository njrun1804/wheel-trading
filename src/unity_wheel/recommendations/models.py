"""Data models for wheel strategy recommendations with full type safety."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PositionType(str, Enum):
    """Valid position types in wheel strategy."""
    STOCK = "stock"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"

class RecommendationAction(str, Enum):
    """Possible recommendation actions."""
    HOLD = "hold"
    ROLL = "roll"
    CLOSE = "close"
    OPEN_CSP = "open_csp"
    OPEN_CC = "open_cc"
    ADJUST_SIZE = "adjust_size"

@dataclass(frozen=True)
class Greeks:
    """Option Greeks with validation."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    def __post_init__(self) -> None:
        """Validate Greeks are within reasonable ranges."""
        if not -1 <= self.delta <= 1:
            logger.warning(f"Delta {self.delta} outside valid range [-1, 1]")
        if self.gamma < 0:
            logger.warning(f"Gamma {self.gamma} should be non-negative")

@dataclass(frozen=True)
class Position:
    """Immutable position representation with validation."""
    symbol: str
    position_type: PositionType
    quantity: int
    cost_basis: float
    current_price: float
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    premium_collected: Optional[float] = None
    greeks: Optional[Greeks] = None
    
    def __post_init__(self) -> None:
        """Validate position consistency."""
        if self.position_type in (PositionType.COVERED_CALL, PositionType.CASH_SECURED_PUT):
            if not self.strike:
                raise ValueError(f"Options position {self.position_type} requires strike price")
            if not self.expiry:
                raise ValueError(f"Options position {self.position_type} requires expiry date")
        
        logger.debug(f"Created position: {self.symbol} {self.position_type} qty={self.quantity}")
    
    @property
    def days_to_expiry(self) -> Optional[int]:
        """Calculate days to expiry for options."""
        if self.expiry:
            return max(0, (self.expiry - datetime.now()).days)
        return None
    
    @property
    def intrinsic_value(self) -> float:
        """Calculate intrinsic value for options."""
        if not self.strike:
            return 0.0
        
        if self.position_type == PositionType.COVERED_CALL:
            return max(0, self.current_price - self.strike)
        elif self.position_type == PositionType.CASH_SECURED_PUT:
            return max(0, self.strike - self.current_price)
        return 0.0

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
            logger.error(f"Margin used ({self.margin_used}) exceeds available ({self.margin_available})")
        
        logger.debug(f"Account state: value=${self.total_value:,.2f}, cash=${self.cash:,.2f}")
    
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