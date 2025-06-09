"""Account model for tracking portfolio state."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Account:
    """
    Immutable account snapshot with validation.
    
    Represents the state of a trading account at a specific point in time.
    All monetary values are in USD.
    
    Attributes
    ----------
    cash_balance : float
        Available cash in the account
    buying_power : float
        Total buying power (includes margin if available)
    margin_used : float
        Amount of margin currently in use
    timestamp : datetime
        When this snapshot was taken (UTC)
    
    Properties
    ----------
    net_liquidation_value : float
        Total account value (cash + margin - margin_used)
    margin_available : float
        Remaining margin capacity
    margin_utilization : float
        Percentage of margin capacity used
    
    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> account = Account(
    ...     cash_balance=50000.0,
    ...     buying_power=100000.0,
    ...     margin_used=25000.0,
    ...     timestamp=datetime.now(timezone.utc)
    ... )
    >>> account.margin_available
    25000.0
    >>> account.margin_utilization
    0.5
    """

    cash_balance: float
    buying_power: float
    margin_used: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate account values."""
        # Validate cash balance
        if self.cash_balance < 0:
            raise ValueError(f"Cash balance cannot be negative, got {self.cash_balance}")
        
        # Validate buying power
        if self.buying_power < 0:
            raise ValueError(f"Buying power cannot be negative, got {self.buying_power}")
        
        # Validate margin used
        if self.margin_used < 0:
            raise ValueError(f"Margin used cannot be negative, got {self.margin_used}")
        
        # Buying power should be at least cash balance
        if self.buying_power < self.cash_balance:
            raise ValueError(
                f"Buying power ({self.buying_power}) cannot be less than "
                f"cash balance ({self.cash_balance})"
            )
        
        # Margin used cannot exceed total margin available
        total_margin = self.buying_power - self.cash_balance
        if self.margin_used > total_margin:
            raise ValueError(
                f"Margin used ({self.margin_used}) exceeds total margin "
                f"available ({total_margin})"
            )
        
        # Ensure timestamp is timezone-aware
        if self.timestamp.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        
        logger.debug(
            "Account snapshot created",
            extra={
                "cash_balance": self.cash_balance,
                "buying_power": self.buying_power,
                "margin_used": self.margin_used,
                "timestamp": self.timestamp.isoformat(),
            },
        )

    @property
    def net_liquidation_value(self) -> float:
        """Calculate total account value."""
        # This is simplified - in practice would include position values
        return self.cash_balance

    @property
    def margin_available(self) -> float:
        """Calculate remaining margin capacity."""
        total_margin = self.buying_power - self.cash_balance
        return total_margin - self.margin_used

    @property
    def margin_utilization(self) -> float:
        """Calculate margin utilization percentage."""
        total_margin = self.buying_power - self.cash_balance
        if total_margin == 0:
            return 0.0
        return self.margin_used / total_margin

    @property
    def is_margin_account(self) -> bool:
        """Check if this is a margin account."""
        return self.buying_power > self.cash_balance

    @property
    def has_sufficient_buying_power(self) -> float:
        """Get available buying power after margin."""
        return self.buying_power - self.margin_used

    def validate_position_size(self, required_capital: float) -> bool:
        """
        Check if account can support a position requiring given capital.
        
        Parameters
        ----------
        required_capital : float
            Capital required for the position
        
        Returns
        -------
        bool
            True if position can be taken, False otherwise
        """
        return required_capital <= self.has_sufficient_buying_power

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for serialization."""
        return {
            "cash_balance": self.cash_balance,
            "buying_power": self.buying_power,
            "margin_used": self.margin_used,
            "timestamp": self.timestamp.isoformat(),
            "net_liquidation_value": self.net_liquidation_value,
            "margin_available": self.margin_available,
            "margin_utilization": self.margin_utilization,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | str]) -> Account:
        """Create Account from dictionary."""
        # Parse timestamp if string
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            cash_balance=float(data["cash_balance"]),
            buying_power=float(data["buying_power"]),
            margin_used=float(data.get("margin_used", 0.0)),
            timestamp=timestamp,
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Account("
            f"cash=${self.cash_balance:,.2f}, "
            f"buying_power=${self.buying_power:,.2f}, "
            f"margin_used=${self.margin_used:,.2f}, "
            f"utilization={self.margin_utilization:.1%})"
        )