from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional


class PositionType(str, Enum):
    """Types of positions in account."""

    STOCK = "STOCK"
    OPTION = "OPTION"
    CASH = "CASH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SchwabPosition:
    """Immutable position data from Schwab."""

    symbol: str
    quantity: Decimal
    position_type: PositionType
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")

    # Option-specific fields
    underlying: Optional[str] = None
    strike: Optional[Decimal] = None
    expiration: Optional[datetime] = None
    option_type: Optional[str] = None  # 'CALL' or 'PUT'

    # Metadata
    last_update: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def is_option(self) -> bool:
        """Check if this is an option position."""
        return self.position_type == PositionType.OPTION

    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.quantity < 0

    def validate(self) -> bool:
        """Validate position data consistency."""
        if self.is_option():
            if not all([self.underlying, self.strike, self.expiration, self.option_type]):
                return False
            if self.strike <= 0:
                return False

        # Market value should match quantity * price (roughly)
        # This is a sanity check, not exact due to bid/ask spread
        if self.quantity == 0 and self.market_value != 0:
            return False

        return True


@dataclass(frozen=True)
class SchwabAccount:
    """Immutable account data from Schwab."""

    account_number: str
    account_type: str  # 'MARGIN', 'CASH', etc.

    # Balances
    total_value: Decimal
    cash_balance: Decimal
    buying_power: Decimal
    margin_balance: Decimal = Decimal("0")

    # Risk metrics
    margin_requirement: Decimal = Decimal("0")
    maintenance_requirement: Decimal = Decimal("0")

    # P&L
    daily_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")

    # Metadata
    last_update: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate account data consistency."""
        # Basic sanity checks
        if self.total_value < 0:
            return False

        if self.buying_power < 0:
            return False

        # Margin requirements should not exceed total value
        if self.margin_requirement > self.total_value:
            return False

        return True

    @property
    def margin_utilization(self) -> Decimal:
        """Calculate margin utilization percentage."""
        if self.buying_power == 0:
            return Decimal("1.0")  # Fully utilized

        # This is a simplified calculation
        # Real calculation depends on account type and broker rules
        if self.total_value > 0:
            return (self.total_value - self.buying_power) / self.total_value

        return Decimal("0")
