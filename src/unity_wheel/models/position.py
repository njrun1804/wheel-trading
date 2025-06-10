"""Position model with symbol validation and type detection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional, Union

logger = logging.getLogger(__name__)


class PositionType(str, Enum):
    """Type of position."""

    STOCK = "stock"
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Position:
    """
    Immutable position representation with symbol validation.

    Attributes
    ----------
    symbol : str
        For stocks: ticker symbol (e.g., "U", "AAPL")
        For options: OCC format (e.g., "U241220C00080000")
    quantity : int
        Signed integer (positive for long, negative for short)

    Properties
    ----------
    position_type : PositionType
        Automatically detected from symbol format
    underlying : str
        Underlying ticker symbol
    strike : Optional[float]
        Strike price for options (None for stocks)
    expiration : Optional[date]
        Expiration date for options (None for stocks)

    Examples
    --------
    >>> pos = Position("U", 100)
    >>> pos.position_type
    <PositionType.STOCK: 'stock'>
    >>> pos.quantity
    100

    >>> opt = Position("U241220C00080000", -1)
    >>> opt.position_type
    <PositionType.CALL: 'call'>
    >>> opt.strike
    80.0
    >>> opt.expiration
    datetime.date(2024, 12, 20)
    """

    symbol: str
    quantity: int
    _position_type: Optional[PositionType] = field(default=None, init=False, repr=False)
    _underlying: Optional[str] = field(default=None, init=False, repr=False)
    _strike: Optional[float] = field(default=None, init=False, repr=False)
    _expiration: Optional[date] = field(default=None, init=False, repr=False)
    _parsed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs and parse symbol."""
        # Validate symbol
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError(f"Symbol must be a non-empty string, got: {self.symbol}")

        # Validate quantity
        if not isinstance(self.quantity, int):
            raise TypeError(f"Quantity must be an integer, got: {type(self.quantity)}")

        if self.quantity == 0:
            raise ValueError("Quantity cannot be zero")

        # Parse symbol to determine type
        self._parse_symbol()

        logger.debug(
            "Position created",
            extra={
                "symbol": self.symbol,
                "quantity": self.quantity,
                "position_type": self._position_type,
                "underlying": self._underlying,
                "strike": self._strike,
                "expiration": self._expiration,
            },
        )

    def _parse_symbol(self) -> None:
        """Parse symbol to determine position type and extract details."""
        # OCC option symbol pattern: TICKER + YYMMDD + C/P + 00000000 (strike * 1000)
        # Example: U241220C00080000 = U, Dec 20 2024, Call, $80 strike
        option_pattern = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

        match = option_pattern.match(self.symbol.upper())

        if match:
            # It's an option
            ticker, date_str, option_type, strike_str = match.groups()

            # Set underlying
            object.__setattr__(self, "_underlying", ticker)

            # Set position type
            if option_type == "C":
                object.__setattr__(self, "_position_type", PositionType.CALL)
            else:
                object.__setattr__(self, "_position_type", PositionType.PUT)

            # Parse expiration date (YYMMDD format)
            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            object.__setattr__(self, "_expiration", date(year, month, day))

            # Parse strike (last 8 digits represent strike * 1000)
            strike = int(strike_str) / 1000.0
            object.__setattr__(self, "_strike", strike)
        else:
            # It's a stock - validate ticker format
            stock_pattern = re.compile(r"^[A-Z]{1,6}$")
            if stock_pattern.match(self.symbol.upper()):
                object.__setattr__(self, "_position_type", PositionType.STOCK)
                object.__setattr__(self, "_underlying", self.symbol.upper())
                object.__setattr__(self, "_strike", None)
                object.__setattr__(self, "_expiration", None)
            else:
                raise ValueError(
                    f"Invalid symbol format: {self.symbol}. "
                    "Expected stock ticker (e.g., 'U') or OCC option format "
                    "(e.g., 'U241220C00080000')"
                )

        object.__setattr__(self, "_parsed", True)

    @property
    def position_type(self) -> PositionType:
        """Get the position type."""
        if not self._parsed:
            self._parse_symbol()
        return self._position_type  # type: ignore

    @property
    def underlying(self) -> str:
        """Get the underlying ticker symbol."""
        if not self._parsed:
            self._parse_symbol()
        return self._underlying  # type: ignore

    @property
    def strike(self) -> Optional[float]:
        """Get the strike price (None for stocks)."""
        if not self._parsed:
            self._parse_symbol()
        return self._strike

    @property
    def expiration(self) -> Optional[date]:
        """Get the expiration date (None for stocks)."""
        if not self._parsed:
            self._parse_symbol()
        return self._expiration

    @property
    def is_long(self) -> bool:
        """Check if position is long (quantity > 0)."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short (quantity < 0)."""
        return self.quantity < 0

    @property
    def abs_quantity(self) -> int:
        """Get absolute quantity."""
        return abs(self.quantity)

    def __str__(self) -> str:
        """Human-readable string representation."""
        direction = "Long" if self.is_long else "Short"

        if self.position_type == PositionType.STOCK:
            return f"{direction} {self.abs_quantity} {self.symbol}"
        else:
            opt_type = "Call" if self.position_type == PositionType.CALL else "Put"
            return (
                f"{direction} {self.abs_quantity} {self.underlying} "
                f"${self.strike:.2f} {opt_type} exp {self.expiration}"
            )

    def to_dict(self) -> dict[str, Union[str, int, float, None]]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "position_type": self.position_type.value,
            "underlying": self.underlying,
            "strike": self.strike,
            "expiration": self.expiration.isoformat() if self.expiration else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Union[str, int, float, None]]) -> Position:
        """Create Position from dictionary."""
        return cls(
            symbol=str(data["symbol"]),
            quantity=int(data["quantity"]),
        )
