"""Data models for wheel trading application."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class Greeks(BaseModel):
    """Options Greeks for risk analysis."""

    delta: Optional[float] = Field(
        None,
        description="Rate of change of option price with respect to underlying price",
    )
    gamma: Optional[float] = Field(
        None, description="Rate of change of delta with respect to underlying price"
    )
    theta: Optional[float] = Field(
        None, description="Rate of change of option price with respect to time"
    )
    vega: Optional[float] = Field(
        None, description="Rate of change of option price with respect to volatility"
    )


class Position(BaseModel):
    """Represents a trading position."""

    symbol: str = Field(..., description="Trading symbol")
    qty: float = Field(..., description="Quantity of shares/contracts")
    avg_price: float = Field(..., description="Average entry price")
    option_type: Optional[str] = Field(
        None, description="Option type: 'call' or 'put' if applicable"
    )
    strike: Optional[float] = Field(None, description="Strike price for options")
    expiry: Optional[date] = Field(None, description="Expiration date for options")


class WheelPosition(BaseModel):
    """Wheel strategy position tracking."""

    symbol: str = Field(..., description="Underlying symbol")
    shares: Optional[int] = Field(None, description="Number of shares owned")
    cash_secured_puts: list[Position] = Field(
        default_factory=list, description="Active cash secured puts"
    )
    covered_calls: list[Position] = Field(default_factory=list, description="Active covered calls")
    cost_basis: Optional[float] = Field(None, description="Average cost basis for shares")
    total_premium_collected: float = Field(0.0, description="Total premium collected from wheel")
