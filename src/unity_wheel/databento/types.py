"""Databento data types optimized for wheel strategy.

These types map Databento's raw data to our internal models with:
- Automatic price normalization (divide by 1e9)
- UTC timestamp handling
- Validation and confidence scoring
"""

import enum
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

import pandas as pd


class OptionType(enum.Enum):
    """Option type enum."""

    CALL = "C"
    PUT = "P"


@dataclass(frozen=True)
class InstrumentDefinition:
    """Option instrument metadata from Databento definition schema."""

    instrument_id: int
    raw_symbol: str  # Full OSI symbol like "U 24 06 21 00055 C"
    underlying: str  # e.g., "U"
    option_type: OptionType
    strike_price: Decimal
    expiration: datetime
    multiplier: int = 100

    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiration from now."""
        return (self.expiration.date() - datetime.now(timezone.utc).date()).days

    @classmethod
    def from_databento(cls, msg) -> "InstrumentDefinition":
        """Create from Databento definition message."""
        # Handle Databento's InstrumentDefMsg object
        # Strike price is already in correct units (not scaled)
        strike = Decimal(str(msg.strike_price))

        # Convert nanosecond timestamp to datetime
        expiration = pd.to_datetime(msg.expiration, unit="ns", utc=True).to_pydatetime()

        return cls(
            instrument_id=msg.instrument_id,
            raw_symbol=msg.raw_symbol,
            underlying=msg.underlying,
            option_type=OptionType(msg.instrument_class),
            strike_price=strike,
            expiration=expiration,
            multiplier=100,  # Standard option multiplier
        )


@dataclass(frozen=True)
class OptionQuote:
    """Real-time option quote from mbp-1 schema."""

    instrument_id: int
    timestamp: datetime
    bid_price: Decimal
    ask_price: Decimal
    bid_size: int
    ask_size: int

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid-point price."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_pct(self) -> Decimal:
        """Spread as percentage of mid price."""
        if self.mid_price == 0:
            return Decimal("inf")
        return (self.spread / self.mid_price) * 100

    @classmethod
    def from_databento(cls, msg: dict) -> "OptionQuote":
        """Create from Databento mbp-1 message."""
        # Prices are in 1e-9 dollars
        scale = Decimal("1e-9")

        level = msg["levels"][0]  # Top of book
        return cls(
            instrument_id=msg["instrument_id"],
            timestamp=pd.to_datetime(msg["ts_event"], unit="ns", utc=True).to_pydatetime(),
            bid_price=Decimal(str(level["bid_px"])) * scale,
            ask_price=Decimal(str(level["ask_px"])) * scale,
            bid_size=level["bid_sz"],
            ask_size=level["ask_sz"],
        )


@dataclass(frozen=True)
class UnderlyingPrice:
    """Underlying equity price data."""

    symbol: str
    timestamp: datetime
    last_price: Decimal
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    volume: Optional[int] = None

    @classmethod
    def from_databento_trade(cls, msg, symbol: str) -> "UnderlyingPrice":
        """Create from Databento trade message."""
        scale = Decimal("1e-9")
        return cls(
            symbol=symbol,
            timestamp=pd.to_datetime(msg.ts_event, unit="ns", utc=True).to_pydatetime(),
            last_price=Decimal(str(msg.price)) * scale,
            volume=getattr(msg, "size", None),
        )


@dataclass
class OptionChain:
    """Complete option chain for a given underlying and expiration."""

    underlying: str
    expiration: datetime
    spot_price: Decimal
    timestamp: datetime
    calls: List[OptionQuote]
    puts: List[OptionQuote]

    def get_atm_strike(self) -> Decimal:
        """Find the at-the-money strike."""
        all_strikes = set()
        for opt in self.calls + self.puts:
            # Need instrument definitions to get strikes
            pass

        # Find closest strike to spot
        return min(all_strikes, key=lambda x: abs(x - self.spot_price))

    def filter_by_delta(self, target_delta: float, option_type: OptionType) -> List[OptionQuote]:
        """Filter options by target delta (requires Greeks calculation)."""
        # This will integrate with our Greeks module
        pass


@dataclass(frozen=True)
class DataQuality:
    """Data quality metrics for validation."""

    symbol: str
    timestamp: datetime
    bid_ask_spread_ok: bool
    sufficient_liquidity: bool
    data_staleness_seconds: float
    confidence_score: float  # 0-1

    @property
    def is_tradeable(self) -> bool:
        """Check if data quality is sufficient for trading."""
        return (
            self.bid_ask_spread_ok
            and self.sufficient_liquidity
            and self.data_staleness_seconds < 60
            and self.confidence_score > 0.8
        )
