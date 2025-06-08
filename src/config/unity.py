"""Unity (U) specific configuration and constants."""

from __future__ import annotations

from decimal import Decimal
from typing import Final

# Unity Software Inc. specific parameters
TICKER: Final[str] = "U"
COMPANY_NAME: Final[str] = "Unity Software Inc."

# Risk parameters for objective function: CAGR - 0.20 × |CVaR₉₅|
CVAR_PERCENTILE: Final[Decimal] = Decimal("0.95")
OBJECTIVE_RISK_WEIGHT: Final[Decimal] = Decimal("0.20")

# Position sizing
KELLY_FRACTION: Final[Decimal] = Decimal("0.5")  # Half-Kelly as specified

# Unity-specific trading characteristics (based on historical data)
# These would be calibrated from actual market data
TYPICAL_IV_RANGE: tuple[Decimal, Decimal] = (Decimal("0.40"), Decimal("0.90"))
AVERAGE_IV: Final[Decimal] = Decimal("0.65")  # Unity tends to have higher vol

# Liquidity constraints
MIN_BID_ASK_SPREAD: Final[Decimal] = Decimal("0.10")  # Allow $0.10 spread for Unity options
MIN_OPEN_INTEREST: Final[int] = 50
MIN_VOLUME: Final[int] = 10

# Strike selection parameters
STRIKE_INTERVALS: Final[list[Decimal]] = [
    Decimal("1.0"),   # For prices < $50
    Decimal("2.5"),   # For prices $50-$100  
    Decimal("5.0"),   # For prices > $100
]

# Market hours (Eastern Time)
MARKET_OPEN: Final[str] = "09:30:00"
MARKET_CLOSE: Final[str] = "16:00:00"

# Options expiration characteristics
WEEKLY_EXPIRATIONS: Final[bool] = True  # Unity has weeklies
PREFERRED_DTE_RANGE: tuple[int, int] = (30, 60)  # 30-60 days preferred

# Risk limits specific to Unity's volatility profile
MAX_DELTA_SHORT_PUT: Final[Decimal] = Decimal("0.35")  # More conservative due to vol
MAX_CONTRACTS_PER_TRADE: Final[int] = 10
MAX_NOTIONAL_PERCENT: Final[Decimal] = Decimal("0.25")  # Max 25% of portfolio

# Data quality thresholds
STALE_DATA_SECONDS: Final[int] = 30
MIN_CONFIDENCE_SCORE: Final[Decimal] = Decimal("0.80")

def validate_unity_strike(strike: Decimal, current_price: Decimal) -> bool:
    """Validate if a strike price is reasonable for Unity."""
    # Strikes should be within 50% of current price for liquidity
    lower_bound = current_price * Decimal("0.5")
    upper_bound = current_price * Decimal("1.5")
    
    return lower_bound <= strike <= upper_bound


def get_strike_interval(price: Decimal) -> Decimal:
    """Get appropriate strike interval for Unity based on price."""
    if price < 50:
        return Decimal("1.0")
    elif price < 100:
        return Decimal("2.5")
    else:
        return Decimal("5.0")


def is_market_hours() -> bool:
    """Check if currently in market hours (ET)."""
    from datetime import datetime
    import pytz
    
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    
    if now.weekday() > 4:  # Weekend
        return False
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close