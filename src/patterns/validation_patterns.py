"""Validation patterns for data integrity and type safety.

This module demonstrates the standard validation patterns used throughout
the codebase. Codex should follow these patterns for consistency.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field, validator

from ..utils.logging import get_logger
from ..utils.validate import die, validate_positive

logger = get_logger(__name__)

T = TypeVar("T")


# PATTERN 1: Pydantic models for complex validation
class MarketDataModel(BaseModel):
    """
    Use Pydantic for complex data structures requiring validation.

    CODEX PATTERN:
    1. Define all fields with types
    2. Use Field() for constraints
    3. Add validators for cross-field logic
    4. Provide clear error messages
    """

    ticker: str = Field(..., min_length=1, max_length=10)
    price: float = Field(..., gt=0, description="Current market price")
    volume: int = Field(0, ge=0, description="Trading volume")
    volatility: Optional[float] = Field(None, ge=0, le=5.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator("ticker")
    def validate_ticker(cls, v):
        """Ensure ticker is uppercase and alphanumeric."""
        if not v.replace(".", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid ticker format: {v}")
        return v.upper()

    @validator("volatility")
    def validate_volatility(cls, v, values):
        """Apply reasonable volatility bounds based on price."""
        if v is not None:
            price = values.get("price", 0)
            if price < 10 and v > 2.0:
                logger.warning(
                    "High volatility for low-priced stock", extra={"price": price, "volatility": v}
                )
        return v

    class Config:
        # Enable assignment validation
        validate_assignment = True
        # Use Enum values
        use_enum_values = True
        # Custom JSON encoders
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
        }


# PATTERN 2: Input validation with early returns
def validate_trade_inputs(
    portfolio_value: float,
    position_size: float,
    max_position_pct: float = 0.20,
) -> Optional[str]:
    """
    Validate inputs with early returns for clarity.

    CODEX PATTERN:
    1. Check each constraint separately
    2. Return specific error messages
    3. Use None for success (no error)
    4. Log validation failures
    """
    # Basic type checks
    if not isinstance(portfolio_value, (int, float)):
        return "Portfolio value must be numeric"

    if not isinstance(position_size, (int, float)):
        return "Position size must be numeric"

    # Value constraints
    if portfolio_value <= 0:
        return f"Portfolio value must be positive, got {portfolio_value}"

    if position_size <= 0:
        return f"Position size must be positive, got {position_size}"

    # Business logic constraints
    position_pct = position_size / portfolio_value
    if position_pct > max_position_pct:
        return (
            f"Position size {position_size:,.2f} exceeds "
            f"{max_position_pct:.0%} limit of {portfolio_value:,.2f}"
        )

    # All checks passed
    return None


# PATTERN 3: Validation with die() for critical paths
def validate_option_data(option_data: dict) -> dict:
    """
    Use die() for data that MUST be present and valid.

    CODEX PATTERN:
    1. die() for required fields
    2. Validate and transform in one pass
    3. Return cleaned data
    """
    # Required fields - will raise if missing
    strike = die(option_data.get("strike"), "Strike price required")
    expiry = die(option_data.get("expiry"), "Expiration date required")
    option_type = die(option_data.get("type"), "Option type required")

    # Validate numeric fields
    strike = die(validate_positive(float(strike)), f"Strike must be positive, got {strike}")

    # Validate option type
    option_type = option_type.lower()
    die(option_type in ["call", "put"], f"Option type must be 'call' or 'put', got {option_type}")

    # Optional fields with defaults and validation
    bid = float(option_data.get("bid", 0))
    ask = float(option_data.get("ask", 0))

    # Validate bid/ask spread
    if bid > 0 and ask > 0 and bid > ask:
        logger.warning("Inverted bid/ask spread", extra={"bid": bid, "ask": ask, "strike": strike})
        # Swap them
        bid, ask = ask, bid

    return {
        "strike": strike,
        "expiry": expiry,
        "type": option_type,
        "bid": max(0, bid),  # Ensure non-negative
        "ask": max(0, ask),
        "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else 0,
    }


# PATTERN 4: Array validation for numerical computations
def validate_price_series(
    prices: Union[List[float], np.ndarray],
    min_length: int = 20,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Validate arrays for numerical computation.

    CODEX PATTERN:
    1. Convert to numpy array
    2. Check shape and size
    3. Validate numeric properties
    4. Return clean array
    """
    # Convert to numpy array
    prices = np.asarray(prices)

    # Check dimensions
    if prices.ndim != 1:
        raise ValueError(f"Expected 1D array, got {prices.ndim}D with shape {prices.shape}")

    # Check length
    if len(prices) < min_length:
        raise ValueError(f"Need at least {min_length} prices, got {len(prices)}")

    if max_length and len(prices) > max_length:
        logger.warning(
            f"Truncating price series from {len(prices)} to {max_length}",
            extra={"original_length": len(prices)},
        )
        prices = prices[-max_length:]  # Keep most recent

    # Check for valid values
    if np.any(np.isnan(prices)):
        raise ValueError("Price series contains NaN values")

    if np.any(prices <= 0):
        raise ValueError("Price series contains non-positive values")

    if np.any(np.isinf(prices)):
        raise ValueError("Price series contains infinite values")

    # Check for reasonable values
    if np.max(prices) / np.min(prices) > 100:
        logger.warning(
            "Large price range detected",
            extra={
                "min": np.min(prices),
                "max": np.max(prices),
                "ratio": np.max(prices) / np.min(prices),
            },
        )

    return prices


# PATTERN 5: Nested validation for complex structures
def validate_portfolio_state(portfolio: dict) -> dict:
    """
    Validate complex nested data structures.

    CODEX PATTERN:
    1. Validate top-level structure
    2. Recursively validate nested data
    3. Ensure consistency across fields
    4. Return normalized structure
    """
    # Top-level required fields
    cash = die(portfolio.get("cash"), "Cash balance required")
    positions = die(portfolio.get("positions"), "Positions list required")

    # Validate cash
    cash = validate_positive(float(cash))

    # Validate positions list
    if not isinstance(positions, list):
        raise ValueError(f"Positions must be a list, got {type(positions)}")

    validated_positions = []
    total_value = cash

    for i, pos in enumerate(positions):
        try:
            # Validate each position
            symbol = die(pos.get("symbol"), f"Position {i}: symbol required")
            quantity = die(pos.get("quantity"), f"Position {i}: quantity required")

            quantity = int(quantity)
            if quantity == 0:
                logger.warning(f"Skipping zero quantity position: {symbol}")
                continue

            # Optional fields
            cost_basis = float(pos.get("cost_basis", 0))
            current_price = float(pos.get("current_price", 0))

            # Calculate position value
            position_value = abs(quantity) * current_price if current_price > 0 else 0
            total_value += position_value

            validated_positions.append(
                {
                    "symbol": symbol.upper(),
                    "quantity": quantity,
                    "cost_basis": max(0, cost_basis),
                    "current_price": max(0, current_price),
                    "position_value": position_value,
                    "position_type": "long" if quantity > 0 else "short",
                }
            )

        except Exception as e:
            logger.error(f"Invalid position at index {i}: {e}", extra={"position": pos})
            # Continue processing other positions

    return {
        "cash": cash,
        "positions": validated_positions,
        "total_value": total_value,
        "position_count": len(validated_positions),
        "buying_power": cash,  # Simplified - add margin logic as needed
    }


# PATTERN 6: Time-based validation
def validate_option_expiry(
    expiry: Union[str, datetime],
    min_dte: int = 0,
    max_dte: int = 365,
) -> datetime:
    """
    Validate option expiration dates.

    CODEX PATTERN:
    1. Parse various date formats
    2. Check business logic constraints
    3. Consider trading calendar
    4. Return standardized datetime
    """
    from ..utils.trading_calendar import is_trading_day

    # Parse expiry date
    if isinstance(expiry, str):
        try:
            # Try ISO format first
            expiry_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        except ValueError:
            # Try other common formats
            from dateutil.parser import parse

            expiry_dt = parse(expiry)
    else:
        expiry_dt = expiry

    # Ensure timezone aware
    if expiry_dt.tzinfo is None:
        import pytz

        expiry_dt = pytz.timezone("US/Eastern").localize(expiry_dt)

    # Calculate days to expiry
    now = datetime.now(expiry_dt.tzinfo)
    dte = (expiry_dt - now).days

    # Validate DTE range
    if dte < min_dte:
        raise ValueError(f"Option expired or too close to expiry: {dte} days")

    if dte > max_dte:
        raise ValueError(f"Option expiry too far out: {dte} days (max {max_dte})")

    # Check if expiry is on a valid trading day
    if not is_trading_day(expiry_dt):
        logger.warning(
            "Expiry is not a trading day",
            extra={"expiry": expiry_dt.date(), "weekday": expiry_dt.strftime("%A")},
        )
        # Adjust to previous trading day
        while not is_trading_day(expiry_dt):
            expiry_dt -= timedelta(days=1)

    return expiry_dt


# PATTERN 7: Cross-validation between related fields
def validate_option_chain(chain_data: dict) -> dict:
    """
    Validate relationships between related data fields.

    CODEX PATTERN:
    1. Check individual field validity
    2. Verify relationships between fields
    3. Ensure data consistency
    4. Log anomalies
    """
    # Basic validation
    underlying_price = die(
        validate_positive(chain_data.get("underlying_price")), "Underlying price required"
    )

    calls = chain_data.get("calls", [])
    puts = chain_data.get("puts", [])

    if not calls and not puts:
        raise ValueError("Option chain must have calls or puts")

    # Validate strike alignment
    call_strikes = sorted([c["strike"] for c in calls])
    put_strikes = sorted([p["strike"] for p in puts])

    # Check for reasonable strike range
    min_strike = min(call_strikes + put_strikes)
    max_strike = max(call_strikes + put_strikes)

    if min_strike < underlying_price * 0.5:
        logger.warning(
            "Very low strikes in chain",
            extra={
                "min_strike": min_strike,
                "underlying": underlying_price,
                "ratio": min_strike / underlying_price,
            },
        )

    if max_strike > underlying_price * 2.0:
        logger.warning(
            "Very high strikes in chain",
            extra={
                "max_strike": max_strike,
                "underlying": underlying_price,
                "ratio": max_strike / underlying_price,
            },
        )

    # Validate put-call parity at ATM
    atm_strike = min(call_strikes + put_strikes, key=lambda x: abs(x - underlying_price))

    atm_call = next((c for c in calls if c["strike"] == atm_strike), None)
    atm_put = next((p for p in puts if p["strike"] == atm_strike), None)

    if atm_call and atm_put:
        # Simple put-call parity check
        call_mid = (atm_call["bid"] + atm_call["ask"]) / 2
        put_mid = (atm_put["bid"] + atm_put["ask"]) / 2

        # C - P ≈ S - K (ignoring interest for simplicity)
        parity_diff = abs((call_mid - put_mid) - (underlying_price - atm_strike))

        if parity_diff > underlying_price * 0.02:  # 2% threshold
            logger.warning(
                "Put-call parity violation detected",
                extra={
                    "strike": atm_strike,
                    "call_mid": call_mid,
                    "put_mid": put_mid,
                    "parity_diff": parity_diff,
                },
            )

    return {
        "underlying_price": underlying_price,
        "calls": sorted(calls, key=lambda x: x["strike"]),
        "puts": sorted(puts, key=lambda x: x["strike"]),
        "strike_range": (min_strike, max_strike),
        "atm_strike": atm_strike,
        "timestamp": chain_data.get("timestamp", datetime.now()),
    }
