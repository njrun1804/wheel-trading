"""Options mathematics with self-validation."""

# Temporary import from deprecated file until migration is complete
from .options_deprecated import (
    CalculationResult,
    black_scholes_price_validated,
    calculate_all_greeks,
    implied_volatility_validated,
    probability_itm_validated,
)

__all__ = [
    "CalculationResult",
    "black_scholes_price_validated",
    "calculate_all_greeks",
    "implied_volatility_validated",
    "probability_itm_validated",
]
