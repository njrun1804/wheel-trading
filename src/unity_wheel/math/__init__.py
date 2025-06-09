"""Options mathematics with self-validation."""

from .options import (
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
