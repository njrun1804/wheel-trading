"""Options mathematics with self-validation and GPU acceleration."""

from .options import (
    CalculationResult,
    black_scholes_price_validated,
    calculate_all_greeks,
    implied_volatility_validated,
    probability_itm_validated,
)

# GPU-accelerated versions
try:
    from .options_gpu import (
        TORCH_AVAILABLE,
        black_scholes,
        black_scholes_numexpr,
        black_scholes_torch,
        calculate_greeks_gpu,
    )

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    TORCH_AVAILABLE = False
    black_scholes = black_scholes_price_validated
    black_scholes_numexpr = None
    black_scholes_torch = None
    calculate_greeks_gpu = None

# Enhanced versions if available
try:
    from .options_enhanced import BlackScholesEnhanced, black_scholes_price_enhanced

    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    BlackScholesEnhanced = None
    black_scholes_price_enhanced = None

__all__ = [
    "CalculationResult",
    "black_scholes_price_validated",
    "calculate_all_greeks",
    "implied_volatility_validated",
    "probability_itm_validated",
    # GPU versions
    "black_scholes",
    "black_scholes_numexpr",
    "black_scholes_torch",
    "calculate_greeks_gpu",
    "GPU_AVAILABLE",
    "TORCH_AVAILABLE",
    # Enhanced versions
    "BlackScholesEnhanced",
    "black_scholes_price_enhanced",
    "ENHANCED_AVAILABLE",
]
