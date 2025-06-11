"""Enhanced options mathematics with advanced performance caching.

This module provides performance-optimized versions of key options calculations
using the new MemoryAwareLRUCache system for significant speed improvements.
"""

import logging
import time
from typing import Dict, Tuple

import numpy as np

from src.unity_wheel.math.options import (
    CalculationResult,
    FloatOrArray,
    OptionType,
    ValidationMetrics,
    calculate_confidence_score,
    norm_cdf_cached,
    validate_inputs,
)
from src.unity_wheel.utils.logging import StructuredLogger
from src.unity_wheel.utils.performance_cache import cache_key_for_options, cached, options_cache

logger = StructuredLogger(logging.getLogger(__name__))


@cached(
    cache_name="options",
    ttl_seconds=3600,  # 1 hour TTL
    key_func=lambda S, K, T, r, sigma, option_type="call": cache_key_for_options(
        float(np.asarray(S).flat[0]) if hasattr(S, "flat") else float(S),
        float(np.asarray(K).flat[0]) if hasattr(K, "flat") else float(K),
        float(np.asarray(T).flat[0]) if hasattr(T, "flat") else float(T),
        float(np.asarray(r).flat[0]) if hasattr(r, "flat") else float(r),
        float(np.asarray(sigma).flat[0]) if hasattr(sigma, "flat") else float(sigma),
        option_type,
    ),
)
def black_scholes_price_enhanced(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> CalculationResult:
    """
    Enhanced Black-Scholes price calculation with advanced caching.

    This function provides the same functionality as black_scholes_price_validated
    but with enhanced performance through the new caching system.
    """
    start_time = time.time()

    # Convert inputs
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    # Validate inputs
    is_valid, warnings_list = validate_inputs(S, K, T, r, sigma)
    if (
        not is_valid
        and len(warnings_list) > 0
        and "Stock price must be positive" in warnings_list[0]
    ):
        return CalculationResult(np.nan, 0.0, warnings_list)

    # Initialize validation metrics
    validation_metrics = ValidationMetrics()

    try:
        # For expired options
        if np.all(T <= 0):
            if option_type == "call":
                value = np.maximum(S - K, 0)
            else:
                value = np.maximum(K - S, 0)

            confidence = 1.0 if not warnings_list else 0.9
            return CalculationResult(
                float(value) if np.ndim(value) == 0 else value, confidence, warnings_list
            )

        # Calculate d1 and d2
        sqrt_T = np.sqrt(np.maximum(T, 0))

        # Handle zero volatility
        if np.all(sigma == 0):
            if option_type == "call":
                value = np.maximum(S - K * np.exp(-r * T), 0)
            else:
                value = np.maximum(K * np.exp(-r * T) - S, 0)

            confidence = 0.8
            warnings_list.append("Zero volatility - using intrinsic value")
            return CalculationResult(
                float(value) if np.ndim(value) == 0 else value, confidence, warnings_list
            )

        # Standard Black-Scholes calculation
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T

        # Handle numerical issues
        if np.any(~np.isfinite(d1)) or np.any(~np.isfinite(d2)):
            validation_metrics.numerical_stability = 0.5
            warnings_list.append("Numerical instability detected in d1/d2 calculation")

        # Calculate prices
        if option_type == "call":
            call_price = S * norm_cdf_cached(d1) - K * np.exp(-r * T) * norm_cdf_cached(d2)
            value = call_price
        else:
            put_price = K * np.exp(-r * T) * norm_cdf_cached(-d2) - S * norm_cdf_cached(-d1)
            value = put_price

        # Validate bounds
        if option_type == "call":
            lower_bound = np.maximum(S - K * np.exp(-r * T), 0)
            upper_bound = S
        else:
            lower_bound = np.maximum(K * np.exp(-r * T) - S, 0)
            upper_bound = K * np.exp(-r * T)

        if np.any(value < lower_bound - 1e-10) or np.any(value > upper_bound + 1e-10):
            validation_metrics.bounds_check_passed = False
            warnings_list.append("Price violates theoretical bounds")

        # Put-call parity check
        if T > 0 and sigma > 0:
            call_val = S * norm_cdf_cached(d1) - K * np.exp(-r * T) * norm_cdf_cached(d2)
            put_val = K * np.exp(-r * T) * norm_cdf_cached(-d2) - S * norm_cdf_cached(-d1)

            parity_lhs = call_val - put_val
            parity_rhs = S - K * np.exp(-r * T)
            parity_error = np.abs(parity_lhs - parity_rhs)

            validation_metrics.put_call_parity_error = float(
                np.max(parity_error / np.maximum(S, 1e-10))
            )

            if validation_metrics.put_call_parity_error > 0.001:
                warnings_list.append(
                    f"Put-call parity violation: {validation_metrics.put_call_parity_error:.4f}"
                )

        # Calculate confidence score
        confidence = calculate_confidence_score(validation_metrics, warnings_list)

        # Log performance metrics
        computation_time_ms = (time.time() - start_time) * 1000
        logger.debug(
            "enhanced_black_scholes_completed",
            extra={
                "option_type": option_type,
                "confidence": confidence,
                "computation_time_ms": computation_time_ms,
                "cache_used": True,
                "warnings_count": len(warnings_list),
            },
        )

        return CalculationResult(
            float(value) if np.ndim(value) == 0 else value, confidence, warnings_list
        )

    except Exception as e:
        logger.error(
            "enhanced_black_scholes_error", extra={"error": str(e), "option_type": option_type}
        )
        return CalculationResult(np.nan, 0.0, [f"Calculation error: {str(e)}"])


@cached(
    cache_name="options",
    ttl_seconds=1800,  # 30 minutes TTL for Greeks
    key_func=lambda S, K, T, r, sigma, option_type="call": f"greeks_{cache_key_for_options(float(S), float(K), float(T), float(r), float(sigma), option_type)}",
)
def calculate_all_greeks_enhanced(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> Tuple[Dict[str, FloatOrArray], float]:
    """
    Enhanced Greeks calculation with advanced caching.

    Provides significant performance improvements for repeated calculations
    with the same parameters.
    """
    start_time = time.time()

    # Convert inputs
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    # Validate inputs
    is_valid, warnings_list = validate_inputs(S, K, T, r, sigma)

    greeks = {}
    confidence = 1.0

    try:
        # For expired options
        if T <= 0:
            if option_type == "call":
                greeks["delta"] = 1.0 if S > K else 0.0
            else:
                greeks["delta"] = -1.0 if S < K else 0.0

            greeks["gamma"] = 0.0
            greeks["theta"] = 0.0
            greeks["vega"] = 0.0
            greeks["rho"] = 0.0

            return greeks, 0.9 if not warnings_list else 0.8

        # Calculate d1 and d2
        sqrt_T = np.sqrt(T)

        # Handle zero volatility
        if sigma == 0:
            forward = S / (K * np.exp(-r * T))
            if option_type == "call":
                greeks["delta"] = 1.0 if forward > 1 else 0.0
            else:
                greeks["delta"] = -1.0 if forward < 1 else 0.0

            greeks["gamma"] = 0.0
            greeks["theta"] = 0.0
            greeks["vega"] = 0.0
            greeks["rho"] = K * T * np.exp(-r * T) if forward < 1 else 0.0

            return greeks, 0.7

        # Standard Greeks calculation
        from scipy.stats import norm

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Delta
        if option_type == "call":
            greeks["delta"] = norm_cdf_cached(d1)
        else:
            greeks["delta"] = norm_cdf_cached(d1) - 1

        # Gamma
        greeks["gamma"] = norm.pdf(d1) / (S * sigma * sqrt_T)

        # Theta
        term1 = -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
        if option_type == "call":
            term2 = -r * K * np.exp(-r * T) * norm_cdf_cached(d2)
            greeks["theta"] = (term1 + term2) / 365
        else:
            term2 = r * K * np.exp(-r * T) * norm_cdf_cached(-d2)
            greeks["theta"] = (term1 + term2) / 365

        # Vega
        greeks["vega"] = S * norm.pdf(d1) * sqrt_T / 100

        # Rho
        if option_type == "call":
            greeks["rho"] = K * T * np.exp(-r * T) * norm_cdf_cached(d2) / 100
        else:
            greeks["rho"] = -K * T * np.exp(-r * T) * norm_cdf_cached(-d2) / 100

        # Validate Greeks
        if option_type == "call":
            if greeks["delta"] < -0.001 or greeks["delta"] > 1.001:
                confidence *= 0.8
                warnings_list.append("Delta outside expected range for call")
        else:
            if greeks["delta"] < -1.001 or greeks["delta"] > 0.001:
                confidence *= 0.8
                warnings_list.append("Delta outside expected range for put")

        if greeks["gamma"] < -0.001:
            confidence *= 0.8
            warnings_list.append("Negative gamma detected")

        confidence *= 1.0 - 0.1 * len(warnings_list)
        confidence = max(0.0, min(1.0, confidence))

        # Log performance metrics
        computation_time_ms = (time.time() - start_time) * 1000
        logger.debug(
            "enhanced_greeks_completed",
            extra={
                "option_type": option_type,
                "confidence": confidence,
                "computation_time_ms": computation_time_ms,
                "cache_used": True,
            },
        )

        return greeks, confidence

    except Exception as e:
        logger.error("enhanced_greeks_error", extra={"error": str(e), "option_type": option_type})
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }, 0.0


@cached(
    cache_name="options",
    ttl_seconds=1800,
    key_func=lambda option_price, spot_price, strike_price, time_to_expiry, risk_free_rate, option_type="call": f"iv_{option_price:.4f}_{spot_price:.2f}_{strike_price:.2f}_{time_to_expiry:.4f}_{risk_free_rate:.4f}_{option_type}",
)
def implied_volatility_enhanced(
    option_price: float,
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType = "call",
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> CalculationResult:
    """
    Enhanced implied volatility calculation with caching.

    Uses Newton-Raphson method with fallback to bisection for robustness.
    Results are cached to improve performance for repeated calculations.
    """
    start_time = time.time()

    # Validate inputs
    if option_price <= 0:
        return CalculationResult(np.nan, 0.0, ["Option price must be positive"])

    if spot_price <= 0 or strike_price <= 0:
        return CalculationResult(np.nan, 0.0, ["Spot and strike prices must be positive"])

    if time_to_expiry <= 0:
        return CalculationResult(np.nan, 0.0, ["Time to expiry must be positive"])

    # Check bounds
    if option_type == "call":
        intrinsic = max(spot_price - strike_price, 0)
        upper_bound = spot_price
    else:
        intrinsic = max(strike_price - spot_price, 0)
        upper_bound = strike_price

    if option_price < intrinsic:
        return CalculationResult(np.nan, 0.0, ["Option price below intrinsic value"])

    if option_price > upper_bound:
        return CalculationResult(np.nan, 0.0, ["Option price above maximum possible value"])

    # Initial guess based on at-the-money approximation
    moneyness = spot_price / strike_price
    if option_type == "call":
        initial_guess = abs(np.log(moneyness)) / np.sqrt(time_to_expiry) + 0.20
    else:
        initial_guess = abs(np.log(moneyness)) / np.sqrt(time_to_expiry) + 0.20

    initial_guess = max(0.01, min(5.0, initial_guess))  # Bound initial guess

    try:
        # Newton-Raphson method
        sigma = initial_guess
        warnings_list = []

        for i in range(max_iterations):
            # Calculate price and vega
            bs_result = black_scholes_price_enhanced(
                spot_price, strike_price, time_to_expiry, risk_free_rate, sigma, option_type
            )

            if bs_result.confidence < 0.5:
                warnings_list.extend(bs_result.warnings)
                break

            calculated_price = bs_result.value

            # Calculate vega for Newton-Raphson
            greeks, _ = calculate_all_greeks_enhanced(
                spot_price, strike_price, time_to_expiry, risk_free_rate, sigma, option_type
            )
            vega = greeks["vega"] * 100  # Convert back to per unit change

            # Check for convergence
            price_diff = calculated_price - option_price
            if abs(price_diff) < tolerance:
                confidence = 0.95 if not warnings_list else 0.85

                computation_time_ms = (time.time() - start_time) * 1000
                logger.debug(
                    "enhanced_iv_completed",
                    extra={
                        "iterations": i + 1,
                        "final_sigma": sigma,
                        "confidence": confidence,
                        "computation_time_ms": computation_time_ms,
                    },
                )

                return CalculationResult(sigma, confidence, warnings_list)

            # Newton-Raphson update
            if abs(vega) > 1e-10:
                sigma_new = sigma - price_diff / vega
                sigma_new = max(0.001, min(10.0, sigma_new))  # Bound sigma
                sigma = sigma_new
            else:
                warnings_list.append("Low vega - switching to bisection method")
                break

        # Fallback to bisection method if Newton-Raphson fails
        logger.debug("IV calculation falling back to bisection method")

        vol_low, vol_high = 0.001, 5.0

        for i in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2

            bs_result = black_scholes_price_enhanced(
                spot_price, strike_price, time_to_expiry, risk_free_rate, vol_mid, option_type
            )

            if bs_result.confidence < 0.5:
                break

            calculated_price = bs_result.value

            if abs(calculated_price - option_price) < tolerance:
                confidence = 0.80 if not warnings_list else 0.70
                warnings_list.append("Used bisection method fallback")

                computation_time_ms = (time.time() - start_time) * 1000
                logger.debug(
                    "enhanced_iv_bisection_completed",
                    extra={
                        "iterations": i + 1,
                        "final_sigma": vol_mid,
                        "confidence": confidence,
                        "computation_time_ms": computation_time_ms,
                    },
                )

                return CalculationResult(vol_mid, confidence, warnings_list)

            if calculated_price > option_price:
                vol_high = vol_mid
            else:
                vol_low = vol_mid

        # If all methods fail
        warnings_list.append("IV calculation did not converge")
        return CalculationResult(initial_guess, 0.30, warnings_list)

    except Exception as e:
        logger.error("enhanced_iv_error", extra={"error": str(e), "option_type": option_type})
        return CalculationResult(np.nan, 0.0, [f"IV calculation error: {str(e)}"])


async def get_cache_performance_stats() -> Dict:
    """Get performance statistics for the options cache."""
    stats = options_cache.get_stats()

    return {
        "cache_name": "options",
        "hit_rate": stats.hit_rate,
        "total_hits": stats.hits,
        "total_misses": stats.misses,
        "evictions": stats.evictions,
        "memory_usage_mb": stats.memory_bytes / (1024 * 1024),
        "avg_computation_time_ms": stats.avg_computation_time_ms,
        "max_computation_time_ms": stats.max_computation_time_ms,
    }


async def clear_options_cache() -> None:
    """Clear the options cache."""
    await options_cache.clear()
    logger.info("options_cache_cleared")


# Convenience functions that use enhanced versions by default
def get_option_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType = "call",
) -> Tuple[float, float]:
    """
    Get option price with confidence score.

    Returns:
        Tuple of (price, confidence)
    """
    result = black_scholes_price_enhanced(
        spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
    )
    return result.value, result.confidence


def get_option_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType = "call",
) -> Tuple[Dict[str, float], float]:
    """
    Get option Greeks with confidence score.

    Returns:
        Tuple of (greeks_dict, confidence)
    """
    return calculate_all_greeks_enhanced(
        spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
    )


def get_implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType = "call",
) -> Tuple[float, float]:
    """
    Get implied volatility with confidence score.

    Returns:
        Tuple of (implied_vol, confidence)
    """
    result = implied_volatility_enhanced(
        option_price, spot, strike, time_to_expiry, risk_free_rate, option_type
    )
    return result.value, result.confidence
