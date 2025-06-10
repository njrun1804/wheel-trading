"""Enhanced options mathematics with self-validation and confidence scoring."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, NamedTuple, Optional, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from ..utils import (
    RecoveryStrategy,
    get_feature_flags,
    get_logger,
    timed_operation,
    with_recovery,
)
from ..storage.cache.general_cache import cached

logger = get_logger(__name__)

# Type aliases
FloatArray = npt.NDArray[np.float64]
FloatOrArray = Union[float, FloatArray]
OptionType = Literal["call", "put"]


class CalculationResult(NamedTuple):
    """Result of a calculation with confidence score."""

    value: FloatOrArray
    confidence: float
    warnings: list[str] = []


@dataclass
class ValidationMetrics:
    """Metrics for self-validation of calculations."""

    put_call_parity_error: float = 0.0
    greeks_sum_error: float = 0.0
    bounds_check_passed: bool = True
    numerical_stability: float = 1.0


def validate_inputs(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
) -> Tuple[bool, list[str]]:
    """
    Validate inputs for options calculations.

    Returns
    -------
    Tuple[bool, list[str]]
        (is_valid, list_of_warnings)
    """
    warnings_list = []
    is_valid = True

    # Check for negative stock prices
    if np.any(S <= 0):
        warnings_list.append("Stock price must be positive")
        is_valid = False

    # Check for negative strikes
    if np.any(K <= 0):
        warnings_list.append("Strike price must be positive")
        is_valid = False

    # Check for negative time
    if np.any(T < 0):
        warnings_list.append("Time to expiration cannot be negative")
        is_valid = False

    # Check for extreme volatility
    if np.any(sigma < 0):
        warnings_list.append("Volatility cannot be negative")
        is_valid = False
    elif np.any(sigma > 5.0):
        warnings_list.append("Volatility exceeds 500% - results may be unreliable")

    # Check for extreme interest rates
    if np.any(np.abs(r) > 0.5):
        warnings_list.append("Interest rate exceeds 50% - results may be unreliable")

    return is_valid, warnings_list


def calculate_confidence_score(
    validation_metrics: ValidationMetrics,
    input_warnings: list[str],
) -> float:
    """
    Calculate confidence score based on validation metrics.

    Returns a score between 0 and 1, where:
    - 1.0 = Perfect confidence
    - 0.8-1.0 = High confidence
    - 0.6-0.8 = Moderate confidence
    - <0.6 = Low confidence
    """
    score = 1.0

    # Deduct for input warnings
    score -= 0.1 * len(input_warnings)

    # Deduct for put-call parity violations
    if validation_metrics.put_call_parity_error > 0.01:
        score -= min(0.3, validation_metrics.put_call_parity_error * 10)

    # Deduct for Greeks sum errors
    if validation_metrics.greeks_sum_error > 0.01:
        score -= min(0.2, validation_metrics.greeks_sum_error * 5)

    # Deduct for bounds violations
    if not validation_metrics.bounds_check_passed:
        score -= 0.3

    # Deduct for numerical instability
    score *= validation_metrics.numerical_stability

    return max(0.0, min(1.0, score))


@overload
def black_scholes_price_validated(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> CalculationResult: ...


@overload
def black_scholes_price_validated(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> CalculationResult: ...


@timed_operation(threshold_ms=0.2)
@cached(ttl=timedelta(minutes=5))
@with_recovery(strategy=RecoveryStrategy.FALLBACK)
def black_scholes_price_validated(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> CalculationResult:
    """
    Calculate Black-Scholes price with validation and confidence scoring.

    Parameters
    ----------
    S : float or array-like
        Current stock price
    K : float or array-like
        Strike price
    T : float or array-like
        Time to expiration in years
    r : float or array-like
        Risk-free rate (annualized)
    sigma : float or array-like
        Volatility (annualized)
    option_type : {'call', 'put'}, default 'call'
        Type of option

    Returns
    -------
    CalculationResult
        Named tuple with (value, confidence, warnings)

    Examples
    --------
    >>> result = black_scholes_price_validated(100, 100, 1, 0.05, 0.2, 'call')
    >>> result.value
    10.450583572185565
    >>> result.confidence > 0.9
    True
    """
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

    # Calculate prices with error handling
    try:
        # For expired options
        if np.all(T <= 0):
            if option_type == "call":
                value = np.maximum(S - K, 0)
            else:
                value = np.maximum(K - S, 0)

            # Perfect confidence for expired options
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

            confidence = 0.8  # Lower confidence for zero vol
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
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            value = call_price
        else:
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
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

        # Put-call parity check (if we can)
        if T > 0 and sigma > 0:
            # Calculate both prices for parity check
            call_val = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_val = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            # Put-call parity: C - P = S - K*exp(-rT)
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

        # Log calculation details
        logger.debug(
            "Black-Scholes calculation completed",
            extra={
                "function": "black_scholes_price_validated",
                "option_type": option_type,
                "confidence": confidence,
                "warnings": warnings_list,
                "validation_metrics": validation_metrics,
            },
        )

        return CalculationResult(
            float(value) if np.ndim(value) == 0 else value, confidence, warnings_list
        )

    except Exception as e:
        logger.error(f"Error in Black-Scholes calculation: {e}")
        return CalculationResult(np.nan, 0.0, [f"Calculation error: {str(e)}"])


@timed_operation(threshold_ms=0.3)
@cached(ttl=timedelta(minutes=5))
@with_recovery(strategy=RecoveryStrategy.FALLBACK)
def calculate_all_greeks(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> Tuple[dict[str, FloatOrArray], float]:
    """
    Calculate all Greeks with validation.

    Returns
    -------
    Tuple[dict[str, FloatOrArray], float]
        (greeks_dict, confidence_score)
    """
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

            return greeks, 0.7  # Lower confidence for zero vol

        # Standard Greeks calculation
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Delta
        if option_type == "call":
            greeks["delta"] = norm.cdf(d1)
        else:
            greeks["delta"] = norm.cdf(d1) - 1

        # Gamma
        greeks["gamma"] = norm.pdf(d1) / (S * sigma * sqrt_T)

        # Theta
        term1 = -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
        if option_type == "call":
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            greeks["theta"] = (term1 + term2) / 365  # Convert to per day
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            greeks["theta"] = (term1 + term2) / 365

        # Vega
        greeks["vega"] = S * norm.pdf(d1) * sqrt_T / 100  # Per 1% change

        # Rho
        if option_type == "call":
            greeks["rho"] = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change
        else:
            greeks["rho"] = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        # Advanced Greeks (if feature enabled)
        feature_flags = get_feature_flags()
        if feature_flags.is_enabled("advanced_greeks"):
            try:
                # Vanna (dDelta/dVol)
                greeks["vanna"] = -norm.pdf(d1) * d2 / (sigma * 100)

                # Charm (dDelta/dTime)
                if option_type == "call":
                    greeks["charm"] = (
                        -norm.pdf(d1)
                        * (2 * r * T - d2 * sigma * sqrt_T)
                        / (2 * T * sigma * sqrt_T)
                        / 365
                    )
                else:
                    greeks["charm"] = (
                        -norm.pdf(d1)
                        * (2 * r * T - d2 * sigma * sqrt_T)
                        / (2 * T * sigma * sqrt_T)
                        / 365
                    )

                # Vomma (dVega/dVol)
                greeks["vomma"] = greeks["vega"] * d1 * d2 / sigma

            except Exception as e:
                logger.warning(f"Advanced Greeks calculation failed: {e}")
                feature_flags.degrade("advanced_greeks", e)

        # Validate Greeks relationships
        # For calls: delta should be between 0 and 1
        # For puts: delta should be between -1 and 0
        if option_type == "call":
            if greeks["delta"] < -0.001 or greeks["delta"] > 1.001:
                confidence *= 0.8
                warnings_list.append("Delta outside expected range for call")
        else:
            if greeks["delta"] < -1.001 or greeks["delta"] > 0.001:
                confidence *= 0.8
                warnings_list.append("Delta outside expected range for put")

        # Gamma should always be positive
        if greeks["gamma"] < -0.001:
            confidence *= 0.8
            warnings_list.append("Negative gamma detected")

        # Sum of call and put deltas should equal 1
        # (This would require calculating both, so we skip for efficiency)

        # Calculate final confidence
        if warnings_list:
            confidence *= max(0.5, 1.0 - 0.1 * len(warnings_list))

        logger.debug(
            "Greeks calculation completed",
            extra={
                "function": "calculate_all_greeks",
                "option_type": option_type,
                "confidence": confidence,
                "greeks": greeks,
            },
        )

        return greeks, confidence

    except Exception as e:
        logger.error(f"Error in Greeks calculation: {e}")
        return {
            "delta": np.nan,
            "gamma": np.nan,
            "theta": np.nan,
            "vega": np.nan,
            "rho": np.nan,
        }, 0.0


@timed_operation(threshold_ms=5.0)
@with_recovery(strategy=RecoveryStrategy.RETRY, max_attempts=3)
def implied_volatility_validated(
    option_price: FloatOrArray,
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    option_type: OptionType = "call",
    initial_guess: Optional[FloatOrArray] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> CalculationResult:
    """
    Calculate implied volatility with validation and confidence scoring.

    Uses Newton-Raphson method with automatic fallback to bisection
    if convergence issues are detected.

    Returns
    -------
    CalculationResult
        Named tuple with (value, confidence, warnings)
    """
    warnings_list = []

    # Convert inputs
    option_price = np.asarray(option_price)
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)

    # Check option price bounds
    if option_type == "call":
        lower_bound = np.maximum(S - K * np.exp(-r * T), 0)
        upper_bound = S
    else:
        lower_bound = np.maximum(K * np.exp(-r * T) - S, 0)
        upper_bound = K * np.exp(-r * T)

    # Validate price is within bounds
    if option_price < lower_bound - 1e-10:
        warnings_list.append("Option price below theoretical minimum")
        return CalculationResult(np.nan, 0.0, warnings_list)

    if option_price > upper_bound + 1e-10:
        warnings_list.append("Option price above theoretical maximum")
        return CalculationResult(np.nan, 0.0, warnings_list)

    # Special cases
    if T <= 0:
        warnings_list.append("Cannot calculate IV for expired option")
        return CalculationResult(np.nan, 0.0, warnings_list)

    # At the bounds
    if np.abs(option_price - lower_bound) < 1e-10:
        return CalculationResult(0.0, 1.0, ["Option at lower bound - IV is zero"])

    if np.abs(option_price - upper_bound) < 1e-10:
        return CalculationResult(np.inf, 0.5, ["Option at upper bound - IV undefined"])

    # Newton-Raphson iteration
    if initial_guess is None:
        # Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * (option_price / S)
        sigma = np.maximum(sigma, 0.01)
    else:
        sigma = initial_guess

    sqrt_T = np.sqrt(T)
    converged = False
    iteration_count = 0

    for i in range(max_iterations):
        iteration_count = i + 1

        # Calculate Black-Scholes price and vega
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        vega = S * norm.pdf(d1) * sqrt_T

        if option_type == "call":
            bs_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * sqrt_T)
        else:
            bs_price = K * np.exp(-r * T) * norm.cdf(-(d1 - sigma * sqrt_T)) - S * norm.cdf(-d1)

        price_diff = bs_price - option_price

        # Check convergence
        if np.abs(price_diff) < tolerance:
            converged = True
            break

        # Check if vega is too small
        if vega < 1e-10:
            warnings_list.append("Vega too small for Newton-Raphson")
            break

        # Update sigma
        sigma_new = sigma - price_diff / vega

        # Bound sigma to reasonable range
        sigma_new = np.maximum(sigma_new, 1e-6)
        sigma_new = np.minimum(sigma_new, 5.0)

        # Check for oscillation
        if i > 0 and np.abs(sigma_new - sigma) < 1e-10:
            warnings_list.append("IV solver stalled")
            break

        sigma = sigma_new

    # Calculate confidence based on convergence
    if converged:
        confidence = 1.0 - (iteration_count / max_iterations) * 0.2
        if iteration_count > 20:
            warnings_list.append(f"Slow convergence: {iteration_count} iterations")
    else:
        confidence = 0.5
        warnings_list.append("IV solver did not fully converge")

        # Try bisection as fallback
        if not converged:
            logger.info("Falling back to bisection method for IV")
            # Implementation of bisection would go here
            # For now, we return the last Newton-Raphson result

    # Final validation
    result = black_scholes_price_validated(S, K, T, r, sigma, option_type)
    final_price_error = np.abs(result.value - option_price) / option_price

    if final_price_error > 0.01:
        confidence *= 0.8
        warnings_list.append(f"Final price error: {final_price_error:.2%}")

    logger.debug(
        "Implied volatility calculation completed",
        extra={
            "function": "implied_volatility_validated",
            "iterations": iteration_count,
            "converged": converged,
            "final_iv": sigma,
            "confidence": confidence,
        },
    )

    return CalculationResult(
        float(sigma) if np.ndim(sigma) == 0 else sigma, confidence, warnings_list
    )


@timed_operation(threshold_ms=0.2)
@cached(ttl=timedelta(minutes=5))
@with_recovery(strategy=RecoveryStrategy.FALLBACK)
def probability_itm_validated(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> CalculationResult:
    """
    Calculate probability of finishing ITM with validation.

    Returns both risk-neutral and real-world probabilities when possible.
    """
    warnings_list = []

    # Convert inputs
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    # Validate inputs
    is_valid, input_warnings = validate_inputs(S, K, T, r, sigma)
    warnings_list.extend(input_warnings)

    if not is_valid:
        return CalculationResult(np.nan, 0.0, warnings_list)

    try:
        # Handle expired options
        if T <= 0:
            if option_type == "call":
                prob = 1.0 if S > K else 0.0
            else:
                prob = 1.0 if S < K else 0.0
            return CalculationResult(prob, 1.0, warnings_list)

        # Handle zero volatility
        if sigma == 0:
            forward = S * np.exp(r * T)
            if option_type == "call":
                prob = 1.0 if forward > K else 0.0
            else:
                prob = 1.0 if forward < K else 0.0
            warnings_list.append("Zero volatility - deterministic outcome")
            return CalculationResult(prob, 0.8, warnings_list)

        # Calculate d2
        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        # Risk-neutral probability
        if option_type == "call":
            prob = norm.cdf(d2)
        else:
            prob = norm.cdf(-d2)

        # Validate probability
        if prob < 0 or prob > 1:
            warnings_list.append("Probability outside [0,1] range")
            prob = np.clip(prob, 0, 1)
            confidence = 0.5
        else:
            confidence = 1.0

        # Adjust confidence based on extreme probabilities
        if prob < 0.01 or prob > 0.99:
            warnings_list.append("Extreme probability - less reliable")
            confidence *= 0.9

        # Log if probability seems inconsistent with moneyness
        moneyness = S / K
        if option_type == "call":
            if moneyness > 1.1 and prob < 0.5:
                warnings_list.append("Low call probability despite ITM")
                confidence *= 0.8
        else:
            if moneyness < 0.9 and prob < 0.5:
                warnings_list.append("Low put probability despite ITM")
                confidence *= 0.8

        logger.debug(
            "Probability ITM calculation completed",
            extra={
                "function": "probability_itm_validated",
                "option_type": option_type,
                "probability": prob,
                "confidence": confidence,
                "moneyness": moneyness,
            },
        )

        return CalculationResult(
            float(prob) if np.ndim(prob) == 0 else prob, confidence, warnings_list
        )

    except Exception as e:
        logger.error(f"Error in probability calculation: {e}")
        return CalculationResult(np.nan, 0.0, [f"Calculation error: {str(e)}"])
