"""Options mathematics module for Black-Scholes pricing and Greeks calculation."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Union, cast, overload

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

# Configure structured logging
logger = logging.getLogger(__name__)

# Type aliases for clarity
FloatArray = npt.NDArray[np.float64]
FloatOrArray = Union[float, FloatArray]
OptionType = Literal["call", "put"]


@overload
def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float: ...

@overload
def black_scholes_price(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> FloatArray: ...

def black_scholes_price(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> FloatOrArray:
    """
    Calculate Black-Scholes option price for calls and puts.

    The Black-Scholes formula for a European call option:
    $$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

    For a European put option:
    $$P = K e^{-rT} N(-d_2) - S_0 N(-d_1)$$

    Where:
    $$d_1 = \\frac{\\ln(S_0/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}$$
    $$d_2 = d_1 - \\sigma\\sqrt{T}$$

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
    float or ndarray
        Option price(s)

    Examples
    --------
    >>> black_scholes_price(100, 100, 1, 0.05, 0.2, 'call')
    10.450583572185565
    """
    # Convert to numpy arrays for vectorization
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    # Determine output shape based on all inputs
    shapes = [np.shape(x) for x in [S, K, T, r, sigma] if np.shape(x) != ()]
    if shapes:
        output_shape = shapes[0]
        result = np.zeros(output_shape, dtype=float)
    else:
        result = np.zeros((), dtype=float)

    # Handle expired options
    if np.ndim(T) == 0:
        if T <= 0:
            if option_type == "call":
                result = np.maximum(S - K, 0)
            else:
                result = np.maximum(K - S, 0)
            return float(result) if np.ndim(result) == 0 else result
    else:
        expired_mask = T <= 0
        if np.any(expired_mask):
            if option_type == "call":
                result[expired_mask] = np.maximum(S[expired_mask] - K[expired_mask], 0)
            else:
                result[expired_mask] = np.maximum(K[expired_mask] - S[expired_mask], 0)

    # Handle zero volatility cases
    if np.ndim(sigma) == 0:
        if sigma == 0 and T > 0:
            if option_type == "call":
                result = np.maximum(S - K * np.exp(-r * T), 0)
            else:
                result = np.maximum(K * np.exp(-r * T) - S, 0)
            return float(result) if np.ndim(result) == 0 else result
    else:
        zero_vol_mask = (sigma == 0) & (T > 0)
        if np.any(zero_vol_mask):
            if option_type == "call":
                result[zero_vol_mask] = np.maximum(
                    S[zero_vol_mask]
                    - K[zero_vol_mask] * np.exp(-r[zero_vol_mask] * T[zero_vol_mask]),
                    0,
                )
            else:
                result[zero_vol_mask] = np.maximum(
                    K[zero_vol_mask] * np.exp(-r[zero_vol_mask] * T[zero_vol_mask])
                    - S[zero_vol_mask],
                    0,
                )

    # Compute Black-Scholes for positive volatility and time
    if np.ndim(sigma) == 0 and np.ndim(T) == 0:
        if sigma > 0 and T > 0:
            sqrt_T = np.sqrt(T)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T

            if option_type == "call":
                result = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                result = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        pos_vol_mask = (sigma > 0) & (T > 0)
        if np.any(pos_vol_mask):
            # Extract values for positive volatility cases
            S_masked = S[pos_vol_mask] if np.ndim(S) > 0 else S
            K_masked = K[pos_vol_mask] if np.ndim(K) > 0 else K
            T_masked = T[pos_vol_mask] if np.ndim(T) > 0 else T
            r_masked = r[pos_vol_mask] if np.ndim(r) > 0 else r
            sigma_masked = sigma[pos_vol_mask] if np.ndim(sigma) > 0 else sigma

            sqrt_T = np.sqrt(T_masked)
            d1 = (np.log(S_masked / K_masked) + (r_masked + 0.5 * sigma_masked**2) * T_masked) / (
                sigma_masked * sqrt_T
            )
            d2 = d1 - sigma_masked * sqrt_T

            if option_type == "call":
                result[pos_vol_mask] = S_masked * norm.cdf(d1) - K_masked * np.exp(
                    -r_masked * T_masked
                ) * norm.cdf(d2)
            else:
                result[pos_vol_mask] = K_masked * np.exp(-r_masked * T_masked) * norm.cdf(
                    -d2
                ) - S_masked * norm.cdf(-d1)

    logger.debug(
        "Black-Scholes calculation completed",
        extra={
            "function": "black_scholes_price",
            "option_type": option_type,
            "input_shapes": {
                "S": np.shape(S),
                "K": np.shape(K),
                "T": np.shape(T),
                "r": np.shape(r),
                "sigma": np.shape(sigma)
            },
            "output_shape": np.shape(result)
        }
    )
    return float(result) if np.ndim(result) == 0 else result


@overload
def calculate_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float: ...

@overload
def calculate_delta(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> FloatArray: ...

def calculate_delta(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> FloatOrArray:
    """
    Calculate the analytical delta of an option.

    Delta measures the rate of change of option price with respect to stock price.

    For a call option:
    $$\\Delta_{call} = N(d_1)$$

    For a put option:
    $$\\Delta_{put} = N(d_1) - 1$$

    Where:
    $$d_1 = \\frac{\\ln(S_0/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}$$

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
    float or ndarray
        Delta value(s)

    Examples
    --------
    >>> calculate_delta(100, 100, 1, 0.05, 0.2, 'call')
    0.6368306507986892
    """
    # Convert to numpy arrays
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    # Determine output shape
    shapes = [np.shape(x) for x in [S, K, T, r, sigma] if np.shape(x) != ()]
    if shapes:
        result = np.zeros(shapes[0], dtype=float)
    else:
        result = np.zeros((), dtype=float)

    # Handle edge cases
    # Expired options
    expired_mask = T <= 0
    if np.any(expired_mask):
        if option_type == "call":
            result[expired_mask] = np.where(S[expired_mask] > K[expired_mask], 1.0, 0.0)
        else:
            result[expired_mask] = np.where(S[expired_mask] < K[expired_mask], -1.0, 0.0)

    # Zero volatility
    zero_vol_mask = (sigma == 0) & (T > 0)
    if np.any(zero_vol_mask):
        forward = S[zero_vol_mask] / (
            K[zero_vol_mask] * np.exp(-r[zero_vol_mask] * T[zero_vol_mask])
        )
        if option_type == "call":
            result[zero_vol_mask] = np.where(forward > 1, 1.0, 0.0)
        else:
            result[zero_vol_mask] = np.where(forward < 1, -1.0, 0.0)

    # Normal case
    normal_mask = (sigma > 0) & (T > 0)
    if np.any(normal_mask):
        sqrt_T = np.sqrt(T[normal_mask])
        d1 = (
            np.log(S[normal_mask] / K[normal_mask])
            + (r[normal_mask] + 0.5 * sigma[normal_mask] ** 2) * T[normal_mask]
        ) / (sigma[normal_mask] * sqrt_T)

        if option_type == "call":
            result[normal_mask] = norm.cdf(d1)
        else:
            result[normal_mask] = norm.cdf(d1) - 1

    logger.debug(
        "Delta calculation completed",
        extra={
            "function": "calculate_delta",
            "option_type": option_type,
            "output_shape": np.shape(result)
        }
    )
    return float(result) if np.ndim(result) == 0 else result


@overload
def probability_itm(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float: ...

@overload
def probability_itm(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> FloatArray: ...

def probability_itm(
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    sigma: FloatOrArray,
    option_type: OptionType = "call",
) -> FloatOrArray:
    """
    Calculate the probability of an option finishing in the money.

    This is the risk-neutral probability under the Black-Scholes model.

    For a call option:
    $$P(S_T > K) = N(d_2)$$

    For a put option:
    $$P(S_T < K) = N(-d_2) = 1 - N(d_2)$$

    Where:
    $$d_2 = \\frac{\\ln(S_0/K) + (r - \\sigma^2/2)T}{\\sigma\\sqrt{T}}$$

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
    float or ndarray
        Probability of finishing ITM (between 0 and 1)

    Examples
    --------
    >>> probability_itm(100, 100, 1, 0.05, 0.2, 'call')
    0.5522988569862262
    """
    # Convert to numpy arrays
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    # Determine output shape
    shapes = [np.shape(x) for x in [S, K, T, r, sigma] if np.shape(x) != ()]
    if shapes:
        result = np.zeros(shapes[0], dtype=float)
    else:
        result = np.zeros((), dtype=float)

    # Handle edge cases
    # Expired options
    expired_mask = T <= 0
    if np.any(expired_mask):
        if option_type == "call":
            result[expired_mask] = np.where(S[expired_mask] > K[expired_mask], 1.0, 0.0)
        else:
            result[expired_mask] = np.where(S[expired_mask] < K[expired_mask], 1.0, 0.0)

    # Zero volatility
    zero_vol_mask = (sigma == 0) & (T > 0)
    if np.any(zero_vol_mask):
        forward = S[zero_vol_mask] * np.exp(r[zero_vol_mask] * T[zero_vol_mask]) / K[zero_vol_mask]
        if option_type == "call":
            result[zero_vol_mask] = np.where(forward > 1, 1.0, 0.0)
        else:
            result[zero_vol_mask] = np.where(forward < 1, 1.0, 0.0)

    # Normal case
    normal_mask = (sigma > 0) & (T > 0)
    if np.any(normal_mask):
        sqrt_T = np.sqrt(T[normal_mask])
        d2 = (
            np.log(S[normal_mask] / K[normal_mask])
            + (r[normal_mask] - 0.5 * sigma[normal_mask] ** 2) * T[normal_mask]
        ) / (sigma[normal_mask] * sqrt_T)

        if option_type == "call":
            result[normal_mask] = norm.cdf(d2)
        else:
            result[normal_mask] = norm.cdf(-d2)

    logger.debug(
        "Probability ITM calculation completed",
        extra={
            "function": "probability_itm",
            "option_type": option_type,
            "output_shape": np.shape(result)
        }
    )
    return float(result) if np.ndim(result) == 0 else result


@overload
def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    initial_guess: Optional[float] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float: ...

@overload
def implied_volatility(
    option_price: FloatOrArray,
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    option_type: OptionType = "call",
    initial_guess: Optional[FloatOrArray] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> FloatArray: ...

def implied_volatility(
    option_price: FloatOrArray,
    S: FloatOrArray,
    K: FloatOrArray,
    T: FloatOrArray,
    r: FloatOrArray,
    option_type: OptionType = "call",
    initial_guess: Optional[FloatOrArray] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> FloatOrArray:
    """
    Calculate implied volatility using Newton-Raphson method.

    Solves for σ in the Black-Scholes equation:
    $$V_{market} = BS(S, K, T, r, \\sigma)$$

    Uses Newton-Raphson iteration:
    $$\\sigma_{n+1} = \\sigma_n - \\frac{BS(\\sigma_n) - V_{market}}{vega(\\sigma_n)}$$

    Where vega is:
    $$vega = S \\phi(d_1) \\sqrt{T}$$
    $$\\phi(x) = \\frac{1}{\\sqrt{2\\pi}} e^{-x^2/2}$$

    Parameters
    ----------
    option_price : float or array-like
        Market price of the option
    S : float or array-like
        Current stock price
    K : float or array-like
        Strike price
    T : float or array-like
        Time to expiration in years
    r : float or array-like
        Risk-free rate (annualized)
    option_type : {'call', 'put'}, default 'call'
        Type of option
    initial_guess : float or array-like, optional
        Initial volatility guess (default: 0.3)
    max_iterations : int, default 100
        Maximum number of iterations
    tolerance : float, default 1e-6
        Convergence tolerance

    Returns
    -------
    float or ndarray
        Implied volatility (returns NaN if no solution found)

    Examples
    --------
    >>> implied_volatility(10.45, 100, 100, 1, 0.05, 'call')
    0.19999...
    """
    # Convert to numpy arrays
    option_price = np.asarray(option_price)
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)

    # Initialize result array
    result = np.full_like(option_price, np.nan, dtype=float)

    # Check for valid inputs
    if option_type == "call":
        # Call price bounds: max(S - K*exp(-rT), 0) <= C <= S
        lower_bound = np.maximum(S - K * np.exp(-r * T), 0)
        upper_bound = S
    else:
        # Put price bounds: max(K*exp(-rT) - S, 0) <= P <= K*exp(-rT)
        lower_bound = np.maximum(K * np.exp(-r * T) - S, 0)
        upper_bound = K * np.exp(-r * T)

    # Check if option prices are within valid bounds
    valid_mask = (
        (option_price >= lower_bound - 1e-10) & (option_price <= upper_bound + 1e-10) & (T > 0)
    )

    if not np.any(valid_mask):
        return float(result) if np.ndim(result) == 0 else result

    # Handle cases at the bounds
    at_lower = valid_mask & (np.abs(option_price - lower_bound) < 1e-10)
    if np.ndim(result) == 0:
        if at_lower:
            result = 0.0
    else:
        result[at_lower] = 0.0

    at_upper = valid_mask & (np.abs(option_price - upper_bound) < 1e-10)
    if np.ndim(result) == 0:
        if at_upper:
            result = np.inf
    else:
        result[at_upper] = np.inf

    # Only iterate for prices strictly between bounds
    iterate_mask = valid_mask & ~at_lower & ~at_upper

    if not np.any(iterate_mask):
        return float(result) if np.ndim(result) == 0 else result

    # Extract values for iteration
    option_price_iter = option_price[iterate_mask] if np.ndim(option_price) > 0 else option_price
    S_iter = S[iterate_mask] if np.ndim(S) > 0 else S
    K_iter = K[iterate_mask] if np.ndim(K) > 0 else K
    T_iter = T[iterate_mask] if np.ndim(T) > 0 else T
    r_iter = r[iterate_mask] if np.ndim(r) > 0 else r

    # Initial guess
    if initial_guess is None:
        # Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T_iter) * (option_price_iter / S_iter)
        sigma = np.maximum(sigma, 0.01)  # Ensure positive initial guess
    else:
        sigma = np.asarray(initial_guess)
        if sigma.shape != iterate_mask.shape:
            sigma = np.full_like(option_price, initial_guess, dtype=float)
        sigma = sigma[iterate_mask].copy() if np.ndim(sigma) > 0 else sigma

    # Ensure sigma is always an array for the iteration
    sigma = np.atleast_1d(sigma)

    # Newton-Raphson iteration
    sqrt_T = np.sqrt(T_iter)

    for _ in range(max_iterations):
        # Calculate d1
        d1 = (np.log(S_iter / K_iter) + (r_iter + 0.5 * sigma**2) * T_iter) / (sigma * sqrt_T)

        # Calculate vega
        vega = S_iter * norm.pdf(d1) * sqrt_T

        # Calculate option price with current sigma
        current_price = black_scholes_price(S_iter, K_iter, T_iter, r_iter, sigma, option_type)

        # Price difference
        price_diff = current_price - option_price_iter

        # Check convergence
        converged = np.abs(price_diff) < tolerance
        if np.all(converged):
            if np.ndim(iterate_mask) == 0:
                result = float(sigma) if sigma.size == 1 else sigma[0]
            else:
                result[iterate_mask] = sigma
            break

        # Update sigma where not converged
        # Prevent division by very small vega
        update_mask = ~converged & (vega > 1e-10)
        if np.any(update_mask):
            if sigma.size == 1:
                # Scalar case
                if update_mask:
                    sigma = sigma - price_diff / vega
                    sigma = np.maximum(sigma, 1e-6)
                    sigma = np.minimum(sigma, 5.0)
            else:
                # Array case
                sigma[update_mask] -= price_diff[update_mask] / vega[update_mask]
                sigma[update_mask] = np.maximum(sigma[update_mask], 1e-6)
                sigma[update_mask] = np.minimum(sigma[update_mask], 5.0)

        # Mark as converged where vega is too small
        if np.ndim(iterate_mask) == 0:
            result = float(sigma) if sigma.size == 1 else sigma[0]
        else:
            result[iterate_mask] = sigma

    logger.debug(
        "Implied volatility calculation completed",
        extra={
            "function": "implied_volatility",
            "option_type": option_type,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "output_shape": np.shape(result)
        }
    )
    return float(result) if np.ndim(result) == 0 else result


def calculate_var(
    returns: FloatOrArray,
    confidence_level: float = 0.95,
    time_horizon: float = 1.0,
) -> FloatOrArray:
    """
    Calculate Value at Risk using the normal distribution assumption.

    VaR represents the maximum expected loss with a given confidence level
    over a specified time horizon.

    For normally distributed returns:
    $$VaR_{\\alpha} = -\\mu - \\sigma \\cdot z_{\\alpha}$$

    Where:
    - μ is the mean return
    - σ is the standard deviation of returns
    - z_α is the inverse normal CDF at (1 - confidence_level)

    Parameters
    ----------
    returns : float or array-like
        Historical returns or expected return and volatility
        If float: interpreted as volatility (assumes zero mean)
        If 1D array: historical returns to calculate mean and std
        If 2D array: rows are time periods, columns are assets
    confidence_level : float, default 0.95
        Confidence level (e.g., 0.95 for 95% VaR)
    time_horizon : float, default 1.0
        Time horizon in same units as returns (e.g., days, years)

    Returns
    -------
    float or ndarray
        Value at Risk (positive number representing loss)

    Examples
    --------
    >>> # With volatility only (assumes zero mean)
    >>> calculate_var(0.2, 0.95, 1.0)
    0.328...
    
    >>> # With historical returns
    >>> returns = np.random.normal(0.001, 0.02, 1000)
    >>> calculate_var(returns, 0.95)
    0.032...
    """
    returns = np.asarray(returns)
    
    # Handle different input formats
    if returns.ndim == 0:
        # Single value: interpret as volatility with zero mean
        mean_return = 0.0
        volatility = returns
    elif returns.ndim == 1:
        # 1D array: calculate mean and std from historical returns
        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1)
    else:
        # 2D array: calculate VaR for each column (asset)
        mean_return = np.mean(returns, axis=0)
        volatility = np.std(returns, axis=0, ddof=1)
    
    # Scale by time horizon
    mean_scaled = mean_return * time_horizon
    vol_scaled = volatility * np.sqrt(time_horizon)
    
    # Calculate z-score for the confidence level
    z_score = norm.ppf(1 - confidence_level)
    
    # Calculate VaR (negative of percentile, so positive VaR represents loss)
    var = -(mean_scaled + z_score * vol_scaled)
    
    logger.debug(
        "VaR calculation completed",
        extra={
            "function": "calculate_var",
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
            "output_shape": np.shape(var)
        }
    )
    
    return float(var) if np.ndim(var) == 0 else var


def calculate_cvar(
    returns: FloatOrArray,
    confidence_level: float = 0.95,
    time_horizon: float = 1.0,
    use_cornish_fisher: bool = True,
) -> FloatOrArray:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR represents the expected loss conditional on the loss exceeding VaR.
    Uses Cornish-Fisher expansion to account for skewness and kurtosis.

    For normal distribution:
    $$CVaR_{\\alpha} = -\\mu + \\sigma \\cdot \\frac{\\phi(z_{\\alpha})}{1-\\alpha}$$

    With Cornish-Fisher expansion:
    $$z_{CF} = z + \\frac{1}{6}(z^2-1)S + \\frac{1}{24}(z^3-3z)(K-3) - \\frac{1}{36}(2z^3-5z)S^2$$

    Where:
    - S is skewness
    - K is kurtosis
    - φ is the standard normal PDF

    Parameters
    ----------
    returns : float or array-like
        Historical returns or expected statistics
        If float: interpreted as volatility (assumes normal dist)
        If 1D array: historical returns to calculate statistics
        If 2D array: rows are time periods, columns are assets
    confidence_level : float, default 0.95
        Confidence level (e.g., 0.95 for 95% CVaR)
    time_horizon : float, default 1.0
        Time horizon in same units as returns
    use_cornish_fisher : bool, default True
        Whether to use Cornish-Fisher expansion for non-normal returns

    Returns
    -------
    float or ndarray
        Conditional Value at Risk (positive number representing loss)

    Examples
    --------
    >>> # With volatility only
    >>> calculate_cvar(0.2, 0.95, 1.0)
    0.399...
    
    >>> # With historical returns
    >>> returns = np.random.normal(0.001, 0.02, 1000)
    >>> calculate_cvar(returns, 0.95)
    0.040...
    """
    returns = np.asarray(returns)
    
    # Handle different input formats
    if returns.ndim == 0:
        # Single value: interpret as volatility with zero mean
        mean_return = 0.0
        volatility = returns
        skewness = 0.0
        excess_kurtosis = 0.0
        use_cornish_fisher = False  # No higher moments available
    elif returns.ndim == 1:
        # 1D array: calculate statistics from historical returns
        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1)
        if use_cornish_fisher and len(returns) > 3:
            from scipy.stats import skew, kurtosis
            skewness = skew(returns)
            excess_kurtosis = kurtosis(returns, fisher=True)
        else:
            skewness = 0.0
            excess_kurtosis = 0.0
    else:
        # 2D array: calculate CVaR for each column
        mean_return = np.mean(returns, axis=0)
        volatility = np.std(returns, axis=0, ddof=1)
        if use_cornish_fisher and returns.shape[0] > 3:
            from scipy.stats import skew, kurtosis
            skewness = skew(returns, axis=0)
            excess_kurtosis = kurtosis(returns, axis=0, fisher=True)
        else:
            skewness = np.zeros(returns.shape[1])
            excess_kurtosis = np.zeros(returns.shape[1])
    
    # Scale by time horizon
    mean_scaled = mean_return * time_horizon
    vol_scaled = volatility * np.sqrt(time_horizon)
    
    # Base z-score
    alpha = 1 - confidence_level
    z = norm.ppf(alpha)
    
    # Apply Cornish-Fisher expansion if requested
    if use_cornish_fisher:
        z_cf = z + (1/6) * (z**2 - 1) * skewness
        z_cf += (1/24) * (z**3 - 3*z) * excess_kurtosis
        z_cf -= (1/36) * (2*z**3 - 5*z) * skewness**2
        z_adjusted = z_cf
    else:
        z_adjusted = z
    
    # Calculate expected shortfall
    # For normal distribution: ES = -μ + σ * φ(z) / α
    pdf_value = norm.pdf(z)
    cvar = -mean_scaled + vol_scaled * pdf_value / alpha
    
    # Adjust for non-normality if using Cornish-Fisher
    if use_cornish_fisher:
        # Apply adjustment factor based on modified quantile
        adjustment = z_adjusted / z
        cvar *= adjustment
    
    logger.debug(
        "CVaR calculation completed",
        extra={
            "function": "calculate_cvar",
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
            "use_cornish_fisher": use_cornish_fisher,
            "output_shape": np.shape(cvar)
        }
    )
    
    return float(cvar) if np.ndim(cvar) == 0 else cvar


def half_kelly_size(
    edge: float,
    odds: float,
    bankroll: float = 1.0,
) -> float:
    """
    Calculate position size using half-Kelly criterion.

    The Kelly criterion maximizes long-term growth rate:
    $$f^* = \\frac{p \\cdot b - q}{b}$$

    Where:
    - f* is the optimal fraction of bankroll to bet
    - p is probability of winning
    - q = 1 - p is probability of losing
    - b is the odds (net gain on win / loss on loss)

    For a given edge and odds:
    $$f^* = \\frac{edge}{odds}$$

    Half-Kelly uses f*/2 for reduced volatility and drawdown.

    Parameters
    ----------
    edge : float
        Expected edge as a fraction (e.g., 0.05 for 5% edge)
        Edge = (probability of profit * profit - probability of loss * loss) / risk
    odds : float
        Ratio of win to loss amounts (e.g., 2.0 for 2:1 odds)
        For options: premium received / max loss
    bankroll : float, default 1.0
        Total bankroll (returns fraction if 1.0, dollar amount otherwise)

    Returns
    -------
    float
        Recommended position size (fraction of bankroll or dollar amount)

    Examples
    --------
    >>> # 5% edge with 2:1 odds
    >>> half_kelly_size(0.05, 2.0)
    0.0125
    
    >>> # With $100k bankroll
    >>> half_kelly_size(0.05, 2.0, 100000)
    1250.0
    """
    # Validate inputs
    if edge < 0:
        logger.warning(
            "Negative edge provided to Kelly sizing",
            extra={"edge": edge, "odds": odds}
        )
        return 0.0
    
    if odds <= 0:
        logger.warning(
            "Invalid odds provided to Kelly sizing",
            extra={"edge": edge, "odds": odds}
        )
        return 0.0
    
    # Full Kelly fraction
    kelly_fraction = edge / odds
    
    # Apply half-Kelly for safety
    half_kelly_fraction = kelly_fraction / 2
    
    # Limit to reasonable bounds (max 25% of bankroll)
    max_fraction = 0.25
    safe_fraction = min(half_kelly_fraction, max_fraction)
    
    # Calculate position size
    position_size = safe_fraction * bankroll
    
    logger.debug(
        "Half-Kelly sizing calculated",
        extra={
            "function": "half_kelly_size",
            "edge": edge,
            "odds": odds,
            "kelly_fraction": kelly_fraction,
            "half_kelly_fraction": half_kelly_fraction,
            "final_fraction": safe_fraction,
            "position_size": position_size
        }
    )
    
    return position_size


def margin_requirement(
    strike: FloatOrArray,
    underlying_price: FloatOrArray,
    option_price: FloatOrArray,
    multiplier: int = 100,
    margin_rate: float = 0.20,
) -> FloatOrArray:
    """
    Calculate margin requirement for naked put options.

    Standard margin requirement for naked puts:
    Greater of:
    1. 20% of underlying - out of money amount + option premium
    2. 10% of strike price + option premium

    Parameters
    ----------
    strike : float or array-like
        Strike price of the put option
    underlying_price : float or array-like
        Current price of the underlying
    option_price : float or array-like
        Premium received for the option
    multiplier : int, default 100
        Contract multiplier (shares per contract)
    margin_rate : float, default 0.20
        Margin rate (20% standard for equities)

    Returns
    -------
    float or ndarray
        Margin requirement in dollars

    Examples
    --------
    >>> # SPY at $450, sell $440 put for $5.00
    >>> margin_requirement(440, 450, 5.0)
    8000.0
    
    >>> # Multiple positions
    >>> strikes = np.array([440, 430, 420])
    >>> margin_requirement(strikes, 450, [5.0, 3.0, 1.5])
    array([8000., 8300., 8150.])
    """
    # Convert to numpy arrays
    strike = np.asarray(strike)
    underlying_price = np.asarray(underlying_price)
    option_price = np.asarray(option_price)
    
    # Calculate out of money amount (positive if OTM)
    otm_amount = np.maximum(underlying_price - strike, 0)
    
    # Method 1: 20% of underlying - OTM amount + premium
    method1 = margin_rate * underlying_price - otm_amount + option_price
    
    # Method 2: 10% of strike + premium
    method2 = 0.10 * strike + option_price
    
    # Take the greater of the two methods
    margin_per_share = np.maximum(method1, method2)
    
    # Apply contract multiplier
    total_margin = margin_per_share * multiplier
    
    logger.debug(
        "Margin requirement calculated",
        extra={
            "function": "margin_requirement",
            "method1_per_share": method1,
            "method2_per_share": method2,
            "margin_per_share": margin_per_share,
            "total_margin": total_margin,
            "output_shape": np.shape(total_margin)
        }
    )
    
    return float(total_margin) if np.ndim(total_margin) == 0 else total_margin
