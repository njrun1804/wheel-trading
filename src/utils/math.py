"""Options mathematics module for Black-Scholes pricing and Greeks calculation."""

from typing import Literal, Optional, Union

import numpy as np
from scipy.stats import norm


def black_scholes_price(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: Literal["call", "put"] = "call",
) -> Union[float, np.ndarray]:
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

    return float(result) if np.ndim(result) == 0 else result


def calculate_delta(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: Literal["call", "put"] = "call",
) -> Union[float, np.ndarray]:
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

    return float(result) if np.ndim(result) == 0 else result


def probability_itm(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: Literal["call", "put"] = "call",
) -> Union[float, np.ndarray]:
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

    return float(result) if np.ndim(result) == 0 else result


def implied_volatility(
    option_price: Union[float, np.ndarray],
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    option_type: Literal["call", "put"] = "call",
    initial_guess: Optional[Union[float, np.ndarray]] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Union[float, np.ndarray]:
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

    return float(result) if np.ndim(result) == 0 else result
