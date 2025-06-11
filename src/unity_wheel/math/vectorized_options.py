"""Vectorized options calculations for high-performance batch processing.

This module provides vectorized implementations of options calculations
that can process multiple strikes, expirations, or scenarios simultaneously,
providing significant performance improvements for wheel strategy analysis.
"""

import logging
import time
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.stats import norm

from src.unity_wheel.math.options import CalculationResult, OptionType
from src.unity_wheel.utils.logging import StructuredLogger
from src.unity_wheel.utils.performance_cache import cached

logger = StructuredLogger(logging.getLogger(__name__))

# Type aliases for vectorized operations
VectorInput = Union[float, np.ndarray]


class VectorizedResults:
    """Container for vectorized calculation results."""

    def __init__(self, values: np.ndarray, confidence: float, shape_info: dict):
        self.values = values
        self.confidence = confidence
        self.shape_info = shape_info
        self.size = values.size

    def __getitem__(self, index):
        """Allow indexing into results."""
        return self.values[index]

    def __len__(self):
        """Return number of results."""
        return self.size

    def to_list(self) -> List[float]:
        """Convert to flat list."""
        return self.values.flatten().tolist()

    def reshape(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Reshape results to specific dimensions."""
        return self.values.reshape(shape)


@cached(
    cache_name="options",
    ttl_seconds=1800,
    key_func=lambda spot, strikes, time_to_expiry, risk_free_rate, volatility, option_type: f"vec_bs_{hash(str(spot))}_k{hash(str(strikes))}_t{time_to_expiry:.4f}_r{risk_free_rate:.4f}_v{volatility:.4f}_{option_type}",
)
def vectorized_black_scholes(
    spot: VectorInput,
    strikes: VectorInput,
    time_to_expiry: VectorInput,
    risk_free_rate: VectorInput,
    volatility: VectorInput,
    option_type: OptionType = "call",
) -> VectorizedResults:
    """
    Vectorized Black-Scholes calculation for multiple scenarios.

    All inputs can be scalars or arrays. Arrays will be broadcast together.

    Args:
        spot: Current stock price(s)
        strikes: Strike price(s)
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free rate(s)
        volatility: Volatility(ies)
        option_type: "call" or "put"

    Returns:
        VectorizedResults with option prices

    Examples:
        # Single spot, multiple strikes
        >>> result = vectorized_black_scholes(100, [95, 100, 105], 0.25, 0.05, 0.20, "call")
        >>> result.values  # Array of 3 option prices

        # Multiple scenarios (spot × strike combinations)
        >>> spots = [98, 100, 102]
        >>> strikes = [95, 100, 105]
        >>> result = vectorized_black_scholes(spots, strikes, 0.25, 0.05, 0.20, "call")
        >>> result.values  # 3x3 matrix of prices
    """
    start_time = time.time()

    # Convert all inputs to numpy arrays
    S = np.asarray(spot, dtype=np.float64)
    K = np.asarray(strikes, dtype=np.float64)
    T = np.asarray(time_to_expiry, dtype=np.float64)
    r = np.asarray(risk_free_rate, dtype=np.float64)
    sigma = np.asarray(volatility, dtype=np.float64)

    # Store original shapes for result metadata
    original_shapes = {
        "spot": S.shape,
        "strikes": K.shape,
        "time_to_expiry": T.shape,
        "risk_free_rate": r.shape,
        "volatility": sigma.shape,
    }

    # Broadcast all arrays to compatible shape
    try:
        S, K, T, r, sigma = np.broadcast_arrays(S, K, T, r, sigma)
    except ValueError as e:
        logger.error("vectorized_broadcast_error", extra={"error": str(e)})
        raise ValueError(f"Cannot broadcast input arrays: {e}")

    result_shape = S.shape
    total_calculations = S.size

    # Vectorized validation
    valid_mask = (S > 0) & (K > 0) & (T >= 0) & (sigma >= 0)
    confidence = np.mean(valid_mask)  # Fraction of valid inputs

    # Initialize result array
    values = np.full_like(S, np.nan)

    # Only calculate for valid inputs
    if np.any(valid_mask):
        S_valid = S[valid_mask]
        K_valid = K[valid_mask]
        T_valid = T[valid_mask]
        r_valid = r[valid_mask]
        sigma_valid = sigma[valid_mask]

        # Handle expired options
        expired_mask = T_valid <= 0
        if np.any(expired_mask):
            if option_type == "call":
                values[valid_mask & (T <= 0)] = np.maximum(
                    S_valid[expired_mask] - K_valid[expired_mask], 0
                )
            else:
                values[valid_mask & (T <= 0)] = np.maximum(
                    K_valid[expired_mask] - S_valid[expired_mask], 0
                )

        # Handle non-expired options
        active_mask = T_valid > 0
        if np.any(active_mask):
            S_act = S_valid[active_mask]
            K_act = K_valid[active_mask]
            T_act = T_valid[active_mask]
            r_act = r_valid[active_mask]
            sigma_act = sigma_valid[active_mask]

            # Vectorized Black-Scholes calculation
            sqrt_T = np.sqrt(T_act)

            # Handle zero volatility cases
            zero_vol_mask = sigma_act == 0
            if np.any(zero_vol_mask):
                if option_type == "call":
                    intrinsic = np.maximum(S_act - K_act * np.exp(-r_act * T_act), 0)
                else:
                    intrinsic = np.maximum(K_act * np.exp(-r_act * T_act) - S_act, 0)

                values[valid_mask & (T > 0) & (sigma[valid_mask] == 0)] = intrinsic[zero_vol_mask]

            # Standard Black-Scholes for non-zero volatility
            nonzero_vol_mask = sigma_act > 0
            if np.any(nonzero_vol_mask):
                S_nz = S_act[nonzero_vol_mask]
                K_nz = K_act[nonzero_vol_mask]
                T_nz = T_act[nonzero_vol_mask]
                r_nz = r_act[nonzero_vol_mask]
                sigma_nz = sigma_act[nonzero_vol_mask]
                sqrt_T_nz = sqrt_T[nonzero_vol_mask]

                # Calculate d1 and d2
                d1 = (np.log(S_nz / K_nz) + (r_nz + 0.5 * sigma_nz**2) * T_nz) / (
                    sigma_nz * sqrt_T_nz
                )
                d2 = d1 - sigma_nz * sqrt_T_nz

                # Calculate option values
                if option_type == "call":
                    option_values = S_nz * norm.cdf(d1) - K_nz * np.exp(-r_nz * T_nz) * norm.cdf(d2)
                else:
                    option_values = K_nz * np.exp(-r_nz * T_nz) * norm.cdf(-d2) - S_nz * norm.cdf(
                        -d1
                    )

                # Store results back in the main array
                full_active_mask = valid_mask & (T > 0)
                full_nonzero_mask = full_active_mask & (sigma > 0)
                values[full_nonzero_mask] = option_values

    computation_time_ms = (time.time() - start_time) * 1000

    logger.debug(
        "vectorized_black_scholes_completed",
        extra={
            "total_calculations": total_calculations,
            "valid_calculations": int(np.sum(valid_mask)),
            "result_shape": result_shape,
            "computation_time_ms": computation_time_ms,
            "option_type": option_type,
            "confidence": confidence,
        },
    )

    return VectorizedResults(
        values=values,
        confidence=confidence,
        shape_info={
            "result_shape": result_shape,
            "original_shapes": original_shapes,
            "total_calculations": total_calculations,
            "computation_time_ms": computation_time_ms,
        },
    )


@cached(
    cache_name="options",
    ttl_seconds=1200,
    key_func=lambda spot, strikes, time_to_expiry, risk_free_rate, volatility, option_type: f"vec_greeks_{hash(str(spot))}_k{hash(str(strikes))}_t{time_to_expiry:.4f}_r{risk_free_rate:.4f}_v{volatility:.4f}_{option_type}",
)
def vectorized_greeks(
    spot: VectorInput,
    strikes: VectorInput,
    time_to_expiry: VectorInput,
    risk_free_rate: VectorInput,
    volatility: VectorInput,
    option_type: OptionType = "call",
) -> Dict[str, VectorizedResults]:
    """
    Vectorized Greeks calculation for multiple scenarios.

    Returns:
        Dictionary with VectorizedResults for each Greek:
        - delta: Price sensitivity to underlying
        - gamma: Delta sensitivity to underlying
        - theta: Time decay (per day)
        - vega: Volatility sensitivity (per 1% vol change)
        - rho: Interest rate sensitivity (per 1% rate change)
    """
    start_time = time.time()

    # Convert inputs to arrays and broadcast
    S = np.asarray(spot, dtype=np.float64)
    K = np.asarray(strikes, dtype=np.float64)
    T = np.asarray(time_to_expiry, dtype=np.float64)
    r = np.asarray(risk_free_rate, dtype=np.float64)
    sigma = np.asarray(volatility, dtype=np.float64)

    try:
        S, K, T, r, sigma = np.broadcast_arrays(S, K, T, r, sigma)
    except ValueError as e:
        raise ValueError(f"Cannot broadcast input arrays: {e}")

    result_shape = S.shape
    total_calculations = S.size

    # Validation mask
    valid_mask = (S > 0) & (K > 0) & (T >= 0) & (sigma >= 0)
    active_mask = valid_mask & (T > 0) & (sigma > 0)
    confidence = np.mean(valid_mask)

    # Initialize all Greeks
    greeks = {
        "delta": np.full_like(S, np.nan),
        "gamma": np.full_like(S, np.nan),
        "theta": np.full_like(S, np.nan),
        "vega": np.full_like(S, np.nan),
        "rho": np.full_like(S, np.nan),
    }

    # Handle expired options
    expired_mask = valid_mask & (T <= 0)
    if np.any(expired_mask):
        if option_type == "call":
            greeks["delta"][expired_mask] = np.where(S[expired_mask] > K[expired_mask], 1.0, 0.0)
        else:
            greeks["delta"][expired_mask] = np.where(S[expired_mask] < K[expired_mask], -1.0, 0.0)

        # All other Greeks are zero for expired options
        for greek in ["gamma", "theta", "vega", "rho"]:
            greeks[greek][expired_mask] = 0.0

    # Calculate Greeks for active options
    if np.any(active_mask):
        S_act = S[active_mask]
        K_act = K[active_mask]
        T_act = T[active_mask]
        r_act = r[active_mask]
        sigma_act = sigma[active_mask]

        sqrt_T = np.sqrt(T_act)

        # Calculate d1 and d2
        d1 = (np.log(S_act / K_act) + (r_act + 0.5 * sigma_act**2) * T_act) / (sigma_act * sqrt_T)
        d2 = d1 - sigma_act * sqrt_T

        # Delta
        if option_type == "call":
            delta_values = norm.cdf(d1)
        else:
            delta_values = norm.cdf(d1) - 1.0
        greeks["delta"][active_mask] = delta_values

        # Gamma (same for calls and puts)
        gamma_values = norm.pdf(d1) / (S_act * sigma_act * sqrt_T)
        greeks["gamma"][active_mask] = gamma_values

        # Theta
        term1 = -S_act * norm.pdf(d1) * sigma_act / (2 * sqrt_T)
        if option_type == "call":
            term2 = -r_act * K_act * np.exp(-r_act * T_act) * norm.cdf(d2)
        else:
            term2 = r_act * K_act * np.exp(-r_act * T_act) * norm.cdf(-d2)

        theta_values = (term1 + term2) / 365  # Convert to per day
        greeks["theta"][active_mask] = theta_values

        # Vega (same for calls and puts)
        vega_values = S_act * norm.pdf(d1) * sqrt_T / 100  # Per 1% change
        greeks["vega"][active_mask] = vega_values

        # Rho
        if option_type == "call":
            rho_values = K_act * T_act * np.exp(-r_act * T_act) * norm.cdf(d2) / 100
        else:
            rho_values = -K_act * T_act * np.exp(-r_act * T_act) * norm.cdf(-d2) / 100
        greeks["rho"][active_mask] = rho_values

    computation_time_ms = (time.time() - start_time) * 1000

    logger.debug(
        "vectorized_greeks_completed",
        extra={
            "total_calculations": total_calculations,
            "active_calculations": int(np.sum(active_mask)),
            "result_shape": result_shape,
            "computation_time_ms": computation_time_ms,
            "option_type": option_type,
            "confidence": confidence,
        },
    )

    shape_info = {
        "result_shape": result_shape,
        "total_calculations": total_calculations,
        "computation_time_ms": computation_time_ms,
    }

    # Return VectorizedResults for each Greek
    return {
        greek_name: VectorizedResults(values, confidence, shape_info)
        for greek_name, values in greeks.items()
    }


def vectorized_wheel_analysis(
    spot_price: float,
    strikes: List[float],
    expirations: List[float],  # In years
    risk_free_rate: float = 0.05,
    volatility: float = 0.60,
    target_delta: float = 0.30,
    min_premium_pct: float = 1.0,
) -> Dict:
    """
    Vectorized analysis of multiple wheel strategy candidates.

    Efficiently evaluates all strike/expiration combinations for wheel strategy.

    Args:
        spot_price: Current stock price
        strikes: List of strike prices to evaluate
        expirations: List of expirations (in years) to evaluate
        risk_free_rate: Risk-free rate
        volatility: Implied volatility
        target_delta: Target delta for put selection
        min_premium_pct: Minimum premium as % of strike

    Returns:
        Dictionary with analysis results including best candidates
    """
    start_time = time.time()

    # Create meshgrid for all combinations
    strikes_grid, exp_grid = np.meshgrid(strikes, expirations)

    # Flatten for vectorized calculation
    strikes_flat = strikes_grid.flatten()
    exp_flat = exp_grid.flatten()

    # Calculate prices and Greeks for all combinations
    put_prices = vectorized_black_scholes(
        spot_price, strikes_flat, exp_flat, risk_free_rate, volatility, "put"
    )

    put_greeks = vectorized_greeks(
        spot_price, strikes_flat, exp_flat, risk_free_rate, volatility, "put"
    )

    # Extract results
    prices = put_prices.values
    deltas = put_greeks["delta"].values
    gammas = put_greeks["gamma"].values
    thetas = put_greeks["theta"].values
    vegas = put_greeks["vega"].values

    # Calculate derived metrics
    premium_pcts = (prices / strikes_flat) * 100
    dte_days = exp_flat * 365
    annualized_yields = (premium_pcts / dte_days) * 365

    # Probability of assignment (absolute delta for puts)
    prob_assignment = np.abs(deltas)

    # Expected return (premium * probability of keeping it)
    prob_profit = 1 - prob_assignment
    expected_returns = premium_pcts * prob_profit

    # Filter candidates based on criteria
    valid_mask = (
        (np.abs(deltas - (-target_delta)) <= 0.10)
        & (premium_pcts >= min_premium_pct)  # Delta within 10% of target
        & (~np.isnan(prices))  # Minimum premium  # Valid prices
    )

    # Create results dataframe-like structure
    results = []
    for i in range(len(strikes_flat)):
        if valid_mask[i]:
            results.append(
                {
                    "strike": strikes_flat[i],
                    "expiration_years": exp_flat[i],
                    "dte_days": int(dte_days[i]),
                    "option_price": prices[i],
                    "delta": deltas[i],
                    "gamma": gammas[i],
                    "theta": thetas[i],
                    "vega": vegas[i],
                    "premium_pct": premium_pcts[i],
                    "annualized_yield": annualized_yields[i],
                    "prob_assignment": prob_assignment[i],
                    "expected_return": expected_returns[i],
                    "score": expected_returns[i] * (1 + premium_pcts[i] / 100),  # Combined score
                }
            )

    # Sort by score (best opportunities first)
    results.sort(key=lambda x: x["score"], reverse=True)

    computation_time_ms = (time.time() - start_time) * 1000

    # Summary statistics
    analysis = {
        "candidates": results,
        "summary": {
            "total_combinations": len(strikes_flat),
            "valid_candidates": len(results),
            "best_candidate": results[0] if results else None,
            "avg_premium_pct": np.mean([r["premium_pct"] for r in results]) if results else 0,
            "avg_annualized_yield": (
                np.mean([r["annualized_yield"] for r in results]) if results else 0
            ),
            "computation_time_ms": computation_time_ms,
        },
        "market_data": {
            "spot_price": spot_price,
            "volatility": volatility,
            "risk_free_rate": risk_free_rate,
            "target_delta": target_delta,
        },
    }

    logger.info(
        "vectorized_wheel_analysis_completed",
        extra={
            "total_combinations": len(strikes_flat),
            "valid_candidates": len(results),
            "computation_time_ms": computation_time_ms,
            "best_score": results[0]["score"] if results else 0,
        },
    )

    return analysis


def compare_scenario_analysis(scenarios: List[Dict], base_case: Dict) -> Dict:
    """
    Vectorized comparison of multiple market scenarios.

    Args:
        scenarios: List of scenario dicts with keys: spot, volatility, rate, etc.
        base_case: Base case scenario for comparison

    Returns:
        Dictionary with comparative analysis
    """
    if not scenarios:
        return {"error": "No scenarios provided"}

    # Extract scenario parameters
    spots = [s.get("spot_price", base_case["spot_price"]) for s in scenarios]
    vols = [s.get("volatility", base_case["volatility"]) for s in scenarios]
    rates = [s.get("risk_free_rate", base_case["risk_free_rate"]) for s in scenarios]

    strike = base_case["strike"]
    expiration = base_case["expiration"]
    option_type = base_case.get("option_type", "put")

    # Vectorized calculation for all scenarios
    prices_result = vectorized_black_scholes(spots, strike, expiration, rates, vols, option_type)

    greeks_result = vectorized_greeks(spots, strike, expiration, rates, vols, option_type)

    # Base case calculation
    base_price = vectorized_black_scholes(
        base_case["spot_price"],
        strike,
        expiration,
        base_case["risk_free_rate"],
        base_case["volatility"],
        option_type,
    )

    base_greeks = vectorized_greeks(
        base_case["spot_price"],
        strike,
        expiration,
        base_case["risk_free_rate"],
        base_case["volatility"],
        option_type,
    )

    # Calculate differences from base case
    price_diffs = prices_result.values - base_price.values[0]
    delta_diffs = greeks_result["delta"].values - base_greeks["delta"].values[0]

    # Compile results
    comparisons = []
    for i, scenario in enumerate(scenarios):
        comparisons.append(
            {
                "scenario": scenario,
                "option_price": prices_result.values[i],
                "price_diff": price_diffs[i],
                "price_diff_pct": (price_diffs[i] / base_price.values[0]) * 100,
                "delta": greeks_result["delta"].values[i],
                "delta_diff": delta_diffs[i],
                "gamma": greeks_result["gamma"].values[i],
                "theta": greeks_result["theta"].values[i],
                "vega": greeks_result["vega"].values[i],
            }
        )

    return {
        "base_case": {
            "price": base_price.values[0],
            "delta": base_greeks["delta"].values[0],
            "gamma": base_greeks["gamma"].values[0],
            "theta": base_greeks["theta"].values[0],
            "vega": base_greeks["vega"].values[0],
        },
        "scenarios": comparisons,
        "summary": {
            "max_price_impact": np.max(np.abs(price_diffs)),
            "max_delta_impact": np.max(np.abs(delta_diffs)),
            "scenario_count": len(scenarios),
        },
    }


# Convenience functions for common use cases
def quick_strike_comparison(
    spot: float, strikes: List[float], dte: int, vol: float = 0.60, option_type: OptionType = "put"
) -> Dict[float, Dict]:
    """Quick comparison of option prices across strikes."""
    exp_years = dte / 365

    prices = vectorized_black_scholes(spot, strikes, exp_years, 0.05, vol, option_type)
    greeks = vectorized_greeks(spot, strikes, exp_years, 0.05, vol, option_type)

    results = {}
    for i, strike in enumerate(strikes):
        results[strike] = {
            "price": prices.values[i],
            "delta": greeks["delta"].values[i],
            "gamma": greeks["gamma"].values[i],
            "theta": greeks["theta"].values[i],
            "premium_pct": (
                (prices.values[i] / strike) * 100
                if option_type == "put"
                else (prices.values[i] / spot) * 100
            ),
        }

    return results


def quick_vol_sensitivity(
    spot: float,
    strike: float,
    dte: int,
    vol_range: Tuple[float, float] = (0.30, 1.00),
    num_points: int = 10,
) -> Dict:
    """Quick volatility sensitivity analysis."""
    vols = np.linspace(vol_range[0], vol_range[1], num_points)
    exp_years = dte / 365

    prices = vectorized_black_scholes(spot, strike, exp_years, 0.05, vols, "put")

    return {
        "volatilities": vols.tolist(),
        "prices": prices.to_list(),
        "vol_sensitivity": np.gradient(prices.values, vols).tolist(),
    }
