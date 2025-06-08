"""Analytics module for trading performance and strategy evaluation."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Configure structured logging
logger = logging.getLogger(__name__)

# Type aliases
FloatArray = npt.NDArray[np.float64]
FloatOrArray = Union[float, FloatArray]


def calculate_edge(
    theoretical_value: FloatOrArray,
    market_price: FloatOrArray,
) -> FloatOrArray:
    """
    Calculate the edge as (theoretical - market) / market.

    Edge represents the percentage advantage of theoretical value over market price.
    A positive edge indicates the theoretical value is higher than market price.

    Edge calculation:
    $$edge = \\frac{V_{theoretical} - V_{market}}{V_{market}}$$

    Parameters
    ----------
    theoretical_value : float or array-like
        Theoretical or fair value of the asset/option
    market_price : float or array-like
        Current market price

    Returns
    -------
    float or ndarray
        Edge as a fraction (e.g., 0.05 for 5% edge)

    Examples
    --------
    >>> # Single option: theoretical $10.50, market $10.00
    >>> calculate_edge(10.50, 10.00)
    0.05

    >>> # Multiple options
    >>> theoretical = np.array([10.50, 5.25, 2.10])
    >>> market = np.array([10.00, 5.00, 2.00])
    >>> calculate_edge(theoretical, market)
    array([0.05, 0.05, 0.05])
    """
    theoretical_value = np.asarray(theoretical_value)
    market_price = np.asarray(market_price)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        edge = (theoretical_value - market_price) / market_price

        # Handle edge cases
        if np.ndim(edge) == 0:
            if market_price == 0:
                edge = (
                    np.inf if theoretical_value > 0 else (-np.inf if theoretical_value < 0 else 0.0)
                )
        else:
            zero_mask = market_price == 0
            if np.any(zero_mask):
                edge[zero_mask] = np.where(
                    theoretical_value[zero_mask] > 0,
                    np.inf,
                    np.where(theoretical_value[zero_mask] < 0, -np.inf, 0.0),
                )

    logger.debug(
        "Edge calculation completed",
        extra={
            "function": "calculate_edge",
            "mean_edge": float(np.mean(edge)) if np.ndim(edge) > 0 else float(edge),
            "output_shape": np.shape(edge),
        },
    )

    return float(edge) if np.ndim(edge) == 0 else edge


def expected_value(
    outcomes: List[float],
    probabilities: List[float],
) -> float:
    """
    Calculate expected value from probability-weighted outcomes.

    Expected value is the probability-weighted average of all possible outcomes:
    $$E[X] = \\sum_{i=1}^{n} p_i \\cdot x_i$$

    Where:
    - x_i is the outcome value
    - p_i is the probability of that outcome
    - Σp_i = 1

    Parameters
    ----------
    outcomes : list of float
        Possible outcome values (profits/losses)
    probabilities : list of float
        Probability of each outcome (must sum to ~1.0)

    Returns
    -------
    float
        Expected value

    Examples
    --------
    >>> # Option expires worthless (70%), small profit (20%), large profit (10%)
    >>> outcomes = [-100, 200, 1000]  # Premium collected minus any losses
    >>> probabilities = [0.7, 0.2, 0.1]
    >>> expected_value(outcomes, probabilities)
    30.0

    >>> # Put assignment scenarios
    >>> outcomes = [500, -1000, -3000]  # Premium kept, small loss, large loss
    >>> probabilities = [0.6, 0.3, 0.1]
    >>> expected_value(outcomes, probabilities)
    0.0
    """
    outcomes_arr = np.asarray(outcomes)
    probabilities_arr = np.asarray(probabilities)

    # Validate probabilities
    prob_sum = np.sum(probabilities_arr)
    if not np.isclose(prob_sum, 1.0, rtol=1e-5):
        logger.warning(
            "Probabilities do not sum to 1.0",
            extra={"prob_sum": prob_sum, "probabilities": probabilities},
        )
        # Normalize probabilities
        probabilities_arr = probabilities_arr / prob_sum

    # Calculate expected value
    ev = np.sum(outcomes_arr * probabilities_arr)

    logger.debug(
        "Expected value calculated",
        extra={
            "function": "expected_value",
            "num_outcomes": len(outcomes),
            "expected_value": ev,
            "min_outcome": np.min(outcomes_arr),
            "max_outcome": np.max(outcomes_arr),
        },
    )

    return float(ev)


def sharpe_ratio(
    returns: FloatOrArray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio for completed trades.

    The Sharpe ratio measures risk-adjusted returns:
    $$SR = \\frac{E[R] - R_f}{\\sigma_R} \\cdot \\sqrt{N}$$

    Where:
    - E[R] is the mean return
    - R_f is the risk-free rate
    - σ_R is the standard deviation of returns
    - N is the annualization factor (periods per year)

    Parameters
    ----------
    returns : array-like
        Array of period returns (as decimals, e.g., 0.01 for 1%)
    risk_free_rate : float, default 0.0
        Risk-free rate for the same period as returns
    periods_per_year : int, default 252
        Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)

    Returns
    -------
    float
        Annualized Sharpe ratio

    Examples
    --------
    >>> # Daily returns for a strategy
    >>> daily_returns = np.array([0.001, -0.002, 0.003, 0.001, -0.001])
    >>> sharpe_ratio(daily_returns, risk_free_rate=0.0001)
    0.894...

    >>> # Monthly returns
    >>> monthly_returns = np.array([0.02, -0.01, 0.03, 0.01, -0.005, 0.015])
    >>> sharpe_ratio(monthly_returns, risk_free_rate=0.002, periods_per_year=12)
    1.789...
    """
    returns = np.asarray(returns)

    if len(returns) < 2:
        logger.warning(
            "Insufficient returns data for Sharpe ratio", extra={"num_returns": len(returns)}
        )
        return 0.0

    # Calculate excess returns
    excess_returns = returns - risk_free_rate

    # Calculate mean and standard deviation
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)  # Sample standard deviation

    # Handle zero volatility
    if std_returns == 0:
        logger.warning(
            "Zero volatility in returns", extra={"mean_return": mean_excess + risk_free_rate}
        )
        return np.inf if mean_excess > 0 else (-np.inf if mean_excess < 0 else 0.0)

    # Calculate Sharpe ratio
    sharpe = mean_excess / std_returns

    # Annualize
    annualized_sharpe = sharpe * np.sqrt(periods_per_year)

    logger.debug(
        "Sharpe ratio calculated",
        extra={
            "function": "sharpe_ratio",
            "num_returns": len(returns),
            "mean_return": float(np.mean(returns)),
            "std_return": float(std_returns),
            "sharpe_ratio": float(sharpe),
            "annualized_sharpe": float(annualized_sharpe),
        },
    )

    return float(annualized_sharpe)


def win_rate(
    returns: FloatOrArray,
    threshold: float = 0.0,
) -> float:
    """
    Calculate the win rate (percentage of profitable trades).

    Win rate is the proportion of returns that exceed a threshold:
    $$WR = \\frac{\\text{count}(R_i > threshold)}{N}$$

    Parameters
    ----------
    returns : array-like
        Array of trade returns
    threshold : float, default 0.0
        Minimum return to consider a "win"

    Returns
    -------
    float
        Win rate as a fraction (0.0 to 1.0)

    Examples
    --------
    >>> returns = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
    >>> win_rate(returns)
    0.6

    >>> # With higher threshold
    >>> win_rate(returns, threshold=0.015)
    0.4
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    wins = np.sum(returns > threshold)
    rate = wins / len(returns)

    logger.debug(
        "Win rate calculated",
        extra={
            "function": "win_rate",
            "num_trades": len(returns),
            "num_wins": int(wins),
            "win_rate": float(rate),
            "threshold": threshold,
        },
    )

    return float(rate)


def profit_factor(
    returns: FloatOrArray,
) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Profit factor measures the ratio of winning trades to losing trades:
    $$PF = \\frac{\\sum_{R_i > 0} R_i}{|\\sum_{R_i < 0} R_i|}$$

    A profit factor > 1.0 indicates a profitable strategy.

    Parameters
    ----------
    returns : array-like
        Array of trade returns

    Returns
    -------
    float
        Profit factor (returns inf if no losses)

    Examples
    --------
    >>> returns = np.array([100, -50, 200, -25, 150])
    >>> profit_factor(returns)
    6.0

    >>> # Strategy with only wins
    >>> profit_factor(np.array([100, 200, 150]))
    inf
    """
    returns = np.asarray(returns)

    gross_profits = np.sum(returns[returns > 0])
    gross_losses = np.abs(np.sum(returns[returns < 0]))

    if gross_losses == 0:
        factor = np.inf if gross_profits > 0 else 0.0
    else:
        factor = gross_profits / gross_losses

    logger.debug(
        "Profit factor calculated",
        extra={
            "function": "profit_factor",
            "gross_profits": float(gross_profits),
            "gross_losses": float(gross_losses),
            "profit_factor": float(factor),
        },
    )

    return float(factor)


def maximum_drawdown(
    cumulative_returns: FloatOrArray,
) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.

    Maximum drawdown is the largest peak-to-trough decline:
    $$MDD = \\max_{t \\in [0,T]} \\left[ \\max_{s \\in [0,t]} R_s - R_t \\right]$$

    Parameters
    ----------
    cumulative_returns : array-like
        Cumulative returns or equity curve

    Returns
    -------
    tuple of (float, int, int)
        (max_drawdown, peak_index, trough_index)

    Examples
    --------
    >>> equity = np.array([10000, 11000, 10500, 12000, 11000, 10000, 11500])
    >>> mdd, peak_idx, trough_idx = maximum_drawdown(equity)
    >>> print(f"Max DD: {mdd:.2%} from index {peak_idx} to {trough_idx}")
    Max DD: 16.67% from index 3 to 5
    """
    cumulative_returns = np.asarray(cumulative_returns)

    if len(cumulative_returns) < 2:
        return 0.0, 0, 0

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Calculate drawdown at each point
    drawdown = (running_max - cumulative_returns) / running_max

    # Find maximum drawdown
    max_dd_idx = np.argmax(drawdown)
    max_dd = drawdown[max_dd_idx]

    # Find the peak (where the drawdown started)
    peak_idx = np.where(cumulative_returns[: max_dd_idx + 1] == running_max[max_dd_idx])[0][-1]

    logger.debug(
        "Maximum drawdown calculated",
        extra={
            "function": "maximum_drawdown",
            "max_drawdown": float(max_dd),
            "peak_index": int(peak_idx),
            "trough_index": int(max_dd_idx),
            "peak_value": float(cumulative_returns[peak_idx]),
            "trough_value": float(cumulative_returns[max_dd_idx]),
        },
    )

    return float(max_dd), int(peak_idx), int(max_dd_idx)


def sortino_ratio(
    returns: FloatOrArray,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (downside deviation adjusted returns).

    The Sortino ratio is similar to Sharpe but only penalizes downside volatility:
    $$SR = \\frac{E[R] - T}{\\sigma_d} \\cdot \\sqrt{N}$$

    Where:
    - E[R] is the mean return
    - T is the target return
    - σ_d is the downside deviation
    - N is the annualization factor

    Parameters
    ----------
    returns : array-like
        Array of period returns
    target_return : float, default 0.0
        Minimum acceptable return (MAR)
    periods_per_year : int, default 252
        Number of periods in a year

    Returns
    -------
    float
        Annualized Sortino ratio

    Examples
    --------
    >>> returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    >>> sortino_ratio(returns, target_return=0.0)
    1.89...
    """
    returns = np.asarray(returns)

    if len(returns) < 2:
        return 0.0

    # Calculate excess returns over target
    excess_returns = returns - target_return
    mean_excess = np.mean(excess_returns)

    # Calculate downside deviation
    downside_returns = np.minimum(excess_returns, 0)
    downside_dev = np.std(downside_returns, ddof=1)

    if downside_dev == 0:
        return np.inf if mean_excess > 0 else 0.0

    # Calculate Sortino ratio
    sortino = mean_excess / downside_dev

    # Annualize
    annualized_sortino = sortino * np.sqrt(periods_per_year)

    logger.debug(
        "Sortino ratio calculated",
        extra={
            "function": "sortino_ratio",
            "mean_return": float(np.mean(returns)),
            "downside_deviation": float(downside_dev),
            "sortino_ratio": float(sortino),
            "annualized_sortino": float(annualized_sortino),
        },
    )

    return float(annualized_sortino)
