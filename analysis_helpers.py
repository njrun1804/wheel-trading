"""Helper functions for mathematical analysis and validation."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def validate_option_params(delta: float, iv: float, dte: int) -> dict[str, Any]:
    """Validate option parameters are within reasonable bounds."""
    issues = []

    if not 0 < delta < 1:
        issues.append(f"Delta {delta} outside valid range (0,1)")
    if not 0 < iv < 5:  # 500% IV is extreme but possible
        issues.append(f"IV {iv} outside reasonable range (0,5)")
    if dte < 0:
        issues.append(f"DTE {dte} cannot be negative")

    return {"valid": len(issues) == 0, "issues": issues}


def calculate_kelly_criterion(
    win_prob: float, win_amt: float, loss_amt: float
) -> float:
    """Calculate optimal position size using Kelly Criterion."""
    if loss_amt == 0:
        return 0

    b = win_amt / loss_amt
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b
    return max(0, min(kelly, 0.25))  # Cap at 25% for safety


def analyze_returns_distribution(returns: pd.Series) -> dict[str, float]:
    """Analyze return distribution for risk metrics."""
    return {
        "mean": returns.mean(),
        "std": returns.std(),
        "skew": stats.skew(returns),
        "kurtosis": stats.kurtosis(returns),
        "sharpe": returns.mean() / returns.std() * np.sqrt(252),
        "var_95": returns.quantile(0.05),
        "cvar_95": returns[returns <= returns.quantile(0.05)].mean(),
    }
