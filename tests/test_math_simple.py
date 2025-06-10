"""Simple tests for options mathematics module."""

from __future__ import annotations

import numpy as np

from src.unity_wheel.math import (
    black_scholes_price_validated,
    calculate_all_greeks,
    implied_volatility_validated,
    probability_itm_validated,
)


def test_black_scholes_call():
    """Test Black-Scholes call pricing."""
    result = black_scholes_price_validated(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert abs(result.value - 10.4506) < 0.001
    assert result.confidence > 0.9


def test_black_scholes_put():
    """Test Black-Scholes put pricing."""
    result = black_scholes_price_validated(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    assert abs(result.value - 5.5735) < 0.001
    assert result.confidence > 0.9


def test_greeks():
    """Test Greeks calculation."""
    greeks, confidence = calculate_all_greeks(
        S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
    )
    assert 0.5 < greeks["delta"] < 0.7
    assert greeks["gamma"] > 0
    assert greeks["theta"] < 0
    assert greeks["vega"] > 0
    assert confidence > 0.9


def test_probability_itm():
    """Test probability ITM calculation."""
    result = probability_itm_validated(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert 0.4 < result.value < 0.6  # Should be around 50% for ATM
    assert result.confidence > 0.9


def test_implied_volatility():
    """Test implied volatility calculation."""
    # First calculate a price with known vol
    price_result = black_scholes_price_validated(
        S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
    )

    # Then recover the vol
    iv_result = implied_volatility_validated(
        option_price=price_result.value, S=100, K=100, T=1, r=0.05, option_type="call"
    )
    assert abs(iv_result.value - 0.2) < 0.001
    assert iv_result.confidence > 0.9
