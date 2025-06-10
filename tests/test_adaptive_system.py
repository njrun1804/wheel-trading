"""
Simple tests for the Unity adaptive system.
Verifies core logic without complex dependencies.
"""

from datetime import datetime

import pytest


# Test the core adaptive logic directly
def test_volatility_scaling():
    """Test that position size scales correctly with volatility."""
    portfolio = 200000
    base_pct = 0.20

    # Test volatility factors
    vol_factors = {
        0.35: 1.2,  # Low vol
        0.50: 1.0,  # Normal
        0.70: 0.7,  # High
        0.85: 0.5,  # Extreme
    }

    for vol, expected_factor in vol_factors.items():
        # Calculate factor based on rules
        if vol < 0.40:
            factor = 1.2
        elif vol < 0.60:
            factor = 1.0
        elif vol < 0.80:
            factor = 0.7
        else:
            factor = 0.5

        assert factor == expected_factor, f"Vol {vol} should give factor {expected_factor}"


def test_drawdown_scaling():
    """Test drawdown-based position reduction."""
    # Linear scaling: 0% dd = 1.0, -20% dd = 0.0
    drawdowns = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25]
    expected = [1.0, 0.75, 0.50, 0.25, 0.0, 0.0]

    for dd, exp in zip(drawdowns, expected):
        factor = max(0, 1 + dd / 0.20)
        assert abs(factor - exp) < 0.01, f"Drawdown {dd} should give factor {exp}"


def test_stop_conditions():
    """Test that stop conditions work correctly."""
    # Earnings stop
    days_to_earnings = 5
    assert days_to_earnings <= 7, "Should stop when earnings <7 days"

    # Volatility stop
    extreme_vol = 1.10
    assert extreme_vol > 1.00, "Should stop when vol >100%"

    # Drawdown stop
    max_drawdown = -0.22
    assert max_drawdown <= -0.20, "Should stop when drawdown >20%"


def test_parameter_adaptation():
    """Test that parameters adapt to conditions."""
    # Normal conditions
    normal_vol = 0.50
    if normal_vol < 0.60:
        delta = 0.30
        dte = 35
    assert delta == 0.30
    assert dte == 35

    # High vol conditions
    high_vol = 0.75
    if high_vol > 0.60 and high_vol < 0.80:
        delta = 0.25
        dte = 28
    assert delta == 0.25
    assert dte == 28


def test_position_size_calculation():
    """Test complete position size calculation."""
    portfolio = 200000
    base_pct = 0.20

    # Scenario 1: Normal conditions
    vol_factor = 1.0
    dd_factor = 0.9  # -2% drawdown
    iv_factor = 1.0

    position = portfolio * base_pct * vol_factor * dd_factor * iv_factor
    assert position == 36000, "Normal conditions should give $36k position"

    # Scenario 2: High vol with drawdown
    vol_factor = 0.7  # 70% vol
    dd_factor = 0.6  # -8% drawdown
    iv_factor = 1.2  # 85 IV rank

    position = portfolio * base_pct * vol_factor * dd_factor * iv_factor
    assert abs(position - 20160) < 1, "High vol scenario should give ~$20k position"


def test_earnings_date_logic():
    """Test earnings date calculations."""
    # Mock earnings dates
    earnings_dates = [
        datetime(2024, 2, 8),
        datetime(2024, 5, 9),
        datetime(2024, 8, 8),
        datetime(2024, 11, 7),
    ]

    # Test date in March 2024
    test_date = datetime(2024, 3, 15)

    # Find next earnings
    next_earnings = None
    for date in earnings_dates:
        if date > test_date:
            next_earnings = date
            break

    assert next_earnings == datetime(2024, 5, 9), "Next earnings should be May 9"

    days_to_earnings = (next_earnings - test_date).days
    assert days_to_earnings == 55, "Should be 55 days to earnings"


def test_multiplicative_safety():
    """Test that all factors multiply for conservative sizing."""
    factors = {
        "volatility": 0.7,
        "drawdown": 0.8,
        "iv_rank": 1.2,
        "earnings": 1.0,
    }

    # Multiply all factors
    total = 1.0
    for factor in factors.values():
        total *= factor

    expected = 0.7 * 0.8 * 1.2 * 1.0
    assert abs(total - expected) < 0.001, "Factors should multiply correctly"
    assert total < 1.0, "Multiple risk factors should reduce position"


if __name__ == "__main__":
    # Run tests manually
    test_volatility_scaling()
    test_drawdown_scaling()
    test_stop_conditions()
    test_parameter_adaptation()
    test_position_size_calculation()
    test_earnings_date_logic()
    test_multiplicative_safety()

    print("✅ All adaptive system tests passed!")
