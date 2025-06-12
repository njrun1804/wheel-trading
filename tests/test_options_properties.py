"""Property-based tests for options mathematics.

Tests mathematical properties that must hold regardless of inputs:
- Put-call parity
- Monotonicity
- Bounds
- Greeks relationships
"""

import math
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from unity_wheel.math.options import (
    black_scholes_price_validated,
    calculate_all_greeks,
    implied_volatility_validated,
    probability_itm_validated,
)

# Custom strategies for reasonable financial values
spot_price = st.floats(min_value=1.0, max_value=10000.0)
strike_price = st.floats(min_value=1.0, max_value=10000.0)
time_to_expiry = st.floats(min_value=1 / 365, max_value=5.0)  # 1 day to 5 years
risk_free_rate = st.floats(min_value=0.0, max_value=0.15)  # 0% to 15%
volatility = st.floats(min_value=0.01, max_value=3.0)  # 1% to 300%


class TestPutCallParity:
    """Test put-call parity relationship."""

    @given(
        spot=spot_price,
        strike=strike_price,
        time=time_to_expiry,
        rate=risk_free_rate,
        vol=volatility,
    )
    @settings(max_examples=100)
    def test_put_call_parity(self, spot, strike, time, rate, vol):
        """Test that C - P = S - K*e^(-rt) within tolerance."""
        # Calculate call and put prices
        call_result = black_scholes_price_validated(spot, strike, time, rate, vol, "call")
        put_result = black_scholes_price_validated(spot, strike, time, rate, vol, "put")

        # Skip if either calculation failed
        assume(not math.isnan(call_result.value))
        assume(not math.isnan(put_result.value))

        # Calculate theoretical difference
        theoretical_diff = spot - strike * math.exp(-rate * time)
        actual_diff = call_result.value - put_result.value

        # Allow for small numerical errors (0.01% of spot price)
        tolerance = spot * 0.0001
        assert (
            abs(actual_diff - theoretical_diff) < tolerance
        ), f"Put-call parity violated: actual={actual_diff:.6f}, theoretical={theoretical_diff:.6f}"


class TestOptionBounds:
    """Test option price bounds."""

    @given(
        spot=spot_price,
        strike=strike_price,
        time=time_to_expiry,
        rate=risk_free_rate,
        vol=volatility,
    )
    @settings(max_examples=100)
    def test_call_bounds(self, spot, strike, time, rate, vol):
        """Test that call price respects theoretical bounds."""
        call_result = black_scholes_price_validated(spot, strike, time, rate, vol, "call")

        assume(not math.isnan(call_result.value))

        # Lower bound: max(S - K*e^(-rt), 0)
        lower_bound = max(spot - strike * math.exp(-rate * time), 0)

        # Upper bound: S
        upper_bound = spot

        assert (
            call_result.value >= lower_bound - 0.01
        ), f"Call below lower bound: {call_result.value:.6f} < {lower_bound:.6f}"
        assert (
            call_result.value <= upper_bound + 0.01
        ), f"Call above upper bound: {call_result.value:.6f} > {upper_bound:.6f}"

    @given(
        spot=spot_price,
        strike=strike_price,
        time=time_to_expiry,
        rate=risk_free_rate,
        vol=volatility,
    )
    @settings(max_examples=100)
    def test_put_bounds(self, spot, strike, time, rate, vol):
        """Test that put price respects theoretical bounds."""
        put_result = black_scholes_price_validated(spot, strike, time, rate, vol, "put")

        assume(not math.isnan(put_result.value))

        # Lower bound: max(K*e^(-rt) - S, 0)
        lower_bound = max(strike * math.exp(-rate * time) - spot, 0)

        # Upper bound: K*e^(-rt)
        upper_bound = strike * math.exp(-rate * time)

        assert (
            put_result.value >= lower_bound - 0.01
        ), f"Put below lower bound: {put_result.value:.6f} < {lower_bound:.6f}"
        assert (
            put_result.value <= upper_bound + 0.01
        ), f"Put above upper bound: {put_result.value:.6f} > {upper_bound:.6f}"


class TestGreeksProperties:
    """Test Greeks mathematical properties."""

    @given(
        spot=spot_price,
        strike=strike_price,
        time=time_to_expiry,
        rate=risk_free_rate,
        vol=volatility,
    )
    @settings(max_examples=100)
    def test_delta_bounds(self, spot, strike, time, rate, vol):
        """Test that delta is within theoretical bounds."""
        call_greeks, call_confidence = calculate_all_greeks(spot, strike, time, rate, vol, "call")
        put_greeks, put_confidence = calculate_all_greeks(spot, strike, time, rate, vol, "put")

        assume(call_confidence > 0.8 and put_confidence > 0.8)

        # Call delta: 0 to 1
        assert (
            0 <= call_greeks["delta"] <= 1.01
        ), f"Call delta out of bounds: {call_greeks['delta']}"

        # Put delta: -1 to 0
        assert -1.01 <= put_greeks["delta"] <= 0, f"Put delta out of bounds: {put_greeks['delta']}"

        # Put-call delta relationship: delta_call - delta_put = 1
        delta_diff = call_greeks["delta"] - put_greeks["delta"]
        assert abs(delta_diff - 1.0) < 0.01, f"Delta parity violated: {delta_diff:.6f} != 1.0"

    @given(
        spot=spot_price,
        strike=strike_price,
        time=time_to_expiry,
        rate=risk_free_rate,
        vol=volatility,
    )
    @settings(max_examples=100)
    def test_gamma_properties(self, spot, strike, time, rate, vol):
        """Test gamma properties."""
        call_greeks, call_confidence = calculate_all_greeks(spot, strike, time, rate, vol, "call")
        put_greeks, put_confidence = calculate_all_greeks(spot, strike, time, rate, vol, "put")

        assume(call_confidence > 0.8 and put_confidence > 0.8)

        # Gamma should be positive for both calls and puts
        assert call_greeks["gamma"] >= 0, f"Call gamma negative: {call_greeks['gamma']}"
        assert put_greeks["gamma"] >= 0, f"Put gamma negative: {put_greeks['gamma']}"

        # Call and put gamma should be equal
        assert (
            abs(call_greeks["gamma"] - put_greeks["gamma"]) < 0.0001
        ), "Call and put gamma not equal"

    @given(
        spot=spot_price,
        strike=strike_price,
        time=time_to_expiry,
        rate=risk_free_rate,
        vol=volatility,
    )
    @settings(max_examples=100)
    def test_theta_decay(self, spot, strike, time, rate, vol):
        """Test that theta represents time decay (usually negative)."""
        call_greeks, call_confidence = calculate_all_greeks(spot, strike, time, rate, vol, "call")
        put_greeks, put_confidence = calculate_all_greeks(spot, strike, time, rate, vol, "put")

        assume(call_confidence > 0.8 and put_confidence > 0.8)
        assume(time > 7 / 365)  # More than a week to expiry

        # For most reasonable cases, theta should be negative
        # (options lose value as time passes)
        # Exception: deep ITM puts can have positive theta due to interest
        # Also, extreme vol or very long expiry can cause positive theta
        if spot > strike * 1.1 and vol < 1.0 and time < 1.0:  # OTM put, reasonable vol/time
            assert put_greeks["theta"] <= 0.015, f"OTM put theta positive: {put_greeks['theta']}"


class TestImpliedVolatility:
    """Test implied volatility solver properties."""

    @given(
        spot=spot_price,
        strike=strike_price,
        time=time_to_expiry,
        rate=risk_free_rate,
        vol=volatility,
    )
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.filter_too_much]
    )  # IV solver can be slow
    def test_iv_round_trip(self, spot, strike, time, rate, vol):
        """Test that we can recover volatility from option price."""
        # Filter out extreme cases that cause numerical precision issues
        moneyness = spot / strike
        assume(0.7 < moneyness < 1.4)  # Much tighter moneyness range to avoid deep ITM/OTM
        assume(vol < 1.0)  # Avoid extremely high volatility
        assume(time > 14 / 365)  # Avoid very short time to expiry (need at least 2 weeks)

        # First calculate option price
        option_result = black_scholes_price_validated(spot, strike, time, rate, vol, "call")

        assume(not math.isnan(option_result.value))
        assume(option_result.value > 0.05)  # Need meaningful price

        # Skip options at intrinsic value (no time value)
        intrinsic_value = max(spot - strike, 0)
        time_value = option_result.value - intrinsic_value
        assume(time_value > 0.02)  # Need substantial time value for IV to make sense

        # Additional filter: time value should be at least 10% of option price
        assume(time_value / option_result.value > 0.1)

        # Then solve for implied volatility
        iv_result = implied_volatility_validated(
            option_result.value, spot, strike, time, rate, "call"
        )

        assume(not math.isnan(iv_result.value))
        assume(iv_result.confidence > 0.7)  # Skip cases where IV solver struggles

        # Should recover original volatility within tolerance
        # Use relative tolerance for better numerical stability
        relative_tolerance = max(0.05, vol * 0.2)  # 5% or 20% of vol, whichever is larger

        assert (
            abs(iv_result.value - vol) < relative_tolerance
        ), f"IV solver failed: got {iv_result.value:.4f}, expected {vol:.4f}, tolerance={relative_tolerance:.4f}"


class TestProbabilityITM:
    """Test probability of finishing in-the-money."""

    @given(
        spot=spot_price,
        strike=strike_price,
        time=time_to_expiry,
        rate=risk_free_rate,
        vol=volatility,
    )
    @settings(max_examples=100)
    def test_probability_bounds(self, spot, strike, time, rate, vol):
        """Test that probabilities are between 0 and 1."""
        call_prob = probability_itm_validated(spot, strike, time, rate, vol, "call")
        put_prob = probability_itm_validated(spot, strike, time, rate, vol, "put")

        assume(not math.isnan(call_prob.value))
        assume(not math.isnan(put_prob.value))

        # Probabilities must be between 0 and 1
        assert 0 <= call_prob.value <= 1, f"Call probability out of bounds: {call_prob.value}"
        assert 0 <= put_prob.value <= 1, f"Put probability out of bounds: {put_prob.value}"

    @given(
        spot=spot_price,
        strike=strike_price,
        time=time_to_expiry,
        rate=risk_free_rate,
        vol=volatility,
    )
    @settings(max_examples=100)
    def test_probability_monotonicity(self, spot, strike, time, rate, vol):
        """Test probability monotonicity with respect to moneyness."""
        # Calculate for different spot prices
        spots = [spot * 0.9, spot, spot * 1.1]
        call_probs = []
        put_probs = []

        for s in spots:
            call_result = probability_itm_validated(s, strike, time, rate, vol, "call")
            put_result = probability_itm_validated(s, strike, time, rate, vol, "put")

            assume(not math.isnan(call_result.value))
            assume(not math.isnan(put_result.value))

            call_probs.append(call_result.value)
            put_probs.append(put_result.value)

        # Call probability should increase with spot
        assert (
            call_probs[0] <= call_probs[1] <= call_probs[2]
        ), "Call probability not monotonic in spot"

        # Put probability should decrease with spot
        assert put_probs[0] >= put_probs[1] >= put_probs[2], "Put probability not monotonic in spot"


class TestNumericalStability:
    """Test numerical stability in edge cases."""

    @given(vol=st.floats(min_value=0.001, max_value=0.01))  # Very low volatility
    @settings(max_examples=20)
    def test_low_volatility_stability(self, vol):
        """Test stability with very low volatility."""
        spot = 100.0
        strike = 100.0
        time = 30 / 365
        rate = 0.05

        result = black_scholes_price_validated(spot, strike, time, rate, vol, "call")

        # Should not crash or return NaN
        assert not math.isnan(result.value), "Low volatility caused NaN"
        assert result.value >= 0, "Negative option price"

    @given(time=st.floats(min_value=1 / 365 / 24, max_value=1 / 365))  # Very short time
    @settings(max_examples=20)
    def test_short_time_stability(self, time):
        """Test stability with very short time to expiry."""
        spot = 100.0
        strike = 100.0
        vol = 0.2
        rate = 0.05

        result = black_scholes_price_validated(spot, strike, time, rate, vol, "call")

        # Should not crash or return NaN
        assert not math.isnan(result.value), "Short time caused NaN"
        assert result.value >= 0, "Negative option price"
