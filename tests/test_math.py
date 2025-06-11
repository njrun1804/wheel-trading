"""Tests for options mathematics module."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.unity_wheel.math.options import (
    black_scholes_price_validated,
    calculate_all_greeks,
    implied_volatility_validated,
    probability_itm_validated,
)
from src.unity_wheel.risk.analytics import RiskAnalyzer


class TestBlackScholesPrice:
    """Test Black-Scholes pricing function."""

    def test_call_option_atm(self) -> None:
        """Test call option pricing at the money."""
        # Known value from standard calculators
        result = black_scholes_price_validated(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert abs(result.value - 10.4506) < 0.0001
        assert result.confidence > 0.9

    def test_put_option_atm(self) -> None:
        """Test put option pricing at the money."""
        result = black_scholes_price_validated(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put"
        )
        assert abs(result.value - 5.5735) < 0.0001

    def test_call_option_itm(self) -> None:
        """Test in-the-money call option."""
        result = black_scholes_price_validated(
            S=110, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert abs(result.value - 17.6630) < 0.0001

    def test_put_option_itm(self) -> None:
        """Test in-the-money put option."""
        result = black_scholes_price_validated(
            S=90, K=100, T=1, r=0.05, sigma=0.2, option_type="put"
        )
        assert abs(result.value - 10.2142) < 0.0001

    def test_deep_otm_call(self) -> None:
        """Test deep out-of-the-money call."""
        result = black_scholes_price_validated(
            S=50, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert result.value < 0.003  # Should be nearly worthless

    def test_deep_itm_call(self) -> None:
        """Test deep in-the-money call."""
        result = black_scholes_price_validated(
            S=200, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        # Should be close to intrinsic value minus discounted strike
        intrinsic = 200 - 100 * np.exp(-0.05 * 1)
        assert abs(result.value - intrinsic) < 0.1

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)."""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        call_result = black_scholes_price_validated(S, K, T, r, sigma, "call")
        put_result = black_scholes_price_validated(S, K, T, r, sigma, "put")
        parity = call_result.value - put_result.value
        expected = S - K * np.exp(-r * T)
        assert abs(parity - expected) < 1e-10

    def test_zero_time_to_expiry(self):
        """Test options at expiration."""
        # Call at expiry
        result = black_scholes_price_validated(
            S=110, K=100, T=0, r=0.05, sigma=0.2, option_type="call"
        )
        assert abs(result.value - 10) < 1e-10  # Intrinsic value

        # Put at expiry
        result = black_scholes_price_validated(
            S=90, K=100, T=0, r=0.05, sigma=0.2, option_type="put"
        )
        assert abs(result.value - 10) < 1e-10  # Intrinsic value

    def test_zero_volatility(self):
        """Test with zero volatility."""
        # Call with zero vol - should be max(S - K*exp(-rT), 0)
        result = black_scholes_price_validated(
            S=110, K=100, T=1, r=0.05, sigma=0, option_type="call"
        )
        expected = 110 - 100 * np.exp(-0.05)
        assert abs(result.value - expected) < 1e-10

        # OTM call with zero vol
        result = black_scholes_price_validated(
            S=90, K=100, T=1, r=0.05, sigma=0, option_type="call"
        )
        assert result.value == 0

    def test_high_volatility(self):
        """Test with very high volatility."""
        result = black_scholes_price_validated(
            S=100, K=100, T=1, r=0.05, sigma=2.0, option_type="call"
        )
        # Should be less than stock price but substantial
        assert 50 < result.value < 100


class TestCalculateGreeks:
    """Test Greeks calculation."""

    def test_call_delta_atm(self):
        """Test ATM call delta."""
        greeks, confidence = calculate_all_greeks(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert abs(greeks["delta"] - 0.6368) < 0.0001
        assert confidence > 0.9

    def test_put_delta_atm(self):
        """Test ATM put delta."""
        greeks, confidence = calculate_all_greeks(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put"
        )
        assert abs(greeks["delta"] - (-0.3632)) < 0.0001
        assert confidence > 0.9

    def test_call_delta_deep_itm(self):
        """Test deep ITM call delta."""
        greeks, confidence = calculate_all_greeks(
            S=200, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert greeks["delta"] > 0.99  # Should be close to 1

    def test_put_delta_deep_itm(self):
        """Test deep ITM put delta."""
        greeks, confidence = calculate_all_greeks(
            S=50, K=100, T=1, r=0.05, sigma=0.2, option_type="put"
        )
        assert greeks["delta"] < -0.99  # Should be close to -1

    def test_gamma_always_positive(self):
        """Test that gamma is always positive."""
        greeks, confidence = calculate_all_greeks(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert greeks["gamma"] > 0

    def test_greeks_dict_keys(self):
        """Test that all expected Greeks are returned."""
        greeks, confidence = calculate_all_greeks(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        expected_keys = {"delta", "gamma", "theta", "vega", "rho"}
        assert set(greeks.keys()) >= expected_keys


class TestProbabilityITM:
    """Test probability of finishing ITM calculation."""

    def test_call_probability_atm(self):
        """Test ATM call probability."""
        result = probability_itm_validated(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        # Should be slightly above 50% due to positive drift
        assert 0.5 < result.value < 0.6
        assert abs(result.value - 0.5596) < 0.0001
        assert result.confidence > 0.9

    def test_put_probability_atm(self):
        """Test ATM put probability."""
        result = probability_itm_validated(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        # Should be slightly below 50% due to positive drift
        assert 0.4 < result.value < 0.5
        assert abs(result.value - 0.4404) < 0.0001

    def test_probability_sum(self):
        """Test that call and put probabilities sum to 1."""
        call_result = probability_itm_validated(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        put_result = probability_itm_validated(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put"
        )
        assert abs(call_result.value + put_result.value - 1.0) < 1e-10

    def test_deep_itm_probability(self):
        """Test deep ITM probability."""
        # Deep ITM call
        result = probability_itm_validated(S=200, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert result.value > 0.99

        # Deep ITM put
        result = probability_itm_validated(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert result.value > 0.99

    def test_zero_time_probability(self):
        """Test probability at expiration."""
        # ITM at expiry
        result = probability_itm_validated(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert result.value == 1.0

        # OTM at expiry
        result = probability_itm_validated(S=90, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert result.value == 0.0

    def test_zero_volatility_probability(self):
        """Test probability with zero volatility."""
        # Forward price > strike
        result = probability_itm_validated(S=100, K=95, T=1, r=0.05, sigma=0, option_type="call")
        assert result.value == 1.0

        # Forward price < strike
        result = probability_itm_validated(S=100, K=110, T=1, r=0.05, sigma=0, option_type="call")
        assert result.value == 0.0


class TestImpliedVolatility:
    """Test implied volatility calculation."""

    def test_call_implied_vol(self):
        """Test IV calculation for call option."""
        # Price a call with known volatility
        true_vol = 0.2
        price_result = black_scholes_price_validated(
            S=100, K=100, T=1, r=0.05, sigma=true_vol, option_type="call"
        )

        # Recover the volatility
        iv_result = implied_volatility_validated(
            price_result.value, S=100, K=100, T=1, r=0.05, option_type="call"
        )
        assert abs(iv_result.value - true_vol) < 1e-6

    def test_put_implied_vol(self):
        """Test IV calculation for put option."""
        true_vol = 0.3
        price_result = black_scholes_price_validated(
            S=100, K=100, T=1, r=0.05, sigma=true_vol, option_type="put"
        )

        iv_result = implied_volatility_validated(
            price_result.value, S=100, K=100, T=1, r=0.05, option_type="put"
        )
        assert abs(iv_result.value - true_vol) < 1e-6

    def test_vectorized_implied_vol(self):
        """Test vectorized IV calculation."""
        vols = np.array([0.1, 0.2, 0.3, 0.4])
        price_results = [
            black_scholes_price_validated(S=100, K=100, T=1, r=0.05, sigma=vol, option_type="call")
            for vol in vols
        ]
        prices = np.array([r.value for r in price_results])

        iv_results = [
            implied_volatility_validated(p, S=100, K=100, T=1, r=0.05, option_type="call")
            for p in prices
        ]
        ivs = np.array([r.value for r in iv_results])
        np.testing.assert_allclose(ivs, vols, rtol=1e-6)

    def test_extreme_prices(self):
        """Test IV with extreme option prices."""
        # Price at lower bound (should return 0)
        lower_bound = 100 - 100 * np.exp(-0.05)
        iv_result = implied_volatility_validated(
            lower_bound, S=100, K=100, T=1, r=0.05, option_type="call"
        )
        assert iv_result.value == 0.0

        # Price at upper bound (should return inf or very high vol)
        iv_result = implied_volatility_validated(
            99.9, S=100, K=100, T=1, r=0.05, option_type="call"
        )
        assert iv_result.value >= 5.0  # Should return very high volatility

    def test_invalid_prices(self):
        """Test IV with invalid option prices."""
        # Price below lower bound
        iv_result = implied_volatility_validated(
            0.01, S=100, K=100, T=1, r=0.05, option_type="call"
        )
        assert np.isnan(iv_result.value)

        # Price above upper bound
        iv_result = implied_volatility_validated(110, S=100, K=100, T=1, r=0.05, option_type="call")
        assert np.isnan(iv_result.value)

    def test_otm_options(self):
        """Test IV for out-of-the-money options."""
        # OTM call
        price_result = black_scholes_price_validated(
            S=90, K=100, T=1, r=0.05, sigma=0.25, option_type="call"
        )
        iv_result = implied_volatility_validated(
            price_result.value, S=90, K=100, T=1, r=0.05, option_type="call"
        )
        assert abs(iv_result.value - 0.25) < 1e-6

        # OTM put
        price_result = black_scholes_price_validated(
            S=110, K=100, T=1, r=0.05, sigma=0.25, option_type="put"
        )
        iv_result = implied_volatility_validated(
            price_result.value, S=110, K=100, T=1, r=0.05, option_type="put"
        )
        assert abs(iv_result.value - 0.25) < 1e-6

    def test_short_maturity(self):
        """Test IV with short time to maturity."""
        # 1 day to expiry
        price_result = black_scholes_price_validated(
            S=100, K=100, T=1 / 365, r=0.05, sigma=0.3, option_type="call"
        )
        iv_result = implied_volatility_validated(
            price_result.value, S=100, K=100, T=1 / 365, r=0.05, option_type="call"
        )
        assert abs(iv_result.value - 0.3) < 1e-5

    def test_high_volatility_recovery(self):
        """Test recovery of high implied volatility."""
        true_vol = 1.5
        price_result = black_scholes_price_validated(
            S=100, K=100, T=0.25, r=0.05, sigma=true_vol, option_type="call"
        )
        iv_result = implied_volatility_validated(
            price_result.value, S=100, K=100, T=0.25, r=0.05, option_type="call"
        )
        assert abs(iv_result.value - true_vol) < 1e-5


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_time(self):
        """Test with very small time to expiration."""
        # 1 minute to expiry
        T = 1 / (365 * 24 * 60)
        result = black_scholes_price_validated(
            S=101, K=100, T=T, r=0.05, sigma=0.2, option_type="call"
        )
        assert abs(result.value - 1.0) < 0.01  # Should be close to intrinsic

    def test_negative_interest_rate(self):
        """Test with negative interest rates."""
        result = black_scholes_price_validated(
            S=100, K=100, T=1, r=-0.01, sigma=0.2, option_type="call"
        )
        assert result.value > 0  # Should still be positive

        # Put-call parity should still hold
        call_result = black_scholes_price_validated(
            S=100, K=100, T=1, r=-0.01, sigma=0.2, option_type="call"
        )
        put_result = black_scholes_price_validated(
            S=100, K=100, T=1, r=-0.01, sigma=0.2, option_type="put"
        )
        parity = call_result.value - put_result.value
        expected = 100 - 100 * np.exp(0.01)
        assert abs(parity - expected) < 1e-10

    def test_zero_stock_price(self):
        """Test with zero stock price."""
        # Call should return NaN with low confidence for invalid input
        result = black_scholes_price_validated(
            S=0, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert result.confidence < 0.5  # Low confidence for invalid input
        assert np.isnan(result.value) or result.value == 0  # Either NaN or 0 is acceptable

        # Put should also return NaN with low confidence for invalid input
        result = black_scholes_price_validated(
            S=0, K=100, T=1, r=0.05, sigma=0.2, option_type="put"
        )
        assert result.confidence < 0.5  # Low confidence for invalid input
        # For puts with S=0, expect either NaN or the theoretical value
        assert np.isnan(result.value) or abs(result.value - 100 * np.exp(-0.05)) < 1e-10

    def test_zero_strike(self):
        """Test with zero strike price."""
        # Call should be worth stock price
        result = black_scholes_price_validated(
            S=100, K=0, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert abs(result.value - 100) < 1e-10

        # Put should be worthless
        result = black_scholes_price_validated(
            S=100, K=0, T=1, r=0.05, sigma=0.2, option_type="put"
        )
        assert result.value == 0

    def test_array_broadcasting(self):
        """Test proper array broadcasting."""
        # Note: The current implementation returns CalculationResult, not arrays
        # Testing with individual values
        S_values = [90, 100, 110]
        results = []
        for s in S_values:
            result = black_scholes_price_validated(
                S=s, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
            )
            results.append(result.value)

        # Verify prices increase with S
        assert results[0] < results[1] < results[2]

    def test_mixed_edge_cases(self):
        """Test combinations of edge cases."""
        # Test individual edge cases

        # First: ATM at expiry
        result1 = black_scholes_price_validated(
            S=100, K=100, T=0, r=0.05, sigma=0.2, option_type="call"
        )
        assert abs(result1.value - 0) < 1e-10

        # Second: Zero strike
        result2 = black_scholes_price_validated(
            S=100, K=0, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert abs(result2.value - 100) < 1e-10

        # Third: Zero stock price
        result3 = black_scholes_price_validated(
            S=0, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert result3.confidence < 0.5  # Low confidence for invalid input
        assert np.isnan(result3.value) or result3.value == 0  # Either NaN or 0 is acceptable

        # Fourth: ITM at expiry with zero vol
        result4 = black_scholes_price_validated(
            S=200, K=100, T=0, r=0.05, sigma=0, option_type="call"
        )
        assert abs(result4.value - 100) < 1e-10


class TestNormCdfCache:
    """Ensure normal CDF caching operates correctly."""

    def test_norm_cdf_caching(self) -> None:
        from src.unity_wheel.math.options import _cached_norm_cdf_scalar, norm_cdf_cached

        _cached_norm_cdf_scalar.cache_clear()
        info_start = _cached_norm_cdf_scalar.cache_info()
        assert info_start.hits == 0 and info_start.misses == 0

        val1 = norm_cdf_cached(0.7)
        assert np.isclose(val1, norm.cdf(0.7))
        info_after_first = _cached_norm_cdf_scalar.cache_info()
        assert info_after_first.misses == 1

        val2 = norm_cdf_cached(0.7)
        info_after_second = _cached_norm_cdf_scalar.cache_info()
        assert val1 == val2
        assert info_after_second.hits == 1


# Risk analytics tests moved to test_risk.py since these are now part of RiskAnalyzer class
