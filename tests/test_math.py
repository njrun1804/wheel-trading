"""Tests for options mathematics module."""

import pytest
import numpy as np
from src.utils.math import black_scholes_price, calculate_delta, probability_itm, implied_volatility


class TestBlackScholesPrice:
    """Test Black-Scholes pricing function."""

    def test_call_option_atm(self):
        """Test call option pricing at the money."""
        # Known value from standard calculators
        price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert abs(price - 10.4506) < 0.0001

    def test_put_option_atm(self):
        """Test put option pricing at the money."""
        price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert abs(price - 5.5735) < 0.0001

    def test_call_option_itm(self):
        """Test in-the-money call option."""
        price = black_scholes_price(S=110, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert abs(price - 17.6630) < 0.0001

    def test_put_option_itm(self):
        """Test in-the-money put option."""
        price = black_scholes_price(S=90, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert abs(price - 10.2142) < 0.0001

    def test_deep_otm_call(self):
        """Test deep out-of-the-money call."""
        price = black_scholes_price(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert price < 0.003  # Should be nearly worthless

    def test_deep_itm_call(self):
        """Test deep in-the-money call."""
        price = black_scholes_price(S=200, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        # Should be close to intrinsic value minus discounted strike
        intrinsic = 200 - 100 * np.exp(-0.05 * 1)
        assert abs(price - intrinsic) < 0.1

    def test_vectorized_pricing(self):
        """Test vectorized computation."""
        S = np.array([90, 100, 110])
        K = 100
        prices = black_scholes_price(S=S, K=K, T=1, r=0.05, sigma=0.2, option_type="call")
        assert len(prices) == 3
        assert prices[0] < prices[1] < prices[2]  # Monotonic in S

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)."""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        call = black_scholes_price(S, K, T, r, sigma, "call")
        put = black_scholes_price(S, K, T, r, sigma, "put")
        parity = call - put
        expected = S - K * np.exp(-r * T)
        assert abs(parity - expected) < 1e-10

    def test_zero_time_to_expiry(self):
        """Test options at expiration."""
        # Call at expiry
        price = black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert abs(price - 10) < 1e-10  # Intrinsic value

        # Put at expiry
        price = black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.2, option_type="put")
        assert abs(price - 10) < 1e-10  # Intrinsic value

    def test_zero_volatility(self):
        """Test with zero volatility."""
        # Call with zero vol - should be max(S - K*exp(-rT), 0)
        price = black_scholes_price(S=110, K=100, T=1, r=0.05, sigma=0, option_type="call")
        expected = 110 - 100 * np.exp(-0.05)
        assert abs(price - expected) < 1e-10

        # OTM call with zero vol
        price = black_scholes_price(S=90, K=100, T=1, r=0.05, sigma=0, option_type="call")
        assert price == 0

    def test_high_volatility(self):
        """Test with very high volatility."""
        price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=2.0, option_type="call")
        # Should be less than stock price but substantial
        assert 50 < price < 100


class TestCalculateDelta:
    """Test delta calculation."""

    def test_call_delta_atm(self):
        """Test ATM call delta."""
        delta = calculate_delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert abs(delta - 0.6368) < 0.0001

    def test_put_delta_atm(self):
        """Test ATM put delta."""
        delta = calculate_delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert abs(delta - (-0.3632)) < 0.0001

    def test_call_delta_deep_itm(self):
        """Test deep ITM call delta."""
        delta = calculate_delta(S=200, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert delta > 0.99  # Should be close to 1

    def test_put_delta_deep_itm(self):
        """Test deep ITM put delta."""
        delta = calculate_delta(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert delta < -0.99  # Should be close to -1

    def test_call_delta_deep_otm(self):
        """Test deep OTM call delta."""
        delta = calculate_delta(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert delta < 0.01  # Should be close to 0

    def test_delta_at_expiry(self):
        """Test delta at expiration."""
        # ITM call at expiry
        delta = calculate_delta(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert delta == 1.0

        # OTM call at expiry
        delta = calculate_delta(S=90, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert delta == 0.0

    def test_delta_zero_volatility(self):
        """Test delta with zero volatility."""
        # ITM call with zero vol
        delta = calculate_delta(S=110, K=100, T=1, r=0.05, sigma=0, option_type="call")
        assert delta == 1.0

        # OTM call with zero vol
        delta = calculate_delta(S=95, K=100, T=1, r=0.05, sigma=0, option_type="call")
        assert delta == 0.0

    def test_put_call_delta_relationship(self):
        """Test that call_delta - put_delta = 1."""
        call_delta = calculate_delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        put_delta = calculate_delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert abs((call_delta - put_delta) - 1.0) < 1e-10


class TestProbabilityITM:
    """Test probability of finishing ITM calculation."""

    def test_call_probability_atm(self):
        """Test ATM call probability."""
        prob = probability_itm(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        # Should be slightly above 50% due to positive drift
        assert 0.5 < prob < 0.6
        assert abs(prob - 0.5596) < 0.0001

    def test_put_probability_atm(self):
        """Test ATM put probability."""
        prob = probability_itm(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        # Should be slightly below 50% due to positive drift
        assert 0.4 < prob < 0.5
        assert abs(prob - 0.4404) < 0.0001

    def test_probability_sum(self):
        """Test that call and put probabilities sum to 1."""
        call_prob = probability_itm(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        put_prob = probability_itm(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert abs(call_prob + put_prob - 1.0) < 1e-10

    def test_deep_itm_probability(self):
        """Test deep ITM probability."""
        # Deep ITM call
        prob = probability_itm(S=200, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert prob > 0.99

        # Deep ITM put
        prob = probability_itm(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert prob > 0.99

    def test_zero_time_probability(self):
        """Test probability at expiration."""
        # ITM at expiry
        prob = probability_itm(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert prob == 1.0

        # OTM at expiry
        prob = probability_itm(S=90, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert prob == 0.0

    def test_zero_volatility_probability(self):
        """Test probability with zero volatility."""
        # Forward price > strike
        prob = probability_itm(S=100, K=95, T=1, r=0.05, sigma=0, option_type="call")
        assert prob == 1.0

        # Forward price < strike
        prob = probability_itm(S=100, K=110, T=1, r=0.05, sigma=0, option_type="call")
        assert prob == 0.0


class TestImpliedVolatility:
    """Test implied volatility calculation."""

    def test_call_implied_vol(self):
        """Test IV calculation for call option."""
        # Price a call with known volatility
        true_vol = 0.2
        price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=true_vol, option_type="call")

        # Recover the volatility
        iv = implied_volatility(price, S=100, K=100, T=1, r=0.05, option_type="call")
        assert abs(iv - true_vol) < 1e-6

    def test_put_implied_vol(self):
        """Test IV calculation for put option."""
        true_vol = 0.3
        price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=true_vol, option_type="put")

        iv = implied_volatility(price, S=100, K=100, T=1, r=0.05, option_type="put")
        assert abs(iv - true_vol) < 1e-6

    def test_vectorized_implied_vol(self):
        """Test vectorized IV calculation."""
        vols = np.array([0.1, 0.2, 0.3, 0.4])
        prices = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=vols, option_type="call")

        ivs = implied_volatility(prices, S=100, K=100, T=1, r=0.05, option_type="call")
        np.testing.assert_allclose(ivs, vols, rtol=1e-6)

    def test_extreme_prices(self):
        """Test IV with extreme option prices."""
        # Price at lower bound (should return 0)
        lower_bound = 100 - 100 * np.exp(-0.05)
        iv = implied_volatility(lower_bound, S=100, K=100, T=1, r=0.05, option_type="call")
        assert iv == 0.0

        # Price at upper bound (should return inf or very high vol)
        iv = implied_volatility(99.9, S=100, K=100, T=1, r=0.05, option_type="call")
        assert iv >= 5.0  # Should return very high volatility

    def test_invalid_prices(self):
        """Test IV with invalid option prices."""
        # Price below lower bound
        iv = implied_volatility(0.01, S=100, K=100, T=1, r=0.05, option_type="call")
        assert np.isnan(iv)

        # Price above upper bound
        iv = implied_volatility(110, S=100, K=100, T=1, r=0.05, option_type="call")
        assert np.isnan(iv)

    def test_otm_options(self):
        """Test IV for out-of-the-money options."""
        # OTM call
        price = black_scholes_price(S=90, K=100, T=1, r=0.05, sigma=0.25, option_type="call")
        iv = implied_volatility(price, S=90, K=100, T=1, r=0.05, option_type="call")
        assert abs(iv - 0.25) < 1e-6

        # OTM put
        price = black_scholes_price(S=110, K=100, T=1, r=0.05, sigma=0.25, option_type="put")
        iv = implied_volatility(price, S=110, K=100, T=1, r=0.05, option_type="put")
        assert abs(iv - 0.25) < 1e-6

    def test_short_maturity(self):
        """Test IV with short time to maturity."""
        # 1 day to expiry
        price = black_scholes_price(S=100, K=100, T=1 / 365, r=0.05, sigma=0.3, option_type="call")
        iv = implied_volatility(price, S=100, K=100, T=1 / 365, r=0.05, option_type="call")
        assert abs(iv - 0.3) < 1e-5

    def test_high_volatility_recovery(self):
        """Test recovery of high implied volatility."""
        true_vol = 1.5
        price = black_scholes_price(
            S=100, K=100, T=0.25, r=0.05, sigma=true_vol, option_type="call"
        )
        iv = implied_volatility(price, S=100, K=100, T=0.25, r=0.05, option_type="call")
        assert abs(iv - true_vol) < 1e-5


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_time(self):
        """Test with very small time to expiration."""
        # 1 minute to expiry
        T = 1 / (365 * 24 * 60)
        price = black_scholes_price(S=101, K=100, T=T, r=0.05, sigma=0.2, option_type="call")
        assert abs(price - 1.0) < 0.01  # Should be close to intrinsic

    def test_negative_interest_rate(self):
        """Test with negative interest rates."""
        price = black_scholes_price(S=100, K=100, T=1, r=-0.01, sigma=0.2, option_type="call")
        assert price > 0  # Should still be positive

        # Put-call parity should still hold
        call = black_scholes_price(S=100, K=100, T=1, r=-0.01, sigma=0.2, option_type="call")
        put = black_scholes_price(S=100, K=100, T=1, r=-0.01, sigma=0.2, option_type="put")
        parity = call - put
        expected = 100 - 100 * np.exp(0.01)
        assert abs(parity - expected) < 1e-10

    def test_zero_stock_price(self):
        """Test with zero stock price."""
        # Call should be worthless
        price = black_scholes_price(S=0, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert price == 0

        # Put should be worth discounted strike
        price = black_scholes_price(S=0, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert abs(price - 100 * np.exp(-0.05)) < 1e-10

    def test_zero_strike(self):
        """Test with zero strike price."""
        # Call should be worth stock price
        price = black_scholes_price(S=100, K=0, T=1, r=0.05, sigma=0.2, option_type="call")
        assert abs(price - 100) < 1e-10

        # Put should be worthless
        price = black_scholes_price(S=100, K=0, T=1, r=0.05, sigma=0.2, option_type="put")
        assert price == 0

    def test_array_broadcasting(self):
        """Test proper array broadcasting."""
        S = np.array([90, 100, 110])
        K = np.array([100])
        T = 1

        prices = black_scholes_price(S=S, K=K, T=T, r=0.05, sigma=0.2, option_type="call")
        assert prices.shape == (3,)

        # Test with multiple dimensions
        S = np.array([[90, 100], [110, 120]])
        prices = black_scholes_price(S=S, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert prices.shape == (2, 2)

    def test_mixed_edge_cases(self):
        """Test combinations of edge cases."""
        S = np.array([100, 100, 0, 200])
        K = np.array([100, 0, 100, 100])
        T = np.array([0, 1, 1, 0])
        sigma = np.array([0.2, 0.2, 0.2, 0])

        calls = black_scholes_price(S=S, K=K, T=T, r=0.05, sigma=sigma, option_type="call")

        # First: ATM at expiry
        assert abs(calls[0] - 0) < 1e-10

        # Second: Zero strike
        assert abs(calls[1] - 100) < 1e-10

        # Third: Zero stock price
        assert calls[2] == 0

        # Fourth: ITM at expiry with zero vol
        assert abs(calls[3] - 100) < 1e-10
