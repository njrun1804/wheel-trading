"""Tests for options mathematics module."""

from __future__ import annotations

import numpy as np

from unity_wheel.math.options import (
    black_scholes_price,
    calculate_delta,
    implied_volatility,
    probability_itm,
)
from unity_wheel.risk.analytics import (
    calculate_var,
    calculate_cvar,
    half_kelly_size,
    margin_requirement,
)


class TestBlackScholesPrice:
    """Test Black-Scholes pricing function."""

    def test_call_option_atm(self) -> None:
        """Test call option pricing at the money."""
        # Known value from standard calculators
        price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert abs(price - 10.4506) < 0.0001

    def test_put_option_atm(self) -> None:
        """Test put option pricing at the money."""
        price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert abs(price - 5.5735) < 0.0001

    def test_call_option_itm(self) -> None:
        """Test in-the-money call option."""
        price = black_scholes_price(S=110, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert abs(price - 17.6630) < 0.0001

    def test_put_option_itm(self) -> None:
        """Test in-the-money put option."""
        price = black_scholes_price(S=90, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert abs(price - 10.2142) < 0.0001

    def test_deep_otm_call(self) -> None:
        """Test deep out-of-the-money call."""
        price = black_scholes_price(S=50, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert price < 0.003  # Should be nearly worthless

    def test_deep_itm_call(self) -> None:
        """Test deep in-the-money call."""
        price = black_scholes_price(S=200, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        # Should be close to intrinsic value minus discounted strike
        intrinsic = 200 - 100 * np.exp(-0.05 * 1)
        assert abs(price - intrinsic) < 0.1

    def test_vectorized_pricing(self) -> None:
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


class TestCalculateVaR:
    """Test Value at Risk calculation."""

    def test_var_with_volatility_only(self):
        """Test VaR with volatility input (assumes zero mean)."""
        # 20% annual volatility, 95% confidence
        var = calculate_var(0.2, confidence_level=0.95)
        # VaR = -μ - σ * z_0.05 = 0 - 0.2 * (-1.645) = 0.329
        assert abs(var - 0.329) < 0.001

    def test_var_with_historical_returns(self):
        """Test VaR with historical returns."""
        # Generate returns with known statistics
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var = calculate_var(returns, confidence_level=0.95)
        # Should be approximately 1.645 * 0.02 = 0.0329
        assert 0.025 < var < 0.035

    def test_var_time_scaling(self):
        """Test VaR scaling with time horizon."""
        daily_vol = 0.02

        # Daily VaR
        var_1d = calculate_var(daily_vol, confidence_level=0.95, time_horizon=1)

        # Monthly VaR (21 trading days)
        var_21d = calculate_var(daily_vol, confidence_level=0.95, time_horizon=21)

        # Should scale with sqrt(time)
        assert abs(var_21d / var_1d - np.sqrt(21)) < 0.001

    def test_var_confidence_levels(self):
        """Test VaR at different confidence levels."""
        vol = 0.2

        var_90 = calculate_var(vol, confidence_level=0.90)
        var_95 = calculate_var(vol, confidence_level=0.95)
        var_99 = calculate_var(vol, confidence_level=0.99)

        # Higher confidence = higher VaR
        assert var_90 < var_95 < var_99

        # Check specific values
        assert abs(var_90 - 0.2 * 1.282) < 0.001
        assert abs(var_95 - 0.2 * 1.645) < 0.001
        assert abs(var_99 - 0.2 * 2.326) < 0.001

    def test_var_with_positive_mean(self):
        """Test VaR with positive expected returns."""
        returns = np.random.normal(0.01, 0.02, 1000)  # 1% mean, 2% vol
        var = calculate_var(returns, confidence_level=0.95)

        # VaR should be lower due to positive drift
        var_zero_mean = calculate_var(0.02, confidence_level=0.95)
        assert var < var_zero_mean

    def test_var_multidimensional(self):
        """Test VaR for multiple assets."""
        # Returns for 3 assets over 100 periods
        returns = np.random.normal(0.001, 0.02, size=(100, 3))
        returns[:, 1] *= 1.5  # Second asset more volatile
        returns[:, 2] *= 0.5  # Third asset less volatile

        var = calculate_var(returns, confidence_level=0.95)

        assert var.shape == (3,)
        assert var[1] > var[0] > var[2]  # Reflects volatility differences


class TestCalculateCVaR:
    """Test Conditional Value at Risk calculation."""

    def test_cvar_greater_than_var(self):
        """Test that CVaR > VaR for same parameters."""
        vol = 0.2
        var = calculate_var(vol, confidence_level=0.95)
        cvar = calculate_cvar(vol, confidence_level=0.95, use_cornish_fisher=False)

        # CVaR should be about 20-30% higher than VaR for normal distribution
        assert cvar > var
        assert 1.2 < cvar / var < 1.3

    def test_cvar_normal_distribution(self):
        """Test CVaR for normal distribution."""
        # For normal dist, CVaR = σ * φ(z_α) / α
        vol = 0.2
        alpha = 0.05
        z = -1.645  # 95% confidence

        cvar = calculate_cvar(vol, confidence_level=0.95, use_cornish_fisher=False)

        # Expected CVaR formula
        from scipy.stats import norm

        expected = vol * norm.pdf(z) / alpha

        assert abs(cvar - expected) < 0.001

    def test_cvar_with_skewness_kurtosis(self):
        """Test CVaR with non-normal returns using Cornish-Fisher."""
        # Generate skewed returns
        np.random.seed(42)
        returns = np.random.standard_t(df=5, size=1000) * 0.02  # Fat tails

        cvar_normal = calculate_cvar(returns, use_cornish_fisher=False)
        cvar_cf = calculate_cvar(returns, use_cornish_fisher=True)

        # Cornish-Fisher should give different (usually higher) CVaR for fat-tailed dist
        assert cvar_cf != cvar_normal

    def test_cvar_time_scaling(self):
        """Test CVaR time scaling."""
        daily_vol = 0.02

        cvar_1d = calculate_cvar(daily_vol, time_horizon=1)
        cvar_21d = calculate_cvar(daily_vol, time_horizon=21)

        # Should scale with sqrt(time) for normal distribution
        assert abs(cvar_21d / cvar_1d - np.sqrt(21)) < 0.01

    def test_cvar_historical_returns(self):
        """Test CVaR with actual return data."""
        # Simulate returns with known properties
        np.random.seed(42)
        returns = np.concatenate(
            [
                np.random.normal(0.001, 0.02, 950),  # Normal market
                np.random.normal(-0.05, 0.05, 50),  # Crisis periods
            ]
        )
        np.random.shuffle(returns)

        var = calculate_var(returns, confidence_level=0.95)
        cvar = calculate_cvar(returns, confidence_level=0.95)

        # CVaR captures tail risk better
        assert cvar > var  # CVaR should always be higher than VaR
        # For this distribution with fat tails, CVaR should be notably higher
        assert cvar > var * 1.2  # At least 20% higher due to crisis periods


class TestHalfKellySize:
    """Test half-Kelly position sizing."""

    def test_basic_kelly_calculation(self):
        """Test basic Kelly criterion calculation."""
        edge = 0.05  # 5% edge
        odds = 2.0  # 2:1 odds

        size = half_kelly_size(edge, odds)

        # Full Kelly = 0.05 / 2 = 0.025
        # Half Kelly = 0.0125
        assert abs(size - 0.0125) < 1e-6

    def test_kelly_with_bankroll(self):
        """Test Kelly sizing with specific bankroll."""
        edge = 0.10
        odds = 3.0
        bankroll = 100000

        size = half_kelly_size(edge, odds, bankroll)

        # Full Kelly = 0.10 / 3 = 0.0333
        # Half Kelly = 0.0167
        # Position = 0.0167 * 100000 = 1667
        assert abs(size - 1666.67) < 1

    def test_kelly_maximum_cap(self):
        """Test Kelly maximum position size cap."""
        # Very high edge should still be capped
        edge = 0.8
        odds = 2.0

        size = half_kelly_size(edge, odds)

        # Full Kelly would be 0.4, half = 0.2
        # But capped at 0.25 max
        assert size == 0.2  # Half-Kelly gives 0.2, which is under the 0.25 cap

    def test_negative_edge(self):
        """Test Kelly with negative edge."""
        edge = -0.05
        odds = 2.0

        size = half_kelly_size(edge, odds)

        # Should return 0 (don't bet on negative edge)
        assert size == 0.0

    def test_invalid_odds(self):
        """Test Kelly with invalid odds."""
        edge = 0.05

        # Zero odds
        size = half_kelly_size(edge, 0)
        assert size == 0.0

        # Negative odds
        size = half_kelly_size(edge, -1)
        assert size == 0.0

    def test_realistic_option_scenario(self):
        """Test Kelly sizing for option trading."""
        # Selling a put: collect $2 premium, risk $18 if assigned
        # If edge is 5% on the $20 risk
        premium = 2.0
        max_loss = 18.0
        odds = premium / max_loss  # 0.111
        edge = 0.05
        bankroll = 50000

        size = half_kelly_size(edge, odds, bankroll)

        # Full Kelly = 0.05 / 0.111 = 0.45 (would be capped)
        # Half Kelly = 0.225, not capped since it's under 0.25
        # Position = 0.225 * 50000 = 11250
        assert abs(size - 11250) < 1


class TestMarginRequirement:
    """Test margin requirement calculations."""

    def test_basic_margin_calculation(self):
        """Test standard margin requirement."""
        # SPY at $450, sell $440 put for $5
        margin = margin_requirement(strike=440, underlying_price=450, option_price=5.0)

        # Method 1: 20% * 450 - 10 + 5 = 90 - 10 + 5 = 85
        # Method 2: 10% * 440 + 5 = 44 + 5 = 49
        # Max(85, 49) = 85 per share * 100 = 8500
        assert margin == 8500

    def test_deep_otm_margin(self):
        """Test margin for deep OTM put."""
        margin = margin_requirement(strike=400, underlying_price=500, option_price=1.0)

        # Method 1: 20% * 500 - 100 + 1 = 100 - 100 + 1 = 1
        # Method 2: 10% * 400 + 1 = 40 + 1 = 41
        # Max(1, 41) = 41 per share * 100 = 4100
        assert margin == 4100

    def test_itm_margin(self):
        """Test margin for ITM put."""
        margin = margin_requirement(strike=500, underlying_price=480, option_price=25.0)

        # Method 1: 20% * 480 - 0 + 25 = 96 + 25 = 121
        # Method 2: 10% * 500 + 25 = 50 + 25 = 75
        # Max(121, 75) = 121 per share * 100 = 12100
        assert margin == 12100

    def test_vectorized_margin(self):
        """Test margin calculation for multiple positions."""
        strikes = np.array([440, 430, 420])
        underlying = 450
        premiums = np.array([5.0, 3.0, 1.5])

        margins = margin_requirement(strikes, underlying, premiums)

        assert margins.shape == (3,)
        # Verify specific margin values for each strike
        # Strike 440: Method1 = 90 - 10 + 5 = 85, Method2 = 44 + 5 = 49, Max = 85
        # Strike 430: Method1 = 90 - 20 + 3 = 73, Method2 = 43 + 3 = 46, Max = 73
        # Strike 420: Method1 = 90 - 30 + 1.5 = 61.5, Method2 = 42 + 1.5 = 43.5, Max = 61.5
        assert abs(margins[0] - 8500) < 1
        assert abs(margins[1] - 7300) < 1
        assert abs(margins[2] - 6150) < 1

    def test_custom_margin_rate(self):
        """Test with custom margin rate."""
        # Some brokers or products may have different rates
        margin = margin_requirement(
            strike=100,
            underlying_price=100,
            option_price=2.0,
            margin_rate=0.15,  # 15% instead of 20%
        )

        # Method 1: 15% * 100 - 0 + 2 = 15 + 2 = 17
        # Method 2: 10% * 100 + 2 = 10 + 2 = 12
        # Max(17, 12) = 17 per share * 100 = 1700
        assert margin == 1700

    def test_custom_multiplier(self):
        """Test with different contract multiplier."""
        # SPX options have 100 multiplier like SPY
        # But some futures options may differ
        margin = margin_requirement(
            strike=4400,
            underlying_price=4500,
            option_price=50.0,
            multiplier=1,  # Index options quoted per point
        )

        # Method 1: 20% * 4500 - 100 + 50 = 900 - 100 + 50 = 850
        # Method 2: 10% * 4400 + 50 = 440 + 50 = 490
        # Max(850, 490) = 850 per share * 1 = 850
        assert margin == 850
