"""Tests for analytics module."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.analytics import (
    calculate_edge,
    expected_value,
    maximum_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


class TestCalculateEdge:
    """Test edge calculation."""

    def test_positive_edge(self):
        """Test positive edge calculation."""
        theoretical = 10.50
        market = 10.00
        
        edge = calculate_edge(theoretical, market)
        assert abs(edge - 0.05) < 1e-6

    def test_negative_edge(self):
        """Test negative edge calculation."""
        theoretical = 9.50
        market = 10.00
        
        edge = calculate_edge(theoretical, market)
        assert abs(edge - (-0.05)) < 1e-6

    def test_zero_edge(self):
        """Test zero edge when prices equal."""
        theoretical = 10.00
        market = 10.00
        
        edge = calculate_edge(theoretical, market)
        assert edge == 0.0

    def test_vectorized_edge(self):
        """Test edge calculation for multiple assets."""
        theoretical = np.array([10.50, 5.25, 2.10])
        market = np.array([10.00, 5.00, 2.00])
        
        edges = calculate_edge(theoretical, market)
        
        assert edges.shape == (3,)
        np.testing.assert_allclose(edges, [0.05, 0.05, 0.05])

    def test_zero_market_price(self):
        """Test edge with zero market price."""
        # Positive theoretical value
        edge = calculate_edge(10.0, 0.0)
        assert edge == np.inf
        
        # Zero theoretical value
        edge = calculate_edge(0.0, 0.0)
        assert edge == 0.0
        
        # Negative theoretical value (shouldn't happen but test anyway)
        edge = calculate_edge(-10.0, 0.0)
        assert edge == -np.inf

    def test_mixed_edges(self):
        """Test mix of positive, negative, and zero edges."""
        theoretical = np.array([11.0, 9.0, 10.0, 5.0])
        market = np.array([10.0, 10.0, 10.0, 0.0])
        
        edges = calculate_edge(theoretical, market)
        
        assert abs(edges[0] - 0.1) < 1e-6    # 10% edge
        assert abs(edges[1] - (-0.1)) < 1e-6  # -10% edge
        assert edges[2] == 0.0                # No edge
        assert edges[3] == np.inf             # Infinite edge


class TestExpectedValue:
    """Test expected value calculation."""

    def test_simple_expected_value(self):
        """Test basic EV calculation."""
        outcomes = [100, -50]
        probabilities = [0.6, 0.4]
        
        ev = expected_value(outcomes, probabilities)
        # EV = 0.6 * 100 + 0.4 * (-50) = 60 - 20 = 40
        assert abs(ev - 40.0) < 1e-6

    def test_option_expiry_scenarios(self):
        """Test EV for option expiration scenarios."""
        # Selling a put for $5 premium
        # 70% expires worthless (keep $500)
        # 20% small loss ($1000 loss - $500 premium = -$500)
        # 10% large loss ($3000 loss - $500 premium = -$2500)
        outcomes = [500, -500, -2500]
        probabilities = [0.7, 0.2, 0.1]
        
        ev = expected_value(outcomes, probabilities)
        # EV = 0.7 * 500 + 0.2 * (-500) + 0.1 * (-2500)
        # EV = 350 - 100 - 250 = 0
        assert abs(ev - 0.0) < 1e-6

    def test_positive_ev_strategy(self):
        """Test positive expected value strategy."""
        # Win 60% of the time, 2:1 risk/reward
        outcomes = [200, -100]
        probabilities = [0.6, 0.4]
        
        ev = expected_value(outcomes, probabilities)
        # EV = 0.6 * 200 + 0.4 * (-100) = 120 - 40 = 80
        assert abs(ev - 80.0) < 1e-6

    def test_probability_normalization(self):
        """Test that probabilities are normalized if they don't sum to 1."""
        outcomes = [100, 0, -100]
        probabilities = [0.5, 0.3, 0.1]  # Sum = 0.9
        
        ev = expected_value(outcomes, probabilities)
        # Normalized: [0.556, 0.333, 0.111]
        # EV = 0.556 * 100 + 0.333 * 0 + 0.111 * (-100) = 55.6 - 11.1 = 44.4
        assert abs(ev - 44.44) < 0.1

    def test_multi_outcome_scenarios(self):
        """Test with many possible outcomes."""
        # Simulating a complex strategy with multiple exit points
        outcomes = [1000, 500, 200, 0, -200, -500, -1000]
        probabilities = [0.05, 0.15, 0.25, 0.20, 0.20, 0.10, 0.05]
        
        ev = expected_value(outcomes, probabilities)
        # EV = 50 + 75 + 50 + 0 - 40 - 50 - 50 = 35
        assert abs(ev - 35.0) < 1e-6


class TestSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_basic_sharpe_ratio(self):
        """Test basic Sharpe ratio calculation."""
        # Daily returns with known statistics
        returns = np.array([0.001, 0.002, -0.001, 0.003, 0.000])
        
        sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
        
        # Mean = 0.001, Std ≈ 0.00158
        # Sharpe = 0.001 / 0.00158 * sqrt(252) ≈ 10.04
        assert 9 < sharpe < 11

    def test_sharpe_with_risk_free_rate(self):
        """Test Sharpe ratio with non-zero risk-free rate."""
        returns = np.array([0.002, 0.003, 0.001, 0.004, 0.000])
        rf_rate = 0.0001  # 0.01% daily risk-free rate
        
        sharpe = sharpe_ratio(returns, risk_free_rate=rf_rate)
        
        # Excess returns mean = 0.002 - 0.0001 = 0.0019
        assert sharpe > 0

    def test_negative_sharpe_ratio(self):
        """Test negative Sharpe ratio for losing strategy."""
        returns = np.array([-0.001, -0.002, 0.001, -0.003, -0.001])
        
        sharpe = sharpe_ratio(returns)
        
        assert sharpe < 0

    def test_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        # Constant returns
        returns = np.array([0.001, 0.001, 0.001, 0.001])
        
        sharpe = sharpe_ratio(returns)
        
        assert sharpe == np.inf

    def test_monthly_sharpe(self):
        """Test Sharpe ratio with monthly returns."""
        monthly_returns = np.array([0.02, -0.01, 0.03, 0.01, -0.005, 0.015])
        
        sharpe = sharpe_ratio(monthly_returns, periods_per_year=12)
        
        # Should be positive with these returns
        assert sharpe > 0

    def test_insufficient_data(self):
        """Test Sharpe ratio with insufficient data."""
        returns = np.array([0.01])  # Only one return
        
        sharpe = sharpe_ratio(returns)
        
        assert sharpe == 0.0


class TestWinRate:
    """Test win rate calculation."""

    def test_basic_win_rate(self):
        """Test basic win rate calculation."""
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        
        rate = win_rate(returns)
        
        # 3 wins out of 5 trades
        assert abs(rate - 0.6) < 1e-6

    def test_win_rate_with_threshold(self):
        """Test win rate with custom threshold."""
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        
        # Only count as win if return > 1.5%
        rate = win_rate(returns, threshold=0.015)
        
        # 2 wins out of 5 trades
        assert abs(rate - 0.4) < 1e-6

    def test_all_wins(self):
        """Test win rate when all trades win."""
        returns = np.array([0.01, 0.02, 0.03, 0.05])
        
        rate = win_rate(returns)
        assert rate == 1.0

    def test_all_losses(self):
        """Test win rate when all trades lose."""
        returns = np.array([-0.01, -0.02, -0.03, -0.05])
        
        rate = win_rate(returns)
        assert rate == 0.0

    def test_empty_returns(self):
        """Test win rate with no trades."""
        returns = np.array([])
        
        rate = win_rate(returns)
        assert rate == 0.0


class TestProfitFactor:
    """Test profit factor calculation."""

    def test_basic_profit_factor(self):
        """Test basic profit factor calculation."""
        returns = np.array([100, -50, 200, -25, 150])
        
        factor = profit_factor(returns)
        
        # Gross profits = 100 + 200 + 150 = 450
        # Gross losses = 50 + 25 = 75
        # Factor = 450 / 75 = 6.0
        assert abs(factor - 6.0) < 1e-6

    def test_breakeven_profit_factor(self):
        """Test profit factor at breakeven."""
        returns = np.array([100, -100, 50, -50])
        
        factor = profit_factor(returns)
        
        # Factor = 150 / 150 = 1.0
        assert abs(factor - 1.0) < 1e-6

    def test_no_losses(self):
        """Test profit factor with no losses."""
        returns = np.array([100, 200, 150])
        
        factor = profit_factor(returns)
        
        assert factor == np.inf

    def test_no_profits(self):
        """Test profit factor with no profits."""
        returns = np.array([-100, -200, -150])
        
        factor = profit_factor(returns)
        
        assert factor == 0.0

    def test_single_trade(self):
        """Test profit factor with single trade."""
        # Single win
        factor = profit_factor(np.array([100]))
        assert factor == np.inf
        
        # Single loss
        factor = profit_factor(np.array([-100]))
        assert factor == 0.0


class TestMaximumDrawdown:
    """Test maximum drawdown calculation."""

    def test_basic_drawdown(self):
        """Test basic maximum drawdown calculation."""
        equity = np.array([10000, 11000, 10500, 12000, 11000, 10000, 11500])
        
        mdd, peak_idx, trough_idx = maximum_drawdown(equity)
        
        # Peak at 12000 (index 3), trough at 10000 (index 5)
        # Drawdown = (12000 - 10000) / 12000 = 0.1667
        assert abs(mdd - 0.1667) < 0.001
        assert peak_idx == 3
        assert trough_idx == 5

    def test_no_drawdown(self):
        """Test with monotonically increasing equity."""
        equity = np.array([10000, 11000, 12000, 13000, 14000])
        
        mdd, peak_idx, trough_idx = maximum_drawdown(equity)
        
        assert mdd == 0.0
        assert peak_idx == 0
        assert trough_idx == 0

    def test_continuous_drawdown(self):
        """Test with continuous decline."""
        equity = np.array([10000, 9000, 8000, 7000, 6000])
        
        mdd, peak_idx, trough_idx = maximum_drawdown(equity)
        
        # Drawdown = (10000 - 6000) / 10000 = 0.4
        assert abs(mdd - 0.4) < 1e-6
        assert peak_idx == 0
        assert trough_idx == 4

    def test_multiple_drawdowns(self):
        """Test with multiple drawdown periods."""
        equity = np.array([
            10000, 9000, 8500, 10500,  # First drawdown: 15%
            11000, 9500, 9000, 10000   # Second drawdown: 18.2%
        ])
        
        mdd, peak_idx, trough_idx = maximum_drawdown(equity)
        
        # Maximum is the second drawdown
        assert abs(mdd - 0.182) < 0.001
        assert peak_idx == 4
        assert trough_idx == 6

    def test_recovery_after_drawdown(self):
        """Test drawdown with full recovery."""
        equity = np.array([10000, 11000, 9000, 8000, 12000])
        
        mdd, peak_idx, trough_idx = maximum_drawdown(equity)
        
        # Drawdown = (11000 - 8000) / 11000 = 0.273
        assert abs(mdd - 0.273) < 0.001
        assert peak_idx == 1
        assert trough_idx == 3


class TestSortinoRatio:
    """Test Sortino ratio calculation."""

    def test_basic_sortino_ratio(self):
        """Test basic Sortino ratio calculation."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        
        sortino = sortino_ratio(returns, target_return=0.0)
        
        # Should penalize only downside volatility
        assert sortino > 0

    def test_sortino_vs_sharpe(self):
        """Test that Sortino > Sharpe for asymmetric returns."""
        # Returns with upside bias
        returns = np.array([0.05, -0.01, 0.04, -0.01, 0.03, -0.01])
        
        sharpe = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)
        
        # Sortino should be higher as it doesn't penalize upside volatility
        assert sortino > sharpe

    def test_sortino_with_target(self):
        """Test Sortino ratio with non-zero target."""
        returns = np.array([0.02, 0.01, 0.03, 0.015, 0.025])
        target = 0.015  # 1.5% target return
        
        sortino = sortino_ratio(returns, target_return=target)
        
        # All returns meet or exceed target except one
        assert sortino > 0

    def test_all_above_target(self):
        """Test Sortino when all returns exceed target."""
        returns = np.array([0.02, 0.03, 0.04, 0.025])
        target = 0.01
        
        sortino = sortino_ratio(returns, target_return=target)
        
        # No downside deviation, should be infinite
        assert sortino == np.inf

    def test_monthly_sortino(self):
        """Test Sortino ratio with monthly data."""
        monthly_returns = np.array([0.03, -0.02, 0.04, 0.01, -0.01, 0.02])
        
        sortino = sortino_ratio(monthly_returns, periods_per_year=12)
        
        assert sortino > 0

    def test_negative_sortino(self):
        """Test negative Sortino ratio."""
        returns = np.array([-0.01, -0.02, 0.005, -0.03, -0.01])
        
        sortino = sortino_ratio(returns)
        
        assert sortino < 0