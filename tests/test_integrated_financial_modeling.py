"""Test integrated financial modeling components."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.unity_wheel.api import WheelAdvisor
from src.unity_wheel.risk import (
    AdvancedFinancialModeling,
    BorrowingCostAnalyzer,
    RiskLimits,
)
from src.unity_wheel.strategy import WheelParameters


class TestIntegratedFinancialModeling:
    """Test that all financial modeling components work together."""

    @pytest.fixture
    def advisor(self):
        """Create advisor with all components."""
        wheel_params = WheelParameters(target_delta=0.30, target_dte=45, max_position_size=0.20)
        risk_limits = RiskLimits(max_var_95=0.05, max_cvar_95=0.075, max_margin_utilization=0.5)
        return WheelAdvisor(wheel_params, risk_limits)

    @pytest.fixture
    def market_snapshot(self):
        """Create a market snapshot."""
        return {
            "timestamp": "2024-01-01T10:00:00",
            "ticker": "U",
            "current_price": 35.0,
            "implied_volatility": 0.60,
            "risk_free_rate": 0.05,
            "buying_power": 100000,
            "available_cash": 20000,  # Limited cash to trigger borrowing
            "positions": [],
            "option_chain": [
                {
                    "strike": 30.0,
                    "expiry_date": "2024-02-15",
                    "option_type": "put",
                    "bid": 0.80,
                    "ask": 0.90,
                    "volume": 500,
                    "open_interest": 1000,
                    "implied_volatility": 0.58,
                },
                {
                    "strike": 32.5,
                    "expiry_date": "2024-02-15",
                    "option_type": "put",
                    "bid": 1.40,
                    "ask": 1.50,
                    "volume": 800,
                    "open_interest": 2000,
                    "implied_volatility": 0.60,
                },
                {
                    "strike": 35.0,
                    "expiry_date": "2024-02-15",
                    "option_type": "put",
                    "bid": 2.20,
                    "ask": 2.30,
                    "volume": 1200,
                    "open_interest": 3000,
                    "implied_volatility": 0.62,
                },
            ],
        }

    def test_advisor_has_financial_components(self, advisor):
        """Test that advisor has all financial modeling components."""
        assert hasattr(advisor, "borrowing_analyzer")
        assert isinstance(advisor.borrowing_analyzer, BorrowingCostAnalyzer)

        assert hasattr(advisor, "financial_modeler")
        assert isinstance(advisor.financial_modeler, AdvancedFinancialModeling)

    def test_borrowing_analysis_limits_position_size(self, advisor, market_snapshot):
        """Test that borrowing analysis can limit position size."""
        # Mock the strategy methods to control the flow
        with patch.object(advisor.strategy, "find_optimal_put_strike") as mock_strike:
            # Mock strike recommendation
            mock_strike.return_value = Mock(
                strike=32.5,
                delta=-0.30,
                probability_itm=0.30,
                premium=1.45,
                confidence=0.85,
                reason="Target delta match",
            )

            # Run advisor
            recommendation = advisor.advise_position(market_snapshot)

            # Should have borrowing analysis in risk metrics
            assert "borrowing_analysis" in recommendation["risk"]
            borrowing = recommendation["risk"]["borrowing_analysis"]

            # Check borrowing fields
            assert "action" in borrowing
            assert "hurdle_rate" in borrowing
            assert "expected_return" in borrowing

    def test_monte_carlo_included_in_recommendation(self, advisor, market_snapshot):
        """Test that Monte Carlo results are included."""
        with patch.object(advisor.strategy, "find_optimal_put_strike") as mock_strike:
            mock_strike.return_value = Mock(
                strike=32.5,
                delta=-0.30,
                probability_itm=0.30,
                premium=1.45,
                confidence=0.85,
                reason="Target delta match",
            )

            recommendation = advisor.advise_position(market_snapshot)

            # Should have Monte Carlo results
            assert "monte_carlo" in recommendation["risk"]
            mc = recommendation["risk"]["monte_carlo"]

            # Check MC fields
            assert "mean_return" in mc
            assert "probability_profit" in mc
            assert "var_95_mc" in mc
            assert "expected_shortfall" in mc

    def test_borrowing_recommendation_in_details(self, advisor, market_snapshot):
        """Test that borrowing recommendation is in details."""
        with patch.object(advisor.strategy, "find_optimal_put_strike") as mock_strike:
            mock_strike.return_value = Mock(
                strike=32.5,
                delta=-0.30,
                probability_itm=0.30,
                premium=1.45,
                confidence=0.85,
                reason="Target delta match",
            )

            recommendation = advisor.advise_position(market_snapshot)

            # Should have borrowing info in details
            assert "borrowing_recommended" in recommendation["details"]
            assert "borrowing_amount" in recommendation["details"]

    def test_low_return_prevents_borrowing(self, advisor, market_snapshot):
        """Test that low expected returns prevent borrowing."""
        with patch.object(advisor.strategy, "find_optimal_put_strike") as mock_strike:
            # Very low premium = low return
            mock_strike.return_value = Mock(
                strike=32.5,
                delta=-0.30,
                probability_itm=0.30,
                premium=0.50,  # Low premium
                confidence=0.85,
                reason="Target delta match",
            )

            # Mock borrowing analyzer to ensure it recommends paydown
            with patch.object(
                advisor.borrowing_analyzer, "analyze_position_allocation"
            ) as mock_borrowing:
                mock_borrowing.return_value = Mock(
                    action="paydown_debt",
                    reasoning="Expected return below hurdle rate",
                    hurdle_rate=0.07,
                    expected_return=0.05,
                    borrowing_cost=100,
                    net_benefit=-50,
                    source_to_use=None,
                )

                recommendation = advisor.advise_position(market_snapshot)

                # Borrowing should not be recommended
                if "borrowing_recommended" in recommendation["details"]:
                    assert not recommendation["details"]["borrowing_recommended"]


class TestAdvancedFinancialModeling:
    """Test advanced financial modeling functions."""

    @pytest.fixture
    def modeler(self):
        """Create financial modeler."""
        return AdvancedFinancialModeling()

    def test_monte_carlo_basic(self, modeler):
        """Test basic Monte Carlo simulation."""
        result = modeler.monte_carlo_simulation(
            expected_return=0.20,
            volatility=0.60,
            time_horizon=45,
            position_size=35000,
            borrowed_amount=0,
            n_simulations=1000,
            random_seed=42,
        )

        # Check result structure
        assert hasattr(result, "mean_return")
        assert hasattr(result, "std_return")
        assert hasattr(result, "probability_profit")
        assert hasattr(result, "percentiles")
        assert hasattr(result, "expected_shortfall")

        # Check percentiles
        assert 5 in result.percentiles
        assert 95 in result.percentiles
        assert result.percentiles[5] < result.percentiles[95]

    def test_monte_carlo_with_leverage(self, modeler):
        """Test Monte Carlo with borrowed funds."""
        # Without leverage
        result_no_leverage = modeler.monte_carlo_simulation(
            expected_return=0.20,
            volatility=0.60,
            time_horizon=45,
            position_size=35000,
            borrowed_amount=0,
            n_simulations=1000,
            random_seed=42,
        )

        # With leverage
        result_leverage = modeler.monte_carlo_simulation(
            expected_return=0.20,
            volatility=0.60,
            time_horizon=45,
            position_size=35000,
            borrowed_amount=25000,
            n_simulations=1000,
            random_seed=42,
        )

        # Leverage should reduce mean return (due to borrowing cost)
        assert result_leverage.mean_return < result_no_leverage.mean_return

    def test_risk_adjusted_metrics(self, modeler):
        """Test risk-adjusted metrics calculation."""
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)

        metrics = modeler.calculate_risk_adjusted_metrics(
            returns=returns,
            borrowed_capital=20000,
            total_capital=50000,
            risk_free_rate=0.05,
        )

        # Check all metrics exist
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "sortino_ratio")
        assert hasattr(metrics, "adjusted_sharpe")
        assert hasattr(metrics, "leverage_ratio")

        # Leverage ratio should be correct
        expected_leverage = 50000 / (50000 - 20000)
        assert abs(metrics.leverage_ratio - expected_leverage) < 0.01

    def test_optimal_capital_structure(self, modeler):
        """Test optimal leverage calculation."""
        result = modeler.optimize_capital_structure(
            expected_return=0.25, volatility=0.60, max_leverage=2.0, risk_tolerance=0.5
        )

        # Should find optimal leverage
        assert 1.0 <= result.optimal_leverage <= 2.0
        assert 0 <= result.optimal_debt_ratio <= 0.5

        # Should have leverage curve
        assert len(result.leverage_curve) > 0

    def test_var_with_leverage(self, modeler):
        """Test VaR calculation with leverage."""
        np.random.seed(42)
        returns = np.random.standard_t(df=5, size=1000) * 0.03

        var_result = modeler.calculate_var_with_leverage(
            position_size=100000,
            borrowed_amount=60000,
            returns_distribution=returns,
            confidence_level=0.95,
        )

        # Check all VaR metrics
        assert "var_basic" in var_result
        assert "var_leveraged" in var_result
        assert "cvar_basic" in var_result
        assert "cvar_leveraged" in var_result

        # Leveraged VaR should be higher
        assert var_result["var_leveraged"] > var_result["var_basic"]
        assert var_result["cvar_leveraged"] > var_result["cvar_basic"]
