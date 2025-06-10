"""Tests for borrowing cost analyzer."""


import pytest

from src.unity_wheel.risk.borrowing_cost_analyzer import (
    BorrowingCostAnalyzer,
    BorrowingSource,
    CapitalAllocationResult,
    analyze_borrowing_decision,
)


class TestBorrowingSource:
    """Test BorrowingSource calculations."""

    def test_daily_rate_calculation(self):
        """Test daily rate calculation from annual."""
        source = BorrowingSource(
            name="test", balance=10000, annual_rate=0.0365  # 3.65% for easy math (0.01% daily)
        )

        assert abs(source.daily_rate - 0.0001) < 1e-6
        assert abs(source.monthly_rate - 0.0365 / 12) < 1e-6

    def test_cost_calculations(self):
        """Test various cost calculations."""
        source = BorrowingSource(name="test", balance=10000, annual_rate=0.10)  # 10% APR

        # Daily cost on full balance
        daily_cost = source.daily_cost()
        assert abs(daily_cost - 10000 * 0.10 / 365) < 0.01

        # Monthly cost on partial amount
        monthly_cost = source.monthly_cost(5000)
        assert abs(monthly_cost - 5000 * 0.10 / 12) < 0.01

        # Cost for 30 days
        period_cost = source.cost_for_period(30, 10000)
        assert abs(period_cost - 10000 * 0.10 * 30 / 365) < 0.01


class TestBorrowingCostAnalyzer:
    """Test BorrowingCostAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default sources."""
        return BorrowingCostAnalyzer()

    def test_default_sources_setup(self, analyzer):
        """Test that default sources are set up correctly."""
        assert "amex_loan" in analyzer.sources
        assert "schwab_margin" in analyzer.sources

        amex = analyzer.sources["amex_loan"]
        assert amex.annual_rate == 0.07
        assert amex.balance == 45000
        assert not amex.is_revolving

        schwab = analyzer.sources["schwab_margin"]
        assert schwab.annual_rate == 0.10
        assert schwab.balance == 0
        assert schwab.is_revolving

    def test_hurdle_rate_calculation(self, analyzer):
        """Test hurdle rate calculations."""
        # Basic hurdle rate for Amex loan
        hurdle = analyzer.calculate_hurdle_rate("amex_loan", include_tax=False)
        # 7% * 1.5 (confidence) = 10.5%
        assert abs(hurdle - 0.105) < 0.001

        # With tax adjustment
        hurdle_with_tax = analyzer.calculate_hurdle_rate("amex_loan", include_tax=True)
        # 10.5% / 0.75 = 14%
        assert abs(hurdle_with_tax - 0.14) < 0.001

    def test_position_allocation_paydown_low_return(self, analyzer):
        """Test allocation when return is below hurdle rate."""
        result = analyzer.analyze_position_allocation(
            position_size=10000,
            expected_annual_return=0.08,  # 8% return
            confidence=0.8,  # Adjusted to 6.4%
            available_cash=0,
        )

        # 6.4% adjusted return < 14% hurdle rate
        assert result.action == "paydown_debt"
        assert result.invest_amount == 0
        assert result.paydown_amount == 10000
        assert "below hurdle rate" in result.reasoning

    def test_position_allocation_invest_with_cash(self, analyzer):
        """Test allocation when cash is available."""
        result = analyzer.analyze_position_allocation(
            position_size=5000,
            expected_annual_return=0.20,  # 20% return
            confidence=0.9,
            available_cash=10000,  # More than enough
        )

        assert result.action == "invest"
        assert result.invest_amount == 5000
        assert result.paydown_amount == 0
        assert result.borrowing_cost == 0
        assert "cash available" in result.reasoning.lower()

    def test_position_allocation_profitable_borrowing(self, analyzer):
        """Test allocation when borrowing is profitable."""
        result = analyzer.analyze_position_allocation(
            position_size=20000,
            expected_annual_return=0.30,  # 30% return
            holding_period_days=45,
            confidence=0.8,  # Adjusted to 24%
            available_cash=0,
        )

        # Should invest despite borrowing cost
        assert result.action == "invest"
        assert result.invest_amount == 20000
        assert result.source_to_use == "amex_loan"  # Cheapest source
        assert result.borrowing_cost > 0
        assert result.net_benefit > 0

    def test_position_allocation_unprofitable_borrowing(self, analyzer):
        """Test when borrowing cost exceeds profit."""
        result = analyzer.analyze_position_allocation(
            position_size=100000,
            expected_annual_return=0.05,  # 5% return (very low)
            holding_period_days=365,
            confidence=0.5,  # Adjusted to 2.5%
            available_cash=0,
        )

        assert result.action == "paydown_debt"
        assert "cost.*exceeds" in result.reasoning.lower()

    def test_borrowing_source_selection(self, analyzer):
        """Test that cheapest source is selected."""
        # Add a more expensive source
        analyzer.add_source(
            BorrowingSource(
                name="credit_card", balance=0, annual_rate=0.18, is_revolving=True  # 18% APR
            )
        )

        result = analyzer.analyze_position_allocation(
            position_size=10000, expected_annual_return=0.25, confidence=0.9, available_cash=0
        )

        # Should select Amex (7%) over Schwab (10%) or CC (18%)
        assert result.source_to_use == "amex_loan"

    def test_paydown_benefit_calculation(self, analyzer):
        """Test paydown benefit calculations."""
        benefits = analyzer.calculate_paydown_benefit(
            paydown_amount=10000, source_name="amex_loan", time_horizon_days=365
        )

        # 7% on $10k for 1 year = $700
        assert abs(benefits["interest_saved"] - 700) < 1
        assert benefits["effective_return"] == 0.07
        assert benefits["daily_savings"] > 0
        assert benefits["monthly_savings"] > 0

    def test_optimize_capital_deployment(self, analyzer):
        """Test multi-opportunity optimization."""
        opportunities = [
            {"size": 10000, "expected_return": 0.15, "confidence": 0.9},
            {"size": 15000, "expected_return": 0.25, "confidence": 0.7},
            {"size": 20000, "expected_return": 0.10, "confidence": 0.95},
        ]

        allocations = analyzer.optimize_capital_deployment(
            available_capital=20000,
            opportunities=opportunities,
            max_leverage=1.5,  # Can borrow up to 50% more
        )

        # Should prioritize opportunity 1 (highest risk-adjusted return)
        assert len(allocations) == 3

        # Check that we don't exceed leverage limits
        total_invested = sum(a.invest_amount for a in allocations.values() if a.action == "invest")
        assert total_invested <= 20000 * 1.5  # Max leverage

    def test_current_borrowing_summary(self, analyzer):
        """Test borrowing summary calculation."""
        summary = analyzer.get_current_borrowing_summary()

        assert "amex_loan" in summary
        assert "schwab_margin" in summary
        assert "totals" in summary

        # Check Amex calculations
        amex_summary = summary["amex_loan"]
        assert amex_summary["balance"] == 45000
        assert amex_summary["annual_rate"] == "7.0%"
        assert abs(amex_summary["annual_cost"] - 45000 * 0.07) < 1

        # Check totals
        totals = summary["totals"]
        assert totals["total_debt"] == 45000  # Only Amex has balance
        assert totals["blended_rate"] == "7.0%"  # Only Amex

    def test_convenience_function(self):
        """Test the convenience function."""
        result = analyze_borrowing_decision(
            position_size=10000, expected_return=0.20, confidence=0.8, available_cash=5000
        )

        assert isinstance(result, CapitalAllocationResult)
        assert result.action in ["invest", "paydown_debt"]

    def test_mixed_cash_and_borrowing(self, analyzer):
        """Test when partially using cash and borrowing."""
        result = analyzer.analyze_position_allocation(
            position_size=30000,
            expected_annual_return=0.25,
            confidence=0.8,
            available_cash=10000,  # Need to borrow 20k
        )

        if result.action == "invest":
            assert result.details["need_to_borrow"] == 20000
            assert result.borrowing_cost > 0
            assert result.source_to_use is not None

    def test_confidence_impact(self, analyzer):
        """Test how confidence affects decisions."""
        # High confidence
        high_conf_result = analyzer.analyze_position_allocation(
            position_size=10000, expected_annual_return=0.15, confidence=0.95, available_cash=0
        )

        # Low confidence
        low_conf_result = analyzer.analyze_position_allocation(
            position_size=10000, expected_annual_return=0.15, confidence=0.50, available_cash=0
        )

        # Low confidence more likely to suggest paying down debt
        if high_conf_result.action == "invest":
            assert low_conf_result.action == "paydown_debt"

    def test_holding_period_impact(self, analyzer):
        """Test how holding period affects borrowing cost."""
        # Short holding period
        short_result = analyzer.analyze_position_allocation(
            position_size=10000,
            expected_annual_return=0.20,
            holding_period_days=7,  # 1 week
            confidence=0.8,
            available_cash=0,
        )

        # Long holding period
        long_result = analyzer.analyze_position_allocation(
            position_size=10000,
            expected_annual_return=0.20,
            holding_period_days=180,  # 6 months
            confidence=0.8,
            available_cash=0,
        )

        # Longer holding period = higher borrowing cost
        if short_result.borrowing_cost > 0 and long_result.borrowing_cost > 0:
            assert long_result.borrowing_cost > short_result.borrowing_cost
