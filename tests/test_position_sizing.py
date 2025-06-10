"""Tests for dynamic position sizing utilities."""

from unittest.mock import Mock, patch

import pytest

from src.unity_wheel.utils.position_sizing import DynamicPositionSizer, PositionSizeResult


class TestDynamicPositionSizer:
    """Test dynamic position sizing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sizer = DynamicPositionSizer()

    def test_basic_position_sizing(self):
        """Test basic position size calculation."""
        result = self.sizer.calculate_position_size(
            portfolio_value=100000,
            option_price=2.50,
            strike_price=35.0,
            buying_power=100000,
            kelly_fraction=0.25,
        )

        assert isinstance(result, PositionSizeResult)
        assert result.contracts >= 0
        assert result.notional_value > 0
        assert result.margin_required > 0
        assert 0 <= result.position_pct <= 1.0
        assert 0 <= result.confidence <= 1.0
        assert result.is_valid

    def test_kelly_constraint(self):
        """Test Kelly criterion constraint."""
        # High Kelly fraction should limit position
        result = self.sizer.calculate_position_size(
            portfolio_value=100000,
            option_price=2.50,
            strike_price=35.0,
            buying_power=100000,
            kelly_fraction=0.10,  # Very conservative
        )

        # Should result in smaller position
        assert result.contracts <= 5  # Conservative sizing
        assert result.sizing_method == "kelly"

    def test_margin_constraint(self):
        """Test margin requirement constraint."""
        # Limited buying power should constrain position
        result = self.sizer.calculate_position_size(
            portfolio_value=100000,
            option_price=2.50,
            strike_price=35.0,
            buying_power=10000,  # Only 10k available
            kelly_fraction=0.50,
        )

        # Should be limited by margin
        assert result.margin_required <= 10000
        assert "margin" in result.sizing_method
        assert "Limited by margin" in str(result.warnings)

    def test_position_limit_constraint(self):
        """Test maximum position size constraint."""
        # Try to size a very large position
        result = self.sizer.calculate_position_size(
            portfolio_value=100000,
            option_price=0.50,  # Very cheap option
            strike_price=10.0,  # Low strike
            buying_power=100000,
            kelly_fraction=1.0,  # Max Kelly
        )

        # Should be limited by position limit (20% default)
        assert result.position_pct <= 0.20
        assert "position_limit" in result.sizing_method

    def test_minimum_contract_requirement(self):
        """Test minimum contract requirement."""
        # Very small portfolio
        result = self.sizer.calculate_position_size(
            portfolio_value=5000,  # Small account
            option_price=2.50,
            strike_price=35.0,
            buying_power=5000,
            kelly_fraction=0.25,
        )

        # Should get 0 contracts if below minimum
        if result.contracts == 0:
            assert "Position too small" in str(result.warnings)

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Normal conditions - high confidence
        result1 = self.sizer.calculate_position_size(
            portfolio_value=100000,
            option_price=2.50,
            strike_price=35.0,
            buying_power=100000,
            kelly_fraction=0.25,
        )
        assert result1.confidence > 0.85

        # Constrained position - lower confidence
        result2 = self.sizer.calculate_position_size(
            portfolio_value=100000,
            option_price=2.50,
            strike_price=35.0,
            buying_power=5000,  # Very limited
            kelly_fraction=0.05,  # Very conservative
        )
        assert result2.confidence < result1.confidence

    def test_adjust_for_small_account(self):
        """Test small account adjustments."""
        # Account below $25k threshold
        contracts, message = self.sizer.adjust_for_small_account(
            portfolio_value=15000,
            contracts=3,
            min_trade_value=5000,
        )

        # Should reduce contracts for small account
        assert contracts <= 3
        assert contracts >= 1  # But maintain minimum
        assert "small account" in message.lower()

    def test_zero_price_handling(self):
        """Test handling of zero/invalid prices."""
        result = self.sizer.calculate_position_size(
            portfolio_value=100000,
            option_price=0,  # Invalid
            strike_price=35.0,
            buying_power=100000,
            kelly_fraction=0.25,
        )

        # Should handle gracefully
        assert result.contracts == 0
        assert not result.is_valid

    def test_extreme_volatility_adjustment(self):
        """Test position sizing under extreme volatility."""
        # Extreme Kelly fraction (high volatility scenario)
        result = self.sizer.calculate_position_size(
            portfolio_value=100000,
            option_price=5.00,  # High premium (high vol)
            strike_price=35.0,
            buying_power=100000,
            kelly_fraction=0.05,  # Very low due to high vol
        )

        # Should have reduced confidence
        assert result.confidence < 0.85
        assert result.contracts < 3  # Very conservative
