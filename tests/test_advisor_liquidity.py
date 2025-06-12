#!/usr/bin/env python3
"""
Test liquidity validation in advisor module.
Validates that option liquidity checks work correctly.
"""

import pytest
from unity_wheel.api.advisor import WheelAdvisor


class TestAdvisorLiquidity:
    """Test liquidity validation functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.advisor = WheelAdvisor()

    def test_validate_option_liquidity_success(self):
        """Test that valid option data passes liquidity checks."""
        # Create option data that should pass all checks
        option_data = {
            "bid": 1.00,
            "ask": 1.05,  # Small spread
            "volume": 200,  # Good volume
            "open_interest": 150,  # Good open interest
        }

        result = self.advisor._validate_option_liquidity(option_data)
        assert result is True

    def test_validate_option_liquidity_wide_spread(self):
        """Test that wide bid-ask spread fails validation."""
        option_data = {
            "bid": 1.00,
            "ask": 10.00,  # Very wide spread
            "volume": 200,
            "open_interest": 150,
        }

        result = self.advisor._validate_option_liquidity(option_data)
        assert result is False

    def test_validate_option_liquidity_low_volume(self):
        """Test that low volume fails validation."""
        option_data = {
            "bid": 1.00,
            "ask": 1.05,
            "volume": 5,  # Very low volume
            "open_interest": 150,
        }

        result = self.advisor._validate_option_liquidity(option_data)
        assert result is False

    def test_validate_option_liquidity_low_open_interest(self):
        """Test that low open interest fails validation."""
        option_data = {
            "bid": 1.00,
            "ask": 1.05,
            "volume": 200,
            "open_interest": 5,  # Very low open interest
        }

        result = self.advisor._validate_option_liquidity(option_data)
        assert result is False

    def test_validate_option_liquidity_missing_data(self):
        """Test that missing data uses reasonable defaults."""
        option_data = {}  # Empty data

        # Should use defaults: bid=0, ask=inf, volume=0, open_interest=0
        # This should fail due to low volume and open interest
        result = self.advisor._validate_option_liquidity(option_data)
        assert result is False

    def test_validate_option_liquidity_partial_data(self):
        """Test with partial option data."""
        option_data = {
            "bid": 1.00,
            "ask": 1.05,
            # Missing volume and open_interest - should use defaults of 0
        }

        # Should fail due to missing volume and open interest
        result = self.advisor._validate_option_liquidity(option_data)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
