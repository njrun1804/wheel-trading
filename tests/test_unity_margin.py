"""Tests for Unity-specific margin calculations."""

from unittest.mock import Mock, patch

import pytest

from unity_wheel.risk.unity_margin import (
    MarginResult,
    UnityMarginCalculator,
    calculate_unity_margin_requirement,
)


class TestUnityMarginCalculator:
    """Test Unity margin calculations with volatility adjustments."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator instance."""
        return UnityMarginCalculator()

    def test_ira_account_full_cash_securing(self, calculator):
        """Test that IRA accounts require 100% cash securing."""
        result = calculator.calculate_unity_margin(
            contracts=10,
            strike=35.0,
            current_price=36.0,
            premium_received=150.0,  # $1.50 per share
            account_type="ira",
            option_type="put",
        )

        # IRA should require full notional value
        expected_margin = 10 * 100 * 35.0  # 35,000
        assert result.margin_required == expected_margin
        assert result.margin_type == "cash"
        assert result.calculation_method == "ira_full_cash"
        assert result.confidence == 0.99

    def test_cash_account_secured_puts(self, calculator):
        """Test that cash accounts require full securing for puts."""
        result = calculator.calculate_unity_margin(
            contracts=5,
            strike=40.0,
            current_price=42.0,
            premium_received=200.0,
            account_type="cash",
            option_type="put",
        )

        # Cash account puts should be fully secured
        expected_margin = 5 * 100 * 40.0  # 20,000
        assert result.margin_required == expected_margin
        assert result.margin_type == "cash"
        assert result.calculation_method == "cash_secured_put"

    def test_margin_account_standard_calculation(self, calculator):
        """Test standard margin calculation with Unity adjustment."""
        # Test ITM put
        result = calculator.calculate_unity_margin(
            contracts=10,
            strike=40.0,
            current_price=38.0,  # ITM by $2
            premium_received=300.0,  # $3.00 per share
            account_type="margin",
            option_type="put",
        )

        # Standard margin calculation:
        # Method 1: 20% of current - OTM + premium
        # For ITM put, OTM = 0
        method1_base = 0.20 * 38.0 * 100  # $760
        method1 = max(0, method1_base - 300.0)  # $460

        # Method 2: 10% of strike + premium
        method2_base = 0.10 * 40.0 * 100  # $400
        method2 = max(0, method2_base - 300.0)  # $100

        # Method 3: $2.50 per share + premium
        method3_base = 2.50 * 100  # $250
        method3 = max(0, method3_base - 300.0)  # $0 (premium covers it)

        # Standard margin is max of all methods
        standard_margin_per_contract = max(method1, method2, method3)  # $460
        standard_total = 10 * standard_margin_per_contract  # $4,600

        # Unity adjustment: 1.5x
        expected_margin = standard_total * 1.5  # $6,900

        assert result.margin_required == expected_margin
        assert result.margin_type == "unity_adjusted"
        assert result.details["standard_margin"] == standard_total
        assert result.details["unity_multiplier"] == 1.5

    def test_margin_account_otm_put(self, calculator):
        """Test margin calculation for OTM put."""
        result = calculator.calculate_unity_margin(
            contracts=5,
            strike=35.0,
            current_price=40.0,  # OTM by $5
            premium_received=100.0,  # $1.00 per share
            account_type="margin",
            option_type="put",
        )

        # For OTM put
        otm_amount = 40.0 - 35.0  # $5

        # Method 1: 20% of current - OTM + premium
        method1_base = (0.20 * 40.0 * 100) - (5.0 * 100)  # $800 - $500 = $300
        method1 = max(0, method1_base - 100.0)  # $200

        # Method 2: 10% of strike
        method2_base = 0.10 * 35.0 * 100  # $350
        method2 = max(0, method2_base - 100.0)  # $250

        # Method 3: $2.50 per share
        method3_base = 2.50 * 100  # $250
        method3 = max(0, method3_base - 100.0)  # $150

        # Standard margin is max
        standard_per_contract = max(method1, method2, method3)  # $250
        standard_total = 5 * standard_per_contract  # $1,250

        # Unity adjustment
        expected_margin = standard_total * 1.5  # $1,875

        assert result.margin_required == expected_margin
        assert result.details["otm_amount"] == 5.0

    def test_portfolio_margin_calculation(self, calculator):
        """Test portfolio margin with stress testing."""
        result = calculator.calculate_portfolio_margin(
            contracts=10,
            strike=40.0,
            current_price=42.0,
            premium_received=200.0,
            implied_volatility=0.60,  # 60% IV
            account_type="portfolio",
        )

        # Stress move: 15% * (1 + 0.60) = 24%
        stress_move = 0.15 * (1 + 0.60)
        stressed_price = 42.0 * (1 - stress_move)  # $31.92

        # Loss if assigned at stressed price
        loss_per_contract = (40.0 - 31.92) * 100  # $808
        margin_per_contract = max(0, loss_per_contract - 200.0)  # $608
        standard_portfolio_margin = 10 * margin_per_contract  # $6,080

        # Unity adjustment even for portfolio margin
        expected_margin = standard_portfolio_margin * 1.5  # $9,120

        assert result.margin_required == expected_margin
        assert result.margin_type == "portfolio_unity_adjusted"
        assert abs(result.details["stressed_price"] - 31.92) < 0.01

    def test_minimum_margin_binding(self, calculator):
        """Test when minimum margin rules apply."""
        # Very low premium, far OTM
        result = calculator.calculate_unity_margin(
            contracts=1,
            strike=20.0,
            current_price=40.0,  # Very far OTM
            premium_received=10.0,  # $0.10 per share
            account_type="margin",
            option_type="put",
        )

        # All methods should result in minimum binding
        # Method 3 ($2.50/share) should be highest
        method3_base = 2.50 * 100  # $250
        method3 = max(0, method3_base - 10.0)  # $240

        standard_margin = method3  # $240
        expected_margin = standard_margin * 1.5  # $360

        assert result.margin_required == expected_margin
        assert result.calculation_method == "minimum_per_share"

    def test_zero_contracts_handling(self, calculator):
        """Test handling of zero contracts."""
        result = calculator.calculate_unity_margin(
            contracts=0,
            strike=35.0,
            current_price=35.0,
            premium_received=100.0,
            account_type="margin",
        )

        assert result.margin_required == 0
        assert result.margin_type == "invalid"
        assert result.confidence == 0.0

    def test_convenience_function(self):
        """Test the convenience function."""
        margin, details = calculate_unity_margin_requirement(
            contracts=5,
            strike=35.0,
            current_price=36.0,
            premium_received=150.0,
            account_type="margin",
        )

        assert margin > 0
        assert isinstance(details, dict)
        assert "standard_margin" in details
        assert "unity_multiplier" in details

    def test_get_margin_by_account_type_routing(self, calculator):
        """Test that account type routing works correctly."""
        # Test IRA routing
        ira_result = calculator.get_margin_by_account_type(
            contracts=10,
            strike=35.0,
            current_price=36.0,
            premium_received=150.0,
            account_type="IRA",  # Test case insensitive
        )
        assert ira_result.margin_type == "cash"

        # Test portfolio routing with IV
        portfolio_result = calculator.get_margin_by_account_type(
            contracts=10,
            strike=35.0,
            current_price=36.0,
            premium_received=150.0,
            account_type="portfolio",
            implied_volatility=0.50,
        )
        assert portfolio_result.margin_type == "portfolio_unity_adjusted"

        # Test margin routing
        margin_result = calculator.get_margin_by_account_type(
            contracts=10,
            strike=35.0,
            current_price=36.0,
            premium_received=150.0,
            account_type="margin",
        )
        assert margin_result.margin_type == "unity_adjusted"

    def test_unity_volatility_multiplier_impact(self, calculator):
        """Test that Unity volatility multiplier is properly applied."""
        # Calculate standard margin first
        standard_calc = UnityMarginCalculator()
        standard_calc.UNITY_VOLATILITY_MULTIPLIER = 1.0  # No adjustment

        standard_result = standard_calc.calculate_unity_margin(
            contracts=10,
            strike=35.0,
            current_price=35.0,
            premium_received=100.0,
            account_type="margin",
        )

        # Calculate with Unity adjustment
        unity_result = calculator.calculate_unity_margin(
            contracts=10,
            strike=35.0,
            current_price=35.0,
            premium_received=100.0,
            account_type="margin",
        )

        # Unity margin should be 1.5x standard
        assert unity_result.margin_required == standard_result.margin_required * 1.5

    def test_premium_reduces_margin_requirement(self, calculator):
        """Test that premium received reduces margin requirement."""
        # High premium
        high_premium_result = calculator.calculate_unity_margin(
            contracts=1,
            strike=35.0,
            current_price=35.0,
            premium_received=500.0,  # $5.00 per share
            account_type="margin",
        )

        # Low premium
        low_premium_result = calculator.calculate_unity_margin(
            contracts=1,
            strike=35.0,
            current_price=35.0,
            premium_received=50.0,  # $0.50 per share
            account_type="margin",
        )

        # Higher premium should result in lower margin requirement
        assert high_premium_result.margin_required < low_premium_result.margin_required

        # The difference should be related to premium difference
        margin_diff = low_premium_result.margin_required - high_premium_result.margin_required
        premium_diff = (500.0 - 50.0) * 1.5  # Adjusted for Unity multiplier

        # Margin difference should be close to premium difference
        assert abs(margin_diff - premium_diff) < 100  # Small tolerance for rounding
