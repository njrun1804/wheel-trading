"""Tests for dependency injection in the API layer."""

from unittest.mock import MagicMock, Mock

import pytest

from src.unity_wheel.api.advisor import WheelAdvisor
from src.unity_wheel.api.dependencies import AdvisorDependencies, create_dependencies
from src.unity_wheel.risk.limits import TradingLimits as RiskLimits
from src.unity_wheel.strategy.wheel import WheelParameters


class TestDependencyInjection:
    """Test dependency injection functionality."""

    def test_default_dependencies(self):
        """Test advisor with default dependencies."""
        advisor = WheelAdvisor()

        # Check that dependencies are initialized
        assert advisor.dependencies is not None
        assert advisor.strategy is not None
        assert advisor.risk_analyzer is not None
        assert advisor.assignment_model is not None
        assert advisor.borrowing_analyzer is not None
        assert advisor.financial_modeler is not None

    def test_custom_dependencies(self):
        """Test advisor with custom dependencies."""
        # Create custom parameters
        wheel_params = WheelParameters(target_delta=0.40, target_dte=30)
        risk_limits = RiskLimits(max_position_size=0.15)

        # Create advisor with custom params
        advisor = WheelAdvisor(wheel_params=wheel_params, risk_limits=risk_limits)

        # Verify custom parameters are used
        assert advisor.wheel_params.target_delta == 0.40
        assert advisor.wheel_params.target_dte == 30
        assert advisor.risk_limits.max_position_size == 0.15

    def test_dependency_container(self):
        """Test creating dependency container with overrides."""
        # Create mock components
        mock_validator = Mock()
        mock_detector = Mock()

        # Create dependencies with mocks
        deps = create_dependencies(market_validator=mock_validator, anomaly_detector=mock_detector)

        # Verify mocks are used
        assert deps.market_validator is mock_validator
        assert deps.anomaly_detector is mock_detector

    def test_advisor_with_mock_dependencies(self):
        """Test advisor with mocked dependencies."""
        # Create mock dependencies
        mock_validator = MagicMock()
        mock_validator.validate.return_value = MagicMock(is_valid=True, issues=[])

        mock_detector = MagicMock()
        mock_detector.detect_market_anomalies.return_value = []

        deps = create_dependencies(market_validator=mock_validator, anomaly_detector=mock_detector)

        # Create advisor with mocked deps
        advisor = WheelAdvisor(dependencies=deps)

        # Create test market snapshot
        market_snapshot = {
            "ticker": "U",
            "current_price": 25.0,
            "buying_power": 100000,
            "option_chain": {},
            "positions": [],
            "implied_volatility": 0.87,
        }

        # Call advise_position
        recommendation = advisor.advise_position(market_snapshot)

        # Verify mocks were called
        mock_validator.validate.assert_called_once()
        mock_detector.detect_market_anomalies.assert_called_once()

        # Verify recommendation returned
        assert recommendation is not None
        assert recommendation.action in ["HOLD", "ADJUST"]

    def test_dependency_lazy_loading(self):
        """Test that dependencies are created lazily."""
        deps = AdvisorDependencies()

        # Components should not be created yet
        assert deps._wheel_strategy is None
        assert deps._risk_analyzer is None

        # Access components - should be created on demand
        strategy = deps.wheel_strategy
        assert strategy is not None
        assert deps._wheel_strategy is strategy  # Same instance

        # Access again - should return same instance
        strategy2 = deps.wheel_strategy
        assert strategy2 is strategy

    def test_no_circular_imports(self):
        """Test that dependency injection avoids circular imports."""
        # This test passes if we can import without errors
        from src.unity_wheel.api.advisor import WheelAdvisor
        from src.unity_wheel.api.dependencies import AdvisorDependencies

        # Create instances without circular import issues
        deps = AdvisorDependencies()
        advisor = WheelAdvisor(dependencies=deps)

        assert advisor is not None
        assert deps is not None
