"""
Dependency injection for API layer.
Replaces lazy imports with proper dependency management.
"""

from typing import Optional, Protocol

from ..analytics.anomaly_detector import AnomalyDetector
from ..analytics.unity_assignment import UnityAssignmentModel
from ..data_providers.base.validation import MarketDataValidator
from ..risk.advanced_financial_modeling import AdvancedFinancialModeling
from ..risk.analytics import RiskAnalyzer
from ..risk.borrowing_cost_analyzer import BorrowingCostAnalyzer
from ..risk.limits import TradingLimits
from ..strategy.wheel import WheelParameters, WheelStrategy


class MarketValidatorProtocol(Protocol):
    """Protocol for market data validators."""

    def validate(self, data: dict) -> object:
        """Validate market data."""
        ...


class AnomalyDetectorProtocol(Protocol):
    """Protocol for anomaly detectors."""

    def detect_market_anomalies(self, data: dict) -> list:
        """Detect anomalies in market data."""
        ...


class AdvisorDependencies:
    """
    Container for all WheelAdvisor dependencies.
    Provides centralized dependency management.
    """

    def __init__(
        self,
        market_validator: Optional[MarketValidatorProtocol] = None,
        anomaly_detector: Optional[AnomalyDetectorProtocol] = None,
        wheel_parameters: Optional[WheelParameters] = None,
        risk_limits: Optional[TradingLimits] = None,
    ):
        """
        Initialize dependencies with optional overrides.

        Parameters
        ----------
        market_validator : Optional[MarketValidatorProtocol]
            Market data validator instance
        anomaly_detector : Optional[AnomalyDetectorProtocol]
            Anomaly detector instance
        wheel_parameters : Optional[WheelParameters]
            Wheel strategy parameters
        risk_limits : Optional[TradingLimits]
            Risk limit configuration
        """
        self._market_validator = market_validator
        self._anomaly_detector = anomaly_detector
        self._wheel_parameters = wheel_parameters
        self._risk_limits = risk_limits

        # Component instances (created on demand)
        self._wheel_strategy: Optional[WheelStrategy] = None
        self._risk_analyzer: Optional[RiskAnalyzer] = None
        self._assignment_model: Optional[UnityAssignmentModel] = None
        self._borrowing_analyzer: Optional[BorrowingCostAnalyzer] = None
        self._financial_modeler: Optional[AdvancedFinancialModeling] = None

    @property
    def market_validator(self) -> MarketDataValidator:
        """Get market validator instance."""
        if self._market_validator is None:
            from ..data_providers.base import get_market_validator

            self._market_validator = get_market_validator()
        return self._market_validator

    @property
    def anomaly_detector(self) -> AnomalyDetector:
        """Get anomaly detector instance."""
        if self._anomaly_detector is None:
            from ..data_providers.base import get_anomaly_detector

            self._anomaly_detector = get_anomaly_detector()
        return self._anomaly_detector

    @property
    def wheel_parameters(self) -> WheelParameters:
        """Get wheel parameters."""
        if self._wheel_parameters is None:
            self._wheel_parameters = WheelParameters()
        return self._wheel_parameters

    @property
    def risk_limits(self) -> TradingLimits:
        """Get risk limits."""
        if self._risk_limits is None:
            self._risk_limits = TradingLimits()
        return self._risk_limits

    @property
    def wheel_strategy(self) -> WheelStrategy:
        """Get wheel strategy instance."""
        if self._wheel_strategy is None:
            self._wheel_strategy = WheelStrategy(self.wheel_parameters)
        return self._wheel_strategy

    @property
    def risk_analyzer(self) -> RiskAnalyzer:
        """Get risk analyzer instance."""
        if self._risk_analyzer is None:
            self._risk_analyzer = RiskAnalyzer(self.risk_limits)
        return self._risk_analyzer

    @property
    def assignment_model(self) -> UnityAssignmentModel:
        """Get Unity assignment model."""
        if self._assignment_model is None:
            self._assignment_model = UnityAssignmentModel()
        return self._assignment_model

    @property
    def borrowing_analyzer(self) -> BorrowingCostAnalyzer:
        """Get borrowing cost analyzer."""
        if self._borrowing_analyzer is None:
            self._borrowing_analyzer = BorrowingCostAnalyzer()
        return self._borrowing_analyzer

    @property
    def financial_modeler(self) -> AdvancedFinancialModeling:
        """Get financial modeler."""
        if self._financial_modeler is None:
            self._financial_modeler = AdvancedFinancialModeling(self.borrowing_analyzer)
        return self._financial_modeler


# Global default instance
_default_dependencies = AdvisorDependencies()


def get_default_dependencies() -> AdvisorDependencies:
    """Get the default dependencies instance."""
    return _default_dependencies


def create_dependencies(**overrides) -> AdvisorDependencies:
    """
    Create a new dependencies instance with overrides.

    Examples
    --------
    >>> # Use defaults
    >>> deps = create_dependencies()

    >>> # Override specific components
    >>> deps = create_dependencies(
    ...     wheel_parameters=WheelParameters(target_delta=0.40),
    ...     risk_limits=TradingLimits(max_position_size=0.15)
    ... )
    """
    return AdvisorDependencies(**overrides)
