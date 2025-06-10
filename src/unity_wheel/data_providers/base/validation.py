"""Data quality validation layer for market data integrity."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ...utils import get_logger, StructuredLogger

logger = get_logger(__name__)
structured_logger = StructuredLogger(__name__)


class DataQualityLevel(str, Enum):
    """Data quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class ValidationRule:
    """Single validation rule definition."""

    name: str
    description: str
    severity: str  # "error", "warning", "info"
    check_func: callable
    auto_correct: bool = False
    correction_func: Optional[callable] = None


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    quality_level: DataQualityLevel
    issues: List[ValidationIssue] = field(default_factory=list)
    corrections_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        """Check if validation has any errors."""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if validation has any warnings."""
        return any(issue.severity == "warning" for issue in self.issues)


@dataclass
class ValidationIssue:
    """Single validation issue found."""

    rule_name: str
    severity: str
    message: str
    field_name: Optional[str] = None  # Renamed to avoid conflict with dataclass field
    value: Any = None
    expected: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketDataValidator:
    """
    Comprehensive market data validation system.

    Validates:
    - Data freshness and staleness
    - Price reasonableness
    - Option chain consistency
    - Greeks relationships
    - Spread validity
    - Volume/liquidity requirements
    """

    def __init__(self):
        """Initialize validator with default rules."""
        self.rules: Dict[str, ValidationRule] = {}
        self._initialize_default_rules()

        # Validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "passed": 0,
            "failed": 0,
            "auto_corrected": 0,
        }

    def _initialize_default_rules(self) -> None:
        """Set up default validation rules."""
        # Price validation rules
        self.add_rule(
            "positive_price",
            "Stock price must be positive",
            "error",
            lambda data: data.get("current_price", 0) > 0,
        )

        self.add_rule(
            "reasonable_price",
            "Stock price within reasonable range",
            "warning",
            lambda data: 0.01 <= data.get("current_price", 0) <= 10000,
        )

        # Timestamp validation
        self.add_rule(
            "fresh_data",
            "Data should be fresh (< 5 minutes old)",
            "warning",
            lambda data: self._check_freshness(data.get("timestamp"), minutes=5),
        )

        self.add_rule(
            "not_stale",
            "Data must not be stale (< 1 hour old)",
            "error",
            lambda data: self._check_freshness(data.get("timestamp"), minutes=60),
        )

        # Option chain validation
        self.add_rule(
            "option_chain_exists",
            "Option chain must exist",
            "error",
            lambda data: bool(data.get("option_chain")),
        )

        self.add_rule(
            "option_chain_consistency",
            "Option chain must be internally consistent",
            "error",
            self._validate_option_chain_consistency,
        )

        # Spread validation
        self.add_rule(
            "valid_spreads",
            "Bid-ask spreads must be valid",
            "error",
            self._validate_spreads,
        )

        # Greeks validation
        self.add_rule(
            "greeks_relationships",
            "Greeks must satisfy known relationships",
            "warning",
            self._validate_greeks_relationships,
        )

        # Volatility validation
        self.add_rule(
            "reasonable_iv",
            "Implied volatility within reasonable range",
            "warning",
            lambda data: self._validate_iv_range(data),
        )

        # Volume/liquidity validation
        self.add_rule(
            "minimum_liquidity",
            "Options must meet minimum liquidity requirements",
            "warning",
            self._validate_liquidity,
        )

    def add_rule(
        self,
        name: str,
        description: str,
        severity: str,
        check_func: callable,
        auto_correct: bool = False,
        correction_func: Optional[callable] = None,
    ) -> None:
        """Add a validation rule."""
        self.rules[name] = ValidationRule(
            name=name,
            description=description,
            severity=severity,
            check_func=check_func,
            auto_correct=auto_correct,
            correction_func=correction_func,
        )

    def validate(
        self,
        data: Dict[str, Any],
        rules: Optional[List[str]] = None,
        auto_correct: bool = True,
    ) -> ValidationResult:
        """
        Validate market data against all or specified rules.

        Parameters
        ----------
        data : dict
            Market data to validate
        rules : list, optional
            Specific rules to apply (None = all rules)
        auto_correct : bool
            Whether to apply auto-corrections

        Returns
        -------
        ValidationResult
            Validation results with issues and corrections
        """
        self.validation_stats["total_validations"] += 1

        issues = []
        corrections = []
        rules_to_check = rules or list(self.rules.keys())

        # Apply each rule
        for rule_name in rules_to_check:
            if rule_name not in self.rules:
                logger.warning(f"Unknown validation rule: {rule_name}")
                continue

            rule = self.rules[rule_name]

            try:
                # Check rule
                passed = rule.check_func(data)

                if not passed:
                    issue = ValidationIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=rule.description,
                    )
                    issues.append(issue)

                    # Apply correction if available
                    if auto_correct and rule.auto_correct and rule.correction_func:
                        try:
                            corrected_data = rule.correction_func(data)
                            data.update(corrected_data)
                            corrections.append(f"Applied correction for {rule.name}")
                            self.validation_stats["auto_corrected"] += 1
                        except Exception as e:
                            logger.error(f"Failed to apply correction for {rule.name}: {e}")

            except Exception as e:
                logger.error(f"Validation rule {rule_name} failed: {e}")
                issues.append(
                    ValidationIssue(
                        rule_name=rule.name,
                        severity="error",
                        message=f"Rule execution failed: {str(e)}",
                    )
                )

        # Determine overall validity and quality
        has_errors = any(i.severity == "error" for i in issues)
        is_valid = not has_errors

        if has_errors:
            quality_level = DataQualityLevel.INVALID
        elif len(issues) == 0:
            quality_level = DataQualityLevel.EXCELLENT
        elif len(issues) <= 2:
            quality_level = DataQualityLevel.GOOD
        elif len(issues) <= 5:
            quality_level = DataQualityLevel.ACCEPTABLE
        else:
            quality_level = DataQualityLevel.POOR

        # Update stats
        if is_valid:
            self.validation_stats["passed"] += 1
        else:
            self.validation_stats["failed"] += 1

        # Log validation results
        if is_valid:
            logger.info(
                f"Data validation passed",
                extra={
                    "function": "validate",
                    "quality_level": quality_level.value,
                    "issues_count": len(issues),
                    "corrections_count": len(corrections),
                    "has_errors": has_errors,
                },
            )
        else:
            logger.warning(
                f"Data validation failed",
                extra={
                    "function": "validate",
                    "quality_level": quality_level.value,
                    "issues_count": len(issues),
                    "corrections_count": len(corrections),
                    "has_errors": has_errors,
                },
            )

        return ValidationResult(
            is_valid=is_valid,
            quality_level=quality_level,
            issues=issues,
            corrections_applied=corrections,
            metadata={
                "validation_time": datetime.now(timezone.utc),
                "rules_checked": len(rules_to_check),
            },
        )

    def _check_freshness(self, timestamp: Any, minutes: int) -> bool:
        """Check if data is fresh within specified minutes."""
        if not timestamp:
            return False

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except:
                return False

        if not isinstance(timestamp, datetime):
            return False

        age = datetime.now(timezone.utc) - timestamp
        return age < timedelta(minutes=minutes)

    def _validate_option_chain_consistency(self, data: Dict[str, Any]) -> bool:
        """Validate option chain internal consistency."""
        option_chain = data.get("option_chain", {})
        if not option_chain:
            return False

        current_price = data.get("current_price", 0)
        if current_price <= 0:
            return False

        issues = []

        for strike_str, option_data in option_chain.items():
            try:
                strike = float(strike_str)

                # Check bid <= mid <= ask
                bid = option_data.get("bid", 0)
                ask = option_data.get("ask", float("inf"))
                mid = option_data.get("mid", 0)

                if not (bid <= mid <= ask):
                    issues.append(f"Strike {strike}: Invalid bid/mid/ask relationship")

                # Check reasonable spread
                if ask > 0 and (ask - bid) / ask > 0.5:  # 50% spread
                    issues.append(f"Strike {strike}: Excessive spread")

                # Check delta sign
                delta = option_data.get("delta", 0)
                if delta > 0:  # Put should have negative delta
                    issues.append(f"Strike {strike}: Invalid delta sign for put")

                # Check IV reasonableness
                iv = option_data.get("implied_volatility", 0)
                if iv <= 0 or iv > 5:  # 0% to 500%
                    issues.append(f"Strike {strike}: Unreasonable IV {iv}")

            except Exception as e:
                issues.append(f"Strike {strike_str}: Validation error {e}")

        if issues:
            logger.warning(f"Option chain consistency issues: {issues[:3]}")  # Log first 3

        return len(issues) == 0

    def _validate_spreads(self, data: Dict[str, Any]) -> bool:
        """Validate bid-ask spreads."""
        option_chain = data.get("option_chain", {})

        invalid_spreads = 0
        for strike_str, option_data in option_chain.items():
            bid = option_data.get("bid", 0)
            ask = option_data.get("ask", 0)

            # Check for valid spread
            if bid < 0 or ask < 0:
                invalid_spreads += 1
            elif ask > 0 and bid > ask:
                invalid_spreads += 1
            elif ask == 0 and bid > 0:
                invalid_spreads += 1

        return invalid_spreads == 0

    def _validate_greeks_relationships(self, data: Dict[str, Any]) -> bool:
        """Validate Greeks satisfy known relationships."""
        option_chain = data.get("option_chain", {})

        issues = []
        for strike_str, option_data in option_chain.items():
            # Put delta should be negative
            delta = option_data.get("delta", 0)
            if delta > 0:
                issues.append(f"Strike {strike_str}: Put delta should be negative")

            # Gamma should be positive
            gamma = option_data.get("gamma", 0)
            if gamma < 0:
                issues.append(f"Strike {strike_str}: Gamma should be positive")

            # Theta typically negative
            theta = option_data.get("theta", 0)
            if theta > 0:
                issues.append(f"Strike {strike_str}: Theta typically negative")

            # Vega should be positive
            vega = option_data.get("vega", 0)
            if vega < 0:
                issues.append(f"Strike {strike_str}: Vega should be positive")

        return len(issues) == 0

    def _validate_iv_range(self, data: Dict[str, Any]) -> bool:
        """Validate implied volatility is in reasonable range."""
        # Check overall IV
        iv = data.get("implied_volatility", 0)
        if not (0.05 <= iv <= 3.0):  # 5% to 300%
            return False

        # Check option chain IVs
        option_chain = data.get("option_chain", {})
        for strike_str, option_data in option_chain.items():
            opt_iv = option_data.get("implied_volatility", 0)
            if not (0.05 <= opt_iv <= 3.0):
                return False

        return True

    def _validate_liquidity(self, data: Dict[str, Any]) -> bool:
        """Validate minimum liquidity requirements."""
        option_chain = data.get("option_chain", {})

        # At least some options should have decent liquidity
        liquid_options = 0
        for strike_str, option_data in option_chain.items():
            volume = option_data.get("volume", 0)
            open_interest = option_data.get("open_interest", 0)

            if volume >= 10 and open_interest >= 100:
                liquid_options += 1

        # At least 20% of options should be liquid
        return liquid_options >= len(option_chain) * 0.2

    def validate_position_data(self, position_data: Dict[str, Any]) -> ValidationResult:
        """Validate position-specific data."""
        # Add position-specific rules temporarily
        self.add_rule(
            "valid_quantity",
            "Position quantity must be non-zero",
            "error",
            lambda data: data.get("quantity", 0) != 0,
        )

        self.add_rule(
            "valid_symbol",
            "Position must have valid symbol",
            "error",
            lambda data: bool(data.get("symbol")),
        )

        # Run validation
        result = self.validate(
            position_data,
            rules=[
                "valid_quantity",
                "valid_symbol",
                "positive_price",
            ],
        )

        # Clean up temporary rules
        del self.rules["valid_quantity"]
        del self.rules["valid_symbol"]

        return result

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["total_validations"]
        if total == 0:
            success_rate = 0
            correction_rate = 0
        else:
            success_rate = self.validation_stats["passed"] / total
            correction_rate = self.validation_stats["auto_corrected"] / total

        return {
            **self.validation_stats,
            "success_rate": success_rate,
            "correction_rate": correction_rate,
        }


class DataAnomalyDetector:
    """
    Detect anomalies in market data using statistical methods.
    """

    def __init__(self, lookback_periods: int = 20):
        """Initialize anomaly detector."""
        self.lookback_periods = lookback_periods
        self.historical_data: Dict[str, List[float]] = {}

    def update(self, key: str, value: float) -> None:
        """Update historical data for a key."""
        if key not in self.historical_data:
            self.historical_data[key] = []

        self.historical_data[key].append(value)

        # Keep only lookback periods
        if len(self.historical_data[key]) > self.lookback_periods:
            self.historical_data[key] = self.historical_data[key][-self.lookback_periods :]

    def is_anomaly(self, key: str, value: float, n_std: float = 3.0) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if value is anomalous based on historical data.

        Returns
        -------
        Tuple[bool, dict]
            (is_anomaly, anomaly_info)
        """
        if key not in self.historical_data or len(self.historical_data[key]) < 3:
            return False, {"reason": "Insufficient history"}

        historical = self.historical_data[key]
        mean = statistics.mean(historical)
        stdev = statistics.stdev(historical)

        if stdev == 0:
            # No variation in historical data
            is_anomaly = value != mean
            return is_anomaly, {
                "reason": "No historical variation" if is_anomaly else "Matches constant",
                "historical_mean": mean,
                "value": value,
            }

        # Calculate z-score
        z_score = abs((value - mean) / stdev)
        is_anomaly = z_score > n_std

        return is_anomaly, {
            "z_score": z_score,
            "n_std": n_std,
            "historical_mean": mean,
            "historical_stdev": stdev,
            "value": value,
            "percentile": self._calculate_percentile(value, historical),
        }

    def _calculate_percentile(self, value: float, historical: List[float]) -> float:
        """Calculate what percentile the value falls in."""
        below = sum(1 for h in historical if h < value)
        return (below / len(historical)) * 100

    def detect_market_anomalies(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in market data."""
        anomalies = []

        # Check price anomaly
        price = market_data.get("current_price", 0)
        if price > 0:
            self.update("price", price)
            is_anomaly, info = self.is_anomaly("price", price)
            if is_anomaly:
                anomalies.append(
                    {
                        "type": "price",
                        "severity": "high" if info.get("z_score", 0) > 4 else "medium",
                        "details": info,
                    }
                )

        # Check IV anomaly
        iv = market_data.get("implied_volatility", 0)
        if iv > 0:
            self.update("iv", iv)
            is_anomaly, info = self.is_anomaly("iv", iv, n_std=2.5)  # More sensitive for IV
            if is_anomaly:
                anomalies.append(
                    {
                        "type": "implied_volatility",
                        "severity": "medium",
                        "details": info,
                    }
                )

        # Check spread anomalies
        option_chain = market_data.get("option_chain", {})
        for strike_str, option_data in option_chain.items():
            bid = option_data.get("bid", 0)
            ask = option_data.get("ask", 0)

            if ask > 0:
                spread_pct = (ask - bid) / ask
                self.update(f"spread_{strike_str}", spread_pct)
                is_anomaly, info = self.is_anomaly(f"spread_{strike_str}", spread_pct, n_std=2.0)
                if is_anomaly:
                    anomalies.append(
                        {
                            "type": "spread",
                            "strike": float(strike_str),
                            "severity": "low",
                            "details": info,
                        }
                    )

        return anomalies


# Global instances
_market_validator: Optional[MarketDataValidator] = None
_anomaly_detector: Optional[DataAnomalyDetector] = None


def get_market_validator() -> MarketDataValidator:
    """Get or create global market validator instance."""
    global _market_validator
    if _market_validator is None:
        _market_validator = MarketDataValidator()
    return _market_validator


def get_anomaly_detector() -> DataAnomalyDetector:
    """Get or create global anomaly detector instance."""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = DataAnomalyDetector()
    return _anomaly_detector
