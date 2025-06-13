"""Feature flag system for graceful degradation and controlled rollouts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .logging import get_logger

logger = get_logger(__name__)


class FeatureStatus(str, Enum):
    """Feature flag status."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    DEGRADED = "degraded"
    EXPERIMENTAL = "experimental"


@dataclass
class Feature:
    """Feature definition with metadata."""

    name: str
    description: str
    status: FeatureStatus = FeatureStatus.DISABLED
    dependencies: List[str] = field(default_factory=list)
    fallback: Optional[Callable] = None
    error_count: int = 0
    max_errors: int = 3
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

    def should_auto_disable(self) -> bool:
        """Check if feature should be auto-disabled due to errors."""
        return self.error_count >= self.max_errors


class FeatureFlags:
    """
    Feature flag management with graceful degradation.

    Supports:
    - Dynamic enable/disable of features
    - Automatic degradation on errors
    - Feature dependencies
    - Fallback mechanisms
    - A/B testing capabilities
    """

    # Core features that should rarely be disabled
    CORE_FEATURES = {
        "black_scholes_calculation",
        "risk_validation",
        "position_sizing",
    }

    # Features that can be disabled without breaking core functionality
    OPTIONAL_FEATURES = {
        "ml_predictions",
        "advanced_greeks",
        "real_time_data",
        "performance_tracking",
        "decision_logging",
        "metrics_export",
    }

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize feature flag system."""
        self.config_file = config_file or Path("feature_flags.json")
        self.features: Dict[str, Feature] = {}
        self.disabled_by_error: Set[str] = set()

        # Initialize default features
        self._initialize_default_features()

        # Load saved configuration
        self._load_config()

    def _initialize_default_features(self) -> None:
        """Initialize default feature definitions."""
        # Core features (default enabled)
        self.register(
            "black_scholes_calculation",
            "Black-Scholes option pricing",
            status=FeatureStatus.ENABLED,
        )

        self.register(
            "risk_validation",
            "Risk metric validation and limits",
            status=FeatureStatus.ENABLED,
        )

        self.register(
            "position_sizing",
            "Kelly criterion position sizing",
            status=FeatureStatus.ENABLED,
        )

        # Optional features
        self.register(
            "ml_predictions",
            "Machine learning price predictions",
            status=FeatureStatus.DISABLED,
            fallback=lambda: {"confidence": 0.5, "prediction": None},
        )

        self.register(
            "advanced_greeks",
            "Second-order Greeks calculations",
            status=FeatureStatus.ENABLED,
            dependencies=["black_scholes_calculation"],
        )

        self.register(
            "real_time_data",
            "Real-time market data feed",
            status=FeatureStatus.DISABLED,
            fallback=lambda: {"source": "mock", "delay": 0},
        )

        self.register(
            "performance_tracking",
            "Detailed performance monitoring",
            status=FeatureStatus.ENABLED,
        )

        self.register(
            "decision_logging",
            "Decision audit trail logging",
            status=FeatureStatus.ENABLED,
        )

        self.register(
            "metrics_export",
            "Metrics export to external systems",
            status=FeatureStatus.DISABLED,
        )

        self.register(
            "volatility_smile",
            "Volatility smile modeling",
            status=FeatureStatus.EXPERIMENTAL,
            dependencies=["black_scholes_calculation"],
        )

        self.register(
            "auto_recovery",
            "Automatic error recovery",
            status=FeatureStatus.ENABLED,
        )

    def register(
        self,
        name: str,
        description: str,
        status: FeatureStatus = FeatureStatus.DISABLED,
        dependencies: Optional[List[str]] = None,
        fallback: Optional[Callable] = None,
        max_errors: int = 3,
    ) -> None:
        """Register a new feature flag."""
        self.features[name] = Feature(
            name=name,
            description=description,
            status=status,
            dependencies=dependencies or [],
            fallback=fallback,
            max_errors=max_errors,
        )

        logger.info(
            f"Feature registered: {name}",
            extra={
                "feature": name,
                "status": status.value,
                "has_fallback": fallback is not None,
            },
        )

    def is_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        feature = self.features.get(feature_name)
        if not feature:
            logger.warning(f"Unknown feature: {feature_name}")
            return False

        # Check if disabled by error
        if feature_name in self.disabled_by_error:
            return False

        # Check status
        if feature.status in (FeatureStatus.DISABLED, FeatureStatus.DEGRADED):
            return False

        # Check dependencies
        for dep in feature.dependencies:
            if not self.is_enabled(dep):
                return False

        return True

    def is_experimental(self, feature_name: str) -> bool:
        """Check if a feature is experimental."""
        feature = self.features.get(feature_name)
        return feature and feature.status == FeatureStatus.EXPERIMENTAL

    def enable(self, feature_name: str, force: bool = False) -> bool:
        """Enable a feature."""
        feature = self.features.get(feature_name)
        if not feature:
            logger.error(f"Cannot enable unknown feature: {feature_name}")
            return False

        # Check if it's a core feature being disabled
        if feature_name in self.CORE_FEATURES and not force:
            logger.warning(f"Enabling core feature {feature_name} (already should be enabled)")

        # Check dependencies
        for dep in feature.dependencies:
            if not self.is_enabled(dep):
                logger.error(f"Cannot enable {feature_name}: dependency {dep} is disabled")
                return False

        feature.status = FeatureStatus.ENABLED
        feature.error_count = 0
        self.disabled_by_error.discard(feature_name)

        logger.info(f"Feature enabled: {feature_name}")
        self._save_config()
        return True

    def disable(self, feature_name: str, reason: str = "") -> bool:
        """Disable a feature."""
        feature = self.features.get(feature_name)
        if not feature:
            logger.error(f"Cannot disable unknown feature: {feature_name}")
            return False

        # Warn if disabling core feature
        if feature_name in self.CORE_FEATURES:
            logger.warning(f"Disabling core feature {feature_name}! Reason: {reason}")

        feature.status = FeatureStatus.DISABLED

        logger.info(f"Feature disabled: {feature_name}", extra={"reason": reason})
        self._save_config()
        return True

    def degrade(self, feature_name: str, error: Optional[Exception] = None) -> None:
        """Degrade a feature due to error."""
        feature = self.features.get(feature_name)
        if not feature:
            return

        feature.error_count += 1
        feature.last_error = str(error) if error else "Unknown error"
        feature.last_error_time = datetime.now(timezone.utc)

        if feature.should_auto_disable():
            feature.status = FeatureStatus.DEGRADED
            self.disabled_by_error.add(feature_name)
            logger.error(
                f"Feature auto-degraded after {feature.error_count} errors: {feature_name}",
                extra={
                    "error_count": feature.error_count,
                    "last_error": feature.last_error,
                },
            )
        else:
            logger.warning(
                f"Feature error {feature.error_count}/{feature.max_errors}: {feature_name}",
                extra={"error": str(error)},
            )

    def get_fallback(self, feature_name: str) -> Optional[Callable]:
        """Get fallback function for a feature."""
        feature = self.features.get(feature_name)
        return feature.fallback if feature else None

    def with_feature(self, feature_name: str):
        """
        Decorator to conditionally execute code based on feature flag.

        Usage:
            @feature_flags.with_feature("ml_predictions")
            def use_ml_predictions():
                return ml_model.predict()
        """

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                if self.is_enabled(feature_name):
                    try:
                        return func(*args, **kwargs)
                    except (ValueError, KeyError, AttributeError) as e:
                        self.degrade(feature_name, e)

                        # Try fallback
                        fallback = self.get_fallback(feature_name)
                        if fallback:
                            logger.info(
                                f"Using fallback for {feature_name}", extra={"error": str(e)}
                            )
                            return fallback()
                        raise
                else:
                    # Use fallback if available
                    fallback = self.get_fallback(feature_name)
                    if fallback:
                        return fallback()

                    # Skip if experimental
                    if self.is_experimental(feature_name):
                        logger.debug(f"Skipping experimental feature: {feature_name}")
                        return None

                    raise RuntimeError(f"Feature disabled: {feature_name}")

            return wrapper

        return decorator

    def reset_errors(self, feature_name: str) -> None:
        """Reset error count for a feature."""
        feature = self.features.get(feature_name)
        if feature:
            feature.error_count = 0
            feature.last_error = None
            feature.last_error_time = None
            self.disabled_by_error.discard(feature_name)

            # Re-enable if it was degraded
            if feature.status == FeatureStatus.DEGRADED:
                feature.status = FeatureStatus.ENABLED

            logger.info(f"Reset errors for feature: {feature_name}")

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": {},
            "summary": {
                "total": len(self.features),
                "enabled": 0,
                "disabled": 0,
                "degraded": 0,
                "experimental": 0,
            },
        }

        for name, feature in self.features.items():
            is_enabled = self.is_enabled(name)

            report["features"][name] = {
                "description": feature.description,
                "status": feature.status.value,
                "is_enabled": is_enabled,
                "is_core": name in self.CORE_FEATURES,
                "dependencies": feature.dependencies,
                "error_count": feature.error_count,
                "last_error": feature.last_error,
                "has_fallback": feature.fallback is not None,
            }

            # Update summary
            report["summary"][feature.status.value] += 1

        return report

    def _save_config(self) -> None:
        """Save current configuration to file."""
        config = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": {
                name: {
                    "status": feature.status.value,
                    "error_count": feature.error_count,
                    "last_error": feature.last_error,
                }
                for name, feature in self.features.items()
            },
            "disabled_by_error": list(self.disabled_by_error),
        }

        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to save feature flags config: {e}")

    def _load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)

            # Update feature states
            for name, state in config.get("features", {}).items():
                if name in self.features:
                    feature = self.features[name]
                    feature.status = FeatureStatus(state["status"])
                    feature.error_count = state.get("error_count", 0)
                    feature.last_error = state.get("last_error")

            # Restore disabled by error set
            self.disabled_by_error = set(config.get("disabled_by_error", []))

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to load feature flags config: {e}")


# Global singleton instance
_feature_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    """Get or create global feature flags instance."""
    global _feature_flags
    if _feature_flags is None:
        # Check for config file path in environment
        config_path = os.environ.get("FEATURE_FLAGS_CONFIG")
        if config_path:
            _feature_flags = FeatureFlags(Path(config_path))
        else:
            _feature_flags = FeatureFlags()
    return _feature_flags
