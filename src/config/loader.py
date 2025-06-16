"""
from __future__ import annotations

Configuration loader with environment variable override support.
Provides unified configuration management with tracking and health reporting.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import WheelConfig, validate_config_health

logger = logging.getLogger(__name__)


class ConfigurationLoader:
    """
    Intelligent configuration loader with environment override,
    change tracking, and self-tuning capabilities.
    """

    def __init__(
        self,
        config_path: str | Path = os.getenv("WHEEL_CONFIG_PATH", "config/unified.yaml"),
    ):
        self.config_path = Path(config_path)
        self.config: WheelConfig | None = None
        self.raw_config: dict[str, Any] = {}
        self.overrides: dict[str, Any] = {}
        self.parameter_usage: dict[str, int] = {}
        self.parameter_impact: dict[str, float] = {}
        self.parameter_execution_times: dict[str, list[float]] = {}
        self.config_history: list[dict[str, Any]] = []
        self.health_report: dict[str, Any] = {}

        # Load configuration on init
        self.load()

    def load(self) -> WheelConfig:
        """Load configuration with environment overrides."""
        # Load base config from YAML
        self.raw_config = self._load_yaml()

        # Apply environment variable overrides
        config_with_overrides = self._apply_env_overrides(self.raw_config.copy())

        # Validate and create config object
        try:
            self.config = WheelConfig(**config_with_overrides)
        except ValidationError as e:
            print(f"Configuration validation failed: {e}")
            raise

        # Perform health checks
        self.health_report = validate_config_health(self.config)

        # Track configuration change
        self._track_config_change()

        return self.config

    def _load_yaml(self) -> dict[str, Any]:
        """Load raw configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Ensure metadata exists
        if "metadata" not in config:
            config["metadata"] = {}
        if "last_updated" not in config["metadata"]:
            config["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        return config

    def _apply_env_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Apply environment variable overrides.

        Environment variables follow pattern: WHEEL_<SECTION>__<SUBSECTION>__<PARAM>
        Examples:
            WHEEL_STRATEGY__DELTA_TARGET=0.25
            WHEEL_RISK__MAX_POSITION_SIZE=0.15
            WHEEL_ML__ENABLED=true
        """
        self.overrides = {}
        prefix = "WHEEL_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Parse environment variable
            key_parts = env_key[len(prefix) :].lower().split("__")

            # Navigate to the correct nested dict location
            current = config
            for _i, part in enumerate(key_parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value with type conversion
            final_key = key_parts[-1]
            converted_value = self._convert_env_value(env_value)
            current[final_key] = converted_value

            # Track override
            override_path = ".".join(key_parts)
            self.overrides[override_path] = {
                "original": current.get(final_key),
                "override": converted_value,
                "env_var": env_key,
            }

        return config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Handle booleans
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Handle None
        if value.lower() == "none":
            return None

        # Try numeric conversion
        try:
            # Try int first
            if "." not in value:
                return int(value)
            # Then float
            return float(value)
        except ValueError as e:
            import logging

            logging.debug(f"Exception caught: {e}", exc_info=True)
            pass

        # Handle lists (comma-separated)
        if "," in value:
            return [self._convert_env_value(v.strip()) for v in value.split(",")]

        # Return as string
        return value

    def _track_config_change(self) -> None:
        """Track configuration changes for history."""
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "config_hash": hash(str(self.raw_config)),
            "overrides": self.overrides,
            "health": self.health_report.get("summary", ""),
        }
        self.config_history.append(change_record)

    def track_parameter_usage(
        self, param_path: str, execution_time: float = 0.0
    ) -> None:
        """Track when a configuration parameter is used with performance metrics."""
        if not self.config.metadata.tracking.track_usage:
            return

        self.parameter_usage[param_path] = self.parameter_usage.get(param_path, 0) + 1

        # Track execution time if provided
        if execution_time > 0:
            if param_path not in self.parameter_execution_times:
                self.parameter_execution_times[param_path] = []
            self.parameter_execution_times[param_path].append(execution_time)

    def track_parameter_impact(self, param_path: str, impact_score: float) -> None:
        """Track the impact of a parameter on outcomes."""
        if not self.config.metadata.tracking.track_usage:
            return

        # Use exponential moving average
        alpha = 0.1
        current = self.parameter_impact.get(param_path, 0.0)
        self.parameter_impact[param_path] = alpha * impact_score + (1 - alpha) * current

    def get_unused_parameters(self) -> set[str]:
        """Get parameters that have never been accessed."""
        if not self.config.metadata.tracking.warn_unused:
            return set()

        all_params = self._get_all_parameter_paths()
        used_params = set(self.parameter_usage.keys())
        return all_params - used_params

    def _get_all_parameter_paths(self, obj: Any = None, prefix: str = "") -> set[str]:
        """Recursively get all parameter paths in config."""
        if obj is None:
            obj = self.config.model_dump()

        paths = set()

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict | list):
                    paths.update(self._get_all_parameter_paths(value, new_prefix))
                else:
                    paths.add(new_prefix)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_prefix = f"{prefix}[{i}]"
                if isinstance(item, dict | list):
                    paths.update(self._get_all_parameter_paths(item, new_prefix))
                else:
                    paths.add(new_prefix)

        return paths

    def suggest_parameter_adjustments(self) -> list[dict[str, Any]]:
        """Suggest parameter adjustments based on usage and impact."""
        if not self.config.metadata.tracking.suggest_tuning:
            return []

        suggestions = []

        # Suggest adjustments for high-impact, frequently used parameters
        for param, impact in self.parameter_impact.items():
            usage = self.parameter_usage.get(param, 0)

            if usage > 10 and abs(impact) > 0.1:
                current_value = self._get_param_value(param)
                if isinstance(current_value, int | float):
                    # Suggest small adjustment based on impact direction
                    adjustment = 0.1 if impact > 0 else -0.1
                    suggested_value = current_value * (1 + adjustment)

                    suggestions.append(
                        {
                            "parameter": param,
                            "current": current_value,
                            "suggested": suggested_value,
                            "reason": f"High impact ({impact:.3f}) with {usage} uses",
                            "confidence": min(abs(impact), 1.0),
                        }
                    )

        # Warn about unused parameters
        unused = self.get_unused_parameters()
        for param in unused:
            suggestions.append(
                {
                    "parameter": param,
                    "current": self._get_param_value(param),
                    "suggested": "Consider removing",
                    "reason": "Never used",
                    "confidence": 0.5,
                }
            )

        return sorted(suggestions, key=lambda x: x["confidence"], reverse=True)

    def _get_param_value(self, param_path: str) -> Any:
        """Get parameter value by path."""
        parts = param_path.replace("[", ".").replace("]", "").split(".")
        obj = self.config.model_dump()

        for part in parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = obj.get(part)
            if obj is None:
                return None

        return obj

    def disable_broken_features(self, feature_path: str, error: Exception) -> None:
        """Automatically disable features that are broken."""
        if not self.config.metadata.tracking.auto_disable_broken:
            return

        print(f"Auto-disabling broken feature: {feature_path} due to {error}")

        # Mark feature as disabled in config
        parts = feature_path.split(".")
        current = self.config.model_dump()

        # Navigate to feature
        for part in parts[:-1]:
            current = current[part]

        # Disable if it's a boolean flag
        if isinstance(current[parts[-1]], bool):
            current[parts[-1]] = False
            # Note: In production, you'd want to persist this change

    def generate_health_report(self) -> str:
        """Generate comprehensive configuration health report."""
        report_lines = [
            "=" * 60,
            "CONFIGURATION HEALTH REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]

        # Basic info
        report_lines.extend(
            [
                f"Configuration File: {self.config_path}",
                f"Environment: {self.config.metadata.environment}",
                f"Version: {self.config.metadata.version}",
                f"Trading Mode: {self.config.trading.mode}",
                "",
            ]
        )

        # Environment overrides
        if self.overrides:
            report_lines.extend(
                [
                    "ENVIRONMENT OVERRIDES:",
                    "-" * 30,
                ]
            )
            for _path, info in self.overrides.items():
                report_lines.append(
                    f"  {info['env_var']}: {info['original']} → {info['override']}"
                )
            report_lines.append("")

        # Validation results
        health = self.health_report
        report_lines.extend(
            [
                "VALIDATION RESULTS:",
                "-" * 30,
                f"Status: {'✓ VALID' if health.get('valid', False) else '✗ INVALID'}",
                "",
            ]
        )

        if health.get("errors"):
            report_lines.extend(
                [
                    "ERRORS:",
                ]
            )
            for error in health["errors"]:
                report_lines.append(f"  ✗ {error}")
            report_lines.append("")

        if health.get("warnings"):
            report_lines.extend(
                [
                    "WARNINGS:",
                ]
            )
            for warning in health["warnings"]:
                report_lines.append(f"  ⚠ {warning}")
            report_lines.append("")

        if health.get("recommendations"):
            report_lines.extend(
                [
                    "RECOMMENDATIONS:",
                ]
            )
            for rec in health["recommendations"]:
                report_lines.append(f"  → {rec}")
            report_lines.append("")

        # Parameter usage statistics
        if self.parameter_usage:
            report_lines.extend(
                [
                    "PARAMETER USAGE:",
                    "-" * 30,
                ]
            )
            sorted_usage = sorted(
                self.parameter_usage.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for param, count in sorted_usage:
                report_lines.append(f"  {param}: {count} uses")
            report_lines.append("")

        # Unused parameters
        unused = self.get_unused_parameters()
        if unused:
            report_lines.extend(
                [
                    "UNUSED PARAMETERS:",
                    "-" * 30,
                ]
            )
            for param in sorted(unused)[:10]:
                report_lines.append(f"  - {param}")
            if len(unused) > 10:
                report_lines.append(f"  ... and {len(unused) - 10} more")
            report_lines.append("")

        # Parameter tuning suggestions
        suggestions = self.suggest_parameter_adjustments()
        if suggestions:
            report_lines.extend(
                [
                    "TUNING SUGGESTIONS:",
                    "-" * 30,
                ]
            )
            for sug in suggestions[:5]:
                report_lines.append(
                    f"  {sug['parameter']}: {sug['current']} → {sug['suggested']}"
                )
                report_lines.append(f"    Reason: {sug['reason']}")
            report_lines.append("")

        # Risk profile summary
        report_lines.extend(
            [
                "RISK PROFILE:",
                "-" * 30,
                f"Max Position Size: {self.config.risk.max_position_size:.1%}",
                f"Max Margin Usage: {self.config.risk.max_margin_percent:.1%}",
                f"Max Drawdown: {self.config.risk.max_drawdown_percent:.1%}",
                f"Kelly Fraction: {self.config.risk.kelly_fraction:.1%}",
                f"Delta Target: {self.config.strategy.delta_target:.2f}",
                f"Days to Expiry Target: {self.config.strategy.days_to_expiry_target}",
                "",
            ]
        )

        # ML status
        report_lines.extend(
            [
                "ML STATUS:",
                "-" * 30,
                f"Enabled: {'Yes' if self.config.ml.enabled else 'No'}",
            ]
        )
        if self.config.ml.enabled:
            report_lines.extend(
                [
                    f"Model Path: {self.config.ml.model_path}",
                    f"Features: {sum(1 for v in self.config.ml.features.model_dump().values() if v)} active",
                ]
            )

        report_lines.extend(["", "=" * 60])

        return "\n".join(report_lines)

    def export_config(self, path: str | Path, format: str = "yaml") -> None:
        """Export current configuration with overrides applied."""
        path = Path(path)
        config_dict = self.config.model_dump()

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj

        config_dict = convert_paths(config_dict)

        # Add metadata about export
        config_dict["_export_metadata"] = {
            "exported_at": datetime.now().isoformat(),
            "source_file": str(self.config_path),
            "overrides_applied": len(self.overrides),
        }

        if format == "yaml":
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Singleton instance
_config_loader: ConfigurationLoader | None = None


def get_config_loader(
    config_path: str | Path = os.getenv("WHEEL_CONFIG_PATH", "config_unified.yaml")
) -> ConfigurationLoader:
    """Get or create configuration loader singleton."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigurationLoader(config_path)
    return _config_loader


def get_config() -> WheelConfig:
    """Get current configuration."""
    return get_config_loader().config
