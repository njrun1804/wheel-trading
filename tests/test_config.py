"""
Comprehensive tests for the configuration system.
Tests schema validation, environment overrides, tracking, and health reporting.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from src.config.loader import ConfigurationLoader, get_config, get_config_loader
from src.config.schema import (
    RiskConfig,
    StrategyConfig,
    WheelConfig,
    load_config,
    validate_config_health,
)


@pytest.fixture
def valid_config_dict():
    """Create a valid configuration dictionary."""
    return {
        "strategy": {
            "delta_target": 0.30,
            "days_to_expiry_target": 45,
            "min_days_to_expiry": 21,
            "max_delta_short_put": 0.35,
            "strike_intervals": [1.0, 2.5, 5.0],
            "roll_triggers": {
                "profit_target_percent": 0.50,
                "delta_breach_threshold": 0.45,
                "dte_threshold": 14,
            },
        },
        "risk": {
            "max_position_size": 0.20,
            "max_portfolio_delta": 100.0,
            "max_portfolio_gamma": 50.0,
            "max_portfolio_vega": 1000.0,
            "max_naked_puts": 3,
            "kelly_fraction": 0.50,
            "limits": {
                "max_var_95": 0.05,
                "max_cvar_95": 0.075,
                "max_margin_utilization": 0.50,
                "max_delta_exposure": 100.0,
                "max_gamma_exposure": 10.0,
                "max_vega_exposure": 1000.0,
            },
        },
        "data": {
            "cache_ttl": {
                "options_chain": 300,
                "stock_quote": 60,
                "greeks": 300,
                "implied_volatility": 300,
            },
            "api_timeouts": {
                "connect": 5,
                "read": 30,
                "total": 60,
            },
            "quality": {
                "stale_data_threshold": 600,
                "min_option_volume": 10,
                "min_option_open_interest": 100,
                "max_bid_ask_spread": 0.10,
                "min_underlying_volume": 100000,
            },
        },
        "ml": {
            "enabled": False,
            "features": {
                "use_iv_rank": True,
                "use_iv_skew": True,
                "use_term_structure": True,
                "use_realized_vol": True,
                "use_volume_analysis": True,
                "use_macro_factors": False,
            },
            "models": {
                "probability_model": {
                    "type": "gradient_boost",
                    "confidence_threshold": 0.75,
                    "update_frequency": 86400,
                    "min_training_samples": 1000,
                },
                "volatility_model": {
                    "type": "garch",
                    "lookback_days": 252,
                    "update_frequency": 3600,
                },
            },
            "hyperparameters": {
                "max_features": 20,
                "learning_rate": 0.01,
                "max_depth": 6,
                "n_estimators": 100,
            },
        },
        "broker": {
            "api_base_url": "https://api.schwab.com/v1",
            "auth_url": "https://auth.schwab.com/oauth/token",
            "redirect_uri": "http://localhost:8000/callback",
            "scopes": [
                "read",
                "write",
                "trade",
            ],
        },
        "trading": {
            "mode": "paper",
            "enable_trading": False,
            "max_orders_per_day": 10,
            "order_types_allowed": ["limit"],
            "use_extended_hours": False,
            "commission_per_contract": 0.65,
        },
        "monitoring": {
            "log_level": "INFO",
            "enable_metrics": True,
            "enable_alerts": True,
            "metrics_export_interval": 60,
            "alert_channels": ["log", "email"],
            "performance_tracking": {
                "track_decision_time": True,
                "track_calculation_accuracy": True,
                "track_prediction_outcomes": True,
            },
        },
        "backtest": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000,
            "data_source": "historical",
            "include_transaction_costs": True,
            "slippage_model": "fixed",
            "slippage_bps": 5,
        },
        "metadata": {
            "version": "1.0.0",
            "last_updated": "2024-01-01",
            "environment": "test",
        },
    }


@pytest.fixture
def valid_config_yaml(valid_config_dict, tmp_path):
    """Create a valid configuration YAML file."""
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_dict, f)
    return config_file


class TestConfigSchema:
    """Test configuration schema validation."""

    def test_valid_config_loads(self, valid_config_dict):
        """Test that a valid configuration loads successfully."""
        config = WheelConfig(**valid_config_dict)
        assert config.strategy.delta_target == 0.30
        assert config.risk.max_position_size == 0.20
        assert config.ml.enabled is False

    def test_invalid_delta_target(self, valid_config_dict):
        """Test that invalid delta target is rejected."""
        valid_config_dict["strategy"]["delta_target"] = 1.5
        with pytest.raises(ValidationError):
            WheelConfig(**valid_config_dict)

    def test_invalid_max_position_size(self, valid_config_dict):
        """Test that invalid position size is rejected."""
        valid_config_dict["risk"]["max_position_size"] = -0.1
        with pytest.raises(ValidationError):
            WheelConfig(**valid_config_dict)

    def test_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            WheelConfig(strategy={}, risk={})


class TestConfigLoader:
    """Test configuration loader functionality."""

    def test_load_from_yaml(self, valid_config_yaml):
        """Test loading configuration from YAML file."""
        loader = ConfigurationLoader(str(valid_config_yaml))
        config = loader.config
        assert config.strategy.delta_target == 0.30

    def test_environment_override(self, valid_config_yaml, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("WHEEL_STRATEGY__DELTA_TARGET", "0.25")
        monkeypatch.setenv("WHEEL_ML__ENABLED", "true")

        loader = ConfigurationLoader(str(valid_config_yaml))
        config = loader.config

        assert config.strategy.delta_target == 0.25
        assert config.ml.enabled is True

    def test_nested_environment_override(self, valid_config_yaml, monkeypatch):
        """Test nested environment variable overrides."""
        monkeypatch.setenv("WHEEL_RISK__LIMITS__MAX_VAR_95", "0.03")

        loader = ConfigurationLoader(str(valid_config_yaml))
        config = loader.config

        assert config.risk.limits.max_var_95 == 0.03

    def test_list_environment_override(self, valid_config_yaml, monkeypatch):
        """Test list environment variable overrides."""
        monkeypatch.setenv("WHEEL_TRADING__BROKER__SCOPES", '["read", "trade"]')

        loader = ConfigurationLoader(str(valid_config_yaml))
        config = loader.config

        assert config.trading.broker.scopes == ["read", "trade"]

    def test_invalid_environment_override(self, valid_config_yaml, monkeypatch):
        """Test that invalid environment overrides are handled."""
        monkeypatch.setenv("WHEEL_STRATEGY__DELTA_TARGET", "invalid")

        with pytest.raises(ValidationError):
            ConfigurationLoader(str(valid_config_yaml))


class TestConfigTracking:
    """Test configuration usage tracking."""

    def test_parameter_usage_tracking(self, valid_config_yaml):
        """Test that parameter usage is tracked."""
        loader = ConfigurationLoader(str(valid_config_yaml))

        # Track some usage
        loader.track_parameter_usage("strategy.delta_target")
        loader.track_parameter_usage("strategy.delta_target")
        loader.track_parameter_usage("risk.max_position_size")

        usage = loader.get_parameter_usage()
        assert usage["strategy.delta_target"] == 2
        assert usage["risk.max_position_size"] == 1

    def test_parameter_impact_tracking(self, valid_config_yaml):
        """Test that parameter impact is tracked."""
        loader = ConfigurationLoader(str(valid_config_yaml))

        # Track some impacts
        loader.track_parameter_impact("strategy.delta_target", 0.75)
        loader.track_parameter_impact("strategy.delta_target", 0.80)
        loader.track_parameter_impact("strategy.delta_target", 0.70)

        # Check average confidence
        impacts = loader.parameter_impacts["strategy.delta_target"]
        avg_confidence = sum(impacts) / len(impacts)
        assert avg_confidence == pytest.approx(0.75, rel=0.01)

    def test_unused_parameters_detection(self, valid_config_yaml):
        """Test detection of unused parameters."""
        loader = ConfigurationLoader(str(valid_config_yaml))

        # Track usage of only some parameters
        loader.track_parameter_usage("strategy.delta_target")
        loader.track_parameter_usage("risk.max_position_size")

        unused = loader.get_unused_parameters()
        assert "ml.enabled" in unused
        assert "backtest.start_date" in unused
        assert "strategy.delta_target" not in unused


class TestConfigHealth:
    """Test configuration health checks."""

    def test_health_report_generation(self, valid_config_yaml):
        """Test that health report is generated correctly."""
        loader = ConfigurationLoader(str(valid_config_yaml))

        # Simulate some usage
        loader.track_parameter_usage("strategy.delta_target")
        loader.track_parameter_impact("strategy.delta_target", 0.90)
        loader.track_parameter_impact("risk.max_position_size", 0.40)

        report = loader.generate_health_report()

        assert "warnings" in report
        assert "recommendations" in report
        assert "statistics" in report
        assert report["statistics"]["total_parameters"] > 0

    def test_low_confidence_warning(self, valid_config_yaml):
        """Test that low confidence parameters generate warnings."""
        loader = ConfigurationLoader(str(valid_config_yaml))

        # Track low confidence
        for _ in range(10):
            loader.track_parameter_impact("risk.max_position_size", 0.30)

        report = loader.generate_health_report()
        warnings = report["warnings"]

        assert any("low average confidence" in w for w in warnings)

    def test_tuning_suggestions(self, valid_config_yaml):
        """Test that tuning suggestions are generated."""
        loader = ConfigurationLoader(str(valid_config_yaml))

        # Track varying confidences
        loader.track_parameter_impact("strategy.delta_target", 0.90)
        loader.track_parameter_impact("strategy.delta_target", 0.60)
        loader.track_parameter_impact("strategy.delta_target", 0.95)

        suggestions = loader.suggest_parameter_tuning()
        assert "strategy.delta_target" in suggestions


class TestConfigPersistence:
    """Test configuration persistence and loading."""

    def test_save_and_load_tracking(self, valid_config_yaml, tmp_path):
        """Test saving and loading tracking data."""
        loader = ConfigurationLoader(str(valid_config_yaml))

        # Track some data
        loader.track_parameter_usage("strategy.delta_target")
        loader.track_parameter_impact("strategy.delta_target", 0.85)

        # Save tracking
        tracking_file = tmp_path / "tracking.json"
        loader.save_tracking_data(str(tracking_file))

        # Load into new loader
        new_loader = ConfigurationLoader(str(valid_config_yaml))
        new_loader.load_tracking_data(str(tracking_file))

        assert new_loader.parameter_usage["strategy.delta_target"] == 1
        assert len(new_loader.parameter_impacts["strategy.delta_target"]) == 1


class TestConfigValidation:
    """Test configuration validation functions."""

    def test_validate_config_health(self, valid_config_dict):
        """Test configuration health validation."""
        config = WheelConfig(**valid_config_dict)
        issues = validate_config_health(config)

        # Should have no critical issues with valid config
        assert all(issue["severity"] != "critical" for issue in issues)

    def test_conflicting_settings_detection(self, valid_config_dict):
        """Test detection of conflicting settings."""
        # Create conflict: paper mode but trading enabled
        valid_config_dict["trading"]["mode"] = "paper"
        valid_config_dict["trading"]["enable_trading"] = True

        config = WheelConfig(**valid_config_dict)
        issues = validate_config_health(config)

        # Should detect the conflict
        assert any("paper mode" in issue["message"] for issue in issues)


class TestConfigIntegration:
    """Test configuration integration with application."""

    def test_get_config_singleton(self, valid_config_yaml, monkeypatch):
        """Test that get_config returns singleton."""
        monkeypatch.setenv("WHEEL_CONFIG_PATH", str(valid_config_yaml))

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_get_config_loader_singleton(self, valid_config_yaml, monkeypatch):
        """Test that get_config_loader returns singleton."""
        monkeypatch.setenv("WHEEL_CONFIG_PATH", str(valid_config_yaml))

        loader1 = get_config_loader()
        loader2 = get_config_loader()

        assert loader1 is loader2

    def test_config_reload(self, valid_config_yaml, tmp_path, monkeypatch):
        """Test configuration reload functionality."""
        monkeypatch.setenv("WHEEL_CONFIG_PATH", str(valid_config_yaml))

        loader = get_config_loader()
        original_delta = loader.config.strategy.delta_target

        # Modify config file
        config_dict = yaml.safe_load(valid_config_yaml.read_text())
        config_dict["strategy"]["delta_target"] = 0.35
        valid_config_yaml.write_text(yaml.dump(config_dict))

        # Reload
        loader.reload()

        assert loader.config.strategy.delta_target == 0.35
        assert loader.config.strategy.delta_target != original_delta
