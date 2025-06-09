"""
Integration layer between new YAML configuration system and existing codebase.
Provides compatibility and migration utilities.
"""

from decimal import Decimal
from typing import Optional

from pydantic import SecretStr

from .base import Settings
from .loader import get_config, get_config_loader
from .schema import WheelConfig


class ConfigAdapter:
    """Adapts new WheelConfig to existing Settings interface."""
    
    def __init__(self, wheel_config: Optional[WheelConfig] = None):
        self.wheel_config = wheel_config or get_config()
        self._loader = get_config_loader()
    
    def to_settings(self) -> Settings:
        """Convert WheelConfig to legacy Settings format."""
        config = self.wheel_config
        
        # Extract API credentials if present
        api_key = None
        api_secret = None
        if config.trading.broker.api_key:
            api_key = config.trading.broker.api_key
        if config.trading.broker.api_secret:
            api_secret = config.trading.broker.api_secret
        
        # Create Settings instance with values from WheelConfig
        settings = Settings(
            # Trading settings
            BROKER_API_KEY=api_key,
            BROKER_API_SECRET=api_secret,
            TRADING_MODE=config.trading.mode,
            
            # Strategy settings
            WHEEL_DELTA_TARGET=config.strategy.delta_target,
            DAYS_TO_EXPIRY_TARGET=config.strategy.days_to_expiry_target,
            MAX_POSITION_SIZE=config.risk.max_position_size,
            
            # Logging
            LOG_LEVEL=config.operations.logging.level,
            
            # Google Cloud (if needed)
            GOOGLE_CLOUD_PROJECT=None,
            GOOGLE_APPLICATION_CREDENTIALS=None,
        )
        
        # Track parameter usage
        self._loader.track_parameter_usage("trading.mode")
        self._loader.track_parameter_usage("strategy.delta_target")
        self._loader.track_parameter_usage("strategy.days_to_expiry_target")
        self._loader.track_parameter_usage("risk.max_position_size")
        
        return settings
    
    def get_unity_constants(self) -> dict:
        """Get Unity-specific constants in expected format."""
        unity = self.wheel_config.unity
        
        # Track usage
        self._loader.track_parameter_usage("unity.ticker")
        self._loader.track_parameter_usage("unity.volatility.average")
        
        return {
            "ticker": unity.ticker,
            "company_name": unity.company_name,
            "average_iv": Decimal(str(unity.volatility.average)),
            "typical_iv_range": (
                Decimal(str(unity.volatility.typical_range[0])),
                Decimal(str(unity.volatility.typical_range[1]))
            ),
        }
    
    def get_risk_limits(self) -> dict:
        """Get risk limits in expected format."""
        risk = self.wheel_config.risk
        limits = risk.limits
        
        # Track usage
        self._loader.track_parameter_usage("risk.limits.max_var_95")
        self._loader.track_parameter_usage("risk.limits.max_cvar_95")
        self._loader.track_parameter_usage("risk.kelly_fraction")
        
        return {
            "max_var_95": limits.max_var_95,
            "max_cvar_95": limits.max_cvar_95,
            "max_kelly_fraction": limits.max_kelly_fraction,
            "max_delta_exposure": limits.max_delta_exposure,
            "max_gamma_exposure": limits.max_gamma_exposure,
            "max_vega_exposure": limits.max_vega_exposure,
            "max_margin_utilization": risk.max_margin_percent,
            "volatility_scalar": 1.0,  # Default if not specified
        }
    
    def get_ml_config(self) -> dict:
        """Get ML configuration in expected format."""
        ml = self.wheel_config.ml
        
        # Track usage
        self._loader.track_parameter_usage("ml.enabled")
        
        if not ml.enabled:
            return {"enabled": False}
        
        self._loader.track_parameter_usage("ml.model_path")
        self._loader.track_parameter_usage("ml.features.use_iv_rank")
        
        return {
            "enabled": ml.enabled,
            "model_path": str(ml.model_path),
            "features": ml.features.model_dump(),
            "models": {
                "probability_model": ml.models.probability_model.model_dump(),
                "volatility_model": ml.models.volatility_model.model_dump(),
            }
        }
    
    def report_decision_impact(self, decision: str, outcome: float) -> None:
        """Report the impact of a decision using specific parameters."""
        # Map decisions to parameters
        param_map = {
            "delta_selection": "strategy.delta_target",
            "dte_selection": "strategy.days_to_expiry_target",
            "position_sizing": "risk.max_position_size",
            "kelly_sizing": "risk.kelly_fraction",
            "ml_adjustment": "ml.enabled",
        }
        
        if decision in param_map:
            param = param_map[decision]
            self._loader.track_parameter_impact(param, outcome)
    
    def validate_runtime_health(self) -> bool:
        """Perform runtime health validation."""
        health = self._loader.health_report
        
        if not health.get("valid", False):
            print("Configuration validation failed!")
            for error in health.get("errors", []):
                print(f"  ERROR: {error}")
            return False
        
        # Show warnings but don't fail
        for warning in health.get("warnings", []):
            print(f"  WARNING: {warning}")
        
        return True


# Singleton adapter instance
_adapter: Optional[ConfigAdapter] = None


def get_config_adapter() -> ConfigAdapter:
    """Get or create configuration adapter singleton."""
    global _adapter
    if _adapter is None:
        _adapter = ConfigAdapter()
    return _adapter


def migrate_env_to_yaml() -> None:
    """
    Utility to help migrate from .env configuration to config.yaml.
    Reads current environment variables and suggests YAML configuration.
    """
    import os
    
    print("Environment Variable Migration Helper")
    print("=" * 40)
    print()
    
    # Map of env vars to config paths
    env_map = {
        "WHEEL_DELTA_TARGET": "strategy.delta_target",
        "DAYS_TO_EXPIRY_TARGET": "strategy.days_to_expiry_target",
        "MAX_POSITION_SIZE": "risk.max_position_size",
        "MAX_MARGIN_PERCENT": "risk.max_margin_percent",
        "MAX_DRAWDOWN_PERCENT": "risk.max_drawdown_percent",
        "TRADING_MODE": "trading.mode",
        "LOG_LEVEL": "operations.logging.level",
        "USE_ML_ENHANCEMENT": "ml.enabled",
        "ML_MODEL_PATH": "ml.model_path",
    }
    
    found_vars = {}
    for env_var, config_path in env_map.items():
        value = os.environ.get(env_var)
        if value:
            found_vars[env_var] = (value, config_path)
    
    if found_vars:
        print("Found environment variables to migrate:")
        print()
        for env_var, (value, path) in found_vars.items():
            print(f"  {env_var} = {value}")
            print(f"    → Set in config.yaml at: {path}")
            print()
        
        print("To use environment overrides with new system, rename to:")
        print()
        for env_var, (value, path) in found_vars.items():
            new_name = "WHEEL_" + path.upper().replace(".", "__")
            print(f"  export {new_name}={value}")
        print()
    else:
        print("No legacy environment variables found.")
    
    print()
    print("The new configuration system uses config.yaml as the primary")
    print("configuration source, with environment variables for overrides.")
    print("See config.yaml for all available parameters.")


# Compatibility function for existing code
def get_settings_with_tracking() -> Settings:
    """
    Get Settings instance with automatic parameter tracking.
    This maintains compatibility with existing code while adding tracking.
    """
    adapter = get_config_adapter()
    return adapter.to_settings()