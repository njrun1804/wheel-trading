"""
Minimal configuration schema for Unity Wheel Trading Bot.

This simplified config contains only essential parameters, removing 90% of unused settings.
Focuses on core functionality for local-only operation.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class MetadataConfig(BaseModel):
    """Essential metadata."""
    version: str = Field("2.2.0", description="Configuration version")
    environment: str = Field("production", description="Environment (development/production)")
    created_at: datetime = Field(default_factory=datetime.now, description="Config creation time")


class UnityConfig(BaseModel):
    """Unity-specific trading parameters."""
    ticker: str = Field("U", description="Unity ticker symbol")
    target_delta: float = Field(0.30, ge=0.05, le=0.50, description="Target delta for puts")
    target_dte: int = Field(45, ge=7, le=90, description="Target days to expiry")
    max_concurrent_puts: int = Field(3, ge=1, le=10, description="Maximum concurrent put positions")


class RiskConfig(BaseModel):
    """Essential risk management parameters."""
    max_position_size: float = Field(0.20, ge=0.01, le=0.50, description="Max % of portfolio per position")
    max_var_95: float = Field(0.05, ge=0.01, le=0.20, description="Maximum 95% VaR")
    max_cvar_95: float = Field(0.075, ge=0.01, le=0.30, description="Maximum 95% CVaR")
    max_volatility: float = Field(1.50, ge=0.50, le=3.0, description="Stop trading above this volatility")
    max_drawdown: float = Field(-0.20, le=-0.05, ge=-0.50, description="Circuit breaker drawdown")
    
    @field_validator('max_cvar_95')
    @classmethod
    def validate_cvar_greater_than_var(cls, v, info):
        """Ensure CVaR >= VaR."""
        if 'max_var_95' in info.data and v < info.data['max_var_95']:
            raise ValueError('max_cvar_95 must be >= max_var_95')
        return v


class StrategyConfig(BaseModel):
    """Core strategy parameters."""
    delta_target: float = Field(0.30, ge=0.05, le=0.50, description="Primary delta target")
    
    class ExpirationConfig(BaseModel):
        days_to_expiry_target: int = Field(45, ge=7, le=90, description="Target DTE")
    
    expiration: ExpirationConfig = Field(default_factory=ExpirationConfig)


class GreeksConfig(BaseModel):
    """Options Greeks parameters."""
    delta_target: float = Field(0.30, ge=0.05, le=0.50, description="Delta target")
    max_gamma_exposure: float = Field(100.0, ge=10.0, le=1000.0, description="Max gamma exposure")
    max_vega_exposure: float = Field(1000.0, ge=100.0, le=10000.0, description="Max vega exposure")


class DataConfig(BaseModel):
    """Data provider settings."""
    databento_enabled: bool = Field(True, description="Enable Databento data")
    fred_enabled: bool = Field(True, description="Enable FRED economic data")
    cache_ttl_minutes: int = Field(15, ge=1, le=60, description="Data cache TTL")


class SystemConfig(BaseModel):
    """System and performance settings."""
    log_level: str = Field("INFO", description="Logging level")
    cache_dir: str = Field("~/.wheel_trading/cache", description="Cache directory")
    max_memory_mb: int = Field(500, ge=100, le=2000, description="Memory limit")


class MinimalWheelConfig(BaseModel):
    """Minimal configuration for Unity Wheel Trading Bot."""
    
    # Essential sections only
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    unity: UnityConfig = Field(default_factory=UnityConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    greeks: GreeksConfig = Field(default_factory=GreeksConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    class Config:
        """Pydantic config."""
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinimalWheelConfig":
        """Create from dictionary."""
        return cls(**data)


def migrate_legacy_config(legacy_config: Dict[str, Any]) -> MinimalWheelConfig:
    """Migrate legacy configuration to minimal version.
    
    Args:
        legacy_config: Old configuration dictionary
        
    Returns:
        MinimalWheelConfig with essential parameters only
    """
    # Extract essential values from legacy config
    minimal_data = {}
    
    # Metadata
    minimal_data["metadata"] = {
        "version": legacy_config.get("version", "2.2.0"),
        "environment": legacy_config.get("environment", "production"),
    }
    
    # Unity settings
    unity_section = legacy_config.get("unity", {})
    minimal_data["unity"] = {
        "ticker": unity_section.get("ticker", "U"),
        "target_delta": unity_section.get("target_delta", 0.30),
        "target_dte": unity_section.get("target_dte", 45),
        "max_concurrent_puts": unity_section.get("max_concurrent_puts", 3),
    }
    
    # Risk settings
    risk_section = legacy_config.get("risk", {})
    minimal_data["risk"] = {
        "max_position_size": risk_section.get("max_position_size", 0.20),
        "max_var_95": risk_section.get("max_var_95", 0.05),
        "max_cvar_95": risk_section.get("max_cvar_95", 0.075),
        "max_volatility": risk_section.get("max_volatility", 1.50),
        "max_drawdown": risk_section.get("max_drawdown", -0.20),
    }
    
    # Strategy settings
    strategy_section = legacy_config.get("strategy", {})
    minimal_data["strategy"] = {
        "delta_target": strategy_section.get("delta_target", 0.30),
        "expiration": {
            "days_to_expiry_target": strategy_section.get("expiration", {}).get("days_to_expiry_target", 45)
        }
    }
    
    # Greeks settings
    greeks_section = legacy_config.get("greeks", strategy_section.get("greeks", {}))
    minimal_data["greeks"] = {
        "delta_target": greeks_section.get("delta_target", 0.30),
        "max_gamma_exposure": greeks_section.get("max_gamma_exposure", 100.0),
        "max_vega_exposure": greeks_section.get("max_vega_exposure", 1000.0),
    }
    
    # Data settings
    data_section = legacy_config.get("data", {})
    minimal_data["data"] = {
        "databento_enabled": data_section.get("databento_enabled", True),
        "fred_enabled": data_section.get("fred_enabled", True),
        "cache_ttl_minutes": data_section.get("cache_ttl_minutes", 15),
    }
    
    # System settings
    system_section = legacy_config.get("system", {})
    minimal_data["system"] = {
        "log_level": system_section.get("log_level", "INFO"),
        "cache_dir": system_section.get("cache_dir", "~/.wheel_trading/cache"),
        "max_memory_mb": system_section.get("max_memory_mb", 500),
    }
    
    return MinimalWheelConfig.from_dict(minimal_data)


def create_default_config() -> MinimalWheelConfig:
    """Create a default minimal configuration."""
    return MinimalWheelConfig()


def save_minimal_config(config: MinimalWheelConfig, file_path: Path) -> None:
    """Save minimal config to YAML file."""
    import yaml
    
    with open(file_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)


def load_minimal_config(file_path: Path) -> MinimalWheelConfig:
    """Load minimal config from YAML file."""
    import yaml
    
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return MinimalWheelConfig.from_dict(data)


# Create a compatibility layer for existing code
WheelConfig = MinimalWheelConfig  # Alias for backward compatibility