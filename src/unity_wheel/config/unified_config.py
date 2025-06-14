"""Unified configuration system for Unity Wheel Trading Bot.

This module provides a single source of truth for all configuration,
eliminating hardcoded values throughout the codebase.
"""

from src.config.network_config import network_config
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, validator


class TradingConfig(BaseModel):
    """Trading strategy configuration."""
    symbol: str = Field(default="U", env="TRADING_SYMBOL")
    target_delta: float = Field(default=0.30, ge=0.1, le=0.5, env="TARGET_DELTA")
    target_dte: int = Field(default=30, ge=7, le=60, env="TARGET_DTE")
    max_position_size: float = Field(default=0.25, ge=0.05, le=0.5, env="MAX_POSITION_SIZE")
    max_concurrent_puts: int = Field(default=3, ge=1, le=10, env="MAX_CONCURRENT_PUTS")
    min_confidence: float = Field(default=0.70, ge=0.5, le=1.0, env="MIN_CONFIDENCE")
    contracts_per_trade: int = Field(default=100, ge=1, env="CONTRACTS_PER_TRADE")
    commission_per_contract: float = Field(default=0.65, ge=0, env="COMMISSION_PER_CONTRACT")
    
    class Config:
        env_prefix = "TRADING_"


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_var_95: float = Field(default=0.05, ge=0.01, le=0.2, env="MAX_VAR_95")
    max_cvar_95: float = Field(default=0.075, ge=0.01, le=0.3, env="MAX_CVAR_95")
    max_margin_utilization: float = Field(default=0.5, ge=0.1, le=0.8, env="MAX_MARGIN_UTILIZATION")
    max_portfolio_allocation: float = Field(default=0.25, ge=0.05, le=0.5, env="MAX_PORTFOLIO_ALLOCATION")
    stress_test_scenarios: int = Field(default=1000, ge=100, le=10000, env="STRESS_TEST_SCENARIOS")
    
    class Config:
        env_prefix = "RISK_"


class StorageConfig(BaseModel):
    """Data storage configuration."""
    database_path: str = Field(
        default="data/wheel_trading_optimized.duckdb",
        env="DATABASE_PATH"
    )
    cache_dir: str = Field(default="~/.wheel_trading/cache", env="CACHE_DIR")
    archive_dir: str = Field(default="~/.wheel_trading/archive", env="ARCHIVE_DIR")
    hot_data_retention_days: int = Field(default=30, ge=7, le=90, env="HOT_DATA_DAYS")
    cold_data_retention_years: int = Field(default=7, ge=1, le=10, env="COLD_DATA_YEARS")
    
    @validator("database_path", "cache_dir", "archive_dir")
    def expand_path(cls, v):
        return str(Path(v).expanduser())
    
    class Config:
        env_prefix = "STORAGE_"


class PerformanceConfig(BaseModel):
    """Performance optimization configuration."""
    use_arrow: bool = Field(default=True, env="USE_ARROW")
    use_polars: bool = Field(default=True, env="USE_POLARS")
    cache_ttl_minutes: int = Field(default=15, ge=1, le=60, env="CACHE_TTL")
    batch_size: int = Field(default=1000, ge=100, le=10000, env="BATCH_SIZE")
    max_workers: int = Field(default=4, ge=1, le=16, env="MAX_WORKERS")
    query_timeout_seconds: int = Field(default=30, ge=5, le=300, env="QUERY_TIMEOUT")
    
    class Config:
        env_prefix = "PERF_"


class MCPConfig(BaseModel):
    """MCP server configuration."""
    use_duckdb_mcp: bool = Field(default=True, env="USE_DUCKDB_MCP")
    use_mlflow_mcp: bool = Field(default=True, env="USE_MLFLOW_MCP")
    use_statsource_mcp: bool = Field(default=True, env="USE_STATSOURCE_MCP")
    use_memory_mcp: bool = Field(default=True, env="USE_MEMORY_MCP")
    mlflow_tracking_uri: str = Field(default=network_config.mlflow_tracking_uri, env="MLFLOW_URI")
    statsource_api_key: Optional[str] = Field(default=None, env="STATSOURCE_API_KEY")
    
    class Config:
        env_prefix = "MCP_"


class UnifiedConfig(BaseModel):
    """Unified configuration for the entire system."""
    
    # Sub-configurations
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    
    # Global settings
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    @classmethod
    def from_env(cls) -> UnifiedConfig:
        """Load configuration from environment variables."""
        # Load sub-configs from environment
        trading = TradingConfig()
        risk = RiskConfig()
        storage = StorageConfig()
        performance = PerformanceConfig()
        mcp = MCPConfig()
        
        # Create unified config
        return cls(
            trading=trading,
            risk=risk,
            storage=storage,
            performance=performance,
            mcp=mcp,
            environment=os.getenv("ENVIRONMENT", "production"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    @classmethod
    def from_file(cls, path: str) -> UnifiedConfig:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> UnifiedConfig:
        """Load configuration from file or environment."""
        if config_path and Path(config_path).exists():
            config = cls.from_file(config_path)
            # Override with environment variables
            config = cls(**{
                **config.dict(),
                **cls.from_env().dict(exclude_unset=True)
            })
            return config
        return cls.from_env()
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
    
    def validate_all(self) -> bool:
        """Validate all configuration values."""
        try:
            # Pydantic handles validation automatically
            _ = self.dict()
            
            # Additional custom validations
            if self.trading.target_dte < 7:
                raise ValueError("Target DTE must be at least 7 days")
                
            if self.risk.max_var_95 >= self.risk.max_cvar_95:
                raise ValueError("CVaR must be greater than VaR")
                
            if self.storage.hot_data_retention_days >= self.storage.cold_data_retention_years * 365:
                raise ValueError("Hot data retention must be less than cold data retention")
                
            return True
        except (ValueError, KeyError, AttributeError) as e:
            logger.info("Configuration validation failed: {e}")
            return False


# Global singleton instance
_config: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = UnifiedConfig.load(
            os.getenv("WHEEL_CONFIG_PATH", "config/unified.yaml")
        )
    return _config


def reload_config(config_path: Optional[str] = None) -> UnifiedConfig:
    """Reload configuration from file or environment."""
    global _config
    _config = UnifiedConfig.load(config_path)
    return _config


# Convenience exports
__all__ = [
    "UnifiedConfig",
    "TradingConfig", 
    "RiskConfig",
    "StorageConfig",
    "PerformanceConfig",
    "MCPConfig",
    "get_config",
    "reload_config"
]