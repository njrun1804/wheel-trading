"""Environment variable loader for unified configuration system.

Supports environment variables from:
- Einstein (EINSTEIN_*)
- Bolt (BOLT_*)
- Jarvis (JARVIS_*) 
- Meta (META_*)
- Wheel Trading (WHEEL_*)
"""
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class EnvironmentLoader:
    """Loads configuration values from environment variables."""

    def __init__(self):
        """Initialize environment loader."""
        self._cache = {}

    def load_environment_config(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        if self._cache:
            return self._cache

        config = {}

        # Einstein environment variables
        config.update(self._load_einstein_env())
        
        # Bolt environment variables
        config.update(self._load_bolt_env())
        
        # Jarvis environment variables
        config.update(self._load_jarvis_env())
        
        # Meta environment variables
        config.update(self._load_meta_env())
        
        # Legacy Wheel Trading variables
        config.update(self._load_wheel_env())

        self._cache = config
        return config

    def _load_einstein_env(self) -> dict[str, Any]:
        """Load Einstein environment variables."""
        config = {}
        
        # Performance settings
        if os.getenv("EINSTEIN_MAX_STARTUP_MS"):
            config["performance"] = config.get("performance", {})
            config["performance"]["max_startup_time_ms"] = self._get_float_env("EINSTEIN_MAX_STARTUP_MS")
            
        if os.getenv("EINSTEIN_MAX_SEARCH_MS"):
            config["performance"] = config.get("performance", {})
            config["performance"]["max_search_time_ms"] = self._get_float_env("EINSTEIN_MAX_SEARCH_MS")
            
        if os.getenv("EINSTEIN_MAX_MEMORY_GB"):
            config["performance"] = config.get("performance", {})
            config["performance"]["max_memory_usage_gb"] = self._get_float_env("EINSTEIN_MAX_MEMORY_GB")
        
        # Concurrency settings
        if os.getenv("EINSTEIN_SEARCH_CONCURRENCY"):
            config["performance"] = config.get("performance", {})
            config["performance"]["max_search_concurrency"] = self._get_int_env("EINSTEIN_SEARCH_CONCURRENCY")
            
        if os.getenv("EINSTEIN_FILE_IO_CONCURRENCY"):
            config["performance"] = config.get("performance", {})
            config["performance"]["max_file_io_concurrency"] = self._get_int_env("EINSTEIN_FILE_IO_CONCURRENCY")
        
        # Cache settings
        if os.getenv("EINSTEIN_HOT_CACHE_SIZE"):
            config["cache"] = config.get("cache", {})
            config["cache"]["hot_cache_size"] = self._get_int_env("EINSTEIN_HOT_CACHE_SIZE")
            
        if os.getenv("EINSTEIN_INDEX_CACHE_MB"):
            config["cache"] = config.get("cache", {})
            config["cache"]["index_cache_size_mb"] = self._get_int_env("EINSTEIN_INDEX_CACHE_MB")
        
        # ML settings
        if os.getenv("EINSTEIN_ENABLE_ANE"):
            config["ml"] = config.get("ml", {})
            config["ml"]["enable_ane"] = self._get_bool_env("EINSTEIN_ENABLE_ANE")
            
        if os.getenv("EINSTEIN_ANE_BATCH_SIZE"):
            config["ml"] = config.get("ml", {})
            config["ml"]["ane_batch_size"] = self._get_int_env("EINSTEIN_ANE_BATCH_SIZE")
        
        # Monitoring settings
        if os.getenv("EINSTEIN_LOG_LEVEL"):
            config["monitoring"] = config.get("monitoring", {})
            config["monitoring"]["log_level"] = os.getenv("EINSTEIN_LOG_LEVEL")
            
        # Feature flags
        if os.getenv("EINSTEIN_USE_GPU"):
            config["enable_gpu_acceleration"] = self._get_bool_env("EINSTEIN_USE_GPU")
            
        if os.getenv("EINSTEIN_ADAPTIVE_CONCURRENCY"):
            config["enable_adaptive_concurrency"] = self._get_bool_env("EINSTEIN_ADAPTIVE_CONCURRENCY")
        
        return config
    
    def _load_bolt_env(self) -> dict[str, Any]:
        """Load Bolt environment variables."""
        config = {}
        
        # Agent settings
        if os.getenv("BOLT_MAX_AGENTS"):
            config["max_agents"] = self._get_int_env("BOLT_MAX_AGENTS")
            
        if os.getenv("BOLT_DEFAULT_AGENTS"):
            config["default_agents"] = self._get_int_env("BOLT_DEFAULT_AGENTS")
        
        # Hardware settings
        if os.getenv("BOLT_USE_GPU"):
            config["ml"] = config.get("ml", {})
            config["ml"]["enable_gpu"] = self._get_bool_env("BOLT_USE_GPU")
            
        if os.getenv("BOLT_PREFER_MLX"):
            config["ml"] = config.get("ml", {})
            config["ml"]["prefer_mlx"] = self._get_bool_env("BOLT_PREFER_MLX")
        
        # Performance settings
        if os.getenv("BOLT_MAX_MEMORY_GB"):
            config["performance"] = config.get("performance", {})
            config["performance"]["max_memory_usage_gb"] = self._get_float_env("BOLT_MAX_MEMORY_GB")
            
        if os.getenv("BOLT_ASYNC_TIMEOUT"):
            config["performance"] = config.get("performance", {})
            config["performance"]["async_timeout"] = self._get_float_env("BOLT_ASYNC_TIMEOUT")
            
        if os.getenv("BOLT_BATCH_SIZE"):
            config["ml"] = config.get("ml", {})
            config["ml"]["batch_size"] = self._get_int_env("BOLT_BATCH_SIZE")
        
        # Monitoring settings
        if os.getenv("BOLT_LOG_LEVEL"):
            config["monitoring"] = config.get("monitoring", {})
            config["monitoring"]["log_level"] = os.getenv("BOLT_LOG_LEVEL")
            
        if os.getenv("BOLT_ENABLE_MONITORING"):
            config["monitoring"] = config.get("monitoring", {})
            config["monitoring"]["enable_monitoring"] = self._get_bool_env("BOLT_ENABLE_MONITORING")
        
        return config
    
    def _load_jarvis_env(self) -> dict[str, Any]:
        """Load Jarvis environment variables."""
        config = {}
        
        # Neural network settings
        if os.getenv("JARVIS_EMBEDDING_DIM"):
            config["ml"] = config.get("ml", {})
            config["ml"]["embedding_dimension"] = self._get_int_env("JARVIS_EMBEDDING_DIM")
            
        if os.getenv("JARVIS_LEARNING_RATE"):
            config["ml"] = config.get("ml", {})
            config["ml"]["learning_rate"] = self._get_float_env("JARVIS_LEARNING_RATE")
            
        if os.getenv("JARVIS_BATCH_SIZE"):
            config["ml"] = config.get("ml", {})
            config["ml"]["batch_size"] = self._get_int_env("JARVIS_BATCH_SIZE")
        
        # Search settings
        if os.getenv("JARVIS_SEARCH_TIMEOUT"):
            config["performance"] = config.get("performance", {})
            config["performance"]["search_timeout_seconds"] = self._get_float_env("JARVIS_SEARCH_TIMEOUT")
        
        return config
    
    def _load_meta_env(self) -> dict[str, Any]:
        """Load Meta system environment variables."""
        config = {}
        
        # Timing settings
        if os.getenv("META_COORDINATION_CYCLE"):
            config["timing"] = config.get("timing", {})
            config["timing"]["coordination_cycle_seconds"] = self._get_int_env("META_COORDINATION_CYCLE")
            
        if os.getenv("META_HEALTH_CHECK_INTERVAL"):
            config["timing"] = config.get("timing", {})
            config["timing"]["health_check_interval_seconds"] = self._get_int_env("META_HEALTH_CHECK_INTERVAL")
        
        # Debug settings
        if os.getenv("META_DEBUG_MODE"):
            config["monitoring"] = config.get("monitoring", {})
            config["monitoring"]["log_level"] = "DEBUG" if self._get_bool_env("META_DEBUG_MODE") else "INFO"
        
        return config
    
    def _load_wheel_env(self) -> dict[str, Any]:
        """Load legacy Wheel Trading environment variables."""
        config = {}

        # Database configuration
        if os.getenv("WHEEL_DATABASE_PATH"):
            config["database"] = {"path": os.getenv("WHEEL_DATABASE_PATH")}

        # API configuration
        if os.getenv("DATABENTO_API_KEY"):
            config["apis"] = config.get("apis", {})
            config["apis"]["databento"] = {"api_key": os.getenv("DATABENTO_API_KEY")}

        if os.getenv("FRED_API_KEY"):
            config["apis"] = config.get("apis", {})
            config["apis"]["fred"] = {"api_key": os.getenv("FRED_API_KEY")}

        # Trading configuration
        if os.getenv("WHEEL_TARGET_DELTA"):
            config["trading"] = config.get("trading", {})
            config["trading"]["target_delta"] = self._get_float_env("WHEEL_TARGET_DELTA")

        if os.getenv("WHEEL_MAX_POSITION_SIZE"):
            config["trading"] = config.get("trading", {})
            config["trading"]["max_position_size"] = self._get_int_env("WHEEL_MAX_POSITION_SIZE")

        # Risk configuration
        if os.getenv("WHEEL_MAX_VAR_95"):
            config["risk"] = config.get("risk", {})
            config["risk"]["max_var_95"] = self._get_float_env("WHEEL_MAX_VAR_95")

        if os.getenv("WHEEL_MAX_CVAR_95"):
            config["risk"] = config.get("risk", {})
            config["risk"]["max_cvar_95"] = self._get_float_env("WHEEL_MAX_CVAR_95")

        return config

    def _get_int_env(self, key: str, default: int = 0) -> int:
        """Get integer environment variable with error handling."""
        try:
            value = os.getenv(key)
            return int(value) if value is not None else default
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid integer environment variable {key}={os.getenv(key)}, using default {default}: {e}")
            return default
    
    def _get_float_env(self, key: str, default: float = 0.0) -> float:
        """Get float environment variable with error handling."""
        try:
            value = os.getenv(key)
            return float(value) if value is not None else default
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid float environment variable {key}={os.getenv(key)}, using default {default}: {e}")
            return default
    
    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable with error handling."""
        value = os.getenv(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        else:
            return default
    
    def get_env_value(self, key: str, default: Any | None = None) -> Any | None:
        """Get a single environment variable value."""
        return os.getenv(key, default)

    def apply_overrides(self, config: Any) -> None:
        """Apply environment variable overrides to a configuration object."""
        env_config = self.load_environment_config()
        
        # Apply performance overrides
        if "performance" in env_config and hasattr(config, "performance"):
            perf_env = env_config["performance"]
            for key, value in perf_env.items():
                if hasattr(config.performance, key):
                    setattr(config.performance, key, value)
        
        # Apply cache overrides
        if "cache" in env_config and hasattr(config, "cache"):
            cache_env = env_config["cache"]
            for key, value in cache_env.items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)
        
        # Apply ML overrides
        if "ml" in env_config and hasattr(config, "ml"):
            ml_env = env_config["ml"]
            for key, value in ml_env.items():
                if hasattr(config.ml, key):
                    setattr(config.ml, key, value)
        
        # Apply monitoring overrides
        if "monitoring" in env_config and hasattr(config, "monitoring"):
            monitoring_env = env_config["monitoring"]
            for key, value in monitoring_env.items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)
        
        # Apply timing overrides
        if "timing" in env_config and hasattr(config, "timing"):
            timing_env = env_config["timing"]
            for key, value in timing_env.items():
                if hasattr(config.timing, key):
                    setattr(config.timing, key, value)
        
        # Apply feature flag overrides
        feature_flags = [
            "enable_gpu_acceleration",
            "enable_adaptive_concurrency", 
            "enable_predictive_prefetch",
            "enable_memory_optimization",
            "enable_realtime_indexing"
        ]
        
        for flag in feature_flags:
            if flag in env_config and hasattr(config, flag):
                setattr(config, flag, env_config[flag])
        
        # Apply legacy overrides for backward compatibility
        self._apply_legacy_overrides(config, env_config)
    
    def _apply_legacy_overrides(self, config: Any, env_config: dict[str, Any]) -> None:
        """Apply legacy environment variable overrides."""
        # Apply database overrides
        if "database" in env_config and hasattr(config, "paths"):
            if hasattr(config.paths, "database"):
                config.paths.database = env_config["database"]["path"]

        # Apply trading overrides
        if "trading" in env_config and hasattr(config, "trading"):
            trading_env = env_config["trading"]
            if "target_delta" in trading_env:
                config.trading.target_delta = trading_env["target_delta"]
            if "max_position_size" in trading_env:
                config.trading.max_position_size = trading_env["max_position_size"]

        # Apply risk overrides
        if "risk" in env_config and hasattr(config, "risk"):
            risk_env = env_config["risk"]
            if "max_var_95" in risk_env:
                config.risk.max_var_95 = risk_env["max_var_95"]
            if "max_cvar_95" in risk_env:
                config.risk.max_cvar_95 = risk_env["max_cvar_95"]
    
    def clear_cache(self) -> None:
        """Clear environment variable cache."""
        self._cache = {}