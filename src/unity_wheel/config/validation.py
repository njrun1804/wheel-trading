"""Configuration validation utilities for unified config system."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.info.append(message)
    
    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return bool(self.errors or self.warnings)


class ConfigValidator:
    """Validates unified configuration values."""

    def __init__(self):
        """Initialize validator."""
        self.validators = {
            "hardware": self._validate_hardware,
            "performance": self._validate_performance,
            "cache": self._validate_cache,
            "paths": self._validate_paths,
            "ml": self._validate_ml,
            "monitoring": self._validate_monitoring,
            "timing": self._validate_timing,
        }

    def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate a configuration dictionary."""
        result = ValidationResult(is_valid=True)
        
        # Validate each section
        for section, validator in self.validators.items():
            if section in config:
                validator(config[section], result)
        
        # Validate legacy sections
        self._validate_legacy_config(config, result)
        
        # Validate cross-section consistency
        self._validate_consistency(config, result)
        
        return result
    
    def _validate_hardware(self, hardware_config: dict[str, Any], result: ValidationResult) -> None:
        """Validate hardware configuration."""
        required_fields = ["cpu_cores", "memory_total_gb", "platform_type"]
        
        for field in required_fields:
            if field not in hardware_config:
                result.add_error(f"Hardware configuration missing required field: {field}")
        
        # Validate CPU cores
        if "cpu_cores" in hardware_config:
            cores = hardware_config["cpu_cores"]
            if not isinstance(cores, int) or cores <= 0:
                result.add_error("CPU cores must be a positive integer")
            elif cores > 64:
                result.add_warning(f"CPU cores ({cores}) seems unusually high")
        
        # Validate memory
        if "memory_total_gb" in hardware_config:
            memory = hardware_config["memory_total_gb"]
            if not isinstance(memory, (int, float)) or memory <= 0:
                result.add_error("Total memory must be a positive number")
            elif memory < 4:
                result.add_warning(f"Total memory ({memory}GB) is quite low")
        
        # Validate platform type
        if "platform_type" in hardware_config:
            platform = hardware_config["platform_type"]
            valid_platforms = ["apple_silicon", "intel", "amd", "intel_mac", "unknown"]
            if platform not in valid_platforms:
                result.add_warning(f"Unknown platform type: {platform}")
    
    def _validate_performance(self, perf_config: dict[str, Any], result: ValidationResult) -> None:
        """Validate performance configuration."""
        # Validate timeouts
        timeout_fields = [
            "max_startup_time_ms", "max_search_time_ms", "async_timeout",
            "search_timeout_seconds", "worker_timeout_seconds"
        ]
        
        for field in timeout_fields:
            if field in perf_config:
                value = perf_config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    result.add_error(f"{field} must be a positive number")
        
        # Validate memory limits
        if "max_memory_usage_gb" in perf_config:
            memory = perf_config["max_memory_usage_gb"]
            if not isinstance(memory, (int, float)) or memory <= 0:
                result.add_error("Max memory usage must be a positive number")
            elif memory > 128:
                result.add_warning(f"Max memory usage ({memory}GB) seems very high")
        
        # Validate concurrency limits
        concurrency_fields = [
            "max_search_concurrency", "max_embedding_concurrency",
            "max_file_io_concurrency", "max_analysis_concurrency"
        ]
        
        for field in concurrency_fields:
            if field in perf_config:
                value = perf_config[field]
                if not isinstance(value, int) or value <= 0:
                    result.add_error(f"{field} must be a positive integer")
                elif value > 64:
                    result.add_warning(f"{field} ({value}) seems very high")
    
    def _validate_cache(self, cache_config: dict[str, Any], result: ValidationResult) -> None:
        """Validate cache configuration."""
        cache_size_fields = [
            "hot_cache_size", "warm_cache_size", "search_cache_size",
            "file_cache_size", "max_cache_entries"
        ]
        
        for field in cache_size_fields:
            if field in cache_config:
                value = cache_config[field]
                if not isinstance(value, int) or value < 0:
                    result.add_error(f"{field} must be a non-negative integer")
        
        # Validate memory cache sizes
        memory_fields = ["index_cache_size_mb", "cache_memory_limit_mb"]
        for field in memory_fields:
            if field in cache_config:
                value = cache_config[field]
                if not isinstance(value, (int, float)) or value < 0:
                    result.add_error(f"{field} must be a non-negative number")
        
        # Validate TTL
        if "cache_ttl_seconds" in cache_config:
            ttl = cache_config["cache_ttl_seconds"]
            if not isinstance(ttl, int) or ttl <= 0:
                result.add_error("Cache TTL must be a positive integer")
    
    def _validate_paths(self, paths_config: dict[str, Any], result: ValidationResult) -> None:
        """Validate paths configuration."""
        required_paths = ["base_dir", "cache_dir", "logs_dir"]
        
        for path_name in required_paths:
            if path_name in paths_config:
                path_value = paths_config[path_name]
                if isinstance(path_value, str):
                    path = Path(path_value)
                elif hasattr(path_value, "__fspath__"):
                    path = Path(path_value)
                else:
                    result.add_error(f"{path_name} must be a valid path")
                    continue
                
                # Check if parent directory exists or can be created
                try:
                    if not path.parent.exists():
                        result.add_warning(f"Parent directory for {path_name} does not exist: {path.parent}")
                except Exception as e:
                    result.add_error(f"Cannot access {path_name}: {e}")
    
    def _validate_ml(self, ml_config: dict[str, Any], result: ValidationResult) -> None:
        """Validate ML configuration."""
        # Validate learning rates
        lr_fields = ["learning_rate", "adaptive_learning_rate", "bandit_exploration_rate"]
        for field in lr_fields:
            if field in ml_config:
                value = ml_config[field]
                if not isinstance(value, (int, float)) or not (0 < value <= 1):
                    result.add_error(f"{field} must be a number between 0 and 1")
        
        # Validate dimensions
        dim_fields = ["embedding_dimension", "hidden_dim", "max_sequence_length"]
        for field in dim_fields:
            if field in ml_config:
                value = ml_config[field]
                if not isinstance(value, int) or value <= 0:
                    result.add_error(f"{field} must be a positive integer")
        
        # Validate thresholds
        threshold_fields = ["similarity_threshold", "confidence_threshold", "relevance_threshold"]
        for field in threshold_fields:
            if field in ml_config:
                value = ml_config[field]
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    result.add_error(f"{field} must be a number between 0 and 1")
        
        # Validate batch sizes
        batch_fields = ["batch_size", "ane_batch_size"]
        for field in batch_fields:
            if field in ml_config:
                value = ml_config[field]
                if not isinstance(value, int) or value <= 0:
                    result.add_error(f"{field} must be a positive integer")
                elif value > 1024:
                    result.add_warning(f"{field} ({value}) seems very large")
    
    def _validate_monitoring(self, monitoring_config: dict[str, Any], result: ValidationResult) -> None:
        """Validate monitoring configuration."""
        # Validate log level
        if "log_level" in monitoring_config:
            level = monitoring_config["log_level"]
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level not in valid_levels:
                result.add_error(f"Log level must be one of {valid_levels}")
        
        # Validate intervals
        interval_fields = [
            "monitoring_interval", "memory_check_interval_s", "resource_check_interval",
            "file_watch_interval_s", "cooldown_period_s"
        ]
        
        for field in interval_fields:
            if field in monitoring_config:
                value = monitoring_config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    result.add_error(f"{field} must be a positive number")
        
        # Validate thresholds
        threshold_fields = ["high_usage_threshold", "memory_pressure_threshold"]
        for field in threshold_fields:
            if field in monitoring_config:
                value = monitoring_config[field]
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    result.add_error(f"{field} must be a number between 0 and 1")
    
    def _validate_timing(self, timing_config: dict[str, Any], result: ValidationResult) -> None:
        """Validate timing configuration."""
        time_fields = [
            "recent_activity_window_seconds", "rapid_development_threshold_seconds",
            "coordination_cycle_seconds", "health_check_interval_seconds"
        ]
        
        for field in time_fields:
            if field in timing_config:
                value = timing_config[field]
                if not isinstance(value, int) or value <= 0:
                    result.add_error(f"{field} must be a positive integer")
        
        # Validate counts
        count_fields = ["minimum_observations_for_evolution", "minimum_recent_activity_count"]
        for field in count_fields:
            if field in timing_config:
                value = timing_config[field]
                if not isinstance(value, int) or value <= 0:
                    result.add_error(f"{field} must be a positive integer")
        
        # Validate debounce delays
        debounce_fields = ["file_change_debounce_seconds", "debounce_delay_ms"]
        for field in debounce_fields:
            if field in timing_config:
                value = timing_config[field]
                if not isinstance(value, (int, float)) or value < 0:
                    result.add_error(f"{field} must be a non-negative number")
    
    def _validate_legacy_config(self, config: dict[str, Any], result: ValidationResult) -> None:
        """Validate legacy configuration sections."""
        # Validate database configuration
        if "database" in config:
            db_config = config["database"]
            if "path" in db_config:
                path = db_config["path"]
                if not isinstance(path, str) or not path:
                    result.add_error("Database path must be a non-empty string")

        # Validate trading configuration
        if "trading" in config:
            trading_config = config["trading"]

            if "target_delta" in trading_config:
                delta = trading_config["target_delta"]
                if not isinstance(delta, (int, float)) or not (0 <= delta <= 1):
                    result.add_error("Target delta must be a number between 0 and 1")

            if "max_position_size" in trading_config:
                size = trading_config["max_position_size"]
                if not isinstance(size, int) or size <= 0:
                    result.add_error("Max position size must be a positive integer")

        # Validate risk configuration
        if "risk" in config:
            risk_config = config["risk"]

            for key in ["max_var_95", "max_cvar_95"]:
                if key in risk_config:
                    value = risk_config[key]
                    if not isinstance(value, (int, float)) or value <= 0:
                        result.add_error(f"{key} must be a positive number")
    
    def _validate_consistency(self, config: dict[str, Any], result: ValidationResult) -> None:
        """Validate cross-section consistency."""
        # Check memory consistency
        if "hardware" in config and "performance" in config:
            hardware = config["hardware"]
            performance = config["performance"]
            
            if "memory_total_gb" in hardware and "max_memory_usage_gb" in performance:
                total_memory = hardware["memory_total_gb"]
                max_usage = performance["max_memory_usage_gb"]
                
                if max_usage > total_memory:
                    result.add_error(
                        f"Max memory usage ({max_usage}GB) exceeds total memory ({total_memory}GB)"
                    )
                elif max_usage > total_memory * 0.9:
                    result.add_warning(
                        f"Max memory usage ({max_usage}GB) is very close to total memory ({total_memory}GB)"
                    )
        
        # Check concurrency vs CPU cores
        if "hardware" in config and "performance" in config:
            hardware = config["hardware"]
            performance = config["performance"]
            
            if "cpu_cores" in hardware:
                cpu_cores = hardware["cpu_cores"]
                
                concurrency_fields = [
                    "max_search_concurrency", "max_file_io_concurrency", "max_analysis_concurrency"
                ]
                
                for field in concurrency_fields:
                    if field in performance:
                        concurrency = performance[field]
                        if concurrency > cpu_cores * 2:
                            result.add_warning(
                                f"{field} ({concurrency}) is much higher than CPU cores ({cpu_cores})"
                            )
        
        # Check GPU settings consistency
        if "hardware" in config and "ml" in config:
            hardware = config["hardware"]
            ml = config["ml"]
            
            has_gpu = hardware.get("has_gpu", False)
            enable_gpu = ml.get("enable_gpu", False)
            
            if enable_gpu and not has_gpu:
                result.add_warning("GPU acceleration enabled but no GPU detected")
        
        # Check ANE settings consistency
        if "hardware" in config and "ml" in config:
            hardware = config["hardware"]
            ml = config["ml"]
            
            has_ane = hardware.get("has_ane", False)
            enable_ane = ml.get("enable_ane", False)
            
            if enable_ane and not has_ane:
                result.add_warning("ANE acceleration enabled but ANE not detected")

    def validate_api_keys(self, config: dict[str, Any]) -> ValidationResult:
        """Validate API key configuration."""
        result = ValidationResult(is_valid=True)

        if "apis" in config:
            apis = config["apis"]

            # Check for required API keys
            required_apis = ["databento", "fred"]
            for api in required_apis:
                if api not in apis:
                    result.add_warning(f"Missing {api} API configuration")
                elif "api_key" not in apis[api] or not apis[api]["api_key"]:
                    result.add_warning(f"Missing or empty API key for {api}")

        return result

    def validate(self, config: Any) -> ValidationResult:
        """Validate a configuration object."""
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            # Try to convert to dict
            config_dict = {}
            for attr in dir(config):
                if not attr.startswith("_") and not callable(getattr(config, attr, None)):
                    try:
                        value = getattr(config, attr)
                        # Convert dataclass fields to dict
                        if hasattr(value, '__dict__') and not callable(value):
                            config_dict[attr] = value.__dict__
                        else:
                            config_dict[attr] = value
                    except Exception as e:
                        logger.debug(f"Failed to get attribute {attr}: {e}")

        return self.validate_config(config_dict)
    
    def validate_system_compatibility(self, config: dict[str, Any], system: str) -> ValidationResult:
        """Validate configuration compatibility for a specific system."""
        result = ValidationResult(is_valid=True)
        
        if system == "einstein":
            self._validate_einstein_compatibility(config, result)
        elif system == "bolt":
            self._validate_bolt_compatibility(config, result)
        elif system == "jarvis":
            self._validate_jarvis_compatibility(config, result)
        elif system == "meta":
            self._validate_meta_compatibility(config, result)
        else:
            result.add_error(f"Unknown system: {system}")
        
        return result
    
    def _validate_einstein_compatibility(self, config: dict[str, Any], result: ValidationResult) -> None:
        """Validate Einstein-specific requirements."""
        required_sections = ["hardware", "performance", "cache", "paths"]
        for section in required_sections:
            if section not in config:
                result.add_error(f"Einstein requires {section} configuration")
    
    def _validate_bolt_compatibility(self, config: dict[str, Any], result: ValidationResult) -> None:
        """Validate Bolt-specific requirements."""
        # Bolt needs at least basic hardware info
        if "hardware" not in config:
            result.add_error("Bolt requires hardware configuration")
        
        # Check for agent configuration
        if "max_agents" not in config and "hardware" in config:
            if "cpu_cores" not in config["hardware"]:
                result.add_warning("Cannot determine optimal agent count without CPU core info")
    
    def _validate_jarvis_compatibility(self, config: dict[str, Any], result: ValidationResult) -> None:
        """Validate Jarvis-specific requirements."""
        if "ml" not in config:
            result.add_error("Jarvis requires ML configuration")
        
        # Check for neural network settings
        if "ml" in config:
            ml_config = config["ml"]
            required_ml_fields = ["embedding_dimension", "learning_rate"]
            for field in required_ml_fields:
                if field not in ml_config:
                    result.add_warning(f"Jarvis ML config missing {field}")
    
    def _validate_meta_compatibility(self, config: dict[str, Any], result: ValidationResult) -> None:
        """Validate Meta system requirements."""
        if "timing" not in config:
            result.add_warning("Meta system works better with timing configuration")
        
        if "monitoring" not in config:
            result.add_warning("Meta system needs monitoring configuration")


# Convenience functions for common validation tasks
def validate_unified_config(config: Any) -> ValidationResult:
    """Validate a unified configuration object."""
    validator = ConfigValidator()
    return validator.validate(config)


def validate_system_config(config: dict[str, Any], system: str) -> ValidationResult:
    """Validate configuration for a specific system."""
    validator = ConfigValidator()
    return validator.validate_system_compatibility(config, system)


def validate_api_keys(config: dict[str, Any]) -> ValidationResult:
    """Validate API key configuration."""
    validator = ConfigValidator()
    return validator.validate_api_keys(config)