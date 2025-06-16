"""
Configuration management for Bolt system.

Provides centralized configuration handling with validation and defaults.
Externalizes all hardcoded values found in Bolt components.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, validator


class HardwareConfig(BaseModel):
    """Hardware-specific configuration for Bolt system."""

    # Core allocation (M4 Pro defaults)
    performance_cores: int = Field(
        default=8, ge=1, description="Number of performance cores"
    )
    efficiency_cores: int = Field(
        default=4, ge=0, description="Number of efficiency cores"
    )
    metal_cores: int = Field(default=20, ge=0, description="Number of Metal GPU cores")
    ane_cores: int = Field(
        default=16, ge=0, description="Number of Apple Neural Engine cores"
    )

    # Memory configuration
    total_memory_gb: float = Field(
        default=24.0, gt=0, description="Total system memory in GB"
    )
    max_allocation_gb: float = Field(
        default=18.0, gt=0, description="Maximum memory allocation in GB"
    )
    memory_pressure_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Memory pressure threshold"
    )
    critical_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Critical memory threshold"
    )

    # Buffer alignment
    buffer_alignment_bytes: int = Field(
        default=16, ge=1, description="Buffer alignment in bytes"
    )

    # GPU memory settings
    gpu_memory_threshold_ratio: float = Field(
        default=0.7, ge=0.0, le=1.0, description="GPU memory threshold ratio"
    )


class PerformanceConfig(BaseModel):
    """Performance settings and thresholds."""

    # Timeout configurations
    short_timeout_s: float = Field(
        default=0.1, gt=0, description="Short operation timeout"
    )
    medium_timeout_s: float = Field(
        default=5.0, gt=0, description="Medium operation timeout"
    )
    long_timeout_s: float = Field(
        default=10.0, gt=0, description="Long operation timeout"
    )
    async_timeout_s: float = Field(
        default=300.0, gt=0, description="Async operation timeout"
    )

    # Batch processing
    default_batch_size: int = Field(default=32, ge=1, description="Default batch size")
    gpu_batch_size: int = Field(default=256, ge=1, description="GPU batch size")
    embedding_batch_size: int = Field(
        default=32, ge=1, description="Embedding batch size"
    )

    # Cache sizes
    default_cache_size_mb: int = Field(
        default=256, ge=1, description="Default cache size in MB"
    )
    hot_cache_size: int = Field(default=1000, ge=0, description="Hot cache entries")
    result_cache_size: int = Field(default=1000, ge=0, description="Result cache size")

    # Workload thresholds for CPU/GPU routing
    vector_ops_threshold: int = Field(
        default=10000, ge=1, description="Vector operations CPU/GPU threshold"
    )
    matrix_ops_threshold: int = Field(
        default=250000, ge=1, description="Matrix operations threshold (500x500)"
    )
    batch_ops_threshold: int = Field(
        default=200, ge=1, description="Batch operations threshold"
    )
    similarity_threshold: int = Field(
        default=2000, ge=1, description="Similarity computation threshold"
    )
    embedding_threshold: int = Field(
        default=500, ge=1, description="Embedding operations threshold"
    )
    attention_threshold: int = Field(
        default=128, ge=1, description="Attention operations threshold"
    )

    # GPU overhead compensation (microseconds)
    gpu_initialization_overhead_us: int = Field(
        default=2000, ge=0, description="GPU initialization overhead"
    )
    gpu_memory_transfer_overhead_us: int = Field(
        default=500, ge=0, description="GPU memory transfer overhead"
    )
    gpu_evaluation_overhead_us: int = Field(
        default=300, ge=0, description="GPU evaluation overhead"
    )
    gpu_cleanup_overhead_us: int = Field(
        default=100, ge=0, description="GPU cleanup overhead"
    )


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = Field(
        default=5, ge=1, description="Failures before opening circuit"
    )
    success_threshold: int = Field(
        default=3, ge=1, description="Successes to close circuit"
    )
    timeout_s: float = Field(
        default=60.0, gt=0, description="Timeout before trying half-open"
    )
    reset_timeout_s: float = Field(
        default=300.0, gt=0, description="How long to keep records"
    )
    max_requests_half_open: int = Field(
        default=5, ge=1, description="Max requests in half-open state"
    )


class MemoryConfig(BaseModel):
    """Memory management configuration."""

    # Component memory budgets (as percentage of max allocation)
    duckdb_budget_ratio: float = Field(
        default=0.50, ge=0.0, le=1.0, description="DuckDB memory budget ratio"
    )
    jarvis_budget_ratio: float = Field(
        default=0.17, ge=0.0, le=1.0, description="Jarvis memory budget ratio"
    )
    einstein_budget_ratio: float = Field(
        default=0.08, ge=0.0, le=1.0, description="Einstein memory budget ratio"
    )
    meta_system_budget_ratio: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Meta system memory budget ratio"
    )
    cache_budget_ratio: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Cache memory budget ratio"
    )
    other_budget_ratio: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Other components memory budget ratio"
    )

    # Memory pool settings
    pool_alignment_bytes: int = Field(
        default=16, ge=1, description="Memory pool alignment"
    )
    eviction_priority_threshold: int = Field(
        default=5, ge=1, le=10, description="Eviction priority threshold"
    )


class BoltConfig(BaseModel):
    """Main configuration for Bolt system."""

    # Agent configuration
    max_agents: int = Field(
        default=8, ge=1, le=16, description="Maximum number of agents"
    )
    default_agents: int = Field(
        default=8, ge=1, le=16, description="Default number of agents"
    )

    # Hardware configuration
    hardware: HardwareConfig = Field(
        default_factory=HardwareConfig, description="Hardware configuration"
    )

    # Performance configuration
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance configuration"
    )

    # Memory configuration
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Memory configuration"
    )

    # Circuit breaker configuration
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration",
    )

    # Legacy fields for backward compatibility
    use_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    prefer_mlx: bool = Field(
        default=True, description="Prefer MLX over other GPU backends"
    )
    max_memory_gb: float | None = Field(
        default=None, description="Maximum memory usage in GB"
    )
    cpu_affinity: list | None = Field(default=None, description="CPU cores to use")
    async_timeout: float = Field(
        default=300.0, gt=0, description="Async operation timeout in seconds"
    )
    batch_size: int = Field(
        default=32, ge=1, description="Default batch size for operations"
    )
    cache_size: int = Field(default=1000, ge=0, description="Cache size for results")

    # Monitoring configuration
    enable_monitoring: bool = Field(
        default=True, description="Enable system monitoring"
    )
    monitoring_interval: float = Field(
        default=1.0, gt=0, description="Monitoring interval in seconds"
    )

    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str | None = Field(default=None, description="Log file path")

    # Tool configuration
    accelerated_tools: dict[str, bool] = Field(
        default_factory=lambda: {
            "ripgrep_turbo": True,
            "dependency_graph": True,
            "python_analysis": True,
            "duckdb_turbo": True,
            "trace_turbo": True,
            "python_helpers": True,
        },
        description="Accelerated tools configuration",
    )

    @validator("default_agents")
    def validate_default_agents(cls, v, values):
        """Ensure default_agents <= max_agents."""
        if "max_agents" in values and v > values["max_agents"]:
            raise ValueError("default_agents cannot be greater than max_agents")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @classmethod
    def from_file(cls, config_path: Path) -> "BoltConfig":
        """Load configuration from file."""

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    @classmethod
    def from_env(cls) -> "BoltConfig":
        """Load configuration from environment variables."""
        env_config = {}

        # Hardware configuration from environment
        hardware_config: dict[str, int | float] = {}
        hardware_mapping = {
            "BOLT_PERFORMANCE_CORES": "performance_cores",
            "BOLT_EFFICIENCY_CORES": "efficiency_cores",
            "BOLT_METAL_CORES": "metal_cores",
            "BOLT_ANE_CORES": "ane_cores",
            "BOLT_TOTAL_MEMORY_GB": "total_memory_gb",
            "BOLT_MAX_ALLOCATION_GB": "max_allocation_gb",
            "BOLT_MEMORY_PRESSURE_THRESHOLD": "memory_pressure_threshold",
            "BOLT_CRITICAL_THRESHOLD": "critical_threshold",
            "BOLT_BUFFER_ALIGNMENT": "buffer_alignment_bytes",
            "BOLT_GPU_MEMORY_THRESHOLD": "gpu_memory_threshold_ratio",
        }

        for env_var, config_key in hardware_mapping.items():
            if env_var in os.environ:
                value_str = os.environ[env_var]
                if config_key in [
                    "performance_cores",
                    "efficiency_cores",
                    "metal_cores",
                    "ane_cores",
                    "buffer_alignment_bytes",
                ]:
                    value = int(value_str)
                elif config_key in [
                    "total_memory_gb",
                    "max_allocation_gb",
                    "memory_pressure_threshold",
                    "critical_threshold",
                    "gpu_memory_threshold_ratio",
                ]:
                    value = float(value_str)
                else:
                    value = value_str
                hardware_config[config_key] = value

        if hardware_config:
            env_config["hardware"] = HardwareConfig(**hardware_config)

        # Performance configuration from environment
        performance_config: dict[str, int | float] = {}
        performance_mapping = {
            "BOLT_SHORT_TIMEOUT": "short_timeout_s",
            "BOLT_MEDIUM_TIMEOUT": "medium_timeout_s",
            "BOLT_LONG_TIMEOUT": "long_timeout_s",
            "BOLT_ASYNC_TIMEOUT": "async_timeout_s",
            "BOLT_DEFAULT_BATCH_SIZE": "default_batch_size",
            "BOLT_GPU_BATCH_SIZE": "gpu_batch_size",
            "BOLT_EMBEDDING_BATCH_SIZE": "embedding_batch_size",
            "BOLT_DEFAULT_CACHE_MB": "default_cache_size_mb",
            "BOLT_HOT_CACHE_SIZE": "hot_cache_size",
            "BOLT_RESULT_CACHE_SIZE": "result_cache_size",
        }

        for env_var, config_key in performance_mapping.items():
            if env_var in os.environ:
                value_str = os.environ[env_var]
                if config_key in [
                    "default_batch_size",
                    "gpu_batch_size",
                    "embedding_batch_size",
                    "default_cache_size_mb",
                    "hot_cache_size",
                    "result_cache_size",
                ]:
                    value = int(value_str)
                elif config_key in [
                    "short_timeout_s",
                    "medium_timeout_s",
                    "long_timeout_s",
                    "async_timeout_s",
                ]:
                    value = float(value_str)
                else:
                    value = value_str
                performance_config[config_key] = value

        if performance_config:
            env_config["performance"] = PerformanceConfig(**performance_config)

        # Memory configuration from environment
        memory_config: dict[str, int | float] = {}
        memory_mapping = {
            "BOLT_DUCKDB_BUDGET": "duckdb_budget_ratio",
            "BOLT_JARVIS_BUDGET": "jarvis_budget_ratio",
            "BOLT_EINSTEIN_BUDGET": "einstein_budget_ratio",
            "BOLT_META_BUDGET": "meta_system_budget_ratio",
            "BOLT_CACHE_BUDGET": "cache_budget_ratio",
            "BOLT_OTHER_BUDGET": "other_budget_ratio",
            "BOLT_POOL_ALIGNMENT": "pool_alignment_bytes",
            "BOLT_EVICTION_PRIORITY": "eviction_priority_threshold",
        }

        for env_var, config_key in memory_mapping.items():
            if env_var in os.environ:
                value_str = os.environ[env_var]
                if config_key in [
                    "pool_alignment_bytes",
                    "eviction_priority_threshold",
                ]:
                    value = int(value_str)
                else:
                    value = float(value_str)
                memory_config[config_key] = value

        if memory_config:
            env_config["memory"] = MemoryConfig(**memory_config)

        # Circuit breaker configuration from environment
        cb_config = {}
        cb_mapping = {
            "BOLT_CB_FAILURE_THRESHOLD": "failure_threshold",
            "BOLT_CB_SUCCESS_THRESHOLD": "success_threshold",
            "BOLT_CB_TIMEOUT": "timeout_s",
            "BOLT_CB_RESET_TIMEOUT": "reset_timeout_s",
            "BOLT_CB_MAX_HALF_OPEN": "max_requests_half_open",
        }

        for env_var, config_key in cb_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                if config_key in [
                    "failure_threshold",
                    "success_threshold",
                    "max_requests_half_open",
                ]:
                    value = int(value)
                else:
                    value = float(value)
                cb_config[config_key] = value

        if cb_config:
            env_config["circuit_breaker"] = CircuitBreakerConfig(**cb_config)

        # Legacy configuration mapping
        legacy_mapping = {
            "BOLT_MAX_AGENTS": "max_agents",
            "BOLT_DEFAULT_AGENTS": "default_agents",
            "BOLT_USE_GPU": "use_gpu",
            "BOLT_PREFER_MLX": "prefer_mlx",
            "BOLT_MAX_MEMORY_GB": "max_memory_gb",
            "BOLT_ASYNC_TIMEOUT": "async_timeout",
            "BOLT_BATCH_SIZE": "batch_size",
            "BOLT_CACHE_SIZE": "cache_size",
            "BOLT_ENABLE_MONITORING": "enable_monitoring",
            "BOLT_MONITORING_INTERVAL": "monitoring_interval",
            "BOLT_LOG_LEVEL": "log_level",
            "BOLT_LOG_FILE": "log_file",
        }

        for env_var, config_key in legacy_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                # Type conversion
                if config_key in [
                    "max_agents",
                    "default_agents",
                    "batch_size",
                    "cache_size",
                ]:
                    value = int(value)
                elif config_key in ["use_gpu", "prefer_mlx", "enable_monitoring"]:
                    value = value.lower() in ["true", "1", "yes", "on"]
                elif config_key in [
                    "max_memory_gb",
                    "async_timeout",
                    "monitoring_interval",
                ]:
                    value = float(value)

                env_config[config_key] = value

        return cls(**env_config)

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to file."""

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=True)

    def get_hardware_config(self) -> dict[str, Any]:
        """Get hardware-specific configuration."""
        from .system_info import get_hardware_capabilities

        capabilities = get_hardware_capabilities()

        # Adjust configuration based on hardware
        config = {
            "max_agents": min(self.max_agents, capabilities["cpu_threads"]),
            "use_gpu": self.use_gpu and capabilities["has_gpu"],
            "prefer_mlx": self.prefer_mlx and capabilities["mlx_available"],
            "max_memory_gb": self.max_memory_gb
            or capabilities["memory_gb"] * 0.8,  # Use 80% by default
        }

        return config


def get_default_config() -> BoltConfig:
    """Get default configuration with environment overrides."""

    # Start with environment configuration
    try:
        config = BoltConfig.from_env()
    except Exception:
        config = BoltConfig()

    # Try to load from default config file
    default_config_path = Path.home() / ".bolt" / "config.yaml"
    if default_config_path.exists():
        try:
            file_config = BoltConfig.from_file(default_config_path)
            # Merge with environment config (env takes precedence)
            config = file_config.copy(update=config.dict(exclude_unset=True))
        except Exception:
            pass  # Use environment/default config

    return config


def setup_config_directory() -> Path:
    """Setup default configuration directory."""
    config_dir = Path.home() / ".bolt"
    config_dir.mkdir(exist_ok=True)

    # Create default config file if it doesn't exist
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        default_config = BoltConfig()
        default_config.save_to_file(config_file)

    return config_dir


# Backward compatibility alias
Config = BoltConfig
