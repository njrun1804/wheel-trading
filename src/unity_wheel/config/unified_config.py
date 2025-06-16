"""
Unified Configuration System - Single source of truth for all configuration.

This module consolidates configuration from:
- einstein/einstein_config.py
- bolt/core/config.py
- meta_config.py and meta_daemon_config.py
- jarvis2/config/jarvis_config.py

Provides:
- Single hardware detection mechanism
- Unified environment variable processing
- Consistent validation framework
- Backward compatibility with all systems
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .environment_loader import EnvironmentLoader
from .hardware_config import HardwareConfig, HardwareDetector
from .validation import ConfigValidator

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Unified performance configuration."""

    # Startup performance
    max_startup_time_ms: float = 500.0
    max_critical_path_ms: float = 200.0
    max_background_init_ms: float = 2000.0

    # Search performance
    max_search_time_ms: float = 50.0
    target_text_search_ms: float = 5.0
    target_semantic_search_ms: float = 20.0
    target_structural_search_ms: float = 10.0
    target_analytical_search_ms: float = 15.0
    target_embedding_ms: float = 50.0

    # Memory targets
    max_memory_usage_gb: float = 2.0
    cache_memory_limit_mb: float = 512.0

    # Concurrency limits
    max_search_concurrency: int = 4
    max_embedding_concurrency: int = 8
    max_file_io_concurrency: int = 12
    max_analysis_concurrency: int = 6

    # Timeouts (from various configs)
    async_timeout: float = 300.0
    search_timeout_seconds: float = 30.0
    worker_timeout_seconds: float = 60.0


@dataclass
class CacheConfig:
    """Unified cache configuration."""

    # LRU cache sizes
    hot_cache_size: int = 1000
    warm_cache_size: int = 5000
    search_cache_size: int = 500
    file_cache_size: int = 1000

    # Memory management
    index_cache_size_mb: int = 256
    compress_threshold_bytes: int = 1024

    # Cache policies
    cache_ttl_seconds: int = 3600
    max_cache_entries: int = 10000

    # Prefetch settings
    prefetch_common_patterns: bool = True
    prefetch_cache_size: int = 1000


@dataclass
class PathConfig:
    """Unified path configuration."""

    base_dir: Path
    cache_dir: Path
    logs_dir: Path

    # Einstein paths
    einstein_cache_dir: Path
    analytics_db_path: Path
    embeddings_db_path: Path

    # Jarvis paths
    jarvis_models_dir: Path
    experience_db_path: Path

    # Bolt paths
    bolt_config_dir: Path
    
    # Meta paths (disabled for trading independence)
    meta_evolution_db: Path | None = None
    meta_monitoring_db: Path | None = None

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "PathConfig":
        """Create path config from base directory."""
        return cls(
            base_dir=base_dir,
            cache_dir=base_dir / ".cache",
            logs_dir=base_dir / ".logs",
            # Einstein
            einstein_cache_dir=base_dir / ".einstein",
            analytics_db_path=base_dir / ".einstein" / "analytics.db",
            embeddings_db_path=base_dir / ".einstein" / "embeddings.db",
            # Jarvis
            jarvis_models_dir=base_dir / ".jarvis" / "models",
            experience_db_path=base_dir / ".jarvis" / "experience.db",
            # Meta (disabled for trading independence)
            meta_evolution_db=None,
            meta_monitoring_db=None,
            # Bolt
            bolt_config_dir=Path.home() / ".bolt",
        )


@dataclass
class MLConfig:
    """Unified ML/AI configuration."""

    # Learning parameters
    learning_rate: float = 1e-4
    adaptive_learning_rate: float = 0.1
    bandit_exploration_rate: float = 0.1

    # Model parameters
    embedding_dimension: int = 768  # Standardized across systems
    hidden_dim: int = 512
    max_sequence_length: int = 512

    # Training parameters
    batch_size: int = 32
    max_training_samples: int = 10000
    gradient_clip_norm: float = 1.0
    dropout_rate: float = 0.1

    # Thresholds
    similarity_threshold: float = 0.7
    confidence_threshold: float = 0.3
    relevance_threshold: float = 0.5

    # Hardware acceleration
    enable_gpu: bool = True
    enable_ane: bool = True
    prefer_mlx: bool = True
    ane_batch_size: int = 256
    ane_cache_size_mb: int = 512


@dataclass
class MonitoringConfig:
    """Unified monitoring configuration."""

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None
    enable_performance_logs: bool = True
    enable_cache_stats: bool = True

    # Monitoring intervals
    monitoring_interval: float = 1.0
    memory_check_interval_s: int = 30
    resource_check_interval: float = 5.0
    file_watch_interval_s: int = 60

    # Thresholds
    high_usage_threshold: float = 0.85
    memory_pressure_threshold: float = 0.8

    # Performance tracking
    performance_history_size: int = 50
    system_load_history_size: int = 20
    cooldown_period_s: float = 5.0


@dataclass
class TimingConfig:
    """Unified timing configuration (from meta system)."""

    # Pattern detection
    recent_activity_window_seconds: int = 300
    rapid_development_threshold_seconds: int = 120

    # Evolution readiness
    minimum_observations_for_evolution: int = 10
    minimum_recent_activity_count: int = 3

    # Coordination
    coordination_cycle_seconds: int = 5
    health_check_interval_seconds: int = 30

    # File watching
    file_change_debounce_seconds: float = 1.0
    debounce_delay_ms: float = 250.0


@dataclass
class UnifiedConfig:
    """Complete unified configuration for all systems."""

    hardware: HardwareConfig
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    paths: PathConfig = field(
        default_factory=lambda: PathConfig.from_base_dir(Path.cwd())
    )
    ml: MLConfig = field(default_factory=MLConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)

    # Feature flags
    enable_gpu_acceleration: bool = True
    enable_adaptive_concurrency: bool = True
    enable_predictive_prefetch: bool = True
    enable_memory_optimization: bool = True
    enable_realtime_indexing: bool = True
    enable_profiling: bool = False

    # System identification
    config_version: str = "2.0.0"
    systems_enabled: dict[str, bool] = field(
        default_factory=lambda: {
            "einstein": True,
            "bolt": True,
            "jarvis": True,
            "meta": False,  # Disabled for trading independence
        }
    )

    def __post_init__(self):
        """Post-initialization adjustments and validation."""
        self._adjust_for_hardware()
        self._create_directories()
        self._validate_config()

    def _adjust_for_hardware(self):
        """Adjust configuration based on detected hardware."""
        # Adjust concurrency based on CPU cores
        self.performance.max_file_io_concurrency = min(
            self.hardware.cpu_cores, self.performance.max_file_io_concurrency
        )

        # Adjust memory limits based on available memory
        max_memory = min(
            self.hardware.memory_available_gb * 0.8,
            self.performance.max_memory_usage_gb,
        )
        self.performance.max_memory_usage_gb = max(1.0, max_memory)

        # Adjust cache sizes for lower memory systems
        if self.hardware.memory_total_gb < 16:
            self.cache.hot_cache_size = int(self.cache.hot_cache_size * 0.5)
            self.cache.warm_cache_size = int(self.cache.warm_cache_size * 0.5)
            self.cache.index_cache_size_mb = int(self.cache.index_cache_size_mb * 0.5)

        # Disable GPU acceleration if not available
        if not self.hardware.has_gpu:
            self.enable_gpu_acceleration = False
            self.ml.enable_gpu = False

        # Disable ANE if not available
        if not self.hardware.has_ane:
            self.ml.enable_ane = False

        # Adjust performance targets for slower systems
        if self.hardware.cpu_cores < 8:
            self.performance.max_search_time_ms *= 1.5
            self.performance.max_startup_time_ms *= 1.5

    def _create_directories(self):
        """Create necessary directories."""
        dirs_to_create = [
            self.paths.cache_dir,
            self.paths.logs_dir,
            self.paths.einstein_cache_dir,
            self.paths.jarvis_models_dir.parent,
            self.paths.bolt_config_dir,
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _validate_config(self):
        """Validate configuration."""
        validator = ConfigValidator()
        result = validator.validate(self)

        if not result.is_valid:
            for error in result.errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError(f"Invalid configuration: {result.errors}")

        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"Configuration warning: {warning}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hardware": self.hardware.__dict__,
            "performance": self.performance.__dict__,
            "cache": self.cache.__dict__,
            "paths": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.paths.__dict__.items()
            },
            "ml": self.ml.__dict__,
            "monitoring": self.monitoring.__dict__,
            "timing": self.timing.__dict__,
            "feature_flags": {
                "enable_gpu_acceleration": self.enable_gpu_acceleration,
                "enable_adaptive_concurrency": self.enable_adaptive_concurrency,
                "enable_predictive_prefetch": self.enable_predictive_prefetch,
                "enable_memory_optimization": self.enable_memory_optimization,
                "enable_realtime_indexing": self.enable_realtime_indexing,
                "enable_profiling": self.enable_profiling,
            },
            "config_version": self.config_version,
            "systems_enabled": self.systems_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedConfig":
        """Create configuration from dictionary."""
        # Convert path strings back to Path objects
        if "paths" in data:
            for key, value in data["paths"].items():
                if isinstance(value, str):
                    data["paths"][key] = Path(value)

        # Extract feature flags
        feature_flags = data.pop("feature_flags", {})

        # Handle hardware config - detect if not provided
        hardware_data = data.get("hardware", {})
        if not hardware_data or not all(
            key in hardware_data
            for key in ["cpu_cores", "memory_total_gb", "platform_type"]
        ):
            # Hardware not in config file - detect it
            hardware = HardwareDetector.detect_hardware()
        else:
            # Use hardware from config
            hardware = HardwareConfig(**hardware_data)

        # Handle paths config - create default if not provided
        paths_data = data.get("paths", {})
        if not paths_data or not all(
            key in paths_data for key in ["base_dir", "cache_dir", "logs_dir"]
        ):
            # Paths not in config file - create default
            paths = PathConfig.from_base_dir(Path.cwd())
        else:
            # Use paths from config
            paths = PathConfig(**paths_data)

        # Handle ML config - filter out incompatible fields
        ml_data = data.get("ml", {})
        # Only pass compatible fields to MLConfig
        ml_compatible_fields = {
            "learning_rate",
            "adaptive_learning_rate",
            "bandit_exploration_rate",
            "embedding_dimension",
            "hidden_dim",
            "max_sequence_length",
            "batch_size",
            "max_training_samples",
            "gradient_clip_norm",
            "dropout_rate",
            "similarity_threshold",
            "confidence_threshold",
            "relevance_threshold",
            "enable_gpu",
            "enable_ane",
            "prefer_mlx",
            "ane_batch_size",
            "ane_cache_size_mb",
        }
        ml_filtered = {k: v for k, v in ml_data.items() if k in ml_compatible_fields}

        # Create config
        config = cls(
            hardware=hardware,
            performance=PerformanceConfig(**data.get("performance", {})),
            cache=CacheConfig(**data.get("cache", {})),
            paths=paths,
            ml=MLConfig(**ml_filtered),
            monitoring=MonitoringConfig(**data.get("monitoring", {})),
            timing=TimingConfig(**data.get("timing", {})),
            **feature_flags,
            config_version=data.get("config_version", "2.0.0"),
            systems_enabled=data.get(
                "systems_enabled",
                {"einstein": True, "bolt": True, "jarvis": True, "meta": False},
            ),
        )

        return config

    def save_to_file(self, path: Path):
        """Save configuration to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            if path.suffix == ".yaml":
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> "UnifiedConfig":
        """Load configuration from file."""
        with open(path) as f:
            if path.suffix == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls.from_dict(data)

    def get_system_config(self, system: str) -> dict[str, Any]:
        """Get configuration for specific system (einstein, bolt, jarvis, meta)."""
        if system not in self.systems_enabled:
            raise ValueError(f"Unknown system: {system}")

        if not self.systems_enabled[system]:
            raise ValueError(f"System {system} is not enabled")

        # Build system-specific config
        if system == "einstein":
            return {
                "hardware": self.hardware,
                "performance": self.performance,
                "cache": self.cache,
                "paths": {
                    "base_dir": self.paths.base_dir,
                    "cache_dir": self.paths.einstein_cache_dir,
                    "logs_dir": self.paths.logs_dir,
                    "analytics_db_path": self.paths.analytics_db_path,
                    "embeddings_db_path": self.paths.embeddings_db_path,
                },
                "ml": self.ml,
                "monitoring": self.monitoring,
                "enable_gpu_acceleration": self.enable_gpu_acceleration,
                "enable_adaptive_concurrency": self.enable_adaptive_concurrency,
                "enable_predictive_prefetch": self.enable_predictive_prefetch,
                "enable_memory_optimization": self.enable_memory_optimization,
                "enable_realtime_indexing": self.enable_realtime_indexing,
            }
        elif system == "bolt":
            return {
                "max_agents": self.hardware.cpu_cores,
                "default_agents": min(8, self.hardware.cpu_cores),
                "use_gpu": self.ml.enable_gpu and self.hardware.has_gpu,
                "prefer_mlx": self.ml.prefer_mlx,
                "max_memory_gb": self.performance.max_memory_usage_gb,
                "async_timeout": self.performance.async_timeout,
                "batch_size": self.ml.batch_size,
                "cache_size": self.cache.hot_cache_size,
                "enable_monitoring": True,
                "monitoring_interval": self.monitoring.monitoring_interval,
                "log_level": self.monitoring.log_level,
                "log_file": self.monitoring.log_file,
            }
        elif system == "jarvis":
            return {
                "neural": {
                    "embedding_dim": self.ml.embedding_dimension,
                    "hidden_dim": self.ml.hidden_dim,
                    "learning_rate": self.ml.learning_rate,
                    "batch_size": self.ml.batch_size,
                    "gradient_clip_norm": self.ml.gradient_clip_norm,
                    "dropout_rate": self.ml.dropout_rate,
                },
                "search": {
                    "default_simulations": 2000,
                    "exploration_constant": 1.414,
                    "batch_size": 256,
                    "search_timeout_seconds": self.performance.search_timeout_seconds,
                    "worker_timeout_seconds": self.performance.worker_timeout_seconds,
                },
                "hardware": {
                    "max_memory_gb": self.performance.max_memory_usage_gb,
                    "p_core_utilization": 0.9,
                    "e_core_utilization": 1.0,
                    "gpu_memory_fraction": 0.75,
                },
                "log_level": self.monitoring.log_level,
                "enable_profiling": self.enable_profiling,
            }
        elif system == "meta":
            # Meta system disabled for trading independence
            return {
                "disabled": True,
                "reason": "Trading system operates independently of meta system",
                "mock_config": {
                    "timing": {"coordination_cycle_seconds": 0},
                    "system": {"enabled": False},
                    "database": {"disabled": True},
                },
            }

        return {}


class ConfigLoader:
    """Unified configuration loader with environment support."""

    _instance: UnifiedConfig | None = None

    @classmethod
    def load_config(
        cls, config_file: Path | None = None, project_root: Path | None = None
    ) -> UnifiedConfig:
        """Load configuration with environment overrides."""

        if cls._instance is not None:
            return cls._instance

        if project_root is None:
            project_root = Path.cwd()

        # Check for config file
        if config_file and config_file.exists():
            config = UnifiedConfig.load_from_file(config_file)
        else:
            # Detect hardware
            hardware = HardwareDetector.detect_hardware()

            # Create base config
            config = UnifiedConfig(
                hardware=hardware, paths=PathConfig.from_base_dir(project_root)
            )

        # Apply environment overrides
        env_loader = EnvironmentLoader()
        env_loader.apply_overrides(config)

        cls._instance = config
        return config

    @classmethod
    def reset(cls):
        """Reset cached configuration."""
        cls._instance = None

    @classmethod
    def get_config(cls) -> UnifiedConfig:
        """Get current configuration, loading if necessary."""
        if cls._instance is None:
            return cls.load_config()
        return cls._instance


# Backward compatibility functions
def get_einstein_config(project_root: Path | None = None) -> dict[str, Any]:
    """Get Einstein-compatible configuration."""
    config = ConfigLoader.load_config(project_root=project_root)
    return config.get_system_config("einstein")


def get_bolt_config() -> dict[str, Any]:
    """Get Bolt-compatible configuration."""
    config = ConfigLoader.get_config()
    return config.get_system_config("bolt")


def get_jarvis_config() -> dict[str, Any]:
    """Get Jarvis-compatible configuration."""
    config = ConfigLoader.get_config()
    return config.get_system_config("jarvis")


def get_meta_config() -> dict[str, Any]:
    """Get Meta-compatible configuration."""
    config = ConfigLoader.get_config()
    return config.get_system_config("meta")


# Global access
def get_unified_config() -> UnifiedConfig:
    """Get the unified configuration instance."""
    return ConfigLoader.get_config()


def reset_config():
    """Reset global configuration."""
    ConfigLoader.reset()


if __name__ == "__main__":
    # Test unified configuration
    config = get_unified_config()

    print("🔧 Unified Configuration System")
    print(f"  Version: {config.config_version}")
    print(f"  Hardware: {config.hardware.platform_type}")
    print(
        f"  CPU: {config.hardware.cpu_cores} cores ({config.hardware.cpu_performance_cores}P + {config.hardware.cpu_efficiency_cores}E)"
    )
    print(f"  Memory: {config.hardware.memory_total_gb:.1f}GB")
    print(f"  GPU: {'✅' if config.hardware.has_gpu else '❌'}")

    print("\n📊 System Configurations:")
    for system in config.systems_enabled:
        if config.systems_enabled[system]:
            try:
                sys_config = config.get_system_config(system)
                print(f"  {system}: ✅ Loaded successfully")
            except Exception as e:
                print(f"  {system}: ❌ Error: {e}")

    # Test backward compatibility
    print("\n🔄 Backward Compatibility:")
    print(f"  Einstein: {len(get_einstein_config())} settings")
    print(f"  Bolt: {len(get_bolt_config())} settings")
    print(f"  Jarvis: {len(get_jarvis_config())} settings")
    print(f"  Meta: {len(get_meta_config())} settings")
