"""
Meta System Configuration - Centralized configuration management
Addresses hardcoded values identified in the audit
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class MetaTimingConfig:
    """Timing-related configuration parameters"""

    # Pattern detection thresholds (from audit findings)
    recent_activity_window_seconds: int = 300  # Last 5 minutes
    rapid_development_threshold_seconds: int = 120  # 2 minutes between modifications

    # Evolution readiness criteria
    minimum_observations_for_evolution: int = 10
    minimum_recent_activity_count: int = 3

    # Coordination loop timing
    coordination_cycle_seconds: int = 5
    health_check_interval_seconds: int = 30

    # File watching debounce
    file_change_debounce_seconds: float = 1.0

    # Additional timing configurations
    watcher_rapid_iteration_threshold_seconds: int = 60
    generator_time_window_seconds: int = 3600
    executor_timeout_seconds: int = 5
    reality_bridge_timeout_seconds: float = 5.0
    reality_bridge_learning_timeout_seconds: float = 2.0
    daemon_compliance_check_seconds: int = 300
    daemon_learning_update_seconds: int = 3600


@dataclass
class MetaSystemConfig:
    """Hardware and system resource configuration"""

    # M4 Pro hardware configuration
    p_cores: int = 8
    e_cores: int = 4
    total_cores: int = 12
    gpu_cores: int = 20
    unified_memory_gb: int = 24

    # Memory limits
    max_memory_usage_mb: int = 200
    max_observations_in_memory: int = 1000
    daemon_memory_limit_mb: int = 512

    # Database configuration
    max_db_size_mb: int = 100
    vacuum_threshold_ops: int = 10000

    # File hash configuration
    file_hash_length: int = 16

    # Generator configuration
    generator_cache_size: int = 10000
    generator_code_preview_length: int = 200

    # Quality enforcer configuration
    quality_max_execution_ms: int = 1000
    quality_max_memory_mb: int = 100


@dataclass
class MetaQualityConfig:
    """Quality enforcement configuration"""

    # Audit thresholds
    minimum_function_lines: int = 3
    hardcoded_number_threshold: int = 10
    max_unused_imports: int = 0

    # Compliance scoring
    minimum_compliance_percentage: float = 90.0
    error_weight: float = 1.0
    warning_weight: float = 0.5
    info_weight: float = 0.1


@dataclass
class MetaEvolutionConfig:
    """Evolution and learning configuration"""

    # Evolution triggers
    evolution_readiness_threshold: float = 0.8  # 80%
    max_evolution_attempts_per_hour: int = 10

    # Generation tracking
    max_generations_stored: int = 100
    evolution_history_retention_days: int = 30

    # Learning parameters
    pattern_confidence_threshold: float = 0.7
    learning_rate: float = 0.1
    decay_factor: float = 0.95

    # Additional evolution configurations
    watcher_observation_retention_count: int = 50
    daemon_compliance_score_retention: int = 100

    # Size change thresholds
    major_file_change_threshold: int = 100

    # Pattern detection
    watcher_recent_changes_limit: int = 10


@dataclass
class DatabaseConfig:
    """Database configuration for meta system"""

    # Database files - centralized to avoid hardcoded paths
    evolution_db: str = "meta_evolution.db"
    monitoring_db: str = "meta_monitoring.db"
    reality_db: str = "meta_reality_learning.db"
    daemon_db: str = "meta_monitoring.db"  # Consolidated into monitoring DB

    # Database settings
    connection_timeout: int = 30
    max_connections: int = 10
    backup_interval_hours: int = 24


@dataclass
class MetaConfig:
    """Complete meta system configuration"""

    timing: MetaTimingConfig
    system: MetaSystemConfig
    quality: MetaQualityConfig
    evolution: MetaEvolutionConfig
    database: DatabaseConfig

    # Meta configuration
    config_version: str = "1.0.0"
    debug_mode: bool = False

    def __init__(self):
        self.timing = MetaTimingConfig()
        self.system = MetaSystemConfig()
        self.quality = MetaQualityConfig()
        self.evolution = MetaEvolutionConfig()
        self.database = DatabaseConfig()

        # Allow environment variable overrides
        self._load_from_environment()

    def _load_from_environment(self):
        """Load configuration overrides from environment variables"""

        # Timing overrides
        if os.getenv("META_RECENT_ACTIVITY_WINDOW"):
            self.timing.recent_activity_window_seconds = int(
                os.getenv("META_RECENT_ACTIVITY_WINDOW")
            )

        if os.getenv("META_RAPID_DEV_THRESHOLD"):
            self.timing.rapid_development_threshold_seconds = int(
                os.getenv("META_RAPID_DEV_THRESHOLD")
            )

        if os.getenv("META_MIN_OBSERVATIONS"):
            self.timing.minimum_observations_for_evolution = int(
                os.getenv("META_MIN_OBSERVATIONS")
            )

        if os.getenv("META_MIN_ACTIVITY"):
            self.timing.minimum_recent_activity_count = int(
                os.getenv("META_MIN_ACTIVITY")
            )

        # System overrides
        if os.getenv("META_MAX_MEMORY_MB"):
            self.system.max_memory_usage_mb = int(os.getenv("META_MAX_MEMORY_MB"))

        # Debug mode
        if os.getenv("META_DEBUG"):
            self.debug_mode = os.getenv("META_DEBUG").lower() in ("true", "1", "yes")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timing": self.timing.__dict__,
            "system": self.system.__dict__,
            "quality": self.quality.__dict__,
            "evolution": self.evolution.__dict__,
            "config_version": self.config_version,
            "debug_mode": self.debug_mode,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "MetaConfig":
        """Create configuration from dictionary"""
        config = cls()

        if "timing" in config_dict:
            config.timing = MetaTimingConfig(**config_dict["timing"])
        if "system" in config_dict:
            config.system = MetaSystemConfig(**config_dict["system"])
        if "quality" in config_dict:
            config.quality = MetaQualityConfig(**config_dict["quality"])
        if "evolution" in config_dict:
            config.evolution = MetaEvolutionConfig(**config_dict["evolution"])

        config.config_version = config_dict.get("config_version", "1.0.0")
        config.debug_mode = config_dict.get("debug_mode", False)

        return config

    def validate(self) -> list[str]:
        """Validate configuration values"""

        violations = []

        # Timing validation
        if self.timing.recent_activity_window_seconds < 60:
            violations.append("recent_activity_window_seconds too short (minimum 60s)")

        if self.timing.minimum_observations_for_evolution < 5:
            violations.append("minimum_observations_for_evolution too low (minimum 5)")

        # System validation
        if self.system.max_memory_usage_mb < 50:
            violations.append("max_memory_usage_mb too low (minimum 50MB)")

        # Quality validation
        if self.quality.minimum_compliance_percentage < 50.0:
            violations.append("minimum_compliance_percentage too low (minimum 50%)")

        return violations


# Global configuration instance
_CONFIG = None


def get_meta_config() -> MetaConfig:
    """Get global meta configuration instance"""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = MetaConfig()

        # Validate on first load
        violations = _CONFIG.validate()
        if violations:
            print(f"⚠️ Configuration warnings: {violations}")

    return _CONFIG


def reload_meta_config():
    """Reload configuration from environment"""
    global _CONFIG
    _CONFIG = None
    return get_meta_config()


def save_meta_config(file_path: str):
    """Save configuration to file"""
    import json

    config = get_meta_config()
    with open(file_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def load_meta_config(file_path: str):
    """Load configuration from file"""
    import json

    global _CONFIG

    with open(file_path) as f:
        config_dict = json.load(f)

    _CONFIG = MetaConfig.from_dict(config_dict)
    return _CONFIG


if __name__ == "__main__":
    # Test configuration
    config = get_meta_config()

    print("🔧 Meta System Configuration:")
    print(f"  Recent Activity Window: {config.timing.recent_activity_window_seconds}s")
    print(
        f"  Rapid Development Threshold: {config.timing.rapid_development_threshold_seconds}s"
    )
    print(f"  Minimum Observations: {config.timing.minimum_observations_for_evolution}")
    print(f"  Minimum Activity: {config.timing.minimum_recent_activity_count}")
    print(f"  M4 Cores: P={config.system.p_cores}, E={config.system.e_cores}")
    print(f"  Max Memory: {config.system.max_memory_usage_mb}MB")
    print(
        f"  Evolution Threshold: {config.evolution.evolution_readiness_threshold:.1%}"
    )

    # Test validation
    violations = config.validate()
    if violations:
        print(f"❌ Configuration issues: {violations}")
    else:
        print("✅ Configuration validation passed")

    # Test environment override
    import os

    os.environ["META_DEBUG"] = "true"
    config_with_debug = reload_meta_config()
    print(f"Debug mode: {config_with_debug.debug_mode}")
