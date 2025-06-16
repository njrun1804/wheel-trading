"""
Meta Daemon Configuration - Centralized configuration for continuous meta system
Built following the 10-step coding principles for production quality
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LearningRulesConfig:
    """Configuration for learning rule enforcement"""

    # Pattern 1: Anti-pattern detection regex
    anti_patterns: str = (
        r"TODO|FIXME|dummy|mock|fake|placeholder|NotImplemented|pass\s*#"
    )

    # Pattern 2: Implementation verification
    min_function_lines: int = 3

    # Pattern 3: Dependency verification
    unused_import_tolerance: int = 0

    # Pattern 4: Configuration audit
    hardcoded_number_threshold: int = 10

    # Pattern 5: Async completeness
    require_await_in_async: bool = True

    # Pattern 6: Error handling
    allow_bare_except: bool = False
    allow_empty_except: bool = False

    # Pattern 7: Performance validation
    max_function_execution_ms: int = 1000
    max_memory_usage_mb: int = 100


@dataclass
class FileWatchConfig:
    """Configuration for file system monitoring"""

    # Monitored file patterns
    watch_patterns: list[str] = None

    # Ignored patterns
    ignore_patterns: list[str] = None

    # Real-time vs batch processing
    real_time_processing: bool = True
    batch_interval_seconds: int = 5

    # Quality gate behavior
    block_non_compliant_saves: bool = True
    show_violations_in_editor: bool = True

    def __post_init__(self):
        if self.watch_patterns is None:
            self.watch_patterns = [
                "*.py",
                "*.js",
                "*.ts",
                "*.jsx",
                "*.tsx",
                "*.go",
                "*.rs",
                "*.java",
                "*.cpp",
                "*.c",
            ]

        if self.ignore_patterns is None:
            self.ignore_patterns = [
                "__pycache__/*",
                "*.pyc",
                ".git/*",
                "node_modules/*",
                ".DS_Store",
                "*.log",
            ]


@dataclass
class DaemonConfig:
    """Configuration for meta daemon process"""

    # Process management
    daemon_mode: bool = True
    pid_file: Path = Path("meta_daemon.pid")
    log_file: Path = Path("meta_daemon.log")

    # Health monitoring
    health_check_interval_seconds: int = 30
    restart_on_failure: bool = True
    max_restart_attempts: int = 5

    # Performance
    worker_threads: int = 8  # M4 Pro P-cores
    background_threads: int = 4  # M4 Pro E-cores
    memory_limit_mb: int = 512

    # Integration
    git_pre_commit_hook: bool = True
    editor_integration: bool = True
    ci_cd_integration: bool = True


@dataclass
class QualityGateConfig:
    """Configuration for quality enforcement"""

    # Gate behavior
    enforcement_level: str = "strict"  # strict, warn, log
    fail_fast: bool = True
    show_detailed_errors: bool = True

    # Reporting
    generate_quality_reports: bool = True
    report_frequency_hours: int = 24

    # Learning
    learn_from_violations: bool = True
    update_patterns_automatically: bool = True

    # Metrics
    track_compliance_metrics: bool = True
    alert_on_compliance_drop: bool = True
    minimum_compliance_percentage: float = 90.0


@dataclass
class SystemConfig:
    """System monitoring configuration"""

    health_check_interval: int = 5
    max_cpu_usage: float = 80.0
    max_memory_usage: float = 80.0
    max_disk_usage: float = 90.0
    daemon_memory_limit_mb: int = 512  # Expected by daemon


@dataclass
class TimingConfig:
    """Timing configuration for various operations"""

    compliance_check_interval: int = 30
    learning_update_interval: int = 300  # 5 minutes
    health_monitor_interval: int = 5
    file_watch_debounce: float = 1.0
    daemon_compliance_check_seconds: int = 30  # Expected by daemon
    daemon_learning_update_seconds: int = 300  # Expected by daemon


@dataclass
class MetaDaemonConfig:
    """Complete meta daemon configuration"""

    learning_rules: LearningRulesConfig
    file_watch: FileWatchConfig
    daemon: DaemonConfig
    quality_gate: QualityGateConfig
    system: SystemConfig
    timing: TimingConfig

    # Meta configuration
    config_version: str = "1.0.0"
    created_with_learning_rules: bool = True

    def __init__(self):
        self.learning_rules = LearningRulesConfig()
        self.file_watch = FileWatchConfig()
        self.daemon = DaemonConfig()
        self.quality_gate = QualityGateConfig()
        self.system = SystemConfig()
        self.timing = TimingConfig()

    def validate_configuration(self) -> list[str]:
        """Validate configuration against learning rules"""

        violations = []

        # Check for hardcoded values that should be configurable
        if self.daemon.worker_threads > 20:
            violations.append(
                "worker_threads should be configurable based on available cores"
            )

        if self.daemon.memory_limit_mb < 64:
            violations.append("memory_limit_mb too low for realistic operation")

        # Validate file patterns
        if not self.file_watch.watch_patterns:
            violations.append("watch_patterns cannot be empty")

        # Validate quality thresholds
        if self.quality_gate.minimum_compliance_percentage < 50.0:
            violations.append("minimum_compliance_percentage too low")

        return violations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "learning_rules": self.learning_rules.__dict__,
            "file_watch": self.file_watch.__dict__,
            "daemon": self.daemon.__dict__,
            "quality_gate": self.quality_gate.__dict__,
            "system": self.system.__dict__,
            "timing": self.timing.__dict__,
            "config_version": self.config_version,
            "created_with_learning_rules": self.created_with_learning_rules,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "MetaDaemonConfig":
        """Create configuration from dictionary"""
        config = cls()

        if "learning_rules" in config_dict:
            config.learning_rules = LearningRulesConfig(**config_dict["learning_rules"])
        if "file_watch" in config_dict:
            config.file_watch = FileWatchConfig(**config_dict["file_watch"])
        if "daemon" in config_dict:
            config.daemon = DaemonConfig(**config_dict["daemon"])
        if "quality_gate" in config_dict:
            config.quality_gate = QualityGateConfig(**config_dict["quality_gate"])

        config.config_version = config_dict.get("config_version", "1.0.0")
        config.created_with_learning_rules = config_dict.get(
            "created_with_learning_rules", True
        )

        return config


# Global configuration instance
DAEMON_CONFIG = MetaDaemonConfig()


def get_daemon_config() -> MetaDaemonConfig:
    """Get global daemon configuration"""
    if DAEMON_CONFIG is None:
        raise RuntimeError("Daemon configuration not initialized")
    return DAEMON_CONFIG


def validate_daemon_config() -> None:
    """Validate configuration and raise if invalid"""
    violations = DAEMON_CONFIG.validate_configuration()
    if violations:
        raise ValueError(f"Configuration violations: {violations}")


def save_daemon_config(file_path: Path) -> None:
    """Save configuration to file"""
    import json

    config_dict = DAEMON_CONFIG.to_dict()
    with open(file_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_daemon_config(file_path: Path) -> None:
    """Load configuration from file"""
    import json

    global DAEMON_CONFIG

    with open(file_path) as f:
        config_dict = json.load(f)

    DAEMON_CONFIG = MetaDaemonConfig.from_dict(config_dict)


if __name__ == "__main__":
    # Test configuration compliance with learning rules
    config = get_daemon_config()

    print("🔧 Meta Daemon Configuration Test")
    print(f"  Config Version: {config.config_version}")
    print(f"  Learning Rules Compliant: {config.created_with_learning_rules}")

    # Validate configuration
    violations = config.validate_configuration()
    if violations:
        print(f"❌ Configuration violations: {len(violations)}")
        for violation in violations:
            print(f"  • {violation}")
    else:
        print("✅ Configuration validation passed")

    # Test serialization
    config_dict = config.to_dict()
    restored_config = MetaDaemonConfig.from_dict(config_dict)

    serialization_ok = (
        restored_config.learning_rules.anti_patterns
        == config.learning_rules.anti_patterns
        and restored_config.daemon.worker_threads == config.daemon.worker_threads
    )

    print(f"✅ Serialization test: {'PASS' if serialization_ok else 'FAIL'}")

    # Show key settings
    print("\n📊 Key Settings:")
    print(f"  Anti-patterns: {config.learning_rules.anti_patterns[:50]}...")
    print(f"  Worker threads: {config.daemon.worker_threads}")
    print(f"  Real-time processing: {config.file_watch.real_time_processing}")
    print(f"  Block non-compliant saves: {config.file_watch.block_non_compliant_saves}")
    print(f"  Minimum compliance: {config.quality_gate.minimum_compliance_percentage}%")
