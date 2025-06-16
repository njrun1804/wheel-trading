#!/usr/bin/env python3
"""
Einstein Configuration System

Externalizes all hardcoded configurations with environment variable support and 
auto-detection for hardware specs instead of hardcoding M4 Pro values.

Environment Variables:
    EINSTEIN_CPU_CORES: Number of CPU cores to use
    EINSTEIN_MEMORY_GB: Maximum memory allocation in GB
    EINSTEIN_USE_GPU: Enable GPU acceleration (true/false)
    EINSTEIN_CACHE_DIR: Cache directory path
    EINSTEIN_LOG_LEVEL: Logging level (DEBUG, INFO, WARN, ERROR)
    EINSTEIN_MAX_STARTUP_MS: Maximum startup time in milliseconds
    EINSTEIN_MAX_SEARCH_MS: Maximum search time in milliseconds
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Hardware configuration with auto-detection."""

    cpu_cores: int
    cpu_performance_cores: int
    cpu_efficiency_cores: int
    memory_total_gb: float
    memory_available_gb: float
    has_gpu: bool
    gpu_cores: int
    has_ane: bool  # Apple Neural Engine availability
    ane_cores: int  # Number of ANE cores (typically 16 on M4 Pro)
    platform_type: str  # 'apple_silicon', 'intel', 'amd', 'unknown'
    architecture: str  # 'arm64', 'x86_64', etc.


@dataclass
class PerformanceConfig:
    """Performance targets and thresholds."""

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

    # Search timeouts and limits
    search_timeout_ms: int = 50  # Default timeout for search operations
    rapid_search_limit: int = 100  # Default result limit for rapid searches
    high_performance_timeout_ms: int = 40  # Target sub-40ms for headroom

    # Memory targets
    max_memory_usage_gb: float = 2.0
    cache_memory_limit_mb: float = 512.0

    # CRITICAL FIX: Conservative concurrency limits to prevent CPU overload
    max_search_concurrency: int = 3  # Reduced from 4
    max_embedding_concurrency: int = 4  # Reduced from 8
    max_file_io_concurrency: int = 6  # Reduced from 12
    max_analysis_concurrency: int = 3  # Reduced from 6


@dataclass
class CacheConfig:
    """Cache sizes and policies."""

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
    """File and directory paths."""

    base_dir: Path
    cache_dir: Path
    logs_dir: Path
    analytics_db_path: Path
    embeddings_db_path: Path
    rapid_cache_dir: Path
    optimized_cache_dir: Path
    models_dir: Path
    # Database concurrency settings
    enable_concurrent_db: bool = True
    max_db_connections: int = 4
    db_connection_timeout: float = 30.0
    db_lock_timeout: float = 30.0

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "PathConfig":
        """Create path config from base directory."""
        einstein_dir = base_dir / ".einstein"
        return cls(
            base_dir=base_dir,
            cache_dir=einstein_dir,
            logs_dir=einstein_dir / "logs",
            analytics_db_path=einstein_dir / "analytics.db",
            embeddings_db_path=einstein_dir / "embeddings.db",
            rapid_cache_dir=einstein_dir / "rapid_cache",
            optimized_cache_dir=einstein_dir / "optimized",
            models_dir=einstein_dir / "models",
            # Configure for concurrent access
            enable_concurrent_db=True,
            max_db_connections=4,
            db_connection_timeout=30.0,
            db_lock_timeout=30.0,
        )


@dataclass
class MLConfig:
    """Machine learning and AI parameters."""

    # Learning rates
    adaptive_learning_rate: float = 0.1
    bandit_exploration_rate: float = 0.1

    # Model parameters
    embedding_dimension: int = 1536  # Standard dimension for modern embedding models
    embedding_dimension_minilm: int = 384  # MiniLM dimension for lightweight models
    max_sequence_length: int = 512

    # Training parameters
    batch_size: int = 32
    max_training_samples: int = 10000

    # Thresholds
    similarity_threshold: float = 0.7
    confidence_threshold: float = 0.3
    relevance_threshold: float = 0.5

    # ANE (Apple Neural Engine) Configuration
    enable_ane: bool = True
    ane_batch_size: int = 256  # Optimal batch size for ANE
    ane_cache_size_mb: int = 512
    ane_warmup_on_startup: bool = True
    ane_fallback_on_error: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration and optimization settings."""

    # SQLite pragma settings
    cache_size_pages: int = 10000  # Default cache size in pages
    journal_mode: str = "WAL"  # Write-Ahead Logging for better concurrency
    synchronous_mode: str = "NORMAL"  # Balance between safety and performance
    temp_store: str = "memory"  # Store temporary data in memory

    # Connection settings
    connection_timeout: float = 30.0
    busy_timeout_ms: int = 30000

    # Query limits
    default_query_limit: int = 100  # Default LIMIT for SELECT queries
    max_query_limit: int = 10000  # Maximum allowed LIMIT

    # Performance tuning
    optimize_on_startup: bool = True
    vacuum_on_startup: bool = False  # Only if needed
    analyze_on_startup: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability settings."""

    # File watching
    debounce_delay_ms: float = 250.0
    file_watch_interval_s: int = 60

    # Performance monitoring
    performance_history_size: int = 50
    system_load_history_size: int = 20
    cooldown_period_s: float = 5.0

    # Memory monitoring
    memory_check_interval_s: int = 30
    gc_interval_s: float = 30.0

    # Logging
    log_level: str = "INFO"
    enable_performance_logs: bool = True
    enable_cache_stats: bool = True


@dataclass
class EinsteinConfig:
    """Complete Einstein system configuration."""

    hardware: HardwareConfig
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paths: PathConfig = field(
        default_factory=lambda: PathConfig.from_base_dir(Path.cwd())
    )
    ml: MLConfig = field(default_factory=MLConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Feature flags
    enable_gpu_acceleration: bool = True
    enable_adaptive_concurrency: bool = True
    enable_predictive_prefetch: bool = True
    enable_memory_optimization: bool = True
    enable_realtime_indexing: bool = True

    def __post_init__(self):
        """Post-initialization adjustments based on hardware."""
        self._adjust_for_hardware()
        self._create_directories()

    def _adjust_for_hardware(self):
        """Adjust configuration based on detected hardware."""
        # Adjust concurrency based on CPU cores
        self.performance.max_file_io_concurrency = min(
            self.hardware.cpu_cores, self.performance.max_file_io_concurrency
        )

        # Adjust memory limits based on available memory
        max_memory = min(
            self.hardware.memory_available_gb * 0.8,  # Use max 80% of available
            self.performance.max_memory_usage_gb,
        )
        self.performance.max_memory_usage_gb = max(1.0, max_memory)  # Minimum 1GB

        # Adjust cache sizes for lower memory systems
        if self.hardware.memory_total_gb < 16:
            # Reduce cache sizes for systems with less than 16GB RAM
            self.cache.hot_cache_size = int(self.cache.hot_cache_size * 0.5)
            self.cache.warm_cache_size = int(self.cache.warm_cache_size * 0.5)
            self.cache.index_cache_size_mb = int(self.cache.index_cache_size_mb * 0.5)

        # Disable GPU acceleration if not available
        if not self.hardware.has_gpu:
            self.enable_gpu_acceleration = False

        # Disable ANE if not available
        if not self.hardware.has_ane:
            self.ml.enable_ane = False

        # Adjust performance targets for slower systems
        if self.hardware.cpu_cores < 8:
            self.performance.max_search_time_ms *= 1.5
            self.performance.max_startup_time_ms *= 1.5

    def _create_directories(self):
        """Create necessary directories."""
        for path_attr in [
            "cache_dir",
            "logs_dir",
            "rapid_cache_dir",
            "optimized_cache_dir",
            "models_dir",
        ]:
            path = getattr(self.paths, path_attr)
            path.mkdir(parents=True, exist_ok=True)

        # Initialize database managers for concurrent access
        self._init_database_managers()

    def _init_database_managers(self):
        """Initialize database managers for Einstein databases."""
        if not self.paths.enable_concurrent_db:
            return

        try:
            # Import database concurrency fixes
            from bolt_database_fixes import ConcurrentDatabase, DatabaseConfig

            self._db_managers = {}

            # Initialize analytics database manager
            if self.paths.analytics_db_path:
                config = DatabaseConfig(
                    path=self.paths.analytics_db_path,
                    max_connections=self.paths.max_db_connections,
                    connection_timeout=self.paths.db_connection_timeout,
                    lock_timeout=self.paths.db_lock_timeout,
                    retry_attempts=3,
                    retry_delay=1.0,
                    enable_wal_mode=True,
                    enable_connection_pooling=True,
                )
                # Create ConcurrentDatabase without duplicating path parameter
                self._db_managers["analytics"] = ConcurrentDatabase(config)

            # Initialize embeddings database manager
            if self.paths.embeddings_db_path:
                config = DatabaseConfig(
                    path=self.paths.embeddings_db_path,
                    max_connections=self.paths.max_db_connections,
                    connection_timeout=self.paths.db_connection_timeout,
                    lock_timeout=self.paths.db_lock_timeout,
                    retry_attempts=3,
                    retry_delay=1.0,
                    enable_wal_mode=True,
                    enable_connection_pooling=True,
                )
                # Create ConcurrentDatabase without duplicating path parameter
                self._db_managers["embeddings"] = ConcurrentDatabase(config)

            logger.info(
                f"Initialized {len(self._db_managers)} Einstein database managers"
            )

        except ImportError:
            logger.warning("Database concurrency fixes not available for Einstein")
            self._db_managers = {}
        except Exception as e:
            logger.error(f"Failed to initialize Einstein database managers: {e}")
            self._db_managers = {}

    def get_database_manager(self, db_name: str):
        """Get database manager for Einstein database."""
        return getattr(self, "_db_managers", {}).get(db_name)

    def close_database_managers(self):
        """Close all database managers."""
        if hasattr(self, "_db_managers"):
            for db_name, manager in self._db_managers.items():
                try:
                    manager.close()
                    logger.info(f"Closed Einstein database manager: {db_name}")
                except Exception as e:
                    logger.warning(f"Error closing database manager {db_name}: {e}")
            self._db_managers.clear()


class HardwareDetector:
    """Detects hardware capabilities."""

    @staticmethod
    def detect_hardware() -> HardwareConfig:
        """Detect current hardware configuration."""
        try:
            # Basic system info
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_physical_cores = psutil.cpu_count(logical=False)
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)

            # Platform detection
            platform_type = HardwareDetector._detect_platform_type()
            architecture = platform.machine().lower()

            # Apple Silicon specific detection
            (
                cpu_perf_cores,
                cpu_eff_cores,
            ) = HardwareDetector._detect_apple_silicon_cores(
                platform_type, cpu_cores, cpu_physical_cores
            )

            # GPU detection
            has_gpu, gpu_cores = HardwareDetector._detect_gpu(platform_type)

            # ANE detection
            has_ane, ane_cores = HardwareDetector._detect_ane(platform_type)

            return HardwareConfig(
                cpu_cores=cpu_cores,
                cpu_performance_cores=cpu_perf_cores,
                cpu_efficiency_cores=cpu_eff_cores,
                memory_total_gb=memory_total_gb,
                memory_available_gb=memory_available_gb,
                has_gpu=has_gpu,
                gpu_cores=gpu_cores,
                has_ane=has_ane,
                ane_cores=ane_cores,
                platform_type=platform_type,
                architecture=architecture,
            )

        except Exception as e:
            logger.error(
                f"Hardware detection failed, using defaults: {e}",
                exc_info=True,
                extra={
                    "operation": "detect_hardware",
                    "error_type": type(e).__name__,
                    "cpu_count_available": psutil.cpu_count() is not None,
                    "memory_available": psutil.virtual_memory() is not None,
                    "platform": platform.system(),
                    "machine": platform.machine(),
                    "python_version": platform.python_version(),
                },
            )
            return HardwareDetector._get_default_hardware()

    @staticmethod
    def _detect_platform_type() -> str:
        """Detect platform type."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "darwin" and "arm" in machine:
            return "apple_silicon"
        elif system == "darwin" and "x86" in machine:
            return "intel_mac"
        elif "intel" in platform.processor().lower():
            return "intel"
        elif "amd" in platform.processor().lower():
            return "amd"
        else:
            return "unknown"

    @staticmethod
    def _detect_apple_silicon_cores(
        platform_type: str, total_cores: int, physical_cores: int
    ) -> tuple[int, int]:
        """Detect Apple Silicon P-cores and E-cores."""
        if platform_type != "apple_silicon":
            return total_cores, 0

        try:
            # Try to get detailed CPU info on macOS
            result = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                perf_cores = int(result.stdout.strip())
                eff_cores = total_cores - perf_cores
                return perf_cores, eff_cores
        except (
            subprocess.SubprocessError,
            subprocess.TimeoutExpired,
            ValueError,
            OSError,
        ) as e:
            logger.debug(
                f"Failed to detect Apple Silicon core configuration via sysctl: {e}",
                extra={
                    "operation": "detect_apple_silicon_cores",
                    "error_type": type(e).__name__,
                    "platform_type": platform_type,
                    "total_cores": total_cores,
                    "physical_cores": physical_cores,
                    "sysctl_command": "sysctl -n hw.perflevel0.physicalcpu",
                    "error_details": str(e),
                },
            )

        # Fallback: estimate based on known Apple Silicon configurations
        if total_cores == 8:  # M1
            return 4, 4
        elif total_cores == 10:  # M1 Pro
            return 8, 2
        elif total_cores == 12:  # M4 Pro, M2 Pro
            return 8, 4
        elif total_cores == 16:  # M1 Max, M2 Max
            return 10, 6
        else:
            # Conservative estimate
            return max(4, total_cores // 2), total_cores - max(4, total_cores // 2)

    @staticmethod
    def _detect_gpu(platform_type: str) -> tuple[bool, int]:
        """Detect GPU availability and capabilities."""
        if platform_type == "apple_silicon":
            try:
                # Try to detect Metal GPU
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and "Metal" in result.stdout:
                    # Estimate GPU cores based on known Apple Silicon configs
                    # This is approximate since exact core count isn't easily accessible
                    return True, 20  # Conservative estimate
            except (
                subprocess.SubprocessError,
                subprocess.TimeoutExpired,
                OSError,
            ) as e:
                logger.debug(
                    f"Failed to detect Metal GPU via system_profiler: {e}",
                    extra={
                        "operation": "detect_gpu_metal",
                        "error_type": type(e).__name__,
                        "platform_type": platform_type,
                        "command": "system_profiler SPDisplaysDataType",
                        "timeout": 10,
                        "error_details": str(e),
                    },
                )

        # Try to detect NVIDIA/AMD GPUs
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                return True, len(gpus) * 1000  # Rough estimate
        except ImportError:
            logger.debug("GPUtil not available for GPU detection")
        except Exception as e:
            logger.warning(f"Error detecting GPU hardware: {e}")

        return False, 0

    @staticmethod
    def _detect_ane(platform_type: str) -> tuple[bool, int]:
        """Detect Apple Neural Engine availability and capabilities."""
        if platform_type != "apple_silicon":
            return False, 0

        try:
            # Check if MLX with ANE support is available
            import mlx.core as mx

            if hasattr(mx, "metal") and mx.metal.is_available():
                # Try to get device info
                device_info = mx.metal.device_info()

                # Check for M4 chip family which has 16 ANE cores
                if "M4" in str(device_info):
                    logger.info("ANE detected: M4 chip with 16 ANE cores")
                    return True, 16
                elif "M3" in str(device_info):
                    logger.info("ANE detected: M3 chip with 16 ANE cores")
                    return True, 16
                elif "M2" in str(device_info):
                    logger.info("ANE detected: M2 chip with 16 ANE cores")
                    return True, 16
                elif "M1" in str(device_info):
                    logger.info("ANE detected: M1 chip with 16 ANE cores")
                    return True, 16
                else:
                    # Assume ANE is available on Apple Silicon but unknown core count
                    logger.info(
                        "ANE detected: Apple Silicon with estimated 16 ANE cores"
                    )
                    return True, 16

        except ImportError:
            logger.debug("MLX not available for ANE detection")
        except Exception as e:
            logger.debug(f"ANE detection failed: {e}")

        # Fallback: check for Apple Silicon via system info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                cpu_brand = result.stdout.strip().lower()
                if any(chip in cpu_brand for chip in ["m1", "m2", "m3", "m4"]):
                    logger.info(f"ANE detected via CPU brand: {cpu_brand}")
                    return True, 16  # All modern Apple Silicon has 16 ANE cores
        except (
            subprocess.SubprocessError,
            subprocess.TimeoutExpired,
            ValueError,
            OSError,
        ) as e:
            logger.debug(f"Failed to detect ANE via sysctl: {e}")

        return False, 0

    @staticmethod
    def _get_default_hardware() -> HardwareConfig:
        """Get default hardware configuration as fallback."""
        return HardwareConfig(
            cpu_cores=8,
            cpu_performance_cores=6,
            cpu_efficiency_cores=2,
            memory_total_gb=16.0,
            memory_available_gb=12.0,
            has_gpu=False,
            gpu_cores=0,
            has_ane=False,
            ane_cores=0,
            platform_type="unknown",
            architecture="unknown",
        )


class ConfigLoader:
    """Loads configuration from environment variables and defaults."""

    @staticmethod
    def load_config(project_root: Path | str | None = None) -> EinsteinConfig:
        """Load complete Einstein configuration."""
        if project_root is None:
            project_root = Path.cwd()
        elif isinstance(project_root, str):
            project_root = Path(project_root)

        # Detect hardware
        hardware = HardwareDetector.detect_hardware()

        # Load performance config from environment
        performance = ConfigLoader._load_performance_config(hardware)

        # Load cache config from environment
        cache = ConfigLoader._load_cache_config()

        # Load paths config
        paths = ConfigLoader._load_paths_config(project_root)

        # Load ML config from environment
        ml = ConfigLoader._load_ml_config()

        # Load database config from environment
        database = ConfigLoader._load_database_config()

        # Load monitoring config from environment
        monitoring = ConfigLoader._load_monitoring_config()

        # Load feature flags from environment
        enable_gpu = ConfigLoader._get_bool_env("EINSTEIN_USE_GPU", hardware.has_gpu)
        enable_adaptive = ConfigLoader._get_bool_env(
            "EINSTEIN_ADAPTIVE_CONCURRENCY", True
        )
        enable_prefetch = ConfigLoader._get_bool_env(
            "EINSTEIN_PREDICTIVE_PREFETCH", True
        )
        enable_memory_opt = ConfigLoader._get_bool_env(
            "EINSTEIN_MEMORY_OPTIMIZATION", True
        )
        enable_realtime = ConfigLoader._get_bool_env("EINSTEIN_REALTIME_INDEXING", True)

        return EinsteinConfig(
            hardware=hardware,
            performance=performance,
            cache=cache,
            database=database,
            paths=paths,
            ml=ml,
            monitoring=monitoring,
            enable_gpu_acceleration=enable_gpu,
            enable_adaptive_concurrency=enable_adaptive,
            enable_predictive_prefetch=enable_prefetch,
            enable_memory_optimization=enable_memory_opt,
            enable_realtime_indexing=enable_realtime,
        )

    @staticmethod
    def _load_performance_config(hardware: HardwareConfig) -> PerformanceConfig:
        """Load performance configuration with environment overrides."""
        config = PerformanceConfig()

        # Override with environment variables if present
        config.max_startup_time_ms = ConfigLoader._get_float_env(
            "EINSTEIN_MAX_STARTUP_MS", config.max_startup_time_ms
        )
        config.max_search_time_ms = ConfigLoader._get_float_env(
            "EINSTEIN_MAX_SEARCH_MS", config.max_search_time_ms
        )
        config.max_memory_usage_gb = ConfigLoader._get_float_env(
            "EINSTEIN_MAX_MEMORY_GB", config.max_memory_usage_gb
        )

        # Adjust concurrency based on hardware and environment
        config.max_search_concurrency = ConfigLoader._get_int_env(
            "EINSTEIN_SEARCH_CONCURRENCY",
            min(hardware.cpu_cores // 2, config.max_search_concurrency),
        )
        config.max_file_io_concurrency = ConfigLoader._get_int_env(
            "EINSTEIN_FILE_IO_CONCURRENCY",
            min(hardware.cpu_cores, config.max_file_io_concurrency),
        )

        return config

    @staticmethod
    def _load_cache_config() -> CacheConfig:
        """Load cache configuration with environment overrides."""
        config = CacheConfig()

        config.hot_cache_size = ConfigLoader._get_int_env(
            "EINSTEIN_HOT_CACHE_SIZE", config.hot_cache_size
        )
        config.warm_cache_size = ConfigLoader._get_int_env(
            "EINSTEIN_WARM_CACHE_SIZE", config.warm_cache_size
        )
        config.search_cache_size = ConfigLoader._get_int_env(
            "EINSTEIN_SEARCH_CACHE_SIZE", config.search_cache_size
        )
        config.index_cache_size_mb = ConfigLoader._get_int_env(
            "EINSTEIN_INDEX_CACHE_MB", config.index_cache_size_mb
        )

        return config

    @staticmethod
    def _load_paths_config(project_root: Path) -> PathConfig:
        """Load paths configuration with environment overrides."""
        # Allow overriding the cache directory
        cache_dir_str = os.getenv("EINSTEIN_CACHE_DIR")
        if cache_dir_str:
            Path(cache_dir_str)
        else:
            project_root / ".einstein"

        return PathConfig.from_base_dir(project_root)

    @staticmethod
    def _load_ml_config() -> MLConfig:
        """Load ML configuration with environment overrides."""
        config = MLConfig()

        config.adaptive_learning_rate = ConfigLoader._get_float_env(
            "EINSTEIN_LEARNING_RATE", config.adaptive_learning_rate
        )
        config.similarity_threshold = ConfigLoader._get_float_env(
            "EINSTEIN_SIMILARITY_THRESHOLD", config.similarity_threshold
        )
        config.confidence_threshold = ConfigLoader._get_float_env(
            "EINSTEIN_CONFIDENCE_THRESHOLD", config.confidence_threshold
        )

        # ANE configuration from environment
        config.enable_ane = ConfigLoader._get_bool_env(
            "EINSTEIN_ENABLE_ANE", config.enable_ane
        )
        config.ane_batch_size = ConfigLoader._get_int_env(
            "EINSTEIN_ANE_BATCH_SIZE", config.ane_batch_size
        )
        config.ane_cache_size_mb = ConfigLoader._get_int_env(
            "EINSTEIN_ANE_CACHE_MB", config.ane_cache_size_mb
        )
        config.ane_warmup_on_startup = ConfigLoader._get_bool_env(
            "EINSTEIN_ANE_WARMUP", config.ane_warmup_on_startup
        )
        config.ane_fallback_on_error = ConfigLoader._get_bool_env(
            "EINSTEIN_ANE_FALLBACK", config.ane_fallback_on_error
        )

        return config

    @staticmethod
    def _load_database_config() -> DatabaseConfig:
        """Load database configuration with environment overrides."""
        config = DatabaseConfig()

        config.cache_size_pages = ConfigLoader._get_int_env(
            "EINSTEIN_DB_CACHE_PAGES", config.cache_size_pages
        )
        config.journal_mode = os.getenv("EINSTEIN_DB_JOURNAL_MODE", config.journal_mode)
        config.synchronous_mode = os.getenv(
            "EINSTEIN_DB_SYNCHRONOUS", config.synchronous_mode
        )
        config.connection_timeout = ConfigLoader._get_float_env(
            "EINSTEIN_DB_TIMEOUT", config.connection_timeout
        )
        config.default_query_limit = ConfigLoader._get_int_env(
            "EINSTEIN_DB_QUERY_LIMIT", config.default_query_limit
        )
        config.max_query_limit = ConfigLoader._get_int_env(
            "EINSTEIN_DB_MAX_LIMIT", config.max_query_limit
        )

        return config

    @staticmethod
    def _load_monitoring_config() -> MonitoringConfig:
        """Load monitoring configuration with environment overrides."""
        config = MonitoringConfig()

        config.log_level = os.getenv("EINSTEIN_LOG_LEVEL", config.log_level).upper()
        config.debounce_delay_ms = ConfigLoader._get_float_env(
            "EINSTEIN_DEBOUNCE_MS", config.debounce_delay_ms
        )
        config.memory_check_interval_s = ConfigLoader._get_int_env(
            "EINSTEIN_MEMORY_CHECK_INTERVAL", config.memory_check_interval_s
        )

        return config

    @staticmethod
    def _get_int_env(key: str, default: int) -> int:
        """Get integer environment variable with default."""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid integer environment variable {key}={os.getenv(key)}, using default {default}: {e}",
                extra={
                    "operation": "get_int_env",
                    "error_type": type(e).__name__,
                    "env_key": key,
                    "env_value": os.getenv(key),
                    "default": default,
                    "expected_type": "integer",
                },
            )
            return default

    @staticmethod
    def _get_float_env(key: str, default: float) -> float:
        """Get float environment variable with default."""
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid float environment variable {key}={os.getenv(key)}, using default {default}: {e}",
                extra={
                    "operation": "get_float_env",
                    "error_type": type(e).__name__,
                    "env_key": key,
                    "env_value": os.getenv(key),
                    "default": default,
                    "expected_type": "float",
                },
            )
            return default

    @staticmethod
    def _get_bool_env(key: str, default: bool) -> bool:
        """Get boolean environment variable with default."""
        value = os.getenv(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        else:
            return default


# Global configuration instance
_config: EinsteinConfig | None = None


def get_einstein_config(project_root: Path | str | None = None) -> EinsteinConfig:
    """Get the global Einstein configuration."""
    global _config
    if _config is None:
        _config = ConfigLoader.load_config(project_root)
        logger.info(
            f"Einstein configuration loaded for {_config.hardware.platform_type} with {_config.hardware.cpu_cores} cores"
        )
    return _config


def reset_config():
    """Reset global configuration (useful for testing)."""
    global _config
    if _config and hasattr(_config, "close_database_managers"):
        _config.close_database_managers()
    _config = None


if __name__ == "__main__":
    # Test hardware detection and configuration loading
    print("🔍 Testing Einstein Configuration System...")

    config = get_einstein_config()

    print("\n💻 Hardware Configuration:")
    print(f"   Platform: {config.hardware.platform_type}")
    print(
        f"   CPU Cores: {config.hardware.cpu_cores} ({config.hardware.cpu_performance_cores}P + {config.hardware.cpu_efficiency_cores}E)"
    )
    print(
        f"   Memory: {config.hardware.memory_total_gb:.1f}GB total, {config.hardware.memory_available_gb:.1f}GB available"
    )
    print(
        f"   GPU: {'✅' if config.hardware.has_gpu else '❌'} ({config.hardware.gpu_cores} cores)"
    )
    print(
        f"   ANE: {'✅' if config.hardware.has_ane else '❌'} ({config.hardware.ane_cores} cores)"
    )
    print(f"   Architecture: {config.hardware.architecture}")

    print("\n⚡ Performance Configuration:")
    print(f"   Max startup time: {config.performance.max_startup_time_ms}ms")
    print(f"   Max search time: {config.performance.max_search_time_ms}ms")
    print(f"   Max memory usage: {config.performance.max_memory_usage_gb}GB")
    print(f"   Search concurrency: {config.performance.max_search_concurrency}")
    print(f"   File I/O concurrency: {config.performance.max_file_io_concurrency}")

    print("\n💾 Cache Configuration:")
    print(f"   Hot cache size: {config.cache.hot_cache_size}")
    print(f"   Warm cache size: {config.cache.warm_cache_size}")
    print(f"   Index cache: {config.cache.index_cache_size_mb}MB")

    print("\n📁 Path Configuration:")
    print(f"   Base directory: {config.paths.base_dir}")
    print(f"   Cache directory: {config.paths.cache_dir}")
    print(f"   Logs directory: {config.paths.logs_dir}")

    print("\n🧠 ML Configuration:")
    print(f"   Learning rate: {config.ml.adaptive_learning_rate}")
    print(f"   Embedding dimension: {config.ml.embedding_dimension}")
    print(f"   Similarity threshold: {config.ml.similarity_threshold}")
    print(f"   ANE enabled: {'✅' if config.ml.enable_ane else '❌'}")
    if config.ml.enable_ane:
        print(f"   ANE batch size: {config.ml.ane_batch_size}")
        print(f"   ANE cache: {config.ml.ane_cache_size_mb}MB")

    print("\n🔧 Feature Flags:")
    print(f"   GPU acceleration: {'✅' if config.enable_gpu_acceleration else '❌'}")
    print(
        f"   Adaptive concurrency: {'✅' if config.enable_adaptive_concurrency else '❌'}"
    )
    print(
        f"   Predictive prefetch: {'✅' if config.enable_predictive_prefetch else '❌'}"
    )
    print(
        f"   Memory optimization: {'✅' if config.enable_memory_optimization else '❌'}"
    )
    print(f"   Realtime indexing: {'✅' if config.enable_realtime_indexing else '❌'}")
