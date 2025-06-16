#!/usr/bin/env python3
"""
Einstein Configuration Validation Examples

This script demonstrates how to validate Einstein configuration settings
and provides examples of common validation patterns.
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from einstein.einstein_config import (
    EinsteinConfig,
    get_einstein_config,
    reset_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EinsteinConfigValidator:
    """Validates Einstein configuration settings."""

    def __init__(self, config: EinsteinConfig):
        self.config = config
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def validate_all(self) -> bool:
        """Validate all configuration sections."""
        logger.info("🔍 Starting Einstein configuration validation...")

        # Validate each section
        self.validate_hardware()
        self.validate_performance()
        self.validate_cache()
        self.validate_paths()
        self.validate_ml()
        self.validate_monitoring()
        self.validate_feature_flags()
        self.validate_cross_section_consistency()

        # Report results
        self._report_results()

        return len(self.errors) == 0

    def validate_hardware(self) -> None:
        """Validate hardware configuration."""
        logger.info("Validating hardware configuration...")
        hw = self.config.hardware

        # CPU validation
        if hw.cpu_cores <= 0:
            self.errors.append("CPU cores must be positive")
        elif hw.cpu_cores > 32:
            self.warnings.append(f"Very high CPU core count: {hw.cpu_cores}")

        if hw.cpu_performance_cores + hw.cpu_efficiency_cores != hw.cpu_cores:
            total_specified = hw.cpu_performance_cores + hw.cpu_efficiency_cores
            if total_specified != hw.cpu_cores:
                self.warnings.append(
                    f"P-cores ({hw.cpu_performance_cores}) + E-cores ({hw.cpu_efficiency_cores}) "
                    f"= {total_specified} doesn't match total cores ({hw.cpu_cores})"
                )

        # Memory validation
        if hw.memory_total_gb <= 0:
            self.errors.append("Total memory must be positive")
        elif hw.memory_total_gb < 4:
            self.warnings.append(f"Low memory system: {hw.memory_total_gb:.1f}GB")

        if hw.memory_available_gb > hw.memory_total_gb:
            self.errors.append("Available memory cannot exceed total memory")
        elif hw.memory_available_gb < hw.memory_total_gb * 0.5:
            self.warnings.append("Available memory is less than 50% of total")

        # GPU validation
        if hw.has_gpu and hw.gpu_cores <= 0:
            self.warnings.append("GPU detected but no GPU cores specified")
        elif not hw.has_gpu and hw.gpu_cores > 0:
            self.warnings.append("GPU cores specified but no GPU detected")

        # ANE validation
        if hw.has_ane and hw.ane_cores <= 0:
            self.warnings.append("ANE detected but no ANE cores specified")
        elif not hw.has_ane and hw.ane_cores > 0:
            self.warnings.append("ANE cores specified but no ANE detected")

        # Platform consistency
        if hw.platform_type == "apple_silicon" and hw.architecture != "arm64":
            self.warnings.append("Apple Silicon platform should use arm64 architecture")

    def validate_performance(self) -> None:
        """Validate performance configuration."""
        logger.info("Validating performance configuration...")
        perf = self.config.performance

        # Timing validation
        if perf.max_startup_time_ms <= 0:
            self.errors.append("Startup time must be positive")
        elif perf.max_startup_time_ms > 10000:  # 10 seconds
            self.warnings.append(
                f"Very long startup timeout: {perf.max_startup_time_ms}ms"
            )

        if perf.max_search_time_ms <= 0:
            self.errors.append("Search time must be positive")
        elif perf.max_search_time_ms > 5000:  # 5 seconds
            self.warnings.append(
                f"Very long search timeout: {perf.max_search_time_ms}ms"
            )

        # Memory validation
        if perf.max_memory_usage_gb <= 0:
            self.errors.append("Memory usage limit must be positive")
        elif perf.max_memory_usage_gb > self.config.hardware.memory_available_gb:
            self.errors.append(
                f"Memory limit ({perf.max_memory_usage_gb}GB) exceeds available memory "
                f"({self.config.hardware.memory_available_gb}GB)"
            )

        # Concurrency validation
        max_cores = self.config.hardware.cpu_cores
        if perf.max_search_concurrency > max_cores:
            self.warnings.append(
                f"Search concurrency ({perf.max_search_concurrency}) exceeds CPU cores ({max_cores})"
            )

        if perf.max_file_io_concurrency > max_cores * 2:
            self.warnings.append(
                f"File I/O concurrency ({perf.max_file_io_concurrency}) is very high for {max_cores} cores"
            )

    def validate_cache(self) -> None:
        """Validate cache configuration."""
        logger.info("Validating cache configuration...")
        cache = self.config.cache

        # Cache size validation
        if cache.hot_cache_size <= 0:
            self.errors.append("Hot cache size must be positive")
        if cache.warm_cache_size <= 0:
            self.errors.append("Warm cache size must be positive")
        if cache.hot_cache_size > cache.warm_cache_size:
            self.warnings.append("Hot cache is larger than warm cache")

        # Memory validation
        if cache.index_cache_size_mb <= 0:
            self.errors.append("Index cache size must be positive")
        elif cache.index_cache_size_mb > 2048:  # 2GB
            self.warnings.append(
                f"Very large index cache: {cache.index_cache_size_mb}MB"
            )

        # Policy validation
        if cache.cache_ttl_seconds <= 0:
            self.errors.append("Cache TTL must be positive")
        elif cache.cache_ttl_seconds < 60:  # 1 minute
            self.warnings.append(f"Very short cache TTL: {cache.cache_ttl_seconds}s")

    def validate_paths(self) -> None:
        """Validate path configuration."""
        logger.info("Validating path configuration...")
        paths = self.config.paths

        # Directory existence and permissions
        directories_to_check = [
            ("base_dir", paths.base_dir),
            ("cache_dir", paths.cache_dir),
            ("logs_dir", paths.logs_dir),
            ("rapid_cache_dir", paths.rapid_cache_dir),
            ("optimized_cache_dir", paths.optimized_cache_dir),
            ("models_dir", paths.models_dir),
        ]

        for name, path in directories_to_check:
            if not path.exists():
                self.warnings.append(f"{name} does not exist: {path}")
            elif not path.is_dir():
                self.errors.append(f"{name} is not a directory: {path}")
            elif not os.access(path, os.W_OK):
                self.errors.append(f"{name} is not writable: {path}")

        # Database files
        db_files = [
            ("analytics_db", paths.analytics_db_path),
            ("embeddings_db", paths.embeddings_db_path),
        ]

        for name, db_path in db_files:
            if db_path.exists() and not os.access(db_path, os.R_OK | os.W_OK):
                self.errors.append(f"{name} is not readable/writable: {db_path}")

        # Database concurrency settings
        if paths.max_db_connections <= 0:
            self.errors.append("Maximum database connections must be positive")
        elif paths.max_db_connections > 100:
            self.warnings.append(
                f"Very high database connection limit: {paths.max_db_connections}"
            )

        if paths.db_connection_timeout <= 0:
            self.errors.append("Database connection timeout must be positive")
        if paths.db_lock_timeout <= 0:
            self.errors.append("Database lock timeout must be positive")

    def validate_ml(self) -> None:
        """Validate ML configuration."""
        logger.info("Validating ML configuration...")
        ml = self.config.ml

        # Learning parameters
        if not (0 < ml.adaptive_learning_rate <= 1):
            self.errors.append("Learning rate must be between 0 and 1")
        if not (0 < ml.bandit_exploration_rate <= 1):
            self.errors.append("Exploration rate must be between 0 and 1")

        # Model parameters
        if ml.embedding_dimension <= 0:
            self.errors.append("Embedding dimension must be positive")
        elif ml.embedding_dimension not in [384, 512, 768, 1024, 1536, 2048]:
            self.warnings.append(
                f"Unusual embedding dimension: {ml.embedding_dimension}"
            )

        if ml.max_sequence_length <= 0:
            self.errors.append("Max sequence length must be positive")
        elif ml.max_sequence_length > 2048:
            self.warnings.append(f"Very long sequence length: {ml.max_sequence_length}")

        # Thresholds
        if not (0 <= ml.similarity_threshold <= 1):
            self.errors.append("Similarity threshold must be between 0 and 1")
        if not (0 <= ml.confidence_threshold <= 1):
            self.errors.append("Confidence threshold must be between 0 and 1")
        if not (0 <= ml.relevance_threshold <= 1):
            self.errors.append("Relevance threshold must be between 0 and 1")

        # ANE validation
        if ml.enable_ane and not self.config.hardware.has_ane:
            self.warnings.append("ANE enabled but not detected in hardware")
        if ml.ane_batch_size <= 0:
            self.errors.append("ANE batch size must be positive")
        if ml.ane_cache_size_mb <= 0:
            self.errors.append("ANE cache size must be positive")

    def validate_monitoring(self) -> None:
        """Validate monitoring configuration."""
        logger.info("Validating monitoring configuration...")
        mon = self.config.monitoring

        # Timing validation
        if mon.debounce_delay_ms <= 0:
            self.errors.append("Debounce delay must be positive")
        elif mon.debounce_delay_ms > 5000:  # 5 seconds
            self.warnings.append(f"Very long debounce delay: {mon.debounce_delay_ms}ms")

        # Interval validation
        if mon.file_watch_interval_s <= 0:
            self.errors.append("File watch interval must be positive")
        if mon.memory_check_interval_s <= 0:
            self.errors.append("Memory check interval must be positive")
        if mon.gc_interval_s <= 0:
            self.errors.append("GC interval must be positive")

        # History size validation
        if mon.performance_history_size <= 0:
            self.errors.append("Performance history size must be positive")
        if mon.system_load_history_size <= 0:
            self.errors.append("System load history size must be positive")

        # Log level validation
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if mon.log_level not in valid_levels:
            self.errors.append(
                f"Invalid log level: {mon.log_level}. Must be one of {valid_levels}"
            )

    def validate_feature_flags(self) -> None:
        """Validate feature flag combinations."""
        logger.info("Validating feature flags...")

        # GPU acceleration
        if self.config.enable_gpu_acceleration and not self.config.hardware.has_gpu:
            self.warnings.append("GPU acceleration enabled but no GPU detected")

        # Memory optimization dependencies
        if (
            self.config.enable_memory_optimization
            and not self.config.enable_adaptive_concurrency
        ):
            self.warnings.append(
                "Memory optimization works best with adaptive concurrency"
            )

    def validate_cross_section_consistency(self) -> None:
        """Validate consistency across configuration sections."""
        logger.info("Validating cross-section consistency...")

        # Memory consistency
        total_cache_memory = (
            self.config.cache.index_cache_size_mb
            + self.config.ml.ane_cache_size_mb
            + self.config.performance.cache_memory_limit_mb
        )

        max_memory_mb = self.config.performance.max_memory_usage_gb * 1024
        if total_cache_memory > max_memory_mb * 0.8:  # 80% threshold
            self.warnings.append(
                f"Total cache memory ({total_cache_memory}MB) is high relative to "
                f"max memory usage ({max_memory_mb}MB)"
            )

        # Concurrency vs hardware
        total_concurrency = (
            self.config.performance.max_search_concurrency
            + self.config.performance.max_embedding_concurrency
            + self.config.performance.max_file_io_concurrency
            + self.config.performance.max_analysis_concurrency
        )

        if total_concurrency > self.config.hardware.cpu_cores * 3:
            self.warnings.append(
                f"Total concurrency ({total_concurrency}) is very high for "
                f"{self.config.hardware.cpu_cores} CPU cores"
            )

    def _report_results(self) -> None:
        """Report validation results."""
        print("\n" + "=" * 80)
        print("📋 EINSTEIN CONFIGURATION VALIDATION RESULTS")
        print("=" * 80)

        if not self.errors and not self.warnings:
            print("✅ Configuration is valid with no issues!")
        else:
            if self.errors:
                print(f"\n❌ ERRORS ({len(self.errors)}):")
                for i, error in enumerate(self.errors, 1):
                    print(f"   {i}. {error}")

            if self.warnings:
                print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
                for i, warning in enumerate(self.warnings, 1):
                    print(f"   {i}. {warning}")

        print("\n" + "=" * 80)


def test_valid_configuration():
    """Test with a valid configuration."""
    print("\n🧪 Testing valid configuration...")

    # Reset any existing config
    reset_config()

    # Load default configuration
    try:
        config = get_einstein_config()
        validator = EinsteinConfigValidator(config)
        is_valid = validator.validate_all()

        if is_valid:
            print("✅ Valid configuration test passed!")
        else:
            print("❌ Valid configuration test failed!")

    except Exception as e:
        print(f"❌ Error testing valid configuration: {e}")


def test_invalid_configuration():
    """Test with various invalid configurations."""
    print("\n🧪 Testing invalid configurations...")

    # Reset any existing config
    reset_config()

    # Test invalid memory settings
    print("\n--- Testing invalid memory settings ---")
    os.environ["EINSTEIN_MAX_MEMORY_GB"] = "-1"
    os.environ[
        "EINSTEIN_PERFORMANCE__MAX_MEMORY_USAGE_GB"
    ] = "100"  # More than available

    try:
        config = get_einstein_config()
        validator = EinsteinConfigValidator(config)
        validator.validate_all()
    except Exception as e:
        print(f"Expected error caught: {e}")
    finally:
        # Clean up
        os.environ.pop("EINSTEIN_MAX_MEMORY_GB", None)
        os.environ.pop("EINSTEIN_PERFORMANCE__MAX_MEMORY_USAGE_GB", None)
        reset_config()

    # Test invalid concurrency settings
    print("\n--- Testing invalid concurrency settings ---")
    os.environ["EINSTEIN_SEARCH_CONCURRENCY"] = "0"
    os.environ["EINSTEIN_FILE_IO_CONCURRENCY"] = "1000"

    try:
        config = get_einstein_config()
        validator = EinsteinConfigValidator(config)
        validator.validate_all()
    except Exception as e:
        print(f"Expected error caught: {e}")
    finally:
        # Clean up
        os.environ.pop("EINSTEIN_SEARCH_CONCURRENCY", None)
        os.environ.pop("EINSTEIN_FILE_IO_CONCURRENCY", None)
        reset_config()

    # Test invalid ML settings
    print("\n--- Testing invalid ML settings ---")
    os.environ["EINSTEIN_SIMILARITY_THRESHOLD"] = "2.0"  # > 1.0
    os.environ["EINSTEIN_ML__EMBEDDING_DIMENSION"] = "-100"

    try:
        config = get_einstein_config()
        validator = EinsteinConfigValidator(config)
        validator.validate_all()
    except Exception as e:
        print(f"Expected error caught: {e}")
    finally:
        # Clean up
        os.environ.pop("EINSTEIN_SIMILARITY_THRESHOLD", None)
        os.environ.pop("EINSTEIN_ML__EMBEDDING_DIMENSION", None)
        reset_config()


def test_environment_variable_override():
    """Test environment variable override functionality."""
    print("\n🧪 Testing environment variable overrides...")

    # Reset any existing config
    reset_config()

    # Set environment variables
    test_vars = {
        "EINSTEIN_LOG_LEVEL": "DEBUG",
        "EINSTEIN_HOT_CACHE_SIZE": "2000",
        "EINSTEIN_MAX_STARTUP_MS": "1000",
        "EINSTEIN_ENABLE_GPU_ACCELERATION": "false",
    }

    # Set environment variables
    for key, value in test_vars.items():
        os.environ[key] = value

    try:
        config = get_einstein_config()

        # Verify overrides
        assert config.monitoring.log_level == "DEBUG", "Log level override failed"
        assert config.cache.hot_cache_size == 2000, "Cache size override failed"
        assert (
            config.performance.max_startup_time_ms == 1000.0
        ), "Startup time override failed"
        assert (
            config.enable_gpu_acceleration == False
        ), "GPU acceleration override failed"

        print("✅ Environment variable override test passed!")

        # Validate the overridden configuration
        validator = EinsteinConfigValidator(config)
        validator.validate_all()

    except Exception as e:
        print(f"❌ Environment variable override test failed: {e}")
    finally:
        # Clean up
        for key in test_vars:
            os.environ.pop(key, None)
        reset_config()


def demonstrate_custom_validation():
    """Demonstrate how to add custom validation rules."""
    print("\n🧪 Demonstrating custom validation...")

    class CustomEinsteinValidator(EinsteinConfigValidator):
        """Extended validator with custom business rules."""

        def validate_business_rules(self) -> None:
            """Add custom business logic validation."""
            logger.info("Validating custom business rules...")

            # Custom rule: For production, require higher memory limits
            if os.getenv("ENVIRONMENT") == "production":
                if self.config.performance.max_memory_usage_gb < 4.0:
                    self.errors.append(
                        "Production environment requires at least 4GB memory limit"
                    )

                if self.config.monitoring.log_level == "DEBUG":
                    self.warnings.append("Debug logging not recommended for production")

            # Custom rule: High-performance systems should use larger caches
            if self.config.hardware.cpu_cores >= 16:
                if self.config.cache.hot_cache_size < 2000:
                    self.warnings.append(
                        "High-performance systems should use larger hot cache"
                    )

            # Custom rule: Validate ANE usage makes sense
            if self.config.ml.enable_ane and self.config.ml.ane_batch_size < 128:
                self.warnings.append("ANE works best with larger batch sizes (>=128)")

        def validate_all(self) -> bool:
            """Override to include custom validation."""
            result = super().validate_all()
            self.validate_business_rules()
            self._report_results()  # Report again with custom rules
            return result and len(self.errors) == 0

    # Test custom validation
    reset_config()

    try:
        config = get_einstein_config()
        custom_validator = CustomEinsteinValidator(config)
        custom_validator.validate_all()
        print("✅ Custom validation demonstration completed!")

    except Exception as e:
        print(f"❌ Custom validation failed: {e}")


def main():
    """Main validation demonstration."""
    print("🚀 Einstein Configuration Validation Examples")
    print("=" * 60)

    # Test valid configuration
    test_valid_configuration()

    # Test invalid configurations
    test_invalid_configuration()

    # Test environment variable overrides
    test_environment_variable_override()

    # Demonstrate custom validation
    demonstrate_custom_validation()

    print("\n✅ All validation examples completed!")
    print("\nFor production use:")
    print("1. Run validation before system startup")
    print("2. Add custom business rules as needed")
    print("3. Log validation results for monitoring")
    print("4. Consider failing fast on critical errors")


if __name__ == "__main__":
    main()
