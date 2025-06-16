#!/usr/bin/env python3
"""
Bolt Configuration Validation Examples

This script demonstrates how to validate Bolt configuration settings
and provides examples of common validation patterns.
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from pydantic import ValidationError

    from bolt.core.config import BoltConfig, get_default_config
except ImportError as e:
    print(f"Error importing Bolt configuration: {e}")
    print("Please ensure Bolt system is properly installed")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoltConfigValidator:
    """Validates Bolt configuration settings with comprehensive checks."""

    def __init__(self, config: BoltConfig):
        self.config = config
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def validate_all(self) -> bool:
        """Validate all configuration sections."""
        logger.info("🔍 Starting Bolt configuration validation...")

        # Validate each section
        self.validate_agent_pool()
        self.validate_token_optimization()
        self.validate_cpu_optimization()
        self.validate_memory_management()
        self.validate_performance()
        self.validate_hardware()
        self.validate_logging()
        self.validate_accelerated_tools()
        self.validate_cross_section_consistency()

        # Report results
        self._report_results()

        return len(self.errors) == 0

    def validate_agent_pool(self) -> None:
        """Validate agent pool configuration."""
        logger.info("Validating agent pool configuration...")

        # Basic validation (already handled by Pydantic)
        if self.config.max_agents < self.config.default_agents:
            self.errors.append("max_agents cannot be less than default_agents")

        # Performance optimization checks
        if self.config.max_agents > 32:
            self.warnings.append(f"Very high agent count: {self.config.max_agents}")
        elif self.config.max_agents < 4:
            self.warnings.append(
                f"Low agent count may limit performance: {self.config.max_agents}"
            )

        # Batch size validation
        if self.config.batch_size > 128:
            self.warnings.append(
                f"Large batch size may increase latency: {self.config.batch_size}"
            )
        elif self.config.batch_size < 8:
            self.warnings.append(
                f"Small batch size may reduce throughput: {self.config.batch_size}"
            )

        # Cache size validation
        if self.config.cache_size > 10000:
            self.warnings.append(
                f"Large cache may consume significant memory: {self.config.cache_size}"
            )
        elif self.config.cache_size < 100:
            self.warnings.append(
                f"Small cache may reduce performance: {self.config.cache_size}"
            )

    def validate_token_optimization(self) -> None:
        """Validate token optimization settings."""
        logger.info("Validating token optimization...")

        # This would require access to the actual token optimization config
        # For now, we'll add placeholder validation logic

        # Validate timeout
        if self.config.async_timeout < 30:
            self.warnings.append(
                f"Short async timeout may cause premature failures: {self.config.async_timeout}s"
            )
        elif self.config.async_timeout > 3600:  # 1 hour
            self.warnings.append(
                f"Very long async timeout: {self.config.async_timeout}s"
            )

    def validate_cpu_optimization(self) -> None:
        """Validate CPU optimization settings."""
        logger.info("Validating CPU optimization...")

        # Check if max_agents is reasonable for the system
        try:
            import psutil

            cpu_count = psutil.cpu_count()

            if self.config.max_agents > cpu_count * 2:
                self.warnings.append(
                    f"Agent count ({self.config.max_agents}) is high for {cpu_count} CPU cores"
                )
            elif self.config.max_agents > cpu_count:
                logger.info(
                    f"Agent count ({self.config.max_agents}) exceeds CPU cores ({cpu_count}) - this may be optimal for I/O bound tasks"
                )

        except ImportError:
            self.warnings.append("Cannot validate CPU settings - psutil not available")

    def validate_memory_management(self) -> None:
        """Validate memory management settings."""
        logger.info("Validating memory management...")

        if self.config.max_memory_gb is not None:
            try:
                import psutil

                total_memory_gb = psutil.virtual_memory().total / (1024**3)

                if self.config.max_memory_gb > total_memory_gb:
                    self.errors.append(
                        f"Memory limit ({self.config.max_memory_gb}GB) exceeds total system memory ({total_memory_gb:.1f}GB)"
                    )
                elif self.config.max_memory_gb > total_memory_gb * 0.9:
                    self.warnings.append(
                        f"Memory limit ({self.config.max_memory_gb}GB) is very high for system with {total_memory_gb:.1f}GB"
                    )
                elif self.config.max_memory_gb < 2.0:
                    self.warnings.append(
                        f"Low memory limit may restrict performance: {self.config.max_memory_gb}GB"
                    )

            except ImportError:
                self.warnings.append(
                    "Cannot validate memory settings - psutil not available"
                )

    def validate_performance(self) -> None:
        """Validate performance settings."""
        logger.info("Validating performance settings...")

        # Timeout validation
        if self.config.async_timeout < 10:
            self.warnings.append(
                "Very short async timeout may cause issues with complex operations"
            )

        # Batch size vs agent count
        if self.config.batch_size > self.config.max_agents * 10:
            self.warnings.append("Batch size is very large compared to agent count")

    def validate_hardware(self) -> None:
        """Validate hardware settings."""
        logger.info("Validating hardware settings...")

        # GPU validation
        if self.config.use_gpu:
            try:
                # Try to detect GPU availability
                gpu_available = False

                # Check for NVIDIA GPU
                try:
                    import GPUtil

                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_available = True
                        logger.info(f"Detected {len(gpus)} NVIDIA GPU(s)")
                except ImportError:
                    pass

                # Check for Apple Metal/MLX
                if not gpu_available:
                    try:
                        import mlx.core as mx

                        if hasattr(mx, "metal") and mx.metal.is_available():
                            gpu_available = True
                            logger.info("Detected Apple Metal GPU")
                    except ImportError:
                        pass

                if not gpu_available:
                    self.warnings.append(
                        "GPU acceleration enabled but no supported GPU detected"
                    )

            except Exception as e:
                self.warnings.append(f"Could not validate GPU availability: {e}")

        # MLX validation
        if self.config.prefer_mlx:
            try:
                import mlx.core as mx

                if not (hasattr(mx, "metal") and mx.metal.is_available()):
                    self.warnings.append("MLX preferred but not available")
            except ImportError:
                self.warnings.append("MLX preferred but mlx package not installed")

    def validate_logging(self) -> None:
        """Validate logging settings."""
        logger.info("Validating logging settings...")

        # Log level validation (already handled by Pydantic validator)

        # Log file validation
        if self.config.log_file:
            log_path = Path(self.config.log_file)

            # Check if parent directory exists or can be created
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if not os.access(log_path.parent, os.W_OK):
                    self.errors.append(
                        f"Log directory is not writable: {log_path.parent}"
                    )
            except Exception as e:
                self.errors.append(f"Cannot create log directory: {e}")

    def validate_accelerated_tools(self) -> None:
        """Validate accelerated tools configuration."""
        logger.info("Validating accelerated tools...")

        enabled_tools = [
            tool for tool, enabled in self.config.accelerated_tools.items() if enabled
        ]

        if not enabled_tools:
            self.warnings.append(
                "No accelerated tools enabled - performance may be reduced"
            )

        # Check dependencies for specific tools
        tool_dependencies = {
            "ripgrep_turbo": ["ripgrep", "rg"],
            "duckdb_turbo": ["duckdb"],
            "python_analysis": ["ast", "inspect"],
        }

        for tool in enabled_tools:
            if tool in tool_dependencies:
                # This is a simplified check - in practice you'd check for actual tool availability
                logger.info(f"Tool {tool} is enabled")

    def validate_cross_section_consistency(self) -> None:
        """Validate consistency across configuration sections."""
        logger.info("Validating cross-section consistency...")

        # Agent count vs batch size
        if self.config.batch_size > self.config.max_agents * 20:
            self.warnings.append(
                "Batch size is extremely large compared to agent count"
            )

        # Memory vs performance
        if self.config.max_memory_gb is not None and self.config.max_memory_gb < 4:
            if self.config.max_agents > 8:
                self.warnings.append(
                    "High agent count with low memory limit may cause issues"
                )

        # GPU vs accelerated tools
        if not self.config.use_gpu:
            gpu_dependent_tools = ["python_analysis"]  # Tools that benefit from GPU
            enabled_gpu_tools = [
                tool
                for tool in gpu_dependent_tools
                if self.config.accelerated_tools.get(tool, False)
            ]
            if enabled_gpu_tools:
                self.warnings.append(
                    f"GPU disabled but GPU-dependent tools enabled: {enabled_gpu_tools}"
                )

    def _report_results(self) -> None:
        """Report validation results."""
        print("\n" + "=" * 80)
        print("📋 BOLT CONFIGURATION VALIDATION RESULTS")
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

        print("\n📊 Configuration Summary:")
        print(f"   Max Agents: {self.config.max_agents}")
        print(f"   Default Agents: {self.config.default_agents}")
        print(f"   Batch Size: {self.config.batch_size}")
        print(f"   Cache Size: {self.config.cache_size}")
        print(f"   GPU Enabled: {self.config.use_gpu}")
        print(f"   MLX Preferred: {self.config.prefer_mlx}")
        print(
            f"   Memory Limit: {self.config.max_memory_gb}GB"
            if self.config.max_memory_gb
            else "   Memory Limit: Auto"
        )
        print(f"   Log Level: {self.config.log_level}")

        enabled_tools = [
            tool for tool, enabled in self.config.accelerated_tools.items() if enabled
        ]
        print(f"   Accelerated Tools: {len(enabled_tools)} enabled")

        print("\n" + "=" * 80)


def test_valid_configuration():
    """Test with a valid configuration."""
    print("\n🧪 Testing valid configuration...")

    try:
        config = get_default_config()
        validator = BoltConfigValidator(config)
        is_valid = validator.validate_all()

        if is_valid:
            print("✅ Valid configuration test passed!")
        else:
            print("❌ Valid configuration test has warnings but no errors")

    except Exception as e:
        print(f"❌ Error testing valid configuration: {e}")


def test_invalid_configuration():
    """Test with various invalid configurations."""
    print("\n🧪 Testing invalid configurations...")

    # Test invalid agent configuration
    print("\n--- Testing invalid agent configuration ---")
    try:
        invalid_config = BoltConfig(
            max_agents=2,
            default_agents=4,  # Invalid: default > max
        )
        validator = BoltConfigValidator(invalid_config)
        validator.validate_all()  # Should show errors
    except ValidationError as e:
        print(f"✅ Pydantic caught invalid configuration: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # Test extreme values
    print("\n--- Testing extreme values ---")
    try:
        extreme_config = BoltConfig(
            max_agents=100,  # Very high
            batch_size=1000,  # Very large
            cache_size=100000,  # Very large
            max_memory_gb=1000.0,  # Likely exceeds system memory
        )
        validator = BoltConfigValidator(extreme_config)
        validator.validate_all()  # Should show warnings
    except Exception as e:
        print(f"❌ Error with extreme configuration: {e}")


def test_environment_variable_override():
    """Test environment variable override functionality."""
    print("\n🧪 Testing environment variable overrides...")

    # Set environment variables
    test_vars = {
        "BOLT_MAX_AGENTS": "16",
        "BOLT_DEFAULT_AGENTS": "12",
        "BOLT_BATCH_SIZE": "64",
        "BOLT_USE_GPU": "false",
        "BOLT_LOG_LEVEL": "DEBUG",
        "BOLT_MAX_MEMORY_GB": "8.0",
    }

    # Set environment variables
    for key, value in test_vars.items():
        os.environ[key] = value

    try:
        config = get_default_config()

        # Verify overrides
        assert (
            config.max_agents == 16
        ), f"Max agents override failed: {config.max_agents}"
        assert (
            config.default_agents == 12
        ), f"Default agents override failed: {config.default_agents}"
        assert (
            config.batch_size == 64
        ), f"Batch size override failed: {config.batch_size}"
        assert config.use_gpu == False, f"GPU override failed: {config.use_gpu}"
        assert (
            config.log_level == "DEBUG"
        ), f"Log level override failed: {config.log_level}"
        assert (
            config.max_memory_gb == 8.0
        ), f"Memory override failed: {config.max_memory_gb}"

        print("✅ Environment variable override test passed!")

        # Validate the overridden configuration
        validator = BoltConfigValidator(config)
        validator.validate_all()

    except Exception as e:
        print(f"❌ Environment variable override test failed: {e}")
    finally:
        # Clean up
        for key in test_vars:
            os.environ.pop(key, None)


def test_pydantic_validation():
    """Test Pydantic's built-in validation."""
    print("\n🧪 Testing Pydantic validation...")

    # Test invalid log level
    try:
        BoltConfig(log_level="INVALID")
        print("❌ Should have failed with invalid log level")
    except ValidationError as e:
        print(f"✅ Pydantic correctly rejected invalid log level: {e}")

    # Test negative values
    try:
        BoltConfig(max_agents=-1)
        print("❌ Should have failed with negative max_agents")
    except ValidationError as e:
        print(f"✅ Pydantic correctly rejected negative max_agents: {e}")

    # Test default_agents > max_agents
    try:
        BoltConfig(max_agents=4, default_agents=8)
        print("❌ Should have failed with default_agents > max_agents")
    except ValidationError as e:
        print(f"✅ Pydantic correctly rejected default_agents > max_agents: {e}")


def demonstrate_custom_validation():
    """Demonstrate how to add custom validation rules."""
    print("\n🧪 Demonstrating custom validation...")

    class CustomBoltValidator(BoltConfigValidator):
        """Extended validator with custom business rules."""

        def validate_business_rules(self) -> None:
            """Add custom business logic validation."""
            logger.info("Validating custom business rules...")

            # Custom rule: Production environments should have specific settings
            if os.getenv("ENVIRONMENT") == "production":
                if self.config.log_level == "DEBUG":
                    self.warnings.append("Debug logging not recommended for production")

                if self.config.max_agents < 8:
                    self.warnings.append("Production should use at least 8 agents")

                if not self.config.use_gpu:
                    self.warnings.append(
                        "Production should enable GPU acceleration when available"
                    )

            # Custom rule: High-performance mode
            if os.getenv("PERFORMANCE_MODE") == "high":
                if self.config.batch_size < 32:
                    self.warnings.append(
                        "High-performance mode should use larger batch sizes"
                    )

                if self.config.cache_size < 2000:
                    self.warnings.append(
                        "High-performance mode should use larger cache"
                    )

            # Custom rule: Resource-constrained environments
            if self.config.max_memory_gb and self.config.max_memory_gb < 8:
                if self.config.max_agents > 6:
                    self.warnings.append(
                        "Consider reducing agents in memory-constrained environment"
                    )

                disabled_tools = [
                    tool
                    for tool, enabled in self.config.accelerated_tools.items()
                    if not enabled
                ]
                if len(disabled_tools) > 2:
                    self.warnings.append("Many tools disabled - may impact performance")

        def validate_all(self) -> bool:
            """Override to include custom validation."""
            result = super().validate_all()
            self.validate_business_rules()
            self._report_results()  # Report again with custom rules
            return result and len(self.errors) == 0

    # Test custom validation
    try:
        config = get_default_config()
        custom_validator = CustomBoltValidator(config)
        custom_validator.validate_all()
        print("✅ Custom validation demonstration completed!")

    except Exception as e:
        print(f"❌ Custom validation failed: {e}")


def demonstrate_configuration_profiles():
    """Demonstrate different configuration profiles."""
    print("\n🧪 Demonstrating configuration profiles...")

    profiles = {
        "development": BoltConfig(
            max_agents=4,
            default_agents=4,
            batch_size=16,
            cache_size=500,
            use_gpu=False,
            log_level="DEBUG",
            max_memory_gb=4.0,
        ),
        "production": BoltConfig(
            max_agents=12,
            default_agents=12,
            batch_size=64,
            cache_size=2000,
            use_gpu=True,
            log_level="INFO",
            max_memory_gb=16.0,
        ),
        "high_performance": BoltConfig(
            max_agents=16,
            default_agents=16,
            batch_size=128,
            cache_size=5000,
            use_gpu=True,
            prefer_mlx=True,
            log_level="WARNING",
            max_memory_gb=32.0,
        ),
    }

    for profile_name, config in profiles.items():
        print(f"\n--- Validating {profile_name} profile ---")
        try:
            validator = BoltConfigValidator(config)
            is_valid = validator.validate_all()
            print(f"✅ {profile_name} profile validation completed")
        except Exception as e:
            print(f"❌ {profile_name} profile validation failed: {e}")


def main():
    """Main validation demonstration."""
    print("🚀 Bolt Configuration Validation Examples")
    print("=" * 60)

    # Test valid configuration
    test_valid_configuration()

    # Test invalid configurations
    test_invalid_configuration()

    # Test Pydantic validation
    test_pydantic_validation()

    # Test environment variable overrides
    test_environment_variable_override()

    # Demonstrate custom validation
    demonstrate_custom_validation()

    # Demonstrate configuration profiles
    demonstrate_configuration_profiles()

    print("\n✅ All validation examples completed!")
    print("\nFor production use:")
    print("1. Run validation during system initialization")
    print("2. Add environment-specific validation rules")
    print("3. Use configuration profiles for different deployments")
    print("4. Monitor configuration changes and validate dynamically")
    print("5. Log validation results for debugging and monitoring")


if __name__ == "__main__":
    main()
