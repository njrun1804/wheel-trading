#!/usr/bin/env python3
"""
Unified Configuration Examples and Validation Tests

Demonstrates:
1. Loading unified configuration
2. System-specific configuration extraction
3. Environment variable overrides
4. Configuration validation
5. File-based configuration
6. Runtime configuration updates
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

# Import unified configuration system
from src.unity_wheel.config.unified_config import (
    UnifiedConfig,
    ConfigLoader,
    get_unified_config,
    get_einstein_config,
    get_bolt_config,
    get_jarvis_config,
    get_meta_config,
    reset_config,
)
from src.unity_wheel.config.validation import (
    validate_unified_config,
    validate_system_config,
    validate_api_keys,
)
from src.unity_wheel.config.environment_loader import EnvironmentLoader


def example_basic_usage():
    """Basic unified configuration usage."""
    print("🔧 Basic Unified Configuration Usage")
    print("=" * 50)
    
    # Load unified configuration
    config = get_unified_config()
    
    print(f"Config Version: {config.config_version}")
    print(f"Hardware Platform: {config.hardware.platform_type}")
    print(f"CPU Cores: {config.hardware.cpu_cores}")
    print(f"Memory: {config.hardware.memory_total_gb:.1f}GB")
    print(f"GPU Available: {'✅' if config.hardware.has_gpu else '❌'}")
    
    # Validate configuration
    result = validate_unified_config(config)
    print(f"\nValidation: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    print()


def example_system_configs():
    """Extract system-specific configurations."""
    print("🎯 System-Specific Configuration Extraction")
    print("=" * 50)
    
    systems = ["einstein", "bolt", "jarvis", "meta"]
    
    for system in systems:
        try:
            if system == "einstein":
                config = get_einstein_config()
            elif system == "bolt":
                config = get_bolt_config()
            elif system == "jarvis":
                config = get_jarvis_config()
            elif system == "meta":
                config = get_meta_config()
            
            print(f"\n{system.title()} Configuration:")
            print(f"  Settings: {len(config)} items")
            
            # Show sample settings
            if isinstance(config, dict):
                sample_keys = list(config.keys())[:3]
                for key in sample_keys:
                    value = config[key]
                    if isinstance(value, dict):
                        print(f"  {key}: {len(value)} sub-items")
                    else:
                        print(f"  {key}: {value}")
            
            # Validate system-specific config
            result = validate_system_config(config if isinstance(config, dict) else config.__dict__, system)
            print(f"  Validation: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
            
        except Exception as e:
            print(f"{system.title()}: ❌ Error - {e}")
    
    print()


def example_environment_overrides():
    """Demonstrate environment variable overrides."""
    print("🌍 Environment Variable Overrides")
    print("=" * 50)
    
    # Test environment variables
    test_env = {
        "EINSTEIN_MAX_MEMORY_GB": "8.0",
        "EINSTEIN_LOG_LEVEL": "DEBUG",
        "BOLT_MAX_AGENTS": "4",
        "BOLT_USE_GPU": "false",
        "JARVIS_EMBEDDING_DIM": "512",
        "META_DEBUG_MODE": "true",
    }
    
    print("Setting test environment variables:")
    for key, value in test_env.items():
        print(f"  {key} = {value}")
    
    with patch.dict(os.environ, test_env):
        # Reset and reload configuration
        reset_config()
        config = get_unified_config()
        
        print("\nConfiguration after environment overrides:")
        print(f"  Max Memory: {config.performance.max_memory_usage_gb}GB")
        print(f"  Log Level: {config.monitoring.log_level}")
        print(f"  Embedding Dimension: {config.ml.embedding_dimension}")
        
        # Check system-specific configs
        einstein_config = get_einstein_config()
        bolt_config = get_bolt_config()
        
        print(f"\nEinstein Config:")
        print(f"  Max Memory: {einstein_config.get('performance', {}).get('max_memory_usage_gb', 'N/A')}")
        print(f"  Log Level: {einstein_config.get('monitoring', {}).get('log_level', 'N/A')}")
        
        print(f"\nBolt Config:")
        print(f"  Max Agents: {bolt_config.get('max_agents', 'N/A')}")
        print(f"  Use GPU: {bolt_config.get('use_gpu', 'N/A')}")
    
    # Reset to clean state
    reset_config()
    print()


def example_file_configuration():
    """Demonstrate file-based configuration."""
    print("📁 File-Based Configuration")
    print("=" * 50)
    
    # Create example configuration
    example_config = {
        "config_version": "2.0.0",
        "systems_enabled": {
            "einstein": True,
            "bolt": True,
            "jarvis": False,
            "meta": True,
        },
        "performance": {
            "max_memory_usage_gb": 4.0,
            "max_search_time_ms": 30.0,
            "async_timeout": 120.0,
        },
        "cache": {
            "hot_cache_size": 500,
            "warm_cache_size": 2000,
            "index_cache_size_mb": 128,
        },
        "ml": {
            "embedding_dimension": 512,
            "batch_size": 16,
            "enable_gpu": False,
            "enable_ane": True,
        },
        "monitoring": {
            "log_level": "INFO",
            "monitoring_interval": 2.0,
            "enable_performance_logs": True,
        },
        "timing": {
            "coordination_cycle_seconds": 10,
            "health_check_interval_seconds": 60,
        },
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "example_config.yaml"
        
        # Save configuration to file
        config = get_unified_config()
        config.save_to_file(config_file)
        print(f"Saved config to: {config_file}")
        
        # Load configuration from file
        loaded_config = UnifiedConfig.load_from_file(config_file)
        print(f"Loaded config version: {loaded_config.config_version}")
        
        # Validate loaded configuration
        result = validate_unified_config(loaded_config)
        print(f"Validation: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
    
    print()


def example_validation_scenarios():
    """Demonstrate various validation scenarios."""
    print("✅ Configuration Validation Scenarios")
    print("=" * 50)
    
    # Valid configuration
    valid_config = {
        "hardware": {
            "cpu_cores": 8,
            "memory_total_gb": 16.0,
            "platform_type": "apple_silicon",
        },
        "performance": {
            "max_memory_usage_gb": 12.0,
            "max_search_time_ms": 50.0,
            "max_search_concurrency": 4,
        },
        "ml": {
            "learning_rate": 0.001,
            "embedding_dimension": 768,
            "batch_size": 32,
        },
    }
    
    print("1. Valid Configuration:")
    result = validate_system_config(valid_config, "einstein")
    print(f"   Result: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
    
    # Invalid configuration - memory exceeds hardware
    invalid_config = {
        "hardware": {
            "cpu_cores": 4,
            "memory_total_gb": 8.0,
            "platform_type": "intel",
        },
        "performance": {
            "max_memory_usage_gb": 12.0,  # Exceeds hardware memory
            "max_search_time_ms": -10.0,  # Invalid negative value
            "max_search_concurrency": 0,  # Invalid zero value
        },
        "ml": {
            "learning_rate": 2.0,  # Invalid > 1
            "embedding_dimension": -100,  # Invalid negative
            "batch_size": 0,  # Invalid zero
        },
    }
    
    print("\n2. Invalid Configuration:")
    result = validate_system_config(invalid_config, "einstein")
    print(f"   Result: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
    if result.errors:
        print("   Errors:")
        for error in result.errors[:3]:  # Show first 3 errors
            print(f"     - {error}")
        if len(result.errors) > 3:
            print(f"     ... and {len(result.errors) - 3} more errors")
    
    # Configuration with warnings
    warning_config = {
        "hardware": {
            "cpu_cores": 64,  # Very high
            "memory_total_gb": 2.0,  # Low memory
            "platform_type": "custom_platform",  # Unknown platform
            "has_gpu": False,
        },
        "performance": {
            "max_memory_usage_gb": 1.8,
            "max_search_concurrency": 128,  # Very high
        },
        "ml": {
            "enable_gpu": True,  # GPU enabled but not available
            "batch_size": 2048,  # Very large
        },
    }
    
    print("\n3. Configuration with Warnings:")
    result = validate_system_config(warning_config, "bolt")
    print(f"   Result: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
    if result.warnings:
        print("   Warnings:")
        for warning in result.warnings[:3]:  # Show first 3 warnings
            print(f"     - {warning}")
        if len(result.warnings) > 3:
            print(f"     ... and {len(result.warnings) - 3} more warnings")
    
    print()


def example_runtime_updates():
    """Demonstrate runtime configuration updates."""
    print("🔄 Runtime Configuration Updates")
    print("=" * 50)
    
    # Get initial configuration
    config = get_unified_config()
    print(f"Initial max memory: {config.performance.max_memory_usage_gb}GB")
    print(f"Initial log level: {config.monitoring.log_level}")
    
    # Update configuration at runtime
    print("\nUpdating configuration...")
    config.performance.max_memory_usage_gb = 6.0
    config.monitoring.log_level = "DEBUG"
    config.cache.hot_cache_size = 2000
    
    print(f"Updated max memory: {config.performance.max_memory_usage_gb}GB")
    print(f"Updated log level: {config.monitoring.log_level}")
    print(f"Updated cache size: {config.cache.hot_cache_size}")
    
    # Validate updated configuration
    result = validate_unified_config(config)
    print(f"\nValidation after updates: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
    
    # Test environment loader
    env_loader = EnvironmentLoader()
    env_config = env_loader.load_environment_config()
    
    if env_config:
        print(f"\nEnvironment overrides detected: {len(env_config)} sections")
        for section in env_config:
            print(f"  {section}: {len(env_config[section])} settings")
    else:
        print("\nNo environment overrides detected")
    
    print()


def example_migration_compatibility():
    """Demonstrate migration from legacy configurations."""
    print("🔄 Legacy Configuration Migration")
    print("=" * 50)
    
    # Simulate legacy Einstein config structure
    legacy_einstein = {
        "hardware": {
            "cpu_cores": 12,
            "cpu_performance_cores": 8,
            "cpu_efficiency_cores": 4,
            "memory_total_gb": 24.0,
            "has_gpu": True,
            "gpu_cores": 20,
        },
        "performance": {
            "max_startup_time_ms": 500.0,
            "max_search_time_ms": 50.0,
            "max_memory_usage_gb": 18.0,
        },
    }
    
    # Simulate legacy Bolt config structure
    legacy_bolt = {
        "max_agents": 8,
        "use_gpu": True,
        "prefer_mlx": True,
        "max_memory_gb": 16.0,
        "batch_size": 64,
    }
    
    print("Legacy Configuration Structures:")
    print(f"  Einstein: {len(legacy_einstein)} sections")
    print(f"  Bolt: {len(legacy_bolt)} settings")
    
    # Show how unified config provides backward compatibility
    unified_config = get_unified_config()
    einstein_compat = get_einstein_config()
    bolt_compat = get_bolt_config()
    
    print("\nUnified Config Backward Compatibility:")
    print(f"  Unified Config: {len(unified_config.to_dict())} sections")
    print(f"  Einstein Compatible: {len(einstein_compat)} settings")
    print(f"  Bolt Compatible: {len(bolt_compat)} settings")
    
    # Validate compatibility
    result = validate_system_config(einstein_compat, "einstein")
    print(f"  Einstein Validation: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
    
    result = validate_system_config(bolt_compat, "bolt")
    print(f"  Bolt Validation: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
    
    print()


def main():
    """Run all configuration examples."""
    print("🧪 Unified Configuration System Examples")
    print("=" * 60)
    print()
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("System Configs", example_system_configs),
        ("Environment Overrides", example_environment_overrides),
        ("File Configuration", example_file_configuration),
        ("Validation Scenarios", example_validation_scenarios),
        ("Runtime Updates", example_runtime_updates),
        ("Migration Compatibility", example_migration_compatibility),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print("✨ All examples completed!")


if __name__ == "__main__":
    main()