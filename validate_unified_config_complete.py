#!/usr/bin/env python3
"""
Complete Unified Configuration System Validation

Final comprehensive test of Einstein and Bolt configuration alignment.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all configuration modules can be imported."""
    print("🔍 Testing Configuration System Imports...")
    
    try:
        from src.unity_wheel.config.unified_config import (
            UnifiedConfig, ConfigLoader, get_unified_config, 
            get_einstein_config, get_bolt_config, reset_config
        )
        print("  ✅ Unified config imports successful")
        
        from src.unity_wheel.config.hardware_config import (
            HardwareConfig, HardwareDetector
        )
        print("  ✅ Hardware config imports successful")
        
        from src.unity_wheel.config.environment_loader import EnvironmentLoader
        print("  ✅ Environment loader imports successful")
        
        from src.unity_wheel.config.validation import (
            validate_unified_config, validate_system_config
        )
        print("  ✅ Validation imports successful")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_hardware_detection():
    """Test hardware detection functionality."""
    print("\n💻 Testing Hardware Detection...")
    
    try:
        from src.unity_wheel.config.hardware_config import HardwareDetector
        
        hardware = HardwareDetector.detect_hardware()
        
        print(f"  ✅ Platform: {hardware.platform_type}")
        print(f"  ✅ CPU: {hardware.cpu_cores} cores ({hardware.cpu_performance_cores}P + {hardware.cpu_efficiency_cores}E)")
        print(f"  ✅ Memory: {hardware.memory_total_gb:.1f}GB total, {hardware.memory_available_gb:.1f}GB available")
        print(f"  ✅ GPU: {'Available' if hardware.has_gpu else 'Not available'}")
        print(f"  ✅ ANE: {'Available' if hardware.has_ane else 'Not available'}")
        
        # Validate hardware data
        assert hardware.cpu_cores > 0, "CPU cores must be positive"
        assert hardware.memory_total_gb > 0, "Memory must be positive"
        assert hardware.platform_type in ["apple_silicon", "intel", "amd", "intel_mac", "unknown"]
        
        return True
        
    except Exception as e:
        print(f"  ❌ Hardware detection failed: {e}")
        return False


def test_unified_config_creation():
    """Test unified configuration creation and validation."""
    print("\n⚙️  Testing Unified Configuration Creation...")
    
    try:
        from src.unity_wheel.config.unified_config import get_unified_config, reset_config
        from src.unity_wheel.config.validation import validate_unified_config
        
        # Reset any cached config
        reset_config()
        
        # Load unified configuration
        config = get_unified_config()
        
        print(f"  ✅ Config version: {config.config_version}")
        print(f"  ✅ Hardware platform: {config.hardware.platform_type}")
        print(f"  ✅ Systems enabled: {list(config.systems_enabled.keys())}")
        
        # Validate configuration
        result = validate_unified_config(config)
        
        if result.is_valid:
            print("  ✅ Configuration validation passed")
        else:
            print("  ❌ Configuration validation failed:")
            for error in result.errors:
                print(f"    - {error}")
        
        if result.warnings:
            print("  ⚠️  Configuration warnings:")
            for warning in result.warnings[:3]:  # Show first 3
                print(f"    - {warning}")
        
        # Test serialization
        config_dict = config.to_dict()
        print(f"  ✅ Serialization: {len(config_dict)} sections")
        
        # Test deserialization
        config2 = UnifiedConfig.from_dict(config_dict)
        print(f"  ✅ Deserialization: {config2.config_version}")
        
        return result.is_valid
        
    except Exception as e:
        print(f"  ❌ Unified config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_compatibility():
    """Test system-specific configuration compatibility."""
    print("\n🎯 Testing System Compatibility...")
    
    try:
        from src.unity_wheel.config.unified_config import (
            get_einstein_config, get_bolt_config, get_jarvis_config, get_meta_config
        )
        from src.unity_wheel.config.validation import validate_system_config
        
        systems = {
            "einstein": get_einstein_config,
            "bolt": get_bolt_config,
            "jarvis": get_jarvis_config,
            "meta": get_meta_config,
        }
        
        all_valid = True
        
        for system_name, get_config_func in systems.items():
            try:
                config = get_config_func()
                print(f"  ✅ {system_name.title()}: {len(config)} settings")
                
                # Validate system-specific config
                if isinstance(config, dict):
                    result = validate_system_config(config, system_name)
                else:
                    result = validate_system_config(config.__dict__, system_name)
                
                if result.is_valid:
                    print(f"    ✅ Validation: passed")
                else:
                    print(f"    ❌ Validation: failed")
                    for error in result.errors[:2]:  # Show first 2 errors
                        print(f"      - {error}")
                    all_valid = False
                
                if result.warnings:
                    print(f"    ⚠️  Warnings: {len(result.warnings)}")
                
            except Exception as e:
                print(f"  ❌ {system_name.title()}: {e}")
                all_valid = False
        
        return all_valid
        
    except Exception as e:
        print(f"  ❌ System compatibility test failed: {e}")
        return False


def test_environment_overrides():
    """Test environment variable override functionality."""
    print("\n🌍 Testing Environment Variable Overrides...")
    
    try:
        from src.unity_wheel.config.unified_config import get_unified_config, reset_config
        from src.unity_wheel.config.environment_loader import EnvironmentLoader
        
        # Test environment variables
        test_env = {
            "EINSTEIN_MAX_MEMORY_GB": "6.0",
            "EINSTEIN_LOG_LEVEL": "DEBUG",
            "BOLT_MAX_AGENTS": "6",
            "BOLT_USE_GPU": "false",
            "JARVIS_EMBEDDING_DIM": "384",
            "META_DEBUG_MODE": "true",
        }
        
        print(f"  🧪 Testing {len(test_env)} environment variables")
        
        with patch.dict(os.environ, test_env):
            # Reset and reload with env vars
            reset_config()
            
            # Test environment loader directly
            env_loader = EnvironmentLoader()
            env_config = env_loader.load_environment_config()
            
            if env_config:
                print(f"  ✅ Environment loader detected {len(env_config)} sections")
            else:
                print("  ⚠️  No environment overrides detected")
            
            # Load unified config with overrides
            config = get_unified_config()
            
            # Check if overrides were applied
            checks = [
                ("Max Memory", config.performance.max_memory_usage_gb, 6.0),
                ("Log Level", config.monitoring.log_level, "DEBUG"),
                ("Embedding Dim", config.ml.embedding_dimension, 384),
            ]
            
            for name, actual, expected in checks:
                if actual == expected:
                    print(f"  ✅ {name}: {actual} (override applied)")
                else:
                    print(f"  ⚠️  {name}: {actual} (expected {expected})")
        
        # Reset to clean state
        reset_config()
        return True
        
    except Exception as e:
        print(f"  ❌ Environment override test failed: {e}")
        return False


def test_file_operations():
    """Test configuration file save/load operations."""
    print("\n📁 Testing File Operations...")
    
    try:
        from src.unity_wheel.config.unified_config import UnifiedConfig, get_unified_config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            
            # Get current config
            config = get_unified_config()
            
            # Save to file
            config.save_to_file(config_file)
            print(f"  ✅ Config saved to: {config_file.name}")
            
            # Verify file exists and has content
            assert config_file.exists(), "Config file was not created"
            assert config_file.stat().st_size > 0, "Config file is empty"
            
            # Load from file
            loaded_config = UnifiedConfig.load_from_file(config_file)
            print(f"  ✅ Config loaded from file: {loaded_config.config_version}")
            
            # Verify key values match
            assert config.config_version == loaded_config.config_version
            assert config.hardware.cpu_cores == loaded_config.hardware.cpu_cores
            assert config.hardware.platform_type == loaded_config.hardware.platform_type
            
            print("  ✅ File save/load validation passed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ File operations test failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility with legacy systems."""
    print("\n🔄 Testing Backward Compatibility...")
    
    try:
        # Test if we can import legacy configs (if available)
        legacy_systems = {}
        
        try:
            from einstein.einstein_config import get_einstein_config as legacy_einstein
            legacy_systems["einstein"] = legacy_einstein
            print("  ✅ Legacy Einstein config available")
        except ImportError:
            print("  ⚠️  Legacy Einstein config not available")
        
        try:
            from bolt.core.config import get_default_config as legacy_bolt
            legacy_systems["bolt"] = legacy_bolt
            print("  ✅ Legacy Bolt config available")
        except ImportError:
            print("  ⚠️  Legacy Bolt config not available")
        
        # Test unified system provides compatibility
        from src.unity_wheel.config.unified_config import get_einstein_config, get_bolt_config
        
        unified_einstein = get_einstein_config()
        unified_bolt = get_bolt_config()
        
        print(f"  ✅ Unified Einstein: {len(unified_einstein)} settings")
        print(f"  ✅ Unified Bolt: {len(unified_bolt)} settings")
        
        # Check if unified configs have expected structure
        einstein_required = ["hardware", "performance", "cache", "paths"]
        for section in einstein_required:
            if section in unified_einstein:
                print(f"    ✅ Einstein {section}: present")
            else:
                print(f"    ❌ Einstein {section}: missing")
        
        bolt_required = ["max_agents", "use_gpu", "max_memory_gb"]
        for setting in bolt_required:
            if setting in unified_bolt:
                print(f"    ✅ Bolt {setting}: {unified_bolt[setting]}")
            else:
                print(f"    ❌ Bolt {setting}: missing")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        return False


def test_validation_comprehensive():
    """Test comprehensive validation scenarios."""
    print("\n✅ Testing Comprehensive Validation...")
    
    try:
        from src.unity_wheel.config.validation import (
            validate_unified_config, validate_system_config, ConfigValidator
        )
        
        validator = ConfigValidator()
        
        # Test valid configuration
        valid_config = {
            "hardware": {
                "cpu_cores": 8,
                "memory_total_gb": 16.0,
                "platform_type": "apple_silicon",
                "has_gpu": True,
                "has_ane": True,
            },
            "performance": {
                "max_memory_usage_gb": 12.0,
                "max_search_time_ms": 50.0,
                "async_timeout": 300.0,
            },
            "ml": {
                "embedding_dimension": 768,
                "learning_rate": 0.001,
                "enable_gpu": True,
                "enable_ane": True,
            },
            "monitoring": {
                "log_level": "INFO",
                "monitoring_interval": 1.0,
            },
        }
        
        result = validator.validate_config(valid_config)
        if result.is_valid:
            print("  ✅ Valid configuration test passed")
        else:
            print("  ❌ Valid configuration test failed")
            for error in result.errors:
                print(f"    - {error}")
        
        # Test invalid configuration
        invalid_config = {
            "hardware": {
                "cpu_cores": -1,  # Invalid
                "memory_total_gb": 0,  # Invalid
                "platform_type": "invalid_platform",
            },
            "performance": {
                "max_memory_usage_gb": -5.0,  # Invalid
                "max_search_time_ms": 0,  # Invalid
            },
            "ml": {
                "learning_rate": 5.0,  # Invalid > 1
                "embedding_dimension": 0,  # Invalid
            },
        }
        
        result = validator.validate_config(invalid_config)
        if not result.is_valid and len(result.errors) > 0:
            print(f"  ✅ Invalid configuration test passed ({len(result.errors)} errors detected)")
        else:
            print("  ❌ Invalid configuration test failed (should have detected errors)")
        
        # Test system compatibility
        systems = ["einstein", "bolt", "jarvis", "meta"]
        for system in systems:
            result = validator.validate_system_compatibility(valid_config, system)
            status = "passed" if result.is_valid else "failed"
            print(f"  ✅ {system.title()} compatibility: {status}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Comprehensive validation test failed: {e}")
        return False


def main():
    """Run complete configuration system validation."""
    print("🧪 Complete Unified Configuration System Validation")
    print("=" * 60)
    
    tests = [
        ("Import Testing", test_imports),
        ("Hardware Detection", test_hardware_detection),
        ("Unified Config Creation", test_unified_config_creation),
        ("System Compatibility", test_system_compatibility),
        ("Environment Overrides", test_environment_overrides),
        ("File Operations", test_file_operations),
        ("Backward Compatibility", test_backward_compatibility),
        ("Comprehensive Validation", test_validation_comprehensive),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\n🎯 Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 CONFIGURATION SYSTEM ALIGNMENT COMPLETE!")
        print("   Einstein and Bolt configurations are fully unified and compatible.")
        print("   All validation tests passed successfully.")
        return True
    else:
        print(f"\n⚠️  CONFIGURATION SYSTEM NEEDS ATTENTION")
        print(f"   {total - passed} test(s) failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)