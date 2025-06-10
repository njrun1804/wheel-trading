#!/usr/bin/env python3
"""
Test configuration for Codex environment
Validates that all configuration is working correctly.
"""

import os
import sys
from typing import Any, Dict


def test_environment_variables() -> Dict[str, Any]:
    """Test environment variables are set correctly."""
    results = {}

    # Required environment variables for Codex
    required_vars = {
        "USE_PURE_PYTHON": "true",
        "USE_MOCK_DATA": "true",
        "DATABENTO_SKIP_VALIDATION": "true",
        "SKIP_VALIDATION": "true",
        "OFFLINE_MODE": "true",
    }

    for var, expected in required_vars.items():
        actual = os.getenv(var)
        results[var] = {"expected": expected, "actual": actual, "correct": actual == expected}

    return results


def test_python_path():
    """Test that Python path includes current directory."""
    current_dir = os.getcwd()
    python_path = os.getenv("PYTHONPATH", "")

    return {
        "current_dir": current_dir,
        "python_path": python_path,
        "includes_current": current_dir in python_path.split(":"),
    }


def test_imports():
    """Test critical imports work."""
    import_tests = {}

    # Test standard library
    try:
        import dataclasses
        import datetime
        import json
        import typing

        import_tests["standard_library"] = True
    except ImportError as e:
        import_tests["standard_library"] = f"Failed: {e}"

    # Test Unity Wheel modules
    try:
        sys.path.insert(0, os.getcwd())
        from src.unity_wheel.math.options import black_scholes_price_validated

        import_tests["unity_math"] = True
    except ImportError as e:
        import_tests["unity_math"] = f"Failed: {e}"

    try:
        from src.unity_wheel.strategy.wheel import WheelStrategy

        import_tests["unity_strategy"] = True
    except ImportError as e:
        import_tests["unity_strategy"] = f"Failed: {e}"

    try:
        from src.unity_wheel.utils.position_sizing import DynamicPositionSizer

        import_tests["unity_utils"] = True
    except ImportError as e:
        import_tests["unity_utils"] = f"Failed: {e}"

    return import_tests


def test_functionality():
    """Test that core functionality works."""
    tests = {}

    try:
        sys.path.insert(0, os.getcwd())
        from src.unity_wheel.math.options import black_scholes_price_validated

        result = black_scholes_price_validated(100, 100, 1, 0.05, 0.2, "call")
        tests["black_scholes"] = {
            "success": True,
            "value": result.value,
            "confidence": result.confidence,
        }
    except Exception as e:
        tests["black_scholes"] = {"success": False, "error": str(e)}

    # Test minimal trader as fallback
    try:
        from .minimal_trader import MinimalWheelAdvisor

        advisor = MinimalWheelAdvisor()
        rec = advisor.generate_recommendation(100000)
        tests["minimal_trader"] = {
            "success": True,
            "action": rec.action,
            "confidence": rec.confidence,
        }
    except Exception as e:
        tests["minimal_trader"] = {"success": False, "error": str(e)}

    return tests


def main():
    """Run all configuration tests."""
    print("🧪 CODEX CONFIGURATION TEST")
    print("=" * 50)

    # Test environment variables
    print("\n📋 Environment Variables:")
    env_results = test_environment_variables()
    all_env_correct = True

    for var, result in env_results.items():
        if result["correct"]:
            print(f"   ✅ {var}: {result['actual']}")
        else:
            print(f"   ❌ {var}: expected '{result['expected']}', got '{result['actual']}'")
            all_env_correct = False

    if all_env_correct:
        print("   ✅ All environment variables correct")
    else:
        print("   ⚠️  Some environment variables need fixing")

    # Test Python path
    print("\n📁 Python Path:")
    path_result = test_python_path()
    if path_result["includes_current"]:
        print(f"   ✅ Current directory in PYTHONPATH")
    else:
        print(f"   ❌ Current directory not in PYTHONPATH")
        print(f"      Current: {path_result['current_dir']}")
        print(f"      PYTHONPATH: {path_result['python_path']}")

    # Test imports
    print("\n📦 Import Tests:")
    import_results = test_imports()
    all_imports_ok = True

    for test, result in import_results.items():
        if result is True:
            print(f"   ✅ {test}: successful")
        else:
            print(f"   ❌ {test}: {result}")
            all_imports_ok = False

    if all_imports_ok:
        print("   ✅ All imports successful")
    else:
        print("   ⚠️  Some imports failed")

    # Test functionality
    print("\n🧮 Functionality Tests:")
    func_results = test_functionality()
    all_func_ok = True

    for test, result in func_results.items():
        if result["success"]:
            if "confidence" in result:
                print(f"   ✅ {test}: confidence {result['confidence']:.1%}")
            else:
                print(f"   ✅ {test}: successful")
        else:
            print(f"   ❌ {test}: {result['error']}")
            all_func_ok = False

    # Overall assessment
    print("\n📊 OVERALL ASSESSMENT:")
    if all_env_correct and path_result["includes_current"] and all_imports_ok and all_func_ok:
        print("   🎉 CONFIGURATION PERFECT - Ready for Codex optimization!")
        return True
    elif all_imports_ok and all_func_ok:
        print("   ✅ CONFIGURATION GOOD - Minor environment issues but functional")
        return True
    else:
        print("   ⚠️  CONFIGURATION NEEDS WORK - Run setup script first")
        print("\n🔧 Quick fixes:")

        if not all_env_correct:
            print("   source .codex/activate.sh")

        if not path_result["includes_current"]:
            print(f'   export PYTHONPATH="$PYTHONPATH:{os.getcwd()}"')

        if not all_imports_ok:
            print("   ./.codex/setup_offline.sh")

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
