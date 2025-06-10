#!/usr/bin/env python3
"""
Codex Environment Checker
Validates the environment and reports what's available vs missing.
"""

import sys
import os
import importlib
from typing import Dict, List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def colored(text: str, color: str) -> str:
    """Add color to text output."""
    return f"{color}{text}{RESET}"

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is adequate."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        return True, version_str
    else:
        return False, version_str

def check_package(package_name: str, import_path: str = None) -> Tuple[bool, str]:
    """Check if a package is available and get its version."""
    try:
        if import_path:
            module = importlib.import_module(import_path)
        else:
            module = importlib.import_module(package_name)
        
        # Try to get version
        version = "unknown"
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(module, attr):
                version = str(getattr(module, attr))
                break
        
        return True, version
    except ImportError:
        return False, "not installed"

def check_unity_wheel_modules() -> Dict[str, Tuple[bool, str]]:
    """Check Unity Wheel specific modules."""
    modules_to_check = {
        'unity_trading': 'unity_trading',
        'unity_trading.math': 'unity_trading.math.options',
        'unity_trading.strategy': 'unity_trading.strategy.wheel',
        'unity_trading.api': 'unity_trading.api.advisor',
        'unity_trading.utils': 'unity_trading.utils.position_sizing',
        'unity_trading.risk': 'unity_trading.risk.analytics',
    }
    
    results = {}
    for name, import_path in modules_to_check.items():
        available, info = check_package(name, import_path)
        results[name] = (available, info)
    
    return results

def check_environment_variables() -> Dict[str, str]:
    """Check relevant environment variables."""
    env_vars = [
        'USE_PURE_PYTHON',
        'USE_MOCK_DATA',
        'DATABENTO_SKIP_VALIDATION',
        'SKIP_VALIDATION',
        'OFFLINE_MODE',
        'LOG_LEVEL',
        'PYTHONPATH'
    ]
    
    return {var: os.getenv(var, 'not set') for var in env_vars}

def test_core_functionality() -> List[Tuple[str, bool, str]]:
    """Test core Unity Wheel functionality."""
    tests = []
    
    # Test 1: Basic math operations
    try:
        from unity_trading.math.options import black_scholes_price_validated
        result = black_scholes_price_validated(100, 100, 1, 0.05, 0.2, 'call')
        if result.confidence > 0.9:
            tests.append(("Black-Scholes calculation", True, f"Value: ${result.value:.2f}, Confidence: {result.confidence:.1%}"))
        else:
            tests.append(("Black-Scholes calculation", False, f"Low confidence: {result.confidence:.1%}"))
    except Exception as e:
        tests.append(("Black-Scholes calculation", False, str(e)))
    
    # Test 2: Strategy module
    try:
        from unity_trading.strategy.wheel import WheelStrategy
        strategy = WheelStrategy()
        tests.append(("Wheel strategy creation", True, "Strategy object created"))
    except Exception as e:
        tests.append(("Wheel strategy creation", False, str(e)))
    
    # Test 3: Position sizing
    try:
        from unity_trading.utils.position_sizing import calculate_position_size
        tests.append(("Position sizing import", True, "Module imported successfully"))
    except Exception as e:
        tests.append(("Position sizing import", False, str(e)))
    
    return tests

def print_section(title: str):
    """Print a section header."""
    print(f"\n{colored('=' * 60, BLUE)}")
    print(f"{colored(title.upper(), BLUE)}")
    print(f"{colored('=' * 60, BLUE)}")

def main():
    """Main environment check."""
    print(f"{colored('🚀 CODEX ENVIRONMENT CHECKER', BLUE)}")
    print(f"{colored('=' * 60, BLUE)}")
    
    # Check Python version
    print_section("Python Environment")
    python_ok, python_version = check_python_version()
    if python_ok:
        print(f"✅ Python version: {colored(python_version, GREEN)}")
    else:
        print(f"❌ Python version: {colored(python_version, RED)} (requires 3.8+)")
    
    print(f"📁 Python executable: {sys.executable}")
    print(f"📁 Current working directory: {os.getcwd()}")
    
    # Check critical packages
    print_section("Critical Dependencies")
    critical_packages = [
        ('json', None),
        ('datetime', None),
        ('typing', None),
        ('dataclasses', None),
        ('abc', None),
        ('logging', None),
        ('os', None),
        ('sys', None),
    ]
    
    all_critical_ok = True
    for package, import_path in critical_packages:
        available, version = check_package(package, import_path)
        if available:
            print(f"✅ {package}: {colored('available', GREEN)}")
        else:
            print(f"❌ {package}: {colored('missing', RED)}")
            all_critical_ok = False
    
    # Check optional packages
    print_section("Optional Dependencies")
    optional_packages = [
        ('numpy', None),
        ('pandas', None),
        ('scipy', None),
        ('pydantic', None),
        ('requests', None),
    ]
    
    optional_count = 0
    for package, import_path in optional_packages:
        available, version = check_package(package, import_path)
        if available:
            print(f"✅ {package}: {colored(f'v{version}', GREEN)}")
            optional_count += 1
        else:
            print(f"⚠️  {package}: {colored('not available - using fallbacks', YELLOW)}")
    
    # Check Unity Wheel modules
    print_section("Unity Wheel Modules")
    unity_modules = check_unity_wheel_modules()
    unity_ok = True
    
    for module, (available, info) in unity_modules.items():
        if available:
            print(f"✅ {module}: {colored('available', GREEN)}")
        else:
            print(f"❌ {module}: {colored(f'failed - {info}', RED)}")
            unity_ok = False
    
    # Check environment variables
    print_section("Environment Variables")
    env_vars = check_environment_variables()
    
    for var, value in env_vars.items():
        if value != 'not set':
            print(f"✅ {var}: {colored(value, GREEN)}")
        else:
            print(f"⚠️  {var}: {colored('not set', YELLOW)}")
    
    # Test core functionality
    print_section("Functionality Tests")
    tests = test_core_functionality()
    
    tests_passed = 0
    for test_name, passed, message in tests:
        if passed:
            print(f"✅ {test_name}: {colored(message, GREEN)}")
            tests_passed += 1
        else:
            print(f"❌ {test_name}: {colored(message, RED)}")
    
    # Summary
    print_section("Summary")
    
    if all_critical_ok:
        print(f"✅ Critical dependencies: {colored('ALL AVAILABLE', GREEN)}")
    else:
        print(f"❌ Critical dependencies: {colored('MISSING SOME', RED)}")
    
    print(f"📊 Optional dependencies: {colored(f'{optional_count}/5 available', GREEN if optional_count >= 3 else YELLOW)}")
    
    if unity_ok:
        print(f"✅ Unity Wheel modules: {colored('ALL AVAILABLE', GREEN)}")
    else:
        print(f"❌ Unity Wheel modules: {colored('SOME MISSING', RED)}")
    
    print(f"🧪 Functionality tests: {colored(f'{tests_passed}/{len(tests)} passed', GREEN if tests_passed == len(tests) else YELLOW)}")
    
    # Recommendations
    print_section("Recommendations")
    
    if not all_critical_ok:
        print(f"🚨 {colored('CRITICAL:', RED)} Missing essential Python modules. Python installation may be incomplete.")
    
    if not unity_ok:
        print(f"🔧 {colored('ACTION NEEDED:', YELLOW)} Run setup script:")
        print("   chmod +x .codex/setup_offline.sh")
        print("   ./.codex/setup_offline.sh")
    
    if optional_count < 3:
        print(f"⚠️  {colored('LIMITED PERFORMANCE:', YELLOW)} Consider enabling fallback mode:")
        print("   export USE_PURE_PYTHON=true")
        print("   export USE_MOCK_DATA=true")
    
    if env_vars['PYTHONPATH'] == 'not set':
        print(f"📁 {colored('PATH ISSUE:', YELLOW)} Add project to Python path:")
        print(f"   export PYTHONPATH=\"$PYTHONPATH:{os.getcwd()}\"")
    
    # Overall status
    if all_critical_ok and unity_ok and tests_passed >= len(tests) * 0.8:
        print(f"\n🎉 {colored('ENVIRONMENT STATUS: READY', GREEN)}")
        print("   Codex can proceed with Unity Wheel optimizations!")
    elif all_critical_ok and unity_ok:
        print(f"\n⚠️  {colored('ENVIRONMENT STATUS: LIMITED', YELLOW)}")
        print("   Codex can work but may need fallback modes.")
    else:
        print(f"\n❌ {colored('ENVIRONMENT STATUS: NEEDS SETUP', RED)}")
        print("   Run the setup script before proceeding.")
    
    print(f"\n📖 For detailed setup instructions: .codex/ENVIRONMENT_SETUP.md")

if __name__ == "__main__":
    main()