#!/usr/bin/env python3
"""Simple production validation check."""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

def main():
    """Run basic validation."""
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'CHECKING',
        'tests': {},
        'overall_status': 'UNKNOWN'
    }
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    results['python_version'] = python_version
    results['tests']['python_version'] = 'PASSED' if sys.version_info >= (3, 8) else 'FAILED'
    
    # Check project structure
    project_root = Path(__file__).parent
    required_files = ['run.py', 'pyproject.toml', 'requirements.txt']
    
    file_checks = {}
    for file in required_files:
        file_path = project_root / file
        file_checks[file] = file_path.exists()
    
    results['tests']['required_files'] = 'PASSED' if all(file_checks.values()) else 'FAILED'
    results['file_checks'] = file_checks
    
    # Check modules
    modules_to_check = ['json', 'sys', 'os', 'pathlib', 'datetime']
    module_checks = {}
    
    for module in modules_to_check:
        try:
            __import__(module)
            module_checks[module] = True
        except ImportError:
            module_checks[module] = False
    
    results['tests']['core_modules'] = 'PASSED' if all(module_checks.values()) else 'FAILED'
    results['module_checks'] = module_checks
    
    # Check directories
    directories = ['src', 'data', 'tests']
    dir_checks = {}
    
    for directory in directories:
        dir_path = project_root / directory
        dir_checks[directory] = dir_path.exists()
    
    results['tests']['directories'] = 'PASSED' if dir_checks.get('src', False) else 'FAILED'
    results['directory_checks'] = dir_checks
    
    # Calculate overall status
    passed_tests = sum(1 for test in results['tests'].values() if test == 'PASSED')
    total_tests = len(results['tests'])
    
    if passed_tests == total_tests:
        results['overall_status'] = 'PASSED'
    elif passed_tests >= total_tests * 0.8:
        results['overall_status'] = 'MOSTLY_PASSED'
    else:
        results['overall_status'] = 'FAILED'
    
    results['passed_tests'] = passed_tests
    results['total_tests'] = total_tests
    results['pass_rate'] = passed_tests / total_tests if total_tests > 0 else 0
    
    # Save results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = project_root / f"simple_validation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Simple Validation Results:")
    print(f"Python Version: {python_version}")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Tests: {passed_tests}/{total_tests} passed")
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()