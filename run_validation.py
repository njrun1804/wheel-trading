#!/usr/bin/env python3
"""Run complete validation suite for the wheel trading system."""

import subprocess
import sys
import os
from datetime import datetime


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Failed with exit code: {result.returncode}")
            return False
        else:
            print(f"✅ Success")
            return True
            
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def main():
    """Run all validation steps."""
    print("="*80)
    print("WHEEL TRADING SYSTEM - COMPLETE VALIDATION SUITE")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Track results
    results = {}
    
    # Step 1: Check Python environment
    print("\n\nStep 1: Checking Python Environment")
    results['python'] = run_command(
        "python --version && pip list | grep -E 'databento|duckdb|pandas|aiohttp'",
        "Python environment check"
    )
    
    # Step 2: Database validation
    print("\n\nStep 2: Database Validation")
    results['database'] = run_command(
        "python validate_database_schema.py",
        "Database schema and data validation"
    )
    
    # Step 3: Check for mocks
    print("\n\nStep 3: Mock Usage Detection")
    results['mocks'] = run_command(
        "python detect_mocks.py -o mock_usage_report.txt",
        "Detect mock usage in codebase"
    )
    
    # Step 4: API validation (only if not in CI)
    if not os.environ.get('CI'):
        print("\n\nStep 4: API Connection Validation")
        results['apis'] = run_command(
            "python validate_api_connections.py",
            "Live API connection tests"
        )
    else:
        print("\n\nStep 4: Skipping API validation (CI environment)")
        results['apis'] = None
    
    # Step 5: Run tests without mocks
    print("\n\nStep 5: Running Tests")
    
    # First, run critical tests
    critical_tests = [
        "tests/test_databento.py",
        "tests/test_fred.py",
        "tests/test_wheel.py",
        "tests/test_config.py"
    ]
    
    for test_file in critical_tests:
        if os.path.exists(test_file):
            test_name = os.path.basename(test_file)
            results[f'test_{test_name}'] = run_command(
                f"pytest {test_file} -v --tb=short",
                f"Testing {test_name}"
            )
    
    # Summary
    print("\n\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for step, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
            skipped += 1
        elif result:
            status = "✅ PASSED"
            passed += 1
        else:
            status = "❌ FAILED"
            failed += 1
        
        print(f"{status:15} {step}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    # Critical issues found
    print("\n" + "="*80)
    print("CRITICAL ISSUES FOUND:")
    print("="*80)
    
    issues = []
    
    # Check if database is empty
    if os.path.exists('data/wheel_trading_optimized.duckdb'):
        import duckdb
        conn = duckdb.connect('data/wheel_trading_optimized.duckdb')
        tables = conn.execute("SHOW TABLES").fetchall()
        conn.close()
        
        if not tables:
            issues.append("Database is EMPTY - no tables found")
    else:
        issues.append("Database file is MISSING")
    
    # Check mock report
    if os.path.exists('mock_usage_report.txt'):
        with open('mock_usage_report.txt', 'r') as f:
            content = f.read()
            if 'CRITICAL: Mock usage in non-test files' in content:
                issues.append("Mock usage found in production code")
    
    if issues:
        for issue in issues:
            print(f"❌ {issue}")
    else:
        print("✅ No critical issues found")
    
    # Next steps
    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS:")
    print("="*80)
    
    if not tables:
        print("\n1. RESTORE DATABASE:")
        print("   cp data/archive/wheel_trading_master.duckdb data/wheel_trading_optimized.duckdb")
    
    print("\n2. CHECK API KEYS:")
    print("   - Ensure DATABENTO_API_KEY is set in .env")
    print("   - Ensure FRED_API_KEY is set in .env")
    
    print("\n3. REMOVE MOCKS:")
    print("   - Review mock_usage_report.txt")
    print("   - Replace mocks with real API calls or test fixtures")
    
    print("\n4. RUN FULL TEST SUITE:")
    print("   pytest -v --no-cov")
    
    print("\n" + "="*80)
    print(f"Validation completed at: {datetime.now()}")
    
    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())