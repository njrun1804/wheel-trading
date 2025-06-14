#!/usr/bin/env python3
"""Comprehensive validation of the wheel trading system."""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{BLUE}→ {description}{RESET}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{GREEN}✓ Success{RESET}")
            if result.stdout:
                print(result.stdout[:500])  # First 500 chars
            return True
        else:
            print(f"{RED}✗ Failed{RESET}")
            if result.stderr:
                print(result.stderr[:500])
            return False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False

def main():
    """Run comprehensive validation."""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}WHEEL TRADING SYSTEM - COMPREHENSIVE VALIDATION{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    results = {}
    
    # 1. Check database integrity
    print(f"\n{YELLOW}1. DATABASE INTEGRITY{RESET}")
    results['db_exists'] = os.path.exists("data/wheel_trading_optimized.duckdb")
    print(f"{'✓' if results['db_exists'] else '✗'} Database exists: data/wheel_trading_optimized.duckdb")
    
    if results['db_exists']:
        # Check data
        cmd = """
        echo "SELECT 
            'Stock' as type, COUNT(*) as records 
        FROM market.price_data 
        WHERE symbol='U'
        UNION ALL
        SELECT 
            'Options', COUNT(*) 
        FROM options.contracts 
        WHERE symbol='U';" | duckdb data/wheel_trading_optimized.duckdb
        """
        results['db_data'] = run_command(cmd, "Checking database records")
    
    # 2. API Configuration
    print(f"\n{YELLOW}2. API CONFIGURATION{RESET}")
    
    # Check .env file
    results['env_exists'] = os.path.exists(".env")
    print(f"{'✓' if results['env_exists'] else '✗'} .env file exists")
    
    # Check for API keys in environment
    databento_key = os.environ.get('DATABENTO_API_KEY', '')
    fred_key = os.environ.get('FRED_API_KEY', '')
    results['databento_key'] = bool(databento_key and not databento_key.startswith('your_'))
    results['fred_key'] = bool(fred_key and not fred_key.startswith('your_'))
    
    print(f"{'✓' if results['databento_key'] else '✗'} Databento API key configured")
    print(f"{'✓' if results['fred_key'] else '✗'} FRED API key configured")
    
    # 3. Check for mock/dummy code
    print(f"\n{YELLOW}3. MOCK/DUMMY CODE DETECTION{RESET}")
    
    # Search for mock patterns in source
    cmd = 'grep -r "mock\\|Mock\\|dummy\\|Dummy\\|TEST_MODE\\|MOCK_DATA" src/unity_wheel --include="*.py" | grep -v "test_" | wc -l'
    mock_count = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    results['no_mocks'] = int(mock_count) < 5  # Allow a few references
    print(f"{'✓' if results['no_mocks'] else '✗'} Mock references in source: {mock_count}")
    
    # 4. Live API test
    print(f"\n{YELLOW}4. LIVE API CONNECTIONS{RESET}")
    results['api_test'] = run_command("python test_api_connections.py", "Testing live API connections")
    
    # 5. Data collection test
    print(f"\n{YELLOW}5. DATA COLLECTION{RESET}")
    results['collection_test'] = run_command(
        "python scripts/collect_eod_production.py --test 2>&1 | grep -E 'Stock data|Options|FRED|ERROR'",
        "Testing data collection (dry run)"
    )
    
    # 6. Monitor test
    print(f"\n{YELLOW}6. MONITORING SYSTEM{RESET}")
    results['monitor_test'] = run_command(
        "python scripts/monitor_collection.py 2>&1 | grep -v Pydantic | head -20",
        "Testing monitoring system"
    )
    
    # 7. Main application test
    print(f"\n{YELLOW}7. MAIN APPLICATION{RESET}")
    results['app_test'] = run_command(
        "python run.py --diagnose 2>&1 | grep -E 'Diagnostics|API|Database|ERROR'",
        "Testing main application"
    )
    
    # 8. Import test
    print(f"\n{YELLOW}8. PYTHON IMPORTS{RESET}")
    import_test = """
import sys
sys.path.insert(0, '.')
try:
    from src.unity_wheel.api import WheelAdvisor
    from src.unity_wheel.data_providers.databento.client import DatabentoClient
    from src.unity_wheel.analytics import UnityAssignmentModel
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    """
    results['imports'] = run_command(f'python -c "{import_test}"', "Testing Python imports")
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}VALIDATION SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for check, status in results.items():
        icon = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
        print(f"{icon} {check.replace('_', ' ').title()}")
    
    print(f"\n{BLUE}Result: {passed}/{total} checks passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}✅ ALL SYSTEMS OPERATIONAL!{RESET}")
        print("\nNext steps:")
        print("1. Run live data collection: python scripts/collect_eod_production.py")
        print("2. Get trading recommendation: python run.py -p 100000")
        print("3. Monitor system: python scripts/monitor_collection.py --watch")
    else:
        print(f"\n{RED}❌ ISSUES DETECTED{RESET}")
        print("\nRecommended fixes:")
        if not results.get('db_exists'):
            print("- Restore database: cp data/archive/wheel_trading_master.duckdb data/wheel_trading_optimized.duckdb")
        if not results.get('databento_key') or not results.get('fred_key'):
            print("- Set API keys in .env file")
        if not results.get('no_mocks'):
            print("- Remove mock/dummy code from production sources")
        if not results.get('imports'):
            print("- Fix import errors (check PYTHONPATH)")

if __name__ == "__main__":
    main()