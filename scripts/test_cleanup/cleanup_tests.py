#!/usr/bin/env python3
"""Master test cleanup script."""

import os
import sys
import subprocess
from pathlib import Path

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def run_step(script, description):
    """Run a cleanup step."""
    print(f"\n{BLUE}→ {description}{RESET}")
    try:
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{GREEN}✓ Success{RESET}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"{RED}✗ Failed{RESET}")
            if result.stderr:
                print(result.stderr)
            return False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False

def remove_mock_tests():
    """Remove or update tests that use mocks."""
    print(f"\n{BLUE}→ Updating mock-based tests{RESET}")
    
    mock_count = 0
    updated_count = 0
    
    # Find test files with mocks
    for test_file in Path(".").glob("**/test_*.py"):
        if "test_cleanup" in str(test_file):
            continue
            
        with open(test_file, 'r') as f:
            content = f.read()
        
        if 'mock' in content.lower() or '@patch' in content:
            mock_count += 1
            
            # Simple replacements
            original = content
            content = content.replace('from unittest.mock import', '# from unittest.mock import')
            content = content.replace('from mock import', '# from mock import')
            content = content.replace('@patch(', '# @patch(')
            content = content.replace('@mock.patch(', '# @mock.patch(')
            
            # Comment out mock.patch decorators but keep the function
            import re
            content = re.sub(r'^(\s*)@\w*[Mm]ock\.\w+\([^)]*\)$', r'\1# \g<0>', content, flags=re.MULTILINE)
            content = re.sub(r'^(\s*)@patch\([^)]*\)$', r'\1# \g<0>', content, flags=re.MULTILINE)
            
            if content != original:
                with open(test_file, 'w') as f:
                    f.write(content)
                updated_count += 1
    
    print(f"{GREEN}✓ Found {mock_count} files with mocks, updated {updated_count}{RESET}")

def create_pytest_config():
    """Create optimized pytest configuration."""
    config = """[tool.pytest.ini_options]
minversion = "6.0"
addopts = [
    "-ra",
    "--strict-markers",
    "--ignore=archive",
    "--ignore=.codex",
    "--ignore=data",
    "-n", "12",  # Use all 12 CPU cores
    "--maxfail=5",
    "--tb=short",
    "-q"
]
testpaths = [
    "tests",
    "src/unity_wheel/tests"
]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests"
]

# Use test database
env = [
    "DATABASE_PATH=data/test_wheel_trading.duckdb",
    "TESTING=true"
]
"""
    
    with open("pytest.ini", "w") as f:
        f.write(config)
    
    print(f"{GREEN}✓ Created pytest.ini with M4 Pro optimizations{RESET}")

def create_test_runner():
    """Create fast test runner script."""
    runner = """#!/usr/bin/env python3
\"\"\"Fast test runner for wheel trading system.\"\"\"

import os
import sys
import time
import subprocess
from pathlib import Path

# Set test environment
os.environ['DATABASE_PATH'] = 'data/test_wheel_trading.duckdb'
os.environ['TESTING'] = 'true'
os.environ['PYTHONPATH'] = str(Path.cwd())

def run_tests(pattern=None):
    \"\"\"Run tests with optimizations.\"\"\"
    print("🚀 Unity Wheel Test Runner (M4 Pro Optimized)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-n", "12",  # 12 CPU cores
        "--tb=short",
        "--maxfail=5",
        "-q"
    ]
    
    if pattern:
        cmd.extend(["-k", pattern])
    
    # Run tests
    result = subprocess.run(cmd)
    
    duration = time.time() - start_time
    print(f"\\n⏱️  Tests completed in {duration:.2f} seconds")
    
    return result.returncode

if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_tests(pattern))
"""
    
    runner_path = Path("test_runner.py")
    with open(runner_path, "w") as f:
        f.write(runner)
    
    # Make executable
    os.chmod(runner_path, 0o755)
    
    print(f"{GREEN}✓ Created test_runner.py{RESET}")

def main():
    """Run complete test cleanup."""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}WHEEL TRADING TEST SUITE CLEANUP{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Create test cleanup directory
    cleanup_dir = Path("scripts/test_cleanup")
    cleanup_dir.mkdir(exist_ok=True)
    
    steps = [
        ("01_setup_test_db.py", "Setting up test database"),
        ("02_fix_imports.py", "Fixing import issues"),
        ("03_fix_diagnose.py", "Fixing diagnose mode"),
    ]
    
    # Run steps
    results = {}
    for script, description in steps:
        script_path = cleanup_dir / script
        if script_path.exists():
            results[script] = run_step(str(script_path), description)
        else:
            print(f"{YELLOW}⚠ Skipping {script} - not found{RESET}")
    
    # Additional cleanup
    remove_mock_tests()
    create_pytest_config()
    create_test_runner()
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}CLEANUP SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n{GREEN}✅ Test cleanup complete!{RESET}")
        print("\nNext steps:")
        print("1. Run tests: python test_runner.py")
        print("2. Test diagnose: python run.py --diagnose")
        print("3. Run specific tests: python test_runner.py test_math")
    else:
        print(f"\n{RED}❌ Some steps failed{RESET}")
        for script, status in results.items():
            icon = "✓" if status else "✗"
            print(f"{icon} {script}")

if __name__ == "__main__":
    main()