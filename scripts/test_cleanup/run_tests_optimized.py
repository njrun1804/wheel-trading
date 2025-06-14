#!/usr/bin/env python3
"""Optimized test runner for M4 Pro hardware.

This script:
1. Configures pytest for maximum performance
2. Uses all 12 CPU cores
3. Implements smart test grouping
4. Provides performance metrics
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# M4 Pro optimizations
CPU_CORES = 12
MEMORY_GB = 19  # 80% of 24GB
PARALLEL_WORKERS = CPU_CORES


class OptimizedTestRunner:
    """Run tests with M4 Pro optimizations."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.test_groups = {
            "unit": ["tests/unit", "src/**/tests/unit"],
            "integration": ["tests/integration", "src/**/tests/integration"],
            "system": ["tests/system"],
            "performance": ["tests/performance"],
        }
        
    def setup_environment(self):
        """Set up optimized test environment."""
        print("Setting up M4 Pro optimized test environment...")
        
        # Environment variables for performance
        env_vars = {
            "PYTEST_XDIST_WORKER_COUNT": str(PARALLEL_WORKERS),
            "WHEEL_TRADING_TEST_MODE": "1",
            "DATABASE_PATH": "data/wheel_trading_test.duckdb",
            "PYTHONDONTWRITEBYTECODE": "1",  # Skip .pyc files
            "PYTEST_CACHE_DISABLE": "1",  # Disable pytest cache
            
            # DuckDB optimizations
            "DUCKDB_MEMORY_LIMIT": f"{MEMORY_GB}GB",
            "DUCKDB_THREADS": str(CPU_CORES),
            
            # NumPy/Pandas optimizations
            "OMP_NUM_THREADS": str(CPU_CORES),
            "OPENBLAS_NUM_THREADS": str(CPU_CORES),
            "MKL_NUM_THREADS": str(CPU_CORES),
            "NUMEXPR_NUM_THREADS": str(CPU_CORES),
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            
        print(f"  ✓ Configured for {CPU_CORES} CPU cores")
        print(f"  ✓ Allocated {MEMORY_GB}GB memory")
        
    def create_pytest_config(self):
        """Create optimized pytest configuration."""
        
        config_content = f"""# Optimized pytest configuration for M4 Pro
[tool.pytest.ini_options]
addopts = [
    "-n", "{PARALLEL_WORKERS}",  # Use all CPU cores
    "--dist", "loadgroup",  # Smart work distribution
    "--maxfail", "3",  # Stop after 3 failures
    "-q",  # Quiet output
    "--tb=short",  # Short traceback
    "--no-cov",  # Disable coverage for speed
    "--disable-warnings",
    "--durations=10",  # Show 10 slowest tests
    "--benchmark-only",  # For performance tests
    "--benchmark-autosave",
    "--benchmark-save-data",
    "--benchmark-max-time=0.5",  # 500ms max per benchmark
]

# Test markers
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "performance: marks tests as performance tests",
    "gpu: marks tests that can use GPU acceleration",
]

# Async settings
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"

# Test paths
testpaths = ["tests", "src"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

# Ignore patterns
norecursedirs = [".git", ".tox", "dist", "build", "*.egg", "__pycache__", ".venv", "data"]

# Plugin settings
plugins = ["xdist", "benchmark", "asyncio", "timeout"]

# Timeout settings
timeout = 30  # 30 second timeout per test
timeout_method = "thread"

# Output settings
console_output_style = "progress"
"""
        
        pyproject_path = self.project_root / "pyproject.toml"
        
        # Append to existing pyproject.toml or create new
        if pyproject_path.exists():
            with open(pyproject_path, "a") as f:
                f.write("\n\n" + config_content)
        else:
            with open(pyproject_path, "w") as f:
                f.write(config_content)
                
        print(f"  ✓ Created optimized pytest configuration")
        
    def run_test_group(self, group: str, paths: List[str]) -> Tuple[bool, float, Dict]:
        """Run a specific test group."""
        print(f"\nRunning {group} tests...")
        
        start_time = time.time()
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            f"-n{PARALLEL_WORKERS}",
            "--dist=loadgroup",
            "--tb=short",
            "-v",
        ]
        
        # Add test paths
        for path in paths:
            if "*" in path:
                # Glob pattern
                for p in self.project_root.glob(path):
                    if p.exists():
                        cmd.append(str(p))
            else:
                test_path = self.project_root / path
                if test_path.exists():
                    cmd.append(str(test_path))
                    
        # Add markers for specific groups
        if group == "unit":
            cmd.extend(["-m", "unit or not integration"])
        elif group == "integration":
            cmd.extend(["-m", "integration"])
        elif group == "performance":
            cmd.extend(["--benchmark-only"])
            
        # Run tests
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        # Parse output for metrics
        metrics = self._parse_pytest_output(result.stdout)
        metrics["duration"] = duration
        metrics["group"] = group
        
        success = result.returncode == 0
        
        if not success:
            print(f"  ✗ {group} tests failed")
            print(result.stdout[-1000:])  # Last 1000 chars
        else:
            print(f"  ✓ {group} tests passed in {duration:.2f}s")
            
        return success, duration, metrics
        
    def _parse_pytest_output(self, output: str) -> Dict:
        """Parse pytest output for metrics."""
        metrics = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        }
        
        # Look for summary line
        for line in output.splitlines():
            if "passed" in line and "failed" in line:
                # Parse summary line
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part and i > 0:
                        metrics["passed"] = int(parts[i-1])
                    elif "failed" in part and i > 0:
                        metrics["failed"] = int(parts[i-1])
                    elif "skipped" in part and i > 0:
                        metrics["skipped"] = int(parts[i-1])
                        
        return metrics
        
    def run_all_tests(self) -> bool:
        """Run all test groups with optimizations."""
        print("Starting optimized test run on M4 Pro")
        print("=" * 60)
        
        self.setup_environment()
        self.create_pytest_config()
        
        total_start = time.time()
        all_success = True
        all_metrics = []
        
        # Run test groups in order
        for group, paths in self.test_groups.items():
            success, duration, metrics = self.run_test_group(group, paths)
            all_success = all_success and success
            all_metrics.append(metrics)
            
        total_duration = time.time() - total_start
        
        # Print summary
        self._print_summary(all_metrics, total_duration)
        
        return all_success
        
    def _print_summary(self, metrics: List[Dict], total_duration: float):
        """Print test run summary."""
        print("\n" + "=" * 60)
        print("TEST RUN SUMMARY")
        print("=" * 60)
        
        total_passed = sum(m["passed"] for m in metrics)
        total_failed = sum(m["failed"] for m in metrics)
        total_skipped = sum(m["skipped"] for m in metrics)
        
        print(f"\nTotal Duration: {total_duration:.2f}s")
        print(f"Tests/second: {(total_passed + total_failed) / total_duration:.1f}")
        print(f"\nResults:")
        print(f"  ✓ Passed:  {total_passed}")
        print(f"  ✗ Failed:  {total_failed}")
        print(f"  ⚠ Skipped: {total_skipped}")
        
        print(f"\nPerformance:")
        for m in metrics:
            print(f"  {m['group']:12} {m['duration']:6.2f}s  ({m['passed']} passed)")
            
        print(f"\nHardware Utilization:")
        print(f"  CPU Cores Used: {PARALLEL_WORKERS}")
        print(f"  Memory Allocated: {MEMORY_GB}GB")
        print(f"  Parallelization: {PARALLEL_WORKERS}x speedup potential")
        
    def run_quick_tests(self):
        """Run only fast unit tests for quick feedback."""
        print("Running quick unit tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            f"-n{PARALLEL_WORKERS}",
            "-m", "unit and not slow",
            "--tb=short",
            "--maxfail=1",
            "-q",
        ]
        
        result = subprocess.run(cmd)
        return result.returncode == 0
        
    def run_coverage(self):
        """Run tests with coverage (slower)."""
        print("Running tests with coverage...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            f"-n{PARALLEL_WORKERS}",
            "--cov=unity_wheel",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n✓ Coverage report generated in htmlcov/")
            
        return result.returncode == 0


def create_test_scripts():
    """Create convenience test scripts."""
    
    # Quick test script
    quick_script = """#!/bin/bash
# Quick unit tests (< 5 seconds)
python scripts/test_cleanup/run_tests_optimized.py --quick
"""
    
    quick_path = Path("test-quick.sh")
    quick_path.write_text(quick_script)
    quick_path.chmod(0o755)
    
    # Full test script  
    full_script = """#!/bin/bash
# Full test suite with M4 Pro optimizations
python scripts/test_cleanup/run_tests_optimized.py --all
"""
    
    full_path = Path("test-all.sh")
    full_path.write_text(full_script)
    full_path.chmod(0o755)
    
    # Coverage script
    coverage_script = """#!/bin/bash
# Test with coverage analysis
python scripts/test_cleanup/run_tests_optimized.py --coverage
"""
    
    coverage_path = Path("test-coverage.sh")
    coverage_path.write_text(coverage_script)
    coverage_path.chmod(0o755)
    
    print("Created test scripts:")
    print(f"  ./test-quick.sh    - Quick unit tests")
    print(f"  ./test-all.sh      - Full test suite")
    print(f"  ./test-coverage.sh - Tests with coverage")


def main():
    """Run optimized tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests with M4 Pro optimizations")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--quick", action="store_true", help="Run quick unit tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--create-scripts", action="store_true", help="Create test scripts")
    args = parser.parse_args()
    
    runner = OptimizedTestRunner()
    
    if args.create_scripts:
        create_test_scripts()
        return
        
    if args.quick:
        success = runner.run_quick_tests()
    elif args.coverage:
        success = runner.run_coverage()
    else:
        success = runner.run_all_tests()
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()