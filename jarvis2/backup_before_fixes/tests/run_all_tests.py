"""Run all Jarvis2 tests comprehensively.

This ensures everything works correctly on M4 Pro without shortcuts.
"""
import pytest
import sys
import os
from pathlib import Path

# Set environment to avoid OpenMP issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Test categories
TEST_SUITES = {
    'device_routing': 'test_device_routing.py',
    'process_isolation': 'test_process_isolation.py', 
    'memory_management': 'test_memory_management.py',
    'performance': 'test_performance_benchmarks.py',
    'mcts_correctness': 'test_mcts_correctness.py',
    'integration': 'test_jarvis2_integration.py'
}


def run_test_suite(name: str, test_file: str) -> bool:
    """Run a single test suite."""
    print(f"\n{'='*60}")
    print(f"Running {name} tests...")
    print(f"{'='*60}")
    
    # Run with pytest
    result = pytest.main([
        test_file,
        '-v',
        '--tb=short',
        '--asyncio-mode=auto',
        '-p', 'no:warnings'  # Suppress warnings for cleaner output
    ])
    
    return result == 0


def main():
    """Run all test suites."""
    print("Jarvis2 Comprehensive Test Suite")
    print("For M4 Pro (Serial: KXQ93HN7DP)")
    print("="*60)
    
    # Change to test directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    results = {}
    
    # Run each test suite
    for name, test_file in TEST_SUITES.items():
        if Path(test_file).exists():
            success = run_test_suite(name, test_file)
            results[name] = success
        else:
            print(f"⚠️  {test_file} not found, skipping")
            results[name] = None
            
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, success in results.items():
        if success is None:
            status = "SKIPPED"
            skipped += 1
        elif success:
            status = "✅ PASSED"
            passed += 1
        else:
            status = "❌ FAILED"
            failed += 1
            
        print(f"{name:20s}: {status}")
        
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    # Return non-zero if any failed
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())