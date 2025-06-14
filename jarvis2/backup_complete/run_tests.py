"""Run Jarvis2 tests with proper output."""
import asyncio
import os
import sys

# Set environment variable to avoid OpenMP issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import tests
from jarvis2.tests.test_jarvis2_integration import (
    test_basic_code_generation,
    test_parallel_requests,
    test_memory_management,
    test_device_routing,
    test_context_retrieval,
    test_learning_feedback_loop,
    test_performance_benchmarks,
    test_error_handling,
    test_cpu_efficiency_cores
)


async def run_all_tests():
    """Run all tests."""
    tests = [
        ("Basic Code Generation", test_basic_code_generation),
        ("Parallel Requests", test_parallel_requests),
        ("Memory Management", test_memory_management),
        ("Device Routing", test_device_routing),
        ("Context Retrieval", test_context_retrieval),
        ("Learning Feedback Loop", test_learning_feedback_loop),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    failed = 0
    
    print("Running Jarvis2 Integration Tests")
    print("=" * 60)
    
    for name, test_func in tests:
        print(f"\n>>> Testing: {name}")
        print("-" * 40)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            print(f"✓ {name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    # Run CPU test
    print(f"\n>>> Testing: CPU Efficiency Cores")
    print("-" * 40)
    try:
        test_cpu_efficiency_cores()
        print(f"✓ CPU Efficiency Cores PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ CPU Efficiency Cores FAILED: {e}")
        failed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)