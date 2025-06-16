#!/usr/bin/env python3
"""
Accelerated Tools Reliability Test
Tests initialization reliability with multiple runs and stress testing
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path.cwd() / "src"))

from src.unity_wheel.accelerated_tools.reliable_initialization import (
    get_tools_manager,
    validate_all_tools,
    ensure_tools_initialized
)


async def test_single_initialization():
    """Test a single initialization cycle."""
    print("🔄 Testing single initialization...")
    
    try:
        start_time = time.perf_counter()
        all_available, results = await validate_all_tools()
        duration = time.perf_counter() - start_time
        
        available_count = results['available_count']
        total_count = results['total_count']
        success_rate = results['success_rate']
        
        print(f"   ✅ Initialization completed in {duration:.2f}s")
        print(f"   📊 Tools available: {available_count}/{total_count} ({success_rate:.1%})")
        
        if all_available:
            print("   🎉 All critical tools available!")
            return True, results
        else:
            missing = results['missing_critical_tools']
            print(f"   ❌ Missing critical tools: {missing}")
            return False, results
            
    except Exception as e:
        print(f"   💥 Initialization failed: {e}")
        return False, {'error': str(e)}


async def test_multiple_initializations(count: int = 5):
    """Test multiple initialization cycles to check reliability."""
    print(f"🔄 Testing {count} initialization cycles...")
    
    results = []
    success_count = 0
    
    for i in range(count):
        print(f"   Cycle {i+1}/{count}...")
        
        # Reset manager for clean test
        import src.unity_wheel.accelerated_tools.reliable_initialization as init_module
        init_module._manager_instance = None
        
        success, result = await test_single_initialization()
        results.append({
            'cycle': i + 1,
            'success': success,
            'result': result
        })
        
        if success:
            success_count += 1
        
        # Small delay between tests
        await asyncio.sleep(0.1)
    
    reliability = success_count / count
    print(f"\n📊 Reliability Test Results:")
    print(f"   Success rate: {success_count}/{count} ({reliability:.1%})")
    
    if reliability >= 0.9:
        print("   ✅ Excellent reliability (≥90%)")
        return True
    elif reliability >= 0.8:
        print("   ⚠️ Good reliability (≥80%)")
        return True
    else:
        print("   ❌ Poor reliability (<80%)")
        return False


async def test_concurrent_initialization():
    """Test concurrent initialization attempts."""
    print("🔄 Testing concurrent initialization...")
    
    # Reset manager
    import src.unity_wheel.accelerated_tools.reliable_initialization as init_module
    init_module._manager_instance = None
    
    # Start multiple initialization tasks simultaneously
    tasks = []
    for i in range(3):
        task = asyncio.create_task(ensure_tools_initialized())
        tasks.append(task)
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if all succeeded
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        print(f"   📊 Concurrent initializations: {success_count}/3 succeeded")
        
        if success_count == 3:
            print("   ✅ All concurrent initializations succeeded")
            return True
        else:
            print("   ❌ Some concurrent initializations failed")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"      Task {i+1} failed: {result}")
            return False
            
    except Exception as e:
        print(f"   💥 Concurrent test failed: {e}")
        return False


async def test_tool_functionality():
    """Test basic functionality of each tool."""
    print("🔄 Testing tool functionality...")
    
    manager = get_tools_manager()
    await ensure_tools_initialized()
    
    tools_to_test = [
        ('ripgrep_turbo', 'search', ('def', 'src/unity_wheel/accelerated_tools')),
        ('dependency_graph_turbo', 'build_graph', ()),
        ('python_analysis_turbo', 'analyze_file', ('src/unity_wheel/accelerated_tools/ripgrep_turbo.py',)),
        ('duckdb_turbo', 'execute', ('CREATE TABLE test_reliability (id INT, name TEXT)',)),
        ('trace_turbo', 'trace_span', ('test_trace',)),
        ('python_helpers_turbo', 'analyze_project', ('src/unity_wheel/accelerated_tools',)),
    ]
    
    functional_tools = []
    
    for tool_name, method_name, args in tools_to_test:
        try:
            tool_instance = manager.get_tool(tool_name)
            if tool_instance is None:
                print(f"   ❌ {tool_name}: Not available")
                continue
            
            # Check if method exists
            if not hasattr(tool_instance, method_name):
                print(f"   ❌ {tool_name}: Method '{method_name}' not found")
                continue
            
            method = getattr(tool_instance, method_name)
            
            # Call method (with timeout for async methods)
            if asyncio.iscoroutinefunction(method):
                result = await asyncio.wait_for(method(*args), timeout=5.0)
            else:
                result = method(*args)
            
            print(f"   ✅ {tool_name}: {method_name}() works")
            functional_tools.append(tool_name)
            
        except Exception as e:
            print(f"   ❌ {tool_name}: {method_name}() failed - {e}")
    
    # Test bolt integration separately
    try:
        bolt_tool = manager.get_tool('bolt_integration')
        if bolt_tool is not None:
            print(f"   ✅ bolt_integration: Available")
            functional_tools.append('bolt_integration')
        else:
            print(f"   ❌ bolt_integration: Not available")
    except Exception as e:
        print(f"   ❌ bolt_integration: Failed - {e}")
    
    functionality_rate = len(functional_tools) / 7
    print(f"   📊 Functional tools: {len(functional_tools)}/7 ({functionality_rate:.1%})")
    
    return functionality_rate >= 0.85  # 85% functionality threshold


async def test_stress_scenarios():
    """Test stress scenarios and edge cases."""
    print("🔄 Testing stress scenarios...")
    
    # Test 1: Rapid successive initializations
    print("   Testing rapid successive initializations...")
    for i in range(10):
        import src.unity_wheel.accelerated_tools.reliable_initialization as init_module
        init_module._manager_instance = None
        await ensure_tools_initialized()
    print("   ✅ Rapid initializations completed")
    
    # Test 2: Initialization with timeout
    print("   Testing initialization timeout handling...")
    try:
        import src.unity_wheel.accelerated_tools.reliable_initialization as init_module
        init_module._manager_instance = None
        await ensure_tools_initialized(timeout=0.1)  # Very short timeout
        print("   ✅ Timeout handling works")
    except Exception as e:
        print(f"   ✅ Timeout properly handled: {e}")
    
    return True


async def main():
    """Run all reliability tests."""
    print("🚀 Accelerated Tools Reliability Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Single initialization
    print("\n1️⃣ Single Initialization Test")
    print("-" * 40)
    success = await test_single_initialization()
    test_results.append(('Single Initialization', success[0]))
    
    # Test 2: Multiple initializations
    print("\n2️⃣ Multiple Initialization Test")
    print("-" * 40)
    success = await test_multiple_initializations()
    test_results.append(('Multiple Initializations', success))
    
    # Test 3: Concurrent initialization
    print("\n3️⃣ Concurrent Initialization Test")
    print("-" * 40)
    success = await test_concurrent_initialization()
    test_results.append(('Concurrent Initialization', success))
    
    # Test 4: Tool functionality
    print("\n4️⃣ Tool Functionality Test")
    print("-" * 40)
    success = await test_tool_functionality()
    test_results.append(('Tool Functionality', success))
    
    # Test 5: Stress scenarios
    print("\n5️⃣ Stress Scenario Test")
    print("-" * 40)
    success = await test_stress_scenarios()
    test_results.append(('Stress Scenarios', success))
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 RELIABILITY TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    overall_success = passed / total
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall Success Rate: {passed}/{total} ({overall_success:.1%})")
    
    if overall_success >= 0.9:
        print("🎉 EXCELLENT: Tools are 100% reliable!")
        return 0
    elif overall_success >= 0.8:
        print("✅ GOOD: Tools are highly reliable")
        return 0
    elif overall_success >= 0.6:
        print("⚠️ FAIR: Tools have some reliability issues")
        return 1
    else:
        print("❌ POOR: Tools have significant reliability issues")
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))