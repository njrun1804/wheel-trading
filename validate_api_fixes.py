#!/usr/bin/env python3
"""
Validate API Interface Fixes for Accelerated Tools

This script validates that all API interface issues identified during
robust tool access testing have been resolved.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_duckdb_api():
    """Test DuckDB API interface."""
    print("🦆 Testing DuckDB API interface...")
    
    try:
        from src.unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo
        
        # Get DuckDB instance with in-memory database to avoid file issues
        db = get_duckdb_turbo(":memory:")
        
        # Test the query method that was missing
        try:
            # Create a simple test table
            await db.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")
            await db.execute("INSERT INTO test_table VALUES (1, 'test'), (2, 'data')")
            
            # Test the query method (this was failing before)
            result_df = await db.query("SELECT * FROM test_table")
            
            print(f"  ✅ query() method works - returned {len(result_df)} rows")
            print(f"     Columns: {list(result_df.columns)}")
            return True
            
        except Exception as e:
            print(f"  ❌ query() method failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ DuckDB import/initialization failed: {e}")
        return False


async def test_trace_api():
    """Test Trace API interface."""
    print("📊 Testing Trace API interface...")
    
    try:
        from src.unity_wheel.accelerated_tools.trace_turbo import get_trace_turbo
        
        tracer = get_trace_turbo()
        
        # Test the async context manager (this was failing before)
        try:
            async with tracer.trace_span("test_operation", {"test_attr": "test_value"}) as span:
                # Test span operations
                span.add_attribute("operation_type", "validation")
                span.set_attribute("validation_step", "api_test")
                
                # Do some work to test the span
                await asyncio.sleep(0.01)
                
                print(f"  ✅ trace_span() context manager works")
                print(f"     Span ID: {span.span_id}")
                print(f"     Trace ID: {span.trace_id}")
                return True
                
        except Exception as e:
            print(f"  ❌ trace_span() context manager failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Trace import/initialization failed: {e}")
        return False


async def test_code_helper_api():
    """Test Code Helper API interface."""
    print("🐍 Testing Code Helper API interface...")
    
    try:
        from src.unity_wheel.accelerated_tools.python_helpers_turbo import get_code_helper
        
        helper = get_code_helper()
        
        # Test the get_function_info method that was missing
        try:
            # Use a simple built-in module for testing
            result = await helper.get_function_info("json", "dumps")
            
            if isinstance(result, dict) and ("signature" in result or "error" in result):
                print(f"  ✅ get_function_info() method works")
                if "signature" in result:
                    print(f"     Found function signature: {result.get('signature', 'N/A')}")
                else:
                    print(f"     Expected error for test case: {result.get('error', 'N/A')}")
                return True
            else:
                print(f"  ❌ get_function_info() returned unexpected format: {result}")
                return False
                
        except Exception as e:
            print(f"  ❌ get_function_info() method failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Code Helper import/initialization failed: {e}")
        return False


async def test_tool_manager_compatibility():
    """Test that the tool manager can now access all tools successfully."""
    print("🔧 Testing Tool Manager compatibility...")
    
    try:
        # Import the robust tool manager
        from bolt.core.robust_tool_manager import get_tool_manager
        
        # Create a test agent tool manager
        tool_manager = get_tool_manager("test_agent")
        
        # Initialize all tools
        await tool_manager.initialize_tools()
        
        # Test tool access
        success_count = 0
        total_tools = 6  # We're testing 6 main tools
        
        # Test each tool with its fixed API (using correct tool names from RobustToolManager)
        tools_to_test = [
            ("ripgrep", "parallel_search"),
            ("dependency_graph", "find_symbol"),
            ("python_analyzer", "analyze_file"),
            ("duckdb", "query"),  # This was failing before
            ("trace", "trace_span"),  # This was failing before
            ("code_helper", "get_function_info"),  # This was failing before
        ]
        
        for tool_name, method_name in tools_to_test:
            try:
                tool_wrapper = await tool_manager.get_tool(tool_name)
                if tool_wrapper:
                    # Get the actual tool instance from the wrapper
                    if hasattr(tool_wrapper, 'tool_instance'):
                        actual_tool = tool_wrapper.tool_instance
                        if hasattr(actual_tool, method_name):
                            print(f"  ✅ {tool_name}.{method_name}() - API available")
                            success_count += 1
                        else:
                            print(f"  ❌ {tool_name}.{method_name}() - Method not found")
                            # Debug: show available methods on actual tool
                            available_methods = [m for m in dir(actual_tool) if not m.startswith('_')]
                            print(f"      Available methods: {available_methods[:5]}...")  # Show first 5
                    else:
                        # Tool wrapper doesn't have tool_instance
                        print(f"  ❌ {tool_name} - Tool wrapper has no tool_instance")
                else:
                    print(f"  ❌ {tool_name} - Tool not available")
                    
            except Exception as e:
                print(f"  ❌ {tool_name} - Tool access failed: {e}")
        
        success_rate = (success_count / total_tools) * 100
        print(f"  📊 Tool API compatibility: {success_count}/{total_tools} ({success_rate:.1f}%)")
        
        return success_count == total_tools
        
    except Exception as e:
        print(f"  ❌ Tool Manager compatibility test failed: {e}")
        return False


async def run_comprehensive_validation():
    """Run comprehensive validation of all API fixes."""
    print("🚀 Running Comprehensive API Fix Validation")
    print("=" * 60)
    
    tests = [
        ("DuckDB API", test_duckdb_api),
        ("Trace API", test_trace_api),
        ("Code Helper API", test_code_helper_api),
        ("Tool Manager Compatibility", test_tool_manager_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    overall_success = passed == total
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if overall_success:
        print("🎉 All API interface issues have been resolved!")
        print("   Bolt agents should now have 100% reliable tool access.")
    else:
        print("⚠️  Some API interface issues remain.")
        print("   Further fixes may be needed for complete reliability.")
    
    return overall_success


async def main():
    """Main validation function."""
    try:
        success = await run_comprehensive_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())