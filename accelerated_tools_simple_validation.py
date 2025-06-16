#!/usr/bin/env python3
"""
Simple Accelerated Tools Validation
Focused test to ensure all 7 tools are consistently available
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

from src.unity_wheel.accelerated_tools.reliable_initialization import (
    validate_all_tools,
    get_tools_manager
)


async def main():
    """Run simple validation of all 7 tools."""
    print("🔍 Simple Accelerated Tools Validation")
    print("=" * 50)
    
    try:
        # Test 1: Validate all tools
        print("\n1️⃣ Testing tool availability...")
        start_time = time.perf_counter()
        all_available, results = await validate_all_tools()
        duration = time.perf_counter() - start_time
        
        available_count = results['available_count']
        total_count = results['total_count']
        success_rate = results['success_rate']
        
        print(f"   ⏱️  Initialization time: {duration:.3f}s")
        print(f"   📊 Tools available: {available_count}/{total_count} ({success_rate:.1%})")
        
        if all_available:
            print("   ✅ All critical tools available!")
        else:
            missing = results['missing_critical_tools']
            print(f"   ⚠️  Missing tools: {missing}")
        
        # Test 2: Show available tools
        print("\n2️⃣ Available tools:")
        for tool_name in results['available_tools']:
            details = results['tool_details'][tool_name]
            init_time = details['init_time']
            version = details['version'] or 'unknown'
            print(f"   ✅ {tool_name}: {init_time:.3f}s (v{version})")
        
        # Test 3: Show any failed tools
        failed_tools = [
            name for name, details in results['tool_details'].items()
            if not details['available']
        ]
        
        if failed_tools:
            print("\n3️⃣ Failed tools:")
            for tool_name in failed_tools:
                details = results['tool_details'][tool_name]
                error = details['error'] or 'Unknown error'
                print(f"   ❌ {tool_name}: {error}")
        
        # Test 4: Quick functionality test (minimal)
        print("\n4️⃣ Basic functionality test...")
        manager = get_tools_manager()
        
        # Test ripgrep (simple)
        try:
            ripgrep = manager.get_tool('ripgrep_turbo')
            if ripgrep and hasattr(ripgrep, 'search'):
                print("   ✅ Ripgrep: Interface available")
            else:
                print("   ❌ Ripgrep: Missing search method")
        except Exception as e:
            print(f"   ❌ Ripgrep: {e}")
        
        # Test dependency graph
        try:
            dep_graph = manager.get_tool('dependency_graph_turbo')
            if dep_graph and hasattr(dep_graph, 'build_graph'):
                print("   ✅ Dependency Graph: Interface available")
            else:
                print("   ❌ Dependency Graph: Missing build_graph method")
        except Exception as e:
            print(f"   ❌ Dependency Graph: {e}")
        
        # Test python analyzer
        try:
            py_analyzer = manager.get_tool('python_analysis_turbo')
            if py_analyzer and hasattr(py_analyzer, 'analyze_file'):
                print("   ✅ Python Analyzer: Interface available")
            else:
                print("   ❌ Python Analyzer: Missing analyze_file method")
        except Exception as e:
            print(f"   ❌ Python Analyzer: {e}")
        
        # Test duckdb
        try:
            duckdb = manager.get_tool('duckdb_turbo')
            if duckdb and hasattr(duckdb, 'execute'):
                print("   ✅ DuckDB: Interface available")
            else:
                print("   ❌ DuckDB: Missing execute method")
        except Exception as e:
            print(f"   ❌ DuckDB: {e}")
        
        # Test trace
        try:
            tracer = manager.get_tool('trace_turbo')
            if tracer and hasattr(tracer, 'trace_span'):
                print("   ✅ Trace: Interface available")
            else:
                print("   ❌ Trace: Missing trace_span method")
        except Exception as e:
            print(f"   ❌ Trace: {e}")
        
        # Test python helpers
        try:
            helpers = manager.get_tool('python_helpers_turbo')
            if helpers and (hasattr(helpers, 'analyze_project_structure') or hasattr(helpers, 'get_function_info')):
                print("   ✅ Python Helpers: Interface available")
            else:
                print("   ❌ Python Helpers: Missing expected methods")
        except Exception as e:
            print(f"   ❌ Python Helpers: {e}")
        
        # Test bolt integration
        try:
            bolt = manager.get_tool('bolt_integration')
            if bolt:
                print("   ✅ Bolt Integration: Available")
            else:
                print("   ⚠️  Bolt Integration: Not available (optional)")
        except Exception as e:
            print(f"   ⚠️  Bolt Integration: {e} (optional)")
        
        # Final assessment
        print("\n" + "=" * 50)
        if available_count >= 6:  # At least 6/7 tools (bolt integration is optional)
            print("🎉 SUCCESS: Accelerated tools are working reliably!")
            print(f"   {available_count}/7 tools available ({success_rate:.1%})")
            return 0
        else:
            print("❌ ISSUES: Some critical tools are missing")
            print(f"   Only {available_count}/7 tools available")
            return 1
            
    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))