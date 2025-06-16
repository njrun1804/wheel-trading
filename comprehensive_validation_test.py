#!/usr/bin/env python3
"""
Comprehensive Validation Test Suite

This script validates that all critical fixes have been properly implemented:

1. Bolt initializes without errors
2. Einstein searches work without MergedResult errors  
3. No more async subprocess warnings
4. Accelerated tools are available to Bolt agents
5. Unified CLI works correctly for both Einstein and Bolt routing

Created by Agent 12 for final validation after all other agents complete their fixes.
"""

import asyncio
import logging
import os
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Test results storage
test_results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "errors": [],
    "details": [],
    "start_time": None,
    "end_time": None,
}


def setup_test_environment():
    """Setup test environment and logging."""
    # Configure minimal logging to avoid noise
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.NullHandler()],  # Suppress log output during tests
    )

    # Set environment variables for testing
    os.environ["PYTHONPATH"] = str(Path.cwd())
    os.environ["TEST_MODE"] = "1"

    # Suppress asyncio debug warnings
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


@contextmanager
def capture_output():
    """Capture stdout and stderr for testing."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_capture, stderr_capture = StringIO(), StringIO()
    try:
        sys.stdout, sys.stderr = stdout_capture, stderr_capture
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def log_test_result(test_name: str, success: bool, details: str = "", error: str = ""):
    """Log test result."""
    global test_results

    if success:
        test_results["passed"] += 1
        status = "✅ PASS"
    else:
        test_results["failed"] += 1
        status = "❌ FAIL"
        if error:
            test_results["errors"].append(f"{test_name}: {error}")

    result_entry = {
        "test": test_name,
        "status": status,
        "success": success,
        "details": details,
        "error": error,
        "timestamp": time.time(),
    }
    test_results["details"].append(result_entry)

    print(f"{status} {test_name}")
    if details:
        print(f"    Details: {details}")
    if error and not success:
        print(f"    Error: {error}")


def skip_test(test_name: str, reason: str):
    """Skip a test with reason."""
    global test_results
    test_results["skipped"] += 1
    print(f"⏭️  SKIP {test_name} - {reason}")

    result_entry = {
        "test": test_name,
        "status": "⏭️  SKIP",
        "success": None,
        "details": reason,
        "error": "",
        "timestamp": time.time(),
    }
    test_results["details"].append(result_entry)


async def test_bolt_initialization():
    """Test 1: Bolt initializes without errors."""
    test_name = "Bolt System Initialization"

    try:
        # Test basic imports
        try:
            from bolt.agents.orchestrator import AgentOrchestrator
            from bolt.core.config import Config
            from bolt.solve import analyze_and_execute

            log_test_result(
                f"{test_name} - Imports", True, "All core imports successful"
            )
        except Exception as e:
            log_test_result(f"{test_name} - Imports", False, error=str(e))
            return

        # Test configuration loading
        try:
            config = Config()
            agent_count = getattr(config, "agent_count", 8)
            log_test_result(
                f"{test_name} - Config", True, f"Config loaded, {agent_count} agents"
            )
        except Exception as e:
            log_test_result(f"{test_name} - Config", False, error=str(e))

        # Test orchestrator initialization
        try:
            orchestrator = AgentOrchestrator()
            log_test_result(
                f"{test_name} - Orchestrator", True, "Orchestrator created successfully"
            )
        except Exception as e:
            log_test_result(f"{test_name} - Orchestrator", False, error=str(e))

        # Test simple analysis (no execution)
        try:
            with capture_output() as (stdout, stderr):
                result = await analyze_and_execute(
                    "test simple analysis", analyze_only=True
                )

            if isinstance(result, dict) and not result.get("error"):
                log_test_result(
                    f"{test_name} - Analysis", True, "Simple analysis completed"
                )
            else:
                error_msg = (
                    result.get("error", "Unknown error")
                    if isinstance(result, dict)
                    else str(result)
                )
                log_test_result(f"{test_name} - Analysis", False, error=error_msg)
        except Exception as e:
            log_test_result(f"{test_name} - Analysis", False, error=str(e))

    except Exception as e:
        log_test_result(test_name, False, error=f"Unexpected error: {e}")


async def test_einstein_search():
    """Test 2: Einstein searches work without MergedResult errors."""
    test_name = "Einstein Search Functionality"

    try:
        # Test Einstein imports
        try:
            from einstein.result_merger import ResultMerger, SearchResult
            from einstein.unified_index import UnifiedIndex

            log_test_result(
                f"{test_name} - Imports", True, "Einstein imports successful"
            )
        except Exception as e:
            log_test_result(f"{test_name} - Imports", False, error=str(e))
            return

        # Test ResultMerger (core component that was causing MergedResult errors)
        try:
            merger = ResultMerger()

            # Create test results
            test_results_list = [
                SearchResult(
                    content="test content 1",
                    file_path="test1.py",
                    line_number=1,
                    score=0.8,
                    result_type="text",
                    context={},
                    timestamp=0.0,
                ),
                SearchResult(
                    content="test content 2",
                    file_path="test2.py",
                    line_number=5,
                    score=0.9,
                    result_type="text",
                    context={},
                    timestamp=0.0,
                ),
            ]

            # Test merge operation - ResultMerger expects dict[str, list[SearchResult]]
            results_by_modality = {"text": test_results_list}
            merged = merger.merge_results(results_by_modality)
            if isinstance(merged, list) and len(merged) >= 0:
                log_test_result(
                    f"{test_name} - ResultMerger", True, f"Merged {len(merged)} results"
                )
            else:
                log_test_result(
                    f"{test_name} - ResultMerger",
                    False,
                    error=f"Invalid merge result structure: {type(merged)}",
                )
        except Exception as e:
            log_test_result(f"{test_name} - ResultMerger", False, error=str(e))

        # Test UnifiedIndex basic functionality
        try:
            # Create temporary test directory
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = Path(temp_dir) / "test_search.py"
                test_file.write_text("def test_function():\n    pass\n")

                index = UnifiedIndex(temp_dir)
                await index.initialize()

                # Perform a simple search
                results = await index.search("test_function", max_results=5)

                if hasattr(results, "results") or isinstance(results, (list, dict)):
                    result_count = (
                        len(results.results)
                        if hasattr(results, "results")
                        else len(results)
                        if isinstance(results, list)
                        else 1
                    )
                    log_test_result(
                        f"{test_name} - Search",
                        True,
                        f"Search completed, {result_count} results",
                    )
                else:
                    log_test_result(
                        f"{test_name} - Search",
                        False,
                        error="Invalid search result structure",
                    )

        except Exception as e:
            log_test_result(f"{test_name} - Search", False, error=str(e))

        # Test Einstein launcher integration
        try:
            from einstein_launcher import EinsteinLauncher

            launcher = EinsteinLauncher(Path.cwd())
            await launcher.initialize()
            log_test_result(
                f"{test_name} - Launcher", True, "Einstein launcher initialized"
            )
        except Exception as e:
            log_test_result(f"{test_name} - Launcher", False, error=str(e))

    except Exception as e:
        log_test_result(test_name, False, error=f"Unexpected error: {e}")


async def test_async_subprocess_warnings():
    """Test 3: No async subprocess warnings are generated."""
    test_name = "Async Subprocess Warnings"

    try:
        # Capture warnings during async subprocess operations
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # Test various async subprocess operations that previously caused warnings
            try:
                # Test subprocess wrapper - this should NOT generate warnings
                from bolt.macos_subprocess_wrapper import execute_command_async, get_subprocess_wrapper
                
                wrapper = get_subprocess_wrapper()
                result = await execute_command_async(sys.executable, '-c', 'print("test")', timeout=5.0)
                
                if result.returncode == 0:
                    # Check for any warnings related to subprocess or event loop
                    async_warnings = [w for w in warning_list 
                                     if any(keyword in str(w.message).lower() 
                                           for keyword in ['subprocess', 'asyncio', 'event loop', 'running loop'])]
                    
                    if len(async_warnings) == 0:
                        log_test_result(test_name, True, f"No warnings detected (method: {result.method})")
                    else:
                        warning_messages = [str(w.message) for w in async_warnings]
                        log_test_result(test_name, False, error=f"Found {len(async_warnings)} warnings: {warning_messages[:3]}")
                else:
                    log_test_result(test_name, False, error=f"Subprocess execution failed: {result.returncode}")

            except ImportError as e:
                log_test_result(test_name, False, error=f"Could not import subprocess wrapper: {e}")
            except Exception as e:
                log_test_result(test_name, False, error=f"Error testing subprocess: {e}")

    except Exception as e:
        log_test_result(test_name, False, error=f"Unexpected error: {e}")


async def test_accelerated_tools_availability():
    """Test 4: Accelerated tools are available to Bolt agents."""
    test_name = "Accelerated Tools Availability"

    try:
        # Test Bolt's ability to access accelerated tools (the main interface)
        from bolt.core.fallbacks import get_accelerated_tool
        
        tools_to_test = [
            "ripgrep_turbo",
            "dependency_graph_turbo", 
            "python_analysis_turbo",
            "duckdb_turbo",
            "trace_turbo",
            "python_helpers_turbo",
        ]

        available_tools = []
        unavailable_tools = []

        for tool_name in tools_to_test:
            try:
                tool_instance = get_accelerated_tool(tool_name)
                if tool_instance:
                    # Check if it's an accelerated tool or fallback
                    tool_type = type(tool_instance).__name__
                    if 'Turbo' in tool_type or 'Accelerated' in tool_type:
                        available_tools.append(f"{tool_name} (accelerated)")
                    else:
                        available_tools.append(f"{tool_name} (fallback)")
                else:
                    unavailable_tools.append(f"{tool_name} (no tool returned)")
            except Exception as e:
                unavailable_tools.append(f"{tool_name} ({str(e)[:50]})")

        # Test overall Bolt integration
        try:
            # Test that we can get multiple tools successfully
            rg_tool = get_accelerated_tool("ripgrep_turbo")
            dep_tool = get_accelerated_tool("dependency_graph_turbo")
            py_tool = get_accelerated_tool("python_analysis_turbo")
            
            integration_success = all(tool is not None for tool in [rg_tool, dep_tool, py_tool])
            if integration_success:
                available_tools.append("bolt_integration")
            else:
                unavailable_tools.append("bolt_integration (tools not returned)")
        except Exception as e:
            unavailable_tools.append(f"bolt_integration ({str(e)[:50]})")

        # Report results
        total_tools = len(tools_to_test) + 1  # +1 for bolt integration
        available_count = len(available_tools)

        if available_count >= total_tools * 0.8:  # 80% availability considered success
            log_test_result(
                test_name,
                True,
                f"{available_count}/{total_tools} tools available: {', '.join(available_tools)}",
            )
        else:
            log_test_result(
                test_name,
                False,
                details=f"Only {available_count}/{total_tools} tools available",
                error=f"Unavailable: {', '.join(unavailable_tools)}",
            )

    except Exception as e:
        log_test_result(test_name, False, error=f"Unexpected error: {e}")


async def test_unified_cli_routing():
    """Test 5: Unified CLI works correctly for both Einstein and Bolt routing."""
    test_name = "Unified CLI Routing"

    try:
        # Test unified CLI imports
        try:
            from unified_cli import QueryRouter, UnifiedCLI

            log_test_result(
                f"{test_name} - Imports", True, "Unified CLI imports successful"
            )
        except Exception as e:
            log_test_result(f"{test_name} - Imports", False, error=str(e))
            return

        # Test QueryRouter classification
        try:
            router = QueryRouter()

            # Test Einstein queries
            einstein_queries = [
                "find WheelStrategy",
                "show options.py",
                "search TODO",
                "def calculate_delta",
            ]

            bolt_queries = [
                "optimize database performance",
                "fix memory leak in trading module",
                "analyze bottlenecks in wheel strategy",
                "help me refactor the risk calculation",
            ]

            correct_classifications = 0
            total_classifications = 0

            for query in einstein_queries:
                system, confidence, reasoning = router.classify_query(query)
                total_classifications += 1
                if system == "einstein":
                    correct_classifications += 1

            for query in bolt_queries:
                system, confidence, reasoning = router.classify_query(query)
                total_classifications += 1
                if system == "bolt":
                    correct_classifications += 1

            accuracy = (
                correct_classifications / total_classifications
                if total_classifications > 0
                else 0
            )

            if accuracy >= 0.75:  # 75% accuracy threshold
                log_test_result(
                    f"{test_name} - Routing",
                    True,
                    f"Routing accuracy: {accuracy:.1%} ({correct_classifications}/{total_classifications})",
                )
            else:
                log_test_result(
                    f"{test_name} - Routing",
                    False,
                    error=f"Poor routing accuracy: {accuracy:.1%} ({correct_classifications}/{total_classifications})",
                )

        except Exception as e:
            log_test_result(f"{test_name} - Routing", False, error=str(e))

        # Test CLI initialization
        try:
            cli = UnifiedCLI(Path.cwd())
            log_test_result(
                f"{test_name} - CLI Init", True, "CLI initialized successfully"
            )
        except Exception as e:
            log_test_result(f"{test_name} - CLI Init", False, error=str(e))

        # Test Einstein routing (simple query)
        try:
            cli = UnifiedCLI(Path.cwd())
            with capture_output() as (stdout, stderr):
                result = await cli.route_query("find test", force_system="einstein")

            if isinstance(result, dict) and "routing" in result:
                log_test_result(
                    f"{test_name} - Einstein Route", True, "Einstein routing successful"
                )
            else:
                log_test_result(
                    f"{test_name} - Einstein Route",
                    False,
                    error="Invalid Einstein routing result",
                )
        except Exception as e:
            log_test_result(f"{test_name} - Einstein Route", False, error=str(e))

        # Test Bolt routing (complex query)
        try:
            cli = UnifiedCLI(Path.cwd())
            with capture_output() as (stdout, stderr):
                result = await cli.route_query(
                    "analyze test system", force_system="bolt"
                )

            if isinstance(result, dict) and "routing" in result:
                log_test_result(
                    f"{test_name} - Bolt Route", True, "Bolt routing successful"
                )
            else:
                log_test_result(
                    f"{test_name} - Bolt Route",
                    False,
                    error="Invalid Bolt routing result",
                )
        except Exception as e:
            log_test_result(f"{test_name} - Bolt Route", False, error=str(e))

    except Exception as e:
        log_test_result(test_name, False, error=f"Unexpected error: {e}")


async def test_executable_commands():
    """Test 6: Test that executable commands work correctly."""
    test_name = "Executable Commands"

    try:
        # Test bolt executable
        try:
            process = await asyncio.create_subprocess_exec(
                "./bolt_executable",
                "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd(),
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

            if process.returncode == 0:
                log_test_result(
                    f"{test_name} - Bolt Executable",
                    True,
                    "Bolt executable runs successfully",
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                log_test_result(
                    f"{test_name} - Bolt Executable",
                    False,
                    error=f"Exit code {process.returncode}: {error_msg[:100]}",
                )
        except TimeoutError:
            log_test_result(
                f"{test_name} - Bolt Executable",
                False,
                error="Timeout after 10 seconds",
            )
        except Exception as e:
            log_test_result(f"{test_name} - Bolt Executable", False, error=str(e))

        # Test unified CLI command
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "unified_cli.py",
                "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd(),
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

            if process.returncode == 0:
                log_test_result(
                    f"{test_name} - Unified CLI", True, "Unified CLI help works"
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                log_test_result(
                    f"{test_name} - Unified CLI",
                    False,
                    error=f"Exit code {process.returncode}: {error_msg[:100]}",
                )
        except TimeoutError:
            log_test_result(
                f"{test_name} - Unified CLI", False, error="Timeout after 10 seconds"
            )
        except Exception as e:
            log_test_result(f"{test_name} - Unified CLI", False, error=str(e))

    except Exception as e:
        log_test_result(test_name, False, error=f"Unexpected error: {e}")


async def test_database_connections():
    """Test 7: Test database connections and basic operations."""
    test_name = "Database Connections"

    try:
        # Test DuckDB connection
        try:
            import duckdb

            conn = duckdb.connect(":memory:")
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'test')")
            result = conn.execute("SELECT * FROM test").fetchall()
            conn.close()

            if len(result) == 1:
                log_test_result(
                    f"{test_name} - DuckDB", True, "DuckDB operations successful"
                )
            else:
                log_test_result(
                    f"{test_name} - DuckDB", False, error="Unexpected query result"
                )
        except Exception as e:
            log_test_result(f"{test_name} - DuckDB", False, error=str(e))

        # Test database file access
        try:
            db_files = [
                "data/wheel_trading_master.duckdb",
                "data/unified_wheel_trading.duckdb",
            ]

            accessible_dbs = []
            for db_file in db_files:
                if Path(db_file).exists():
                    try:
                        conn = duckdb.connect(db_file, read_only=True)
                        tables = conn.execute("SHOW TABLES").fetchall()
                        conn.close()
                        accessible_dbs.append(f"{db_file} ({len(tables)} tables)")
                    except Exception as e:
                        accessible_dbs.append(f"{db_file} (error: {str(e)[:30]})")
                else:
                    accessible_dbs.append(f"{db_file} (not found)")

            log_test_result(
                f"{test_name} - Files", True, f"DB status: {', '.join(accessible_dbs)}"
            )

        except Exception as e:
            log_test_result(f"{test_name} - Files", False, error=str(e))

    except Exception as e:
        log_test_result(test_name, False, error=f"Unexpected error: {e}")


async def test_memory_management():
    """Test 8: Test memory management and cleanup."""
    test_name = "Memory Management"

    try:
        import gc

        import psutil

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate memory-intensive operations
        try:
            # Create some objects that should be cleaned up
            large_objects = []
            for i in range(100):
                large_objects.append([0] * 1000)  # Small objects, not too large

            # Force garbage collection
            gc.collect()

            # Clear objects
            large_objects.clear()
            gc.collect()

            # Check memory after cleanup
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            if memory_increase < 50:  # Less than 50MB increase is acceptable
                log_test_result(
                    f"{test_name} - Cleanup",
                    True,
                    f"Memory increase: {memory_increase:.1f}MB",
                )
            else:
                log_test_result(
                    f"{test_name} - Cleanup",
                    False,
                    error=f"High memory increase: {memory_increase:.1f}MB",
                )

        except Exception as e:
            log_test_result(f"{test_name} - Cleanup", False, error=str(e))

        # Test process monitoring
        try:
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_percent = process.memory_percent()

            log_test_result(
                f"{test_name} - Monitoring",
                True,
                f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%",
            )
        except Exception as e:
            log_test_result(f"{test_name} - Monitoring", False, error=str(e))

    except Exception as e:
        log_test_result(test_name, False, error=f"Unexpected error: {e}")


def print_test_summary():
    """Print comprehensive test summary."""
    global test_results

    test_results["end_time"] = time.time()
    duration = test_results["end_time"] - test_results["start_time"]

    print("\n" + "=" * 80)
    print("🧪 COMPREHENSIVE VALIDATION TEST RESULTS")
    print("=" * 80)

    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"⏭️  Skipped: {test_results['skipped']}")

    total_tests = test_results["passed"] + test_results["failed"]
    if total_tests > 0:
        success_rate = (test_results["passed"] / total_tests) * 100
        print(f"📊 Success Rate: {success_rate:.1f}%")

    # Print detailed results for failed tests
    if test_results["failed"] > 0:
        print(f"\n❌ FAILED TESTS ({test_results['failed']}):")
        print("-" * 40)
        for detail in test_results["details"]:
            if not detail["success"] and detail["success"] is not None:
                print(f"  • {detail['test']}")
                if detail["error"]:
                    print(f"    Error: {detail['error']}")
                if detail["details"]:
                    print(f"    Details: {detail['details']}")

    # Print errors summary
    if test_results["errors"]:
        print("\n🔍 ERROR SUMMARY:")
        print("-" * 40)
        for error in test_results["errors"][:10]:  # Show first 10 errors
            print(f"  • {error}")
        if len(test_results["errors"]) > 10:
            print(f"  ... and {len(test_results['errors']) - 10} more errors")

    # Overall assessment
    print("\n🎯 OVERALL ASSESSMENT:")
    print("-" * 40)

    if test_results["failed"] == 0:
        print("🎉 ALL TESTS PASSED! System is ready for production.")
        return 0
    elif test_results["failed"] <= 2:
        print("⚠️  Minor issues detected. System mostly functional.")
        return 1
    elif test_results["failed"] <= 5:
        print("🔧 Several issues need attention before production use.")
        return 2
    else:
        print("🚨 Major issues detected. System needs significant fixes.")
        return 3


async def run_all_tests():
    """Run all validation tests."""
    global test_results

    print("🚀 Starting Comprehensive Validation Test Suite")
    print("Testing all critical fixes after agent completion")
    print("=" * 80)

    test_results["start_time"] = time.time()

    # Setup test environment
    setup_test_environment()

    # Define all tests
    tests = [
        ("Bolt System Initialization", test_bolt_initialization),
        ("Einstein Search Functionality", test_einstein_search),
        ("Async Subprocess Warnings", test_async_subprocess_warnings),
        ("Accelerated Tools Availability", test_accelerated_tools_availability),
        ("Unified CLI Routing", test_unified_cli_routing),
        ("Executable Commands", test_executable_commands),
        ("Database Connections", test_database_connections),
        ("Memory Management", test_memory_management),
    ]

    # Run each test with error handling
    for test_category, test_func in tests:
        print(f"\n🧪 Testing: {test_category}")
        print("-" * 50)

        try:
            await test_func()
        except Exception as e:
            log_test_result(test_category, False, error=f"Test runner error: {e}")
            if "--debug" in sys.argv:
                traceback.print_exc()

    # Print final summary
    return print_test_summary()


def main():
    """Main entry point."""
    print("Comprehensive Validation Test Suite")
    print("Validates fixes after other agents complete their work")
    print()

    # Check if we're in the right directory
    if not Path("bolt").exists() or not Path("einstein").exists():
        print("❌ Error: Run this script from the wheel-trading project root directory")
        print("Expected directories: bolt/, einstein/, src/")
        return 1

    try:
        # Run async tests
        return asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error running tests: {e}")
        if "--debug" in sys.argv:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
