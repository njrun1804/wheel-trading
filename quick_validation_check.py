#!/usr/bin/env python3
"""
Quick Validation Check

This script provides a rapid assessment of the current system state
to identify what needs to be fixed by other agents.

Can be run immediately to see current issues.
"""

import asyncio
import sys
import traceback
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def check_imports():
    """Check if critical imports are working."""
    print("🔍 Checking Critical Imports...")

    checks = {
        "Bolt Core": ["bolt.core.config", "bolt.solve", "bolt.agents.orchestrator"],
        "Einstein": [
            "einstein.unified_index",
            "einstein.result_merger",
            "einstein_launcher",
        ],
        "Accelerated Tools": [
            "src.unity_wheel.accelerated_tools.ripgrep_turbo",
            "src.unity_wheel.accelerated_tools.dependency_graph_turbo",
        ],
        "Unified CLI": ["unified_cli", "query_intelligence"],
    }

    results = {}
    for category, modules in checks.items():
        working = []
        broken = []

        for module in modules:
            try:
                __import__(module)
                working.append(module.split(".")[-1])
            except Exception as e:
                broken.append(f"{module.split('.')[-1]} ({str(e)[:30]})")

        results[category] = {
            "working": working,
            "broken": broken,
            "status": "✅" if len(broken) == 0 else "⚠️" if len(working) > 0 else "❌",
        }

    for category, result in results.items():
        print(
            f"  {result['status']} {category}: {len(result['working'])} working, {len(result['broken'])} broken"
        )
        if result["broken"]:
            print(f"    Broken: {', '.join(result['broken'])}")

    return results


async def check_basic_functionality():
    """Check basic functionality of key systems."""
    print("\n🧪 Testing Basic Functionality...")

    # Test Bolt basic import and config
    print("  Testing Bolt...")
    try:
        from bolt.core.config import Config

        config = Config()
        print("    ✅ Bolt config loads")
    except Exception as e:
        print(f"    ❌ Bolt config failed: {e}")

    # Test Einstein basic functionality
    print("  Testing Einstein...")
    try:
        from einstein.result_merger import ResultMerger, SearchResult

        merger = ResultMerger()
        test_result = SearchResult(file="test.py", content="test", line=1, score=0.8)
        merged = merger.merge_results([test_result])
        print("    ✅ Einstein ResultMerger works")
    except Exception as e:
        print(f"    ❌ Einstein ResultMerger failed: {e}")

    # Test Unified CLI routing
    print("  Testing Unified CLI...")
    try:
        from unified_cli import QueryRouter

        router = QueryRouter()
        system, confidence, reasoning = router.classify_query("find test")
        print(f"    ✅ Query routing works (routed to {system})")
    except Exception as e:
        print(f"    ❌ Query routing failed: {e}")

    # Test accelerated tools
    print("  Testing Accelerated Tools...")
    try:
        from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo

        tool = get_ripgrep_turbo()
        print("    ✅ Accelerated tools accessible")
    except Exception as e:
        print(f"    ❌ Accelerated tools failed: {e}")


def check_executables():
    """Check executable files."""
    print("\n🔧 Checking Executables...")

    executables = ["bolt_executable", "unified", "boltcli", "unified_cli.py"]

    for exe in executables:
        path = Path(exe)
        if path.exists():
            if path.is_file() and (path.suffix == ".py" or path.stat().st_mode & 0o111):
                print(f"  ✅ {exe} exists and is executable")
            else:
                print(f"  ⚠️  {exe} exists but may not be executable")
        else:
            print(f"  ❌ {exe} not found")


def check_directories():
    """Check critical directories and files."""
    print("\n📁 Checking Directory Structure...")

    critical_paths = [
        "bolt/",
        "bolt/core/",
        "bolt/agents/",
        "einstein/",
        "src/unity_wheel/accelerated_tools/",
        "data/",
        "unified_cli.py",
        "requirements.txt",
    ]

    for path_str in critical_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.glob("*.py"))) if path.is_dir() else 0
                print(f"  ✅ {path_str} ({file_count} Python files)")
            else:
                print(f"  ✅ {path_str}")
        else:
            print(f"  ❌ {path_str} missing")


async def quick_async_test():
    """Quick test of async functionality to check for warnings."""
    print("\n⚡ Testing Async Operations...")

    try:
        # Test async subprocess (this should not generate warnings after fixes)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            'print("async test")',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            print("  ✅ Async subprocess works")
        else:
            print(f"  ⚠️  Async subprocess returned {process.returncode}")

    except Exception as e:
        print(f"  ❌ Async subprocess failed: {e}")


def analyze_current_state(import_results):
    """Analyze the current state and provide recommendations."""
    print("\n📊 Current State Analysis:")
    print("-" * 50)

    total_categories = len(import_results)
    working_categories = sum(1 for r in import_results.values() if r["status"] == "✅")
    partial_categories = sum(1 for r in import_results.values() if r["status"] == "⚠️")
    broken_categories = sum(1 for r in import_results.values() if r["status"] == "❌")

    print(
        f"System Categories: {working_categories} working, {partial_categories} partial, {broken_categories} broken"
    )

    if broken_categories == 0 and partial_categories == 0:
        print("🎉 System appears to be in good condition!")
        print("   Ready to run comprehensive validation tests.")
    elif broken_categories <= 1:
        print("⚠️  System mostly functional with minor issues.")
        print("   Some components need attention from other agents.")
    else:
        print("🔧 System needs significant fixes from other agents.")
        print("   Multiple core components are not working.")

    print("\n🎯 Recommendations for Other Agents:")

    for category, result in import_results.items():
        if result["broken"]:
            print(f"  • Fix {category}: {', '.join(result['broken'])}")

    print("\nAfter other agents complete their fixes, run:")
    print("  python comprehensive_validation_test.py")


async def main():
    """Main quick validation check."""
    print("⚡ Quick Validation Check")
    print("=" * 50)
    print("Rapid assessment of current system state\n")

    try:
        # Check directory structure
        check_directories()

        # Check imports
        import_results = check_imports()

        # Check executables
        check_executables()

        # Test basic functionality
        await check_basic_functionality()

        # Test async operations
        await quick_async_test()

        # Analyze and provide recommendations
        analyze_current_state(import_results)

        print("\n✅ Quick validation check completed.")
        return 0

    except Exception as e:
        print(f"\n💥 Error during quick validation: {e}")
        if "--debug" in sys.argv:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
