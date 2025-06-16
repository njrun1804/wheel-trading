#!/usr/bin/env python3
"""
Bolt Einstein CLI - Simple command-line interface for Einstein search.
Provides fast semantic search with robust fallbacks.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))


async def einstein_search(
    query: str, max_results: int = 10, analyze_only: bool = False
):
    """Perform Einstein search with timing and fallback handling."""
    start_time = time.time()

    try:
        from bolt.core.integration import BoltIntegration

        # Initialize with minimal agents for faster startup
        integration = BoltIntegration(num_agents=1, enable_error_handling=False)
        init_time = time.time() - start_time

        print(f"🚀 Initialized in {init_time:.1f}s")

        # Quick initialization for search-only usage
        await integration.einstein_index.initialize()
        search_ready_time = time.time() - start_time

        print(f"🔍 Search ready in {search_ready_time:.1f}s")

        # Perform search
        search_start = time.time()
        results = await integration.einstein_index.search(
            query, max_results=max_results
        )
        search_time = (time.time() - search_start) * 1000

        print(f"📊 Found {len(results)} results in {search_time:.1f}ms")

        if results:
            print("\n📋 Results:")
            for i, result in enumerate(results):
                print(f"  {i+1:2d}. {result.file_path}")
                print(f"      Score: {result.score:.3f} | Type: {result.result_type}")
                if hasattr(result, "line_number") and result.line_number > 0:
                    print(f"      Line: {result.line_number}")
                if len(result.content) > 100:
                    print(f"      Preview: {result.content[:97]}...")
                else:
                    print(f"      Content: {result.content}")
                print()
        else:
            print("🔍 No results found. Try a different query or check the index.")

        # Clean shutdown
        await integration.shutdown()
        total_time = time.time() - start_time
        print(f"⏱️  Total time: {total_time:.1f}s")

        return len(results)

    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback

        traceback.print_exc()
        return 0


async def einstein_solve(instruction: str, analyze_only: bool = True):
    """Perform full Bolt solve workflow with Einstein."""
    start_time = time.time()

    try:
        from bolt.core.integration import BoltIntegration

        # Initialize with more agents for solving
        integration = BoltIntegration(num_agents=4)
        await integration.initialize()
        init_time = time.time() - start_time

        print(f"🚀 System initialized in {init_time:.1f}s")

        # Execute solve
        print(f"🔧 Solving: '{instruction}'")
        if analyze_only:
            print("📊 Analysis mode (no changes will be made)")

        result = await integration.solve(instruction, analyze_only=analyze_only)
        solve_time = time.time() - start_time

        if result["success"]:
            print(f"✅ Completed in {solve_time:.1f}s")
            print(f"📋 Tasks executed: {result['tasks_executed']}")

            synthesis = result["results"]
            if synthesis.get("findings"):
                print(f"\n🔍 Findings ({len(synthesis['findings'])}):")
                for finding in synthesis["findings"][:5]:  # Show top 5
                    print(f"  • {finding}")

            if synthesis.get("recommendations"):
                print(f"\n💡 Recommendations ({len(synthesis['recommendations'])}):")
                for rec in synthesis["recommendations"][:3]:  # Show top 3
                    print(f"  • {rec}")

            if synthesis.get("errors"):
                print(f"\n⚠️  Errors ({len(synthesis['errors'])}):")
                for error in synthesis["errors"]:
                    print(f"  • {error['task']}: {error['error']}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

        await integration.shutdown()
        return result["success"]

    except Exception as e:
        print(f"❌ Solve failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bolt Einstein CLI - Fast semantic search and AI-powered analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick search
  python bolt_einstein_cli.py search "wheel trading strategy"
  
  # Limited results
  python bolt_einstein_cli.py search "options pricing" --max-results 5
  
  # Full analysis
  python bolt_einstein_cli.py solve "optimize trading performance"
  
  # Analysis with modifications
  python bolt_einstein_cli.py solve "fix database performance" --execute
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Semantic search through codebase"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-n",
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results (default: 10)",
    )

    # Solve command
    solve_parser = subparsers.add_parser(
        "solve", help="AI-powered analysis and optimization"
    )
    solve_parser.add_argument("instruction", help="Task instruction for AI system")
    solve_parser.add_argument(
        "--execute", action="store_true", help="Execute changes (default: analyze only)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "search":
        result_count = asyncio.run(einstein_search(args.query, args.max_results))
        return 0 if result_count > 0 else 1

    elif args.command == "solve":
        success = asyncio.run(
            einstein_solve(args.instruction, analyze_only=not args.execute)
        )
        return 0 if success else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
