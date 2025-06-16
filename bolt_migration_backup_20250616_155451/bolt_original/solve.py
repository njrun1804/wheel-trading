#!/usr/bin/env python3
"""
Bolt problem solver with full 8-agent system integration.

Uses the comprehensive integration layer with hardware acceleration,
GPU routing, memory safety, and real-time monitoring for M4 Pro.
"""

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Any

import click

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the integration layer with error handling
try:
    from bolt.core.integration import BoltIntegration

    HAS_INTEGRATION = True
except ImportError as e:
    print(f"Warning: Failed to import BoltIntegration: {e}")
    HAS_INTEGRATION = False


async def analyze_and_execute(query: str, analyze_only: bool = False) -> dict[str, Any]:
    """Analyze and optionally execute a query using the integrated Bolt system.

    This uses the full integration layer with:
    - 8 parallel agents with hardware acceleration
    - Real-time M4 Pro system monitoring
    - GPU/CPU routing optimization
    - Memory safety enforcement
    - Comprehensive task orchestration
    """

    if not HAS_INTEGRATION:
        return await fallback_analyze_and_execute(query, analyze_only)

    # Create integrated system
    integration = None
    try:
        integration = BoltIntegration(num_agents=8)

        # Initialize all components
        await integration.initialize()

        if analyze_only:
            # Just analyze the query
            result = await integration.analyze_query(query)
            print("\n=== Query Analysis ===")
            print(f"Query: {result['query']}")
            print(f"Relevant files: {len(result.get('relevant_files', []))}")
            print(f"Planned tasks: {len(result.get('tasks', []))}")
            print(f"Estimated agents: {result.get('estimated_agents', 0)}")

            print("\n=== Planned Tasks ===")
            for i, task in enumerate(result.get("tasks", []), 1):
                task_desc = task.get("description", "Unknown task")
                task_priority = task.get("priority", "NORMAL")
                if hasattr(task_priority, "name"):
                    priority_name = task_priority.name
                else:
                    priority_name = str(task_priority)
                print(f"{i}. {task_desc} (Priority: {priority_name})")

        else:
            # Use the new solve method
            result = await integration.solve(query, analyze_only=False)

            print("\n=== Execution Results ===")
            print(f"Query: {query}")
            print(f"Success: {result.get('success', False)}")
            print(f"Total duration: {result.get('duration', 0):.2f}s")

            if result.get("system_metrics"):
                metrics = result["system_metrics"]
                print("\nSystem metrics:")
                print(f"  CPU: {metrics.get('cpu_percent', 0):.1f}%")
                print(f"  Memory: {metrics.get('memory_percent', 0):.1f}%")
                print(f"  GPU Memory: {metrics.get('gpu_memory_gb', 0):.1f}GB")
                print(f"  Active agents: {metrics.get('active_agents', 0)}")

            # Show results
            synthesis = result.get("results", {})
            if synthesis:
                if synthesis.get("summary"):
                    print(f"\nSummary: {synthesis['summary']}")

                if synthesis.get("findings"):
                    print("\nFindings:")
                    for finding in synthesis["findings"]:
                        print(f"  • {finding}")

                if synthesis.get("recommendations"):
                    print("\nRecommendations:")
                    for rec in synthesis["recommendations"]:
                        print(f"  • {rec}")

                if synthesis.get("errors"):
                    print("\nErrors:")
                    for error in synthesis["errors"]:
                        print(f"  • {error['task']}: {error['error']}")

        return result

    except Exception as e:
        error_msg = f"Error in analysis/execution: {str(e)}"
        print(f"\n{error_msg}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        return {"error": error_msg}

    finally:
        # Ensure clean shutdown
        if integration:
            try:
                await integration.shutdown()
            except Exception as e:
                print(f"Warning: Shutdown error: {e}", file=sys.stderr)


async def fallback_analyze_and_execute(
    query: str, analyze_only: bool = False
) -> dict[str, Any]:
    """Fallback implementation when integration layer is not available."""
    print("🔧 Using fallback implementation (limited functionality)")
    print(f"📝 Query: {query}")
    print(f"📊 Mode: {'Analysis only' if analyze_only else 'Full execution'}")

    # Basic query analysis
    result = {
        "query": query,
        "analyze_only": analyze_only,
        "fallback": True,
        "message": "Fallback implementation - limited functionality available",
    }

    # Simple pattern matching for basic recommendations
    query_lower = query.lower()
    recommendations = []

    if "optimize" in query_lower:
        recommendations.extend(
            [
                "Profile your code to identify bottlenecks",
                "Look for repeated computations that could be cached",
                "Consider using more efficient algorithms or data structures",
                "Review database queries for optimization opportunities",
            ]
        )
    elif "debug" in query_lower or "fix" in query_lower:
        recommendations.extend(
            [
                "Add logging to trace execution flow",
                "Use a debugger to step through problematic code",
                "Check for common issues like null references or type errors",
                "Review error handling and exception management",
            ]
        )
    elif "refactor" in query_lower:
        recommendations.extend(
            [
                "Identify code duplication and extract common functionality",
                "Break large functions into smaller, focused functions",
                "Improve naming conventions for better readability",
                "Consider design patterns that could improve structure",
            ]
        )
    else:
        recommendations.extend(
            [
                "Break down the problem into smaller, manageable parts",
                "Research best practices for your specific use case",
                "Consider creating a plan before implementing changes",
                "Test your changes thoroughly",
            ]
        )

    result["recommendations"] = recommendations

    print("\n💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    return result


@click.command()
@click.argument("query")
@click.option("--analyze-only", is_flag=True, help="Only analyze, don't execute")
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("--version", is_flag=True, help="Show version information")
def main(query: str, analyze_only: bool, debug: bool, version: bool):
    """Solve problems using integrated 8-agent system with M4 Pro acceleration.

    BOLT - Hardware-Accelerated 8-Agent Problem Solver

    Features:
    - 8 parallel Claude Code agents
    - Hardware-accelerated tools (MLX GPU, Metal)
    - Real-time system monitoring
    - Memory safety enforcement
    - Intelligent task orchestration
    - Einstein semantic search integration

    Examples:
        bolt solve "optimize database queries"
        bolt solve "fix memory leak in trading module" --analyze-only
        bolt solve "refactor wheel strategy code"
        bolt solve "analyze performance bottlenecks" --debug
    """

    if version:
        print("Bolt 8-Agent Problem Solver v1.0.0")
        print("Hardware-accelerated AI agents for M4 Pro")
        print("Features: Einstein integration, Metal GPU, MLX acceleration")
        return

    if not query:
        print("Error: Query is required", file=sys.stderr)
        print('Usage: bolt solve "your query here"')
        print("Use --help for more information")
        sys.exit(1)

    try:
        print("🚀 Bolt 8-Agent Problem Solver")
        print("=" * 50)

        if debug:
            print("🐛 Debug mode enabled")

        if not HAS_INTEGRATION:
            print("⚠️  Integration layer not available - using fallback mode")

        result = asyncio.run(analyze_and_execute(query, analyze_only))

        # Return appropriate exit code
        if result.get("error"):
            print("\n❌ Operation failed", file=sys.stderr)
            sys.exit(1)
        else:
            print("\n✅ Operation completed successfully")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}", file=sys.stderr)
        if debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
