#!/usr/bin/env python3
"""
Optimized Bolt solver with <50ms Einstein integration
Replaces the standard solve.py with ultra-fast query processing
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import click

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import optimized integration
try:
    from bolt.core.optimized_integration import (
        OptimizedEinsteinBoltIntegration,
        benchmark_integration,
        create_optimized_integration,
    )

    HAS_OPTIMIZED_INTEGRATION = True
except ImportError as e:
    print(f"Warning: Failed to import optimized integration: {e}")
    HAS_OPTIMIZED_INTEGRATION = False

# Fallback to standard integration
try:
    from bolt.core.integration import BoltIntegration

    HAS_STANDARD_INTEGRATION = True
except ImportError as e:
    print(f"Warning: Failed to import standard integration: {e}")
    HAS_STANDARD_INTEGRATION = False


class OptimizedSolver:
    """High-performance solver with sub-50ms target."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.integration: OptimizedEinsteinBoltIntegration = None
        self.performance_log = []

    async def initialize(self):
        """Initialize the optimized solver."""
        if not HAS_OPTIMIZED_INTEGRATION:
            raise RuntimeError("Optimized integration not available")

        start_time = time.time()
        self.integration = await create_optimized_integration(self.project_root)
        init_time = (time.time() - start_time) * 1000

        print(f"🚀 Optimized solver initialized in {init_time:.1f}ms")
        return self

    async def solve_fast(self, query: str, execute: bool = True) -> dict[str, Any]:
        """Fast solve with <50ms target."""
        if not self.integration:
            await self.initialize()

        result = await self.integration.query_end_to_end(query, execute=execute)

        # Log performance
        self.performance_log.append(
            {
                "query": query[:50],
                "time_ms": result["total_time_ms"],
                "target_met": result["target_met"],
                "timestamp": time.time(),
            }
        )

        return result

    async def analyze_query(self, query: str) -> dict[str, Any]:
        """Fast query analysis only."""
        return await self.solve_fast(query, execute=False)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self.performance_log:
            return {}

        times = [entry["time_ms"] for entry in self.performance_log]
        target_met = sum(1 for entry in self.performance_log if entry["target_met"])

        return {
            "total_queries": len(self.performance_log),
            "average_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "target_achievement_rate": target_met / len(self.performance_log),
            "sub_50ms_count": target_met,
            "recent_queries": self.performance_log[-5:],  # Last 5 queries
        }


async def solve_with_fallback(query: str, analyze_only: bool = False) -> dict[str, Any]:
    """Solve with optimized integration and fallback to standard."""

    # Try optimized solver first
    if HAS_OPTIMIZED_INTEGRATION:
        try:
            solver = OptimizedSolver()

            if analyze_only:
                result = await solver.analyze_query(query)
            else:
                result = await solver.solve_fast(query, execute=True)

            # Add solver type to result
            result["solver_type"] = "optimized"
            result["fallback_used"] = False

            return result

        except Exception as e:
            print(f"⚠️ Optimized solver failed, falling back to standard: {e}")

    # Fallback to standard integration
    if HAS_STANDARD_INTEGRATION:
        try:
            integration = BoltIntegration(num_agents=8)
            await integration.initialize()

            if analyze_only:
                result = await integration.analyze_query(query)
            else:
                result = await integration.solve(query, analyze_only=False)

            result["solver_type"] = "standard"
            result["fallback_used"] = True

            return result

        except Exception as e:
            print(f"❌ Standard solver also failed: {e}")
            return {
                "success": False,
                "error": f"Both optimized and standard solvers failed: {e}",
                "solver_type": "none",
                "fallback_used": True,
            }

    return {
        "success": False,
        "error": "No solvers available",
        "solver_type": "none",
        "fallback_used": False,
    }


@click.command()
@click.argument("query", required=True)
@click.option(
    "--analyze-only",
    "-a",
    is_flag=True,
    help="Only analyze the query without executing",
)
@click.option("--benchmark", "-b", is_flag=True, help="Run performance benchmark")
@click.option("--iterations", "-i", default=10, help="Number of benchmark iterations")
@click.option("--fast-mode", "-f", is_flag=True, help="Use optimized solver only")
async def main_async(
    query: str, analyze_only: bool, benchmark: bool, iterations: int, fast_mode: bool
):
    """Optimized Bolt solver with sub-50ms performance target."""

    if benchmark:
        await run_benchmark(iterations)
        return

    print(f"🔍 {'Analyzing' if analyze_only else 'Solving'}: {query}")
    print("🎯 Target: <50ms end-to-end processing")
    print("-" * 60)

    start_time = time.time()

    if fast_mode and HAS_OPTIMIZED_INTEGRATION:
        # Use optimized solver only
        try:
            solver = OptimizedSolver()
            if analyze_only:
                result = await solver.analyze_query(query)
            else:
                result = await solver.solve_fast(query, execute=True)
        except Exception as e:
            result = {"success": False, "error": str(e)}
    else:
        # Use fallback approach
        result = await solve_with_fallback(query, analyze_only)

    total_time = (time.time() - start_time) * 1000

    # Display results
    if result.get("success"):
        print("✅ SUCCESS")

        if "total_time_ms" in result:
            processing_time = result["total_time_ms"]
            target_met = result.get("target_met", processing_time < 50.0)
            print(
                f"⏱️  Processing time: {processing_time:.1f}ms {'✅' if target_met else '⚠️'}"
            )

        print(f"🔧 Solver: {result.get('solver_type', 'unknown')}")

        if result.get("results_found", 0) > 0:
            print(f"📊 Results found: {result['results_found']}")

        if result.get("search_results"):
            print("\n📁 Top search results:")
            for i, sr in enumerate(result["search_results"][:3], 1):
                print(
                    f"  {i}. {Path(sr['file']).name}:{sr['line']} ({sr['type']}, score: {sr['score']:.2f})"
                )

        if result.get("execution_result"):
            exec_result = result["execution_result"]
            print(f"⚙️  Execution: {exec_result.get('execution_time_ms', 0):.1f}ms")

        if result.get("system_metrics"):
            metrics = result["system_metrics"]
            print(f"💾 Memory: {metrics.get('memory_mb', 0):.1f}MB")
            print(f"⚡ CPU: {metrics.get('cpu_percent', 0):.1f}%")

    else:
        print("❌ FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")

    print(f"\n🕐 Total time: {total_time:.1f}ms")


async def run_benchmark(iterations: int = 10):
    """Run performance benchmark."""
    print(f"🏁 Running performance benchmark ({iterations} iterations)")
    print("=" * 60)

    if not HAS_OPTIMIZED_INTEGRATION:
        print("❌ Optimized integration not available for benchmarking")
        return

    # Test queries representing different workloads
    test_queries = [
        "find WheelStrategy class",
        "search for risk calculation functions",
        "locate options pricing models",
        "find database connection code",
        "search for error handling patterns",
        "find trading strategy implementation",
        "locate performance optimization code",
        "search for async await patterns",
    ]

    try:
        integration = await create_optimized_integration()
        results = await benchmark_integration(integration, test_queries, iterations)

        print("📊 Benchmark Results:")
        print(f"   Total queries: {results['total_queries']}")
        print(
            f"   Target met: {results['target_met_count']}/{results['total_queries']}"
        )
        print(f"   Success rate: {results['target_achievement_rate']:.1%}")
        print(f"   Average time: {results['average_time_ms']:.1f}ms")
        print(f"   Min time: {results['min_time_ms']:.1f}ms")
        print(f"   Max time: {results['max_time_ms']:.1f}ms")

        if results["target_achievement_rate"] >= 0.8:
            print("🎯 EXCELLENT: >80% of queries met <50ms target")
        elif results["target_achievement_rate"] >= 0.6:
            print("✅ GOOD: >60% of queries met <50ms target")
        else:
            print("⚠️ NEEDS IMPROVEMENT: <60% of queries met target")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        traceback.print_exc()


@click.command()
@click.argument("query", required=True)
@click.option(
    "--analyze-only",
    "-a",
    is_flag=True,
    help="Only analyze the query without executing",
)
@click.option("--benchmark", "-b", is_flag=True, help="Run performance benchmark")
@click.option("--iterations", "-i", default=10, help="Number of benchmark iterations")
@click.option("--fast-mode", "-f", is_flag=True, help="Use optimized solver only")
def main(
    query: str, analyze_only: bool, benchmark: bool, iterations: int, fast_mode: bool
):
    """Optimized Bolt solver CLI."""
    asyncio.run(main_async(query, analyze_only, benchmark, iterations, fast_mode))


if __name__ == "__main__":
    main()
