#!/usr/bin/env python3
"""
Einstein Unified Indexing System Launcher

Replaces fragmented Jarvis indexing with unified system using ALL accelerated tools:
- Ripgrep Turbo (30x faster search)
- Dependency Graph Turbo (hardware-accelerated)
- Python Analysis Turbo (M4 Pro optimized)
- DuckDB Turbo (analytics)
- Sequential Thinking (planning)
- Trace Turbo (performance monitoring)

Provides sub-10ms context retrieval for 235k LOC codebase.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

from einstein.query_router import QueryRouter
from einstein.result_merger import ResultMerger
from einstein.unified_index import get_einstein_hub

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EinsteinLauncher:
    """Main launcher for Einstein unified indexing system."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.hub = get_einstein_hub()
        self.router = QueryRouter()
        self.merger = ResultMerger()

    async def initialize(self) -> None:
        """Initialize Einstein system."""

        print("🧠 Initializing Einstein Unified Indexing System...")
        start_time = time.time()

        await self.hub.initialize()

        init_time = (time.time() - start_time) * 1000
        print(f"✅ Einstein initialized in {init_time:.1f}ms")

        # Show system capabilities
        await self._show_capabilities()

    async def _show_capabilities(self) -> None:
        """Display Einstein system capabilities."""

        stats = await self.hub.get_stats()

        print("\n📋 Einstein System Capabilities:")
        print(f"   Files indexed: {stats.total_files}")
        print("   Search modalities: Text, Semantic, Structural, Analytical")
        print("   Hardware acceleration: M4 Pro (12 cores + Metal GPU)")
        print("   Accelerated tools: 10+ high-performance components")

        # Show search performance
        print("\n⚡ Search Performance:")
        for search_type, avg_time in stats.search_performance_ms.items():
            if avg_time > 0:
                print(f"   {search_type}: {avg_time:.1f}ms average")

    async def search(
        self, query: str, optimize_for: str = "balanced"
    ) -> dict[str, Any]:
        """Perform intelligent search using Einstein system."""

        print(f"\n🔍 Searching: '{query}'")

        # Route query intelligently
        plan = self.router.analyze_query(query)

        # Optimize plan based on preference
        if optimize_for == "speed":
            plan = self.router.optimize_for_latency(plan)
        elif optimize_for == "comprehensive":
            plan = self.router.optimize_for_recall(plan)

        print(f"   Query type: {plan.query_type.value}")
        print(f"   Search strategy: {', '.join(plan.search_modalities)}")
        print(f"   Estimated time: {plan.estimated_time_ms:.1f}ms")
        print(f"   Confidence: {plan.confidence:.1%}")

        # Execute search
        start_time = time.time()
        results = await self.hub.search(query, plan.search_modalities)
        search_time = (time.time() - start_time) * 1000

        # Group results by modality for merging
        results_by_modality = {}
        for result in results:
            modality = result.result_type
            if modality not in results_by_modality:
                results_by_modality[modality] = []
            results_by_modality[modality].append(result)

        # Merge and rank results
        merged_results = self.merger.merge_results(results_by_modality)

        # Apply relevance boosting
        merged_results = self.merger.boost_relevant_results(merged_results, query)

        # Get diverse subset
        final_results = self.merger.get_diversity_subset(merged_results, 20)

        # Generate summary
        summary = self.merger.generate_search_summary(final_results, query)

        print(f"\n✅ Search completed in {search_time:.1f}ms")
        print(
            f"   Found: {summary['total_results']} results across {summary['unique_files']} files"
        )
        print(f"   Top score: {summary['top_score']:.2f}")
        print(f"   Multi-modal: {summary['multi_modality_results']} results")

        return {
            "query": query,
            "plan": plan,
            "results": final_results,
            "summary": summary,
            "search_time_ms": search_time,
            "total_results": len(final_results),
        }

    async def get_intelligent_context(self, query: str) -> dict[str, Any]:
        """Get intelligent context for Jarvis using all accelerated tools."""

        print(f"\n🧠 Generating intelligent context for: '{query}'")

        start_time = time.time()
        context = await self.hub.get_intelligent_context(query)
        context_time = (time.time() - start_time) * 1000

        print(f"\n✅ Context generated in {context_time:.1f}ms")
        print(f"   Search results: {context['total_results']}")
        print(f"   Dependencies: {len(context.get('dependencies', []))}")
        print(f"   Thinking steps: {len(context.get('thinking_plan', []))}")

        return context

    async def benchmark_performance(self) -> dict[str, float]:
        """Benchmark Einstein system performance."""

        print("\n📊 Running performance benchmark...")

        test_queries = [
            "WheelStrategy",
            "options pricing",
            "delta calculation",
            "import pandas",
            "complex functions",
        ]

        results = {}

        for query in test_queries:
            start_time = time.time()
            await self.hub.search(query)
            search_time = (time.time() - start_time) * 1000
            results[query] = search_time

        avg_time = sum(results.values()) / len(results)

        print("\n⚡ Performance Results:")
        for query, time_ms in results.items():
            print(f"   '{query}': {time_ms:.1f}ms")
        print(f"   Average: {avg_time:.1f}ms")

        return results

    async def interactive_mode(self) -> None:
        """Run Einstein in interactive mode."""

        print("\n🚀 Einstein Interactive Mode")
        print("Commands: search <query>, context <query>, benchmark, stats, quit")

        while True:
            try:
                user_input = input("\neinstein> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                parts = user_input.split(" ", 1)
                command = parts[0].lower()

                if command == "search" and len(parts) > 1:
                    await self.search(parts[1])

                elif command == "context" and len(parts) > 1:
                    await self.get_intelligent_context(parts[1])

                elif command == "benchmark":
                    await self.benchmark_performance()

                elif command == "stats":
                    stats = await self.hub.get_stats()
                    print("\n📋 Index Statistics:")
                    print(f"   Files: {stats.total_files}")
                    print(f"   Coverage: {stats.coverage_percentage:.1f}%")
                    print(f"   Performance: {stats.search_performance_ms}")

                else:
                    print("Available commands: search, context, benchmark, stats, quit")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\n👋 Einstein session ended")


async def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Einstein Unified Indexing System")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["init", "search", "context", "benchmark", "interactive"],
        default="interactive",
        help="Command to execute",
    )
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument(
        "--optimize",
        choices=["speed", "balanced", "comprehensive"],
        default="balanced",
        help="Optimization preference",
    )
    parser.add_argument("--project-root", type=Path, help="Project root directory")

    args = parser.parse_args()

    # Initialize Einstein
    launcher = EinsteinLauncher(args.project_root)

    try:
        await launcher.initialize()

        if args.command == "init":
            print("\n✅ Einstein initialization complete")

        elif args.command == "search":
            if not args.query:
                print("Error: --query required for search command")
                sys.exit(1)
            await launcher.search(args.query, args.optimize)

        elif args.command == "context":
            if not args.query:
                print("Error: --query required for context command")
                sys.exit(1)
            await launcher.get_intelligent_context(args.query)

        elif args.command == "benchmark":
            await launcher.benchmark_performance()

        elif args.command == "interactive":
            await launcher.interactive_mode()

    except KeyboardInterrupt:
        print("\n👋 Einstein interrupted")
    except Exception as e:
        print(f"\n❌ Einstein error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
