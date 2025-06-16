#!/usr/bin/env python3
"""
Unified CLI Demo - Demonstrates intelligent routing between Einstein and Bolt

This script shows examples of how queries are automatically routed to the
appropriate system based on their complexity and intent.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_cli import UnifiedCLI


async def demo_routing():
    """Demonstrate the routing capabilities of the unified CLI."""

    print("🚀 Unified CLI - Routing Demonstration")
    print("=" * 60)
    print()

    # Initialize CLI
    cli = UnifiedCLI()

    # Demo queries with expected routing
    demo_queries = [
        # Einstein examples (simple searches)
        ("find WheelStrategy", "Einstein - Simple code search"),
        ("show options.py", "Einstein - File lookup"),
        ("calculate_delta", "Einstein - Function name search"),
        ("import pandas", "Einstein - Import statement search"),
        # Bolt examples (complex analysis)
        ("optimize database performance", "Bolt - Performance optimization"),
        ("fix memory leak in trading module", "Bolt - Debug complex issue"),
        ("analyze bottlenecks in the wheel strategy", "Bolt - System analysis"),
        ("help me refactor the risk calculation logic", "Bolt - Code restructuring"),
        (
            "how can I improve the overall system architecture",
            "Bolt - Architecture review",
        ),
    ]

    print("📋 Demonstration Queries:")
    print("=" * 60)

    for i, (query, description) in enumerate(demo_queries, 1):
        print(f"\n{i}. {description}")
        print(f'   Query: "{query}"')

        # Show routing decision without executing
        system, confidence, reasoning = cli.router.classify_query(query)

        print(f"   → Routed to: {system.upper()}")
        print(f"   → Confidence: {confidence:.1%}")
        print(f"   → Reasoning: {reasoning}")

        # Show what would happen
        if system == "einstein":
            print("   → Action: Fast semantic search and code lookup")
        else:
            print("   → Action: Multi-agent analysis and problem solving")

    print(f"\n{'=' * 60}")
    print("📊 Routing Summary:")

    einstein_count = sum(
        1
        for query, _ in demo_queries
        if cli.router.classify_query(query)[0] == "einstein"
    )
    bolt_count = len(demo_queries) - einstein_count

    print(f"   Einstein queries: {einstein_count}")
    print(f"   Bolt queries: {bolt_count}")
    print(f"   Total: {len(demo_queries)}")

    print(f"\n{'=' * 60}")
    print("🎯 Key Routing Principles:")
    print(
        """
   EINSTEIN (Fast Search):
   • Simple searches: "find X", "show Y"
   • Code elements: function names, class names
   • Technical lookups: imports, definitions
   • Single words or short phrases
   • High confidence for symbol patterns

   BOLT (Complex Analysis):
   • Optimization tasks: "optimize", "improve"
   • Problem solving: "fix", "debug", "analyze"
   • Action-oriented: "help me", "how to"
   • Long complex queries (10+ words)
   • Multi-step reasoning required
   
   FALLBACK LOGIC:
   • If primary system fails, try the other
   • Confidence scoring helps with edge cases
   • Override with --force-einstein or --force-bolt
    """
    )

    print(f"{'=' * 60}")
    print("🚀 Try it yourself:")
    print('   python3 unified_cli.py "find WheelStrategy"')
    print('   python3 unified_cli.py "optimize database queries"')
    print("   python3 unified_cli.py --interactive")
    print('   ./unified "help me fix performance issues"')


if __name__ == "__main__":
    asyncio.run(demo_routing())
