#!/usr/bin/env python3
"""
Demo script showing Query Intelligence features in action.

This demonstrates:
1. Basic autocomplete suggestions
2. Query history learning
3. Refinement suggestions
4. Context-aware suggestions
5. Integration with Einstein/Bolt routing
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from query_intelligence import get_query_intelligence
from unified_cli import UnifiedCLI


async def demo_suggestions():
    """Demo the suggestion system."""
    print("🎯 Demo: Query Intelligence Suggestions")
    print("=" * 60)

    qi = get_query_intelligence()

    # Simulate some query history
    print("📝 Building query history...")
    mock_history = [
        ("find WheelStrategy", "einstein", True, 3, 0.08),
        ("show options.py", "einstein", True, 1, 0.05),
        ("optimize database queries", "bolt", True, 8, 1.45),
        ("search TODO", "einstein", True, 12, 0.15),
        ("analyze risk exposure", "bolt", True, 4, 2.1),
        ("find class OptionsCalculator", "einstein", True, 2, 0.12),
        ("fix memory leak", "bolt", False, 0, 3.2),
        ("show risk.py", "einstein", True, 1, 0.06),
    ]

    for query, system, success, results, time_ms in mock_history:
        await qi.record_query(query, system, success, results, time_ms)
        status = "✅" if success else "❌"
        print(f"  {status} {query} → {system} ({results} results)")

    # Add some context
    print("\n📂 Adding context files...")
    context_files = [
        (
            "src/unity_wheel/strategy/wheel.py",
            ["WheelStrategy"],
            ["execute_trade", "calculate_premium"],
        ),
        (
            "src/unity_wheel/risk/analytics.py",
            ["RiskAnalyzer"],
            ["calculate_var", "stress_test"],
        ),
        (
            "src/unity_wheel/math/options.py",
            ["OptionsCalculator"],
            ["black_scholes", "calculate_greeks"],
        ),
    ]

    for file_path, classes, functions in context_files:
        qi.update_context(file_path, classes, functions)
        print(f"  📄 {file_path}")

    # Demo suggestions for different queries
    print("\n💡 Testing intelligent suggestions...")
    test_queries = [
        "",  # Popular suggestions
        "find",  # Autocomplete based on history
        "wheel",  # Context + codebase suggestions
        "opt",  # Pattern-based suggestions
        "unknown_function",  # Refinement suggestions (no results)
    ]

    for query in test_queries:
        if query == "":
            print("\n🔍 Popular suggestions (empty query):")
        else:
            print(f"\n🔍 Suggestions for: '{query}'")

        suggestions = await qi.get_suggestions(query, max_suggestions=4)

        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                confidence_icon = (
                    "🟢"
                    if suggestion.confidence > 0.8
                    else "🟡"
                    if suggestion.confidence > 0.6
                    else "🔴"
                )
                print(f"  {i}. {suggestion.text} {confidence_icon}")
                print(
                    f"     Category: {suggestion.category} | Confidence: {suggestion.confidence:.0%}"
                )
                print(f"     Reasoning: {suggestion.reasoning}")
        else:
            print("  No suggestions available")

    # Demo refinement suggestions
    print("\n🔧 Testing refinement suggestions...")

    # No results case
    refinements = await qi.get_refinement_suggestions("nonexistent_function_xyz", 0)
    print("No results for 'nonexistent_function_xyz':")
    for ref in refinements:
        print(f"  • {ref.text} - {ref.reasoning}")

    # Few results case
    refinements = await qi.get_refinement_suggestions("complex trading algorithm", 2)
    print("\nFew results for 'complex trading algorithm':")
    for ref in refinements:
        print(f"  • {ref.text} - {ref.reasoning}")


async def demo_cli_integration():
    """Demo CLI integration with intelligent routing."""
    print("\n🎯 Demo: CLI Integration with Intelligent Routing")
    print("=" * 60)

    cli = UnifiedCLI()

    if not cli.query_intelligence:
        print("❌ Query intelligence not available")
        return

    # Test queries with different routing
    test_queries = [
        "find WheelStrategy",
        "optimize performance issues",
        "show options.py",
        "analyze bottlenecks in risk calculation",
        "WheelBacktester",
        "fix memory leak in trading module",
    ]

    print("🤖 Testing intelligent query routing...")

    for query in test_queries:
        # Classify the query
        system, confidence, reasoning = cli.router.classify_query(query)

        # Get suggestions
        suggestions = await cli.query_intelligence.get_suggestions(
            query, max_suggestions=2
        )

        print(f"\n📝 Query: '{query}'")
        print(f"   Route: {system.upper()} (confidence: {confidence:.0%})")
        print(f"   Reasoning: {reasoning}")

        if suggestions:
            print("   Suggestions:")
            for suggestion in suggestions:
                print(f"     • {suggestion.text} ({suggestion.category})")


async def demo_stats():
    """Demo system statistics."""
    print("\n🎯 Demo: System Statistics")
    print("=" * 60)

    qi = get_query_intelligence()
    stats = await qi.get_stats()

    print("📊 Query Intelligence Statistics:")
    for key, value in stats.items():
        if key == "popular_terms" and isinstance(value, list):
            print(f"  {key}:")
            for i, term in enumerate(value[:3], 1):
                if isinstance(term, dict):
                    query = term.get("query", "N/A")
                    freq = term.get("frequency", 0)
                    print(f"    {i}. '{query}' ({freq} times)")
        else:
            print(f"  {key}: {value}")


async def main():
    """Run the complete demo."""

    print("🚀 Query Intelligence System - Interactive Demo")
    print("This demo shows all features of the intelligent autocomplete system")
    print("=" * 80)

    try:
        await demo_suggestions()
        await demo_cli_integration()
        await demo_stats()

        print("\n🎉 Demo completed successfully!")
        print("\n💡 Ready to try it yourself?")
        print("   python unified_cli.py --interactive")
        print("   python unified_cli.py 'find WheelStrategy'")
        print("   python unified_cli.py 'optimize database performance'")

        print("\n📋 Features demonstrated:")
        print("   ✅ Autocomplete based on query history")
        print("   ✅ Intelligent suggestions from codebase symbols")
        print("   ✅ Query refinement for poor results")
        print("   ✅ Context-aware suggestions from recent files")
        print("   ✅ Integration with Einstein/Bolt routing")
        print("   ✅ Performance monitoring and statistics")

    except Exception as e:
        print(f"💥 Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
