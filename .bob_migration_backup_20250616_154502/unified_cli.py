#!/usr/bin/env python3
"""
Unified CLI - Intelligent Router for Einstein and Bolt Systems

This CLI intelligently routes queries between:
- Einstein: Fast semantic search, code understanding, simple queries
- Bolt: Complex analysis, multi-step problem solving, optimization tasks

Features:
- Smart query classification and routing
- Automatic fallback handling
- Performance monitoring
- Unified help and examples
"""

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

# Import query intelligence system
try:
    from query_intelligence import QuerySuggestion, get_query_intelligence

    HAS_QUERY_INTELLIGENCE = True
except (ImportError, SyntaxError, Exception):
    HAS_QUERY_INTELLIGENCE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QueryRouter:
    """Intelligent query router that decides between Einstein and Bolt."""

    # Keywords that indicate Einstein (search-focused) queries
    EINSTEIN_KEYWORDS = {
        "find",
        "search",
        "locate",
        "show",
        "list",
        "where",
        "what",
        "which",
        "definition",
        "function",
        "class",
        "method",
        "variable",
        "import",
        "usage",
        "example",
        "reference",
        "documentation",
        "grep",
        "symbol",
    }

    # Keywords that indicate Bolt (analysis/optimization) queries
    BOLT_KEYWORDS = {
        "optimize",
        "fix",
        "debug",
        "analyze",
        "improve",
        "refactor",
        "solve",
        "performance",
        "memory",
        "speed",
        "bottleneck",
        "issue",
        "problem",
        "bug",
        "error",
        "exception",
        "crash",
        "leak",
        "slow",
        "inefficient",
        "architecture",
        "design",
        "pattern",
        "strategy",
        "algorithm",
        "complex",
    }

    # Patterns that strongly indicate complexity (Bolt territory)
    COMPLEX_PATTERNS = [
        r"\b(how to|how can I|help me)\b.*\b(optimize|fix|improve|solve)\b",
        r"\b(analyze|review|audit)\b.*\b(performance|memory|security)\b",
        r"\bmultiple\b.*\b(files|components|systems)\b",
        r"\b(database|sql|query)\b.*\b(optimization|performance)\b",
        r"\b(refactor|redesign|restructure)\b",
        r"\b(integration|coordination|orchestration)\b",
    ]

    def classify_query(self, query: str) -> tuple[str, float, str]:
        """
        Classify query as 'einstein' or 'bolt' with confidence score.

        Returns:
            (system, confidence, reasoning)
        """
        query_lower = query.lower()

        # Check for complex patterns first (high confidence indicators)
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, query_lower):
                return ("bolt", 0.9, f"Complex pattern detected: {pattern}")

        # Count keyword matches
        einstein_matches = sum(
            1 for word in self.EINSTEIN_KEYWORDS if word in query_lower
        )
        bolt_matches = sum(1 for word in self.BOLT_KEYWORDS if word in query_lower)

        # Simple search queries (single term, no action words)
        words = query_lower.split()
        if len(words) <= 3 and any(word in self.EINSTEIN_KEYWORDS for word in words):
            return ("einstein", 0.85, "Simple search query detected")

        # Specific technical searches
        if any(
            pattern in query_lower for pattern in ["class ", "def ", "import ", "from "]
        ):
            return ("einstein", 0.8, "Code element search detected")

        # File or function name patterns
        if re.match(
            r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$", query.strip()
        ):
            return ("einstein", 0.9, "Symbol name pattern detected")

        # Long, complex queries tend to be Bolt territory
        if len(words) > 10:
            return ("bolt", 0.7, "Long, complex query")

        # Action-oriented queries
        action_words = ["create", "build", "implement", "develop", "write", "generate"]
        if any(word in query_lower for word in action_words):
            return ("bolt", 0.8, "Action-oriented query detected")

        # Compare keyword counts
        if bolt_matches > einstein_matches:
            confidence = min(0.9, 0.6 + (bolt_matches - einstein_matches) * 0.1)
            return (
                "bolt",
                confidence,
                f"Bolt keywords: {bolt_matches}, Einstein: {einstein_matches}",
            )
        elif einstein_matches > bolt_matches:
            confidence = min(0.9, 0.6 + (einstein_matches - bolt_matches) * 0.1)
            return (
                "einstein",
                confidence,
                f"Einstein keywords: {einstein_matches}, Bolt: {bolt_matches}",
            )

        # Default fallback - simple queries go to Einstein
        if len(words) <= 5:
            return ("einstein", 0.5, "Default: short query to Einstein")
        else:
            return ("bolt", 0.5, "Default: longer query to Bolt")


class UnifiedCLI:
    """Main unified CLI that routes between Einstein and Bolt."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.router = QueryRouter()
        self.query_intelligence = (
            get_query_intelligence(self.project_root)
            if HAS_QUERY_INTELLIGENCE
            else None
        )
        self.stats = {
            "einstein_queries": 0,
            "bolt_queries": 0,
            "fallbacks": 0,
            "errors": 0,
            "suggestions_shown": 0,
            "suggestions_accepted": 0,
        }

    async def route_query(
        self, query: str, force_system: str | None = None
    ) -> dict[str, Any]:
        """Route query to appropriate system and return results."""

        if force_system:
            system = force_system
            confidence = 1.0
            reasoning = f"Forced to {system}"
        else:
            system, confidence, reasoning = self.router.classify_query(query)

        print(f"🤖 Routing to {system.upper()}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Reasoning: {reasoning}")
        print(f"   Query: {query}")
        print("-" * 60)

        start_time = time.time()

        try:
            if system == "einstein":
                result = await self._run_einstein(query)
                self.stats["einstein_queries"] += 1
            else:
                result = await self._run_bolt(query)
                self.stats["bolt_queries"] += 1

            result["routing"] = {
                "system": system,
                "confidence": confidence,
                "reasoning": reasoning,
                "execution_time": time.time() - start_time,
            }

            # Record query for intelligence learning
            if self.query_intelligence:
                result_count = len(result.get("results", []))
                success = "error" not in result
                await self.query_intelligence.record_query(
                    query, system, success, result_count, time.time() - start_time
                )

            return result

        except Exception as e:
            self.stats["errors"] += 1
            print(f"❌ Error in {system}: {e}")

            # Try fallback to other system
            fallback_system = "bolt" if system == "einstein" else "einstein"
            print(f"🔄 Attempting fallback to {fallback_system}...")

            try:
                if fallback_system == "einstein":
                    result = await self._run_einstein(query)
                else:
                    result = await self._run_bolt(query)

                self.stats["fallbacks"] += 1
                result["routing"] = {
                    "system": fallback_system,
                    "confidence": 0.3,  # Low confidence for fallback
                    "reasoning": f"Fallback from {system} due to error: {e}",
                    "execution_time": time.time() - start_time,
                    "fallback": True,
                }

                return result

            except Exception as fallback_error:
                return {
                    "error": f"Both systems failed. {system}: {e}, {fallback_system}: {fallback_error}",
                    "routing": {
                        "system": "none",
                        "confidence": 0.0,
                        "reasoning": "Complete system failure",
                        "execution_time": time.time() - start_time,
                    },
                }

    async def _run_einstein(self, query: str) -> dict[str, Any]:
        """Execute query using Einstein system."""

        try:
            from einstein_launcher import EinsteinLauncher

            launcher = EinsteinLauncher(self.project_root)
            await launcher.initialize()

            # Try search first, then context if needed
            if len(query.split()) <= 5:
                result = await launcher.search(query)
                result["system"] = "einstein"
                result["mode"] = "search"
            else:
                result = await launcher.get_intelligent_context(query)
                result["system"] = "einstein"
                result["mode"] = "context"

            return result

        except ImportError:
            return {
                "system": "einstein",
                "mode": "fallback",
                "error": "Einstein system not available",
                "recommendation": "Install Einstein dependencies or use --force-bolt",
            }

    async def _run_bolt(self, query: str) -> dict[str, Any]:
        """Execute query using Bolt system."""

        try:
            from bolt.solve import analyze_and_execute

            # Use analysis mode for query classification
            analyze_only = any(
                word in query.lower()
                for word in ["analyze", "review", "assess", "check"]
            )

            result = await analyze_and_execute(query, analyze_only)
            result["system"] = "bolt"
            result["mode"] = "analyze" if analyze_only else "execute"

            return result

        except ImportError:
            return {
                "system": "bolt",
                "mode": "fallback",
                "error": "Bolt system not available",
                "recommendation": "Install Bolt dependencies or use --force-einstein",
            }

    def print_stats(self) -> None:
        """Print usage statistics."""
        query_count = self.stats["einstein_queries"] + self.stats["bolt_queries"]
        if query_count == 0:
            print("\n📊 No queries executed yet")
            return

        print("\n📊 Session Statistics:")
        print(f"   Einstein queries: {self.stats['einstein_queries']}")
        print(f"   Bolt queries: {self.stats['bolt_queries']}")
        print(f"   Fallbacks: {self.stats['fallbacks']}")
        print(f"   Errors: {self.stats['errors']}")

        if self.query_intelligence:
            print(f"   Suggestions shown: {self.stats['suggestions_shown']}")
            print(f"   Suggestions accepted: {self.stats['suggestions_accepted']}")
            if self.stats["suggestions_shown"] > 0:
                acceptance_rate = (
                    self.stats["suggestions_accepted"] / self.stats["suggestions_shown"]
                ) * 100
                print(f"   Suggestion acceptance rate: {acceptance_rate:.1f}%")

        success_rate = ((query_count - self.stats["errors"]) / query_count) * 100
        print(f"   Success rate: {success_rate:.1f}%")

    async def interactive_mode(self) -> None:
        """Run in interactive mode with intelligent suggestions."""

        print("🚀 Unified CLI - Interactive Mode")
        print(
            "   Commands: <query>, !einstein <query>, !bolt <query>, stats, help, quit"
        )
        print("   Features: Auto-suggestions, query history, intelligent routing")
        print("   Examples:")
        print("     find WheelStrategy          → Einstein (search)")
        print("     optimize database queries   → Bolt (analysis)")
        print("     !bolt find WheelStrategy    → Force Bolt")
        if self.query_intelligence:
            print(
                "   Tip: Type 'suggest' to see popular queries or start typing for suggestions"
            )
        print("")

        while True:
            try:
                user_input = input("unified> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if user_input.lower() == "stats":
                    self.print_stats()
                    continue

                if user_input.lower() in ["help", "?"]:
                    self._print_help()
                    continue

                if user_input.lower() == "suggest":
                    if self.query_intelligence:
                        await self._show_suggestions()
                    else:
                        print(
                            "🤖 Query intelligence not available. Try these common queries:"
                        )
                        print("     find WheelStrategy")
                        print("     show options.py")
                        print("     optimize performance")
                        print("     analyze risk")
                    continue

                if not user_input:
                    continue

                # Show suggestions for partial queries and potentially replace query
                suggested_query = None
                if self.query_intelligence and len(user_input.strip()) >= 2:
                    suggested_query = await self._show_inline_suggestions(
                        user_input.strip()
                    )

                # Use suggested query if provided, otherwise use original
                if suggested_query:
                    query_to_use = suggested_query
                else:
                    query_to_use = user_input

                # Handle forced routing
                force_system = None
                if query_to_use.startswith("!einstein "):
                    force_system = "einstein"
                    query = query_to_use[10:]
                elif query_to_use.startswith("!bolt "):
                    force_system = "bolt"
                    query = query_to_use[6:]
                else:
                    query = query_to_use

                print()  # Add spacing
                result = await self.route_query(query, force_system)

                # Display results
                if "error" in result:
                    print(f"❌ {result['error']}")
                else:
                    # Add query to result for refinement suggestions
                    result["query"] = query
                    self._display_result(result)

                print()  # Add spacing

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\n👋 Session ended")
        self.print_stats()

    async def _show_suggestions(self) -> None:
        """Show popular query suggestions."""
        try:
            suggestions = await self.query_intelligence.get_suggestions(
                "", max_suggestions=8
            )

            if suggestions:
                print("\n💡 Popular Suggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    confidence_icon = (
                        "🟢"
                        if suggestion.confidence > 0.8
                        else "🟡"
                        if suggestion.confidence > 0.6
                        else "🔴"
                    )
                    print(f"  {i}. {suggestion.text} {confidence_icon}")
                    print(f"     {suggestion.reasoning}")

                self.stats["suggestions_shown"] += len(suggestions)
            else:
                print(
                    "\n💡 No suggestions available yet. Try some queries to build history!"
                )

        except Exception as e:
            print(f"Error getting suggestions: {e}")

    async def _show_inline_suggestions(self, partial_query: str) -> None:
        """Show inline suggestions for partial query."""
        try:
            suggestions = await self.query_intelligence.get_suggestions(
                partial_query, max_suggestions=3
            )

            if suggestions:
                print(f"\n💡 Suggestions for '{partial_query}':")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion.text} ({suggestion.confidence:.0%})")

                self.stats["suggestions_shown"] += len(suggestions)

                # Ask if user wants to use a suggestion
                try:
                    choice = input(
                        "Use suggestion (1-3) or press Enter to continue: "
                    ).strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(suggestions):
                        selected = suggestions[int(choice) - 1]
                        print(f"Using: {selected.text}")
                        self.stats["suggestions_accepted"] += 1
                        return selected.text
                except (KeyboardInterrupt, EOFError):
                    pass

        except Exception as e:
            print(f"Error getting inline suggestions: {e}")

        return None

    async def _show_refinement_suggestions(
        self, original_query: str, result_count: int
    ) -> None:
        """Show suggestions to refine queries with few results."""
        try:
            refinements = await self.query_intelligence.get_refinement_suggestions(
                original_query, result_count
            )

            if refinements:
                if result_count == 0:
                    print("\n🔍 No results found. Try these refinements:")
                else:
                    print(f"\n🔍 Few results ({result_count}). Try these alternatives:")

                for i, suggestion in enumerate(refinements, 1):
                    print(f"  {i}. {suggestion.text}")
                    print(f"     {suggestion.reasoning}")

        except Exception as e:
            print(f"Error getting refinement suggestions: {e}")

    def _display_result(self, result: dict[str, Any]) -> None:
        """Display formatted results."""

        routing = result.get("routing", {})
        system = routing.get("system", "unknown")
        execution_time = routing.get("execution_time", 0)

        print(f"✅ {system.upper()} completed in {execution_time:.2f}s")

        if routing.get("fallback"):
            print("   ⚠️  Result from fallback system")

        # Display system-specific results
        if system == "einstein":
            if "results" in result:
                results = result["results"]
                print(f"   Found {len(results)} results")

                # Show top results
                for i, res in enumerate(results[:5], 1):
                    # Handle MergedResult objects (dataclass) vs dictionaries
                    if hasattr(res, "file_path"):
                        # MergedResult object - access attributes directly
                        file_path = res.file_path
                        score = res.combined_score
                    else:
                        # Dictionary - use get method
                        file_path = res.get("file", "unknown")
                        score = res.get("score", 0)
                    print(f"   {i}. {file_path} (score: {score:.2f})")

            if "summary" in result:
                summary = result["summary"]
                print(f"   Files: {summary.get('unique_files', 0)}")
                print(f"   Top score: {summary.get('top_score', 0):.2f}")

        elif system == "bolt":
            if "success" in result:
                status = "✅ Success" if result["success"] else "❌ Failed"
                print(f"   Status: {status}")

            if "results" in result and isinstance(result["results"], dict):
                bolt_results = result["results"]

                if bolt_results.get("summary"):
                    print(f"   Summary: {bolt_results['summary']}")

                if bolt_results.get("findings"):
                    print("   Findings:")
                    for finding in bolt_results["findings"][:3]:
                        print(f"     • {finding}")

                if bolt_results.get("recommendations"):
                    print("   Recommendations:")
                    for rec in bolt_results["recommendations"][:3]:
                        print(f"     • {rec}")

        # Show refinement suggestions if few results
        if self.query_intelligence and "query" in result:
            original_query = result["query"]
            result_count = len(result.get("results", []))

            if result_count <= 3:
                asyncio.create_task(
                    self._show_refinement_suggestions(original_query, result_count)
                )

    def _print_help(self) -> None:
        """Print help information."""

        print(
            """
🚀 Unified CLI Help

SYSTEMS:
  Einstein  - Fast semantic search, code understanding, simple queries
  Bolt      - Complex analysis, optimization, multi-step problem solving

INTELLIGENT FEATURES:
  Auto-suggestions     - Smart query completion based on codebase
  Query history        - Learn from successful queries  
  Refinement hints     - Improve queries with few results
  Context awareness    - Suggestions based on recent files

ROUTING LOGIC:
  Einstein: find, search, show, function names, simple queries
  Bolt:     optimize, fix, analyze, complex multi-word queries
  
COMMANDS:
  <query>              - Auto-route query to best system
  !einstein <query>    - Force Einstein system
  !bolt <query>        - Force Bolt system
  suggest             - Show popular query suggestions
  stats               - Show session statistics  
  help                - Show this help
  quit                - Exit

EXAMPLES:
  Simple searches (→ Einstein):
    find WheelStrategy
    show options.py
    WheelBacktester
    search TODO
    
  Complex analysis (→ Bolt):
    optimize database performance
    fix memory leak in trading module
    analyze bottlenecks in wheel strategy
    help me refactor the risk calculation
    
  Forced routing:
    !bolt find WheelStrategy     - Force complex analysis of search
    !einstein optimize queries   - Force simple search for "optimize"

TIP: Let the system auto-route for best results, use forced routing to override when needed.
        """
        )


async def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Unified CLI - Intelligent Router for Einstein and Bolt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  unified_cli.py "find WheelStrategy"
  unified_cli.py "optimize database queries" 
  unified_cli.py --force-bolt "find WheelStrategy"
  unified_cli.py --interactive
  unified_cli.py --benchmark

ROUTING LOGIC:
  Einstein: Fast search, code lookups, simple queries
  Bolt:     Complex analysis, optimization, problem solving
        """,
    )

    parser.add_argument("query", nargs="?", help="Query to execute")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--force-einstein", action="store_true", help="Force routing to Einstein system"
    )
    parser.add_argument(
        "--force-bolt", action="store_true", help="Force routing to Bolt system"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run routing benchmark tests"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--project-root", type=Path, help="Project root directory")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize CLI
    cli = UnifiedCLI(args.project_root)

    try:
        if args.benchmark:
            await run_benchmark(cli)
        elif args.interactive:
            await cli.interactive_mode()
        elif args.query:
            # Determine forced system
            force_system = None
            if args.force_einstein:
                force_system = "einstein"
            elif args.force_bolt:
                force_system = "bolt"

            # Execute single query
            result = await cli.route_query(args.query, force_system)

            if "error" in result:
                print(f"❌ {result['error']}")
                sys.exit(1)
            else:
                cli._display_result(result)
                cli.print_stats()
        else:
            print("Error: Provide a query or use --interactive mode")
            print("Use --help for usage information")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        cli.print_stats()
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def run_benchmark(cli: UnifiedCLI) -> None:
    """Run benchmark tests for routing logic."""

    print("🧪 Running routing benchmark tests...")

    test_cases = [
        # Einstein cases
        ("find WheelStrategy", "einstein"),
        ("show options.py", "einstein"),
        ("WheelBacktester", "einstein"),
        ("search TODO", "einstein"),
        ("def calculate_delta", "einstein"),
        ("import pandas", "einstein"),
        # Bolt cases
        ("optimize database performance", "bolt"),
        ("fix memory leak in trading module", "bolt"),
        ("analyze bottlenecks in wheel strategy", "bolt"),
        ("help me refactor the risk calculation", "bolt"),
        ("debug performance issues", "bolt"),
        ("how to improve query speed", "bolt"),
        ("review database design patterns", "bolt"),
        # Edge cases
        ("simple", "einstein"),  # Single word -> Einstein
        (
            "very long complex query about optimizing multiple database systems with performance analysis",
            "bolt",
        ),
    ]

    correct = 0
    total = len(test_cases)

    print(f"\nTesting {total} routing decisions...")
    print("-" * 80)

    for query, expected in test_cases:
        predicted, confidence, reasoning = cli.router.classify_query(query)

        status = "✅" if predicted == expected else "❌"
        print(f"{status} '{query[:50]}'")
        print(f"    Expected: {expected}, Got: {predicted} ({confidence:.1%})")
        print(f"    Reasoning: {reasoning}")
        print()

        if predicted == expected:
            correct += 1

    accuracy = correct / total
    print(f"📊 Routing Accuracy: {correct}/{total} ({accuracy:.1%})")

    if accuracy >= 0.85:
        print("🎉 Excellent routing performance!")
    elif accuracy >= 0.70:
        print("👍 Good routing performance")
    else:
        print("⚠️  Routing performance needs improvement")


if __name__ == "__main__":
    asyncio.run(main())
