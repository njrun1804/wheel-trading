#!/usr/bin/env python3
"""Jarvis - Meta-coder assistant for Claude Code CLI.

Usage:
    ./jarvis.py "optimize all trading analysis functions"
    ./jarvis.py --explain "refactor the WheelStrategy class"
    ./jarvis.py --interactive
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from jarvis import Jarvis, JarvisConfig


async def main():
    """Main entry point for Jarvis CLI."""
    parser = argparse.ArgumentParser(
        description="Jarvis - Meta-coder for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./jarvis.py "optimize performance of backtesting"
  ./jarvis.py --explain "add comprehensive tests"
  ./jarvis.py --interactive
  ./jarvis.py --verbose "refactor duplicate code"
        """,
    )

    parser.add_argument(
        "query", nargs="?", help="Task description for Jarvis to assist with"
    )

    parser.add_argument(
        "--explain", action="store_true", help="Explain approach without executing"
    )

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Enter interactive mode"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed execution information",
    )

    parser.add_argument(
        "--no-mcts", action="store_true", help="Disable MCTS for faster execution"
    )

    parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="Workspace root directory (default: current directory)",
    )

    args = parser.parse_args()

    # Configure Jarvis
    config = JarvisConfig(
        workspace_root=args.workspace, use_mcts=not args.no_mcts, verbose=args.verbose
    )

    # Initialize Jarvis
    jarvis = Jarvis(config)

    try:
        if args.interactive:
            await interactive_mode(jarvis)
        elif args.query:
            if args.explain:
                # Explain mode
                explanation = await jarvis.explain_approach(args.query)
                print(explanation)
            else:
                # Execute mode
                result = await jarvis.assist(args.query)

                if not args.verbose:
                    # Show summary if not in verbose mode
                    print(f"\n✅ Task completed in {result['total_duration_ms']:.1f}ms")

                    # Show key results
                    impl_data = result["phases"]["implement"]["data"]
                    if "strategy_used" in impl_data:
                        print(f"📋 Strategy: {impl_data['strategy_used']}")

                    if "files_modified" in impl_data:
                        print(f"📝 Files modified: {impl_data['files_modified']}")
        else:
            parser.print_help()

    finally:
        # Cleanup
        await jarvis.cleanup()


async def interactive_mode(jarvis: Jarvis):
    """Run Jarvis in interactive mode."""
    print("🤖 Jarvis Interactive Mode")
    print("=" * 60)
    print("Enter tasks for assistance. Type 'help' for commands or 'quit' to exit.")
    print()

    while True:
        try:
            query = input("jarvis> ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if query.lower() == "help":
                print(
                    """
Commands:
  <task description>    - Execute a task
  explain <task>        - Explain approach without executing
  status               - Show last result status
  clear                - Clear screen
  help                 - Show this help
  quit/exit            - Exit Jarvis
                """
                )
                continue

            if query.lower() == "clear":
                print("\033[2J\033[H")  # Clear screen
                continue

            if query.lower() == "status":
                if jarvis.last_result:
                    print(f"Last task: {jarvis.last_result['query']}")
                    print(f"Success: {jarvis.last_result['success']}")
                    print(f"Duration: {jarvis.last_result['total_duration_ms']:.1f}ms")
                else:
                    print("No tasks executed yet.")
                continue

            if query.lower().startswith("explain "):
                task = query[8:]
                explanation = await jarvis.explain_approach(task)
                print(explanation)
            else:
                # Execute task
                result = await jarvis.assist(query)

                if not jarvis.config.verbose:
                    print(f"\n✅ Completed in {result['total_duration_ms']:.1f}ms")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
