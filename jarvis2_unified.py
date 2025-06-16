#!/usr/bin/env python3
"""
Jarvis2 Unified Entry Point

Consolidated entry point eliminating duplicate functionality.
Replaces: jarvis2_complete.py, jarvis2_core.py, jarvis2_mcts.py, jarvis2_optimized.py
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from database_manager import get_database_manager

# Import Jarvis2 core
from jarvis2.core.jarvis2 import Jarvis2
from jarvis2.core.orchestrator import Jarvis2Orchestrator
from neural_backend_manager import get_neural_backend_manager

# Import unified systems
from unified_config import get_unified_config


class UnifiedJarvis2:
    """Unified Jarvis2 system with all capabilities."""

    def __init__(self, enable_meta: bool = False):
        self.config = get_unified_config()
        self.db_manager = get_database_manager()
        self.neural_backend = get_neural_backend_manager()

        # Initialize core Jarvis2 with lazy meta loading
        self.jarvis2 = Jarvis2(enable_meta=enable_meta)
        self.orchestrator = Jarvis2Orchestrator()

        print("🧠 Unified Jarvis2 initialized")
        print(f"   Neural backend: {self.neural_backend.get_current_backend()}")
        print(f"   CPU cores: {self.config.get_jarvis2_cpu_cores()}")
        print(f"   Meta system: {'enabled' if enable_meta else 'disabled'}")

    async def generate_code(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """Generate code using Jarvis2."""
        return await self.jarvis2.generate_code(prompt, context or {})

    async def analyze_codebase(self, target_path: str = ".") -> dict[str, Any]:
        """Analyze codebase structure and patterns."""
        return await self.jarvis2.analyze_codebase(Path(target_path))

    async def optimize_code(
        self, code: str, optimization_type: str = "performance"
    ) -> str:
        """Optimize given code."""
        return await self.jarvis2.optimize_code(code, optimization_type)

    async def search_solutions(
        self, problem: str, max_iterations: int = 1000
    ) -> list[dict[str, Any]]:
        """Search for solutions using MCTS."""
        # Use orchestrator for complex search operations
        return await self.orchestrator.search_solutions(problem, max_iterations)

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        neural_status = self.neural_backend.get_backend_status()
        db_stats = self.db_manager.get_database_stats()

        return {
            "jarvis2_ready": True,
            "neural_backend": neural_status["current"],
            "available_backends": neural_status["functional_backends"],
            "cpu_cores_allocated": self.config.get_jarvis2_cpu_cores(),
            "memory_limit_gb": self.config.hardware.memory_limit_gb,
            "database_stats": db_stats,
            "config_version": self.config.config_version,
        }


async def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Jarvis2 Unified System")
    parser.add_argument(
        "command",
        choices=["generate", "analyze", "optimize", "search", "status", "interactive"],
        help="Command to execute",
    )
    parser.add_argument("--prompt", "-p", help="Prompt for generation/optimization")
    parser.add_argument("--target", "-t", default=".", help="Target path for analysis")
    parser.add_argument("--meta", action="store_true", help="Enable meta system")
    parser.add_argument(
        "--optimization-type",
        default="performance",
        choices=["performance", "readability", "memory"],
        help="Type of optimization",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=1000, help="Maximum MCTS iterations"
    )

    args = parser.parse_args()

    # Initialize unified Jarvis2
    jarvis = UnifiedJarvis2(enable_meta=args.meta)

    try:
        if args.command == "generate":
            if not args.prompt:
                print("❌ --prompt required for generate command")
                sys.exit(1)
            result = await jarvis.generate_code(args.prompt)
            print("\n📝 Generated Code:")
            print(result)

        elif args.command == "analyze":
            result = await jarvis.analyze_codebase(args.target)
            print("\n📊 Codebase Analysis:")
            for key, value in result.items():
                print(f"  {key}: {value}")

        elif args.command == "optimize":
            if not args.prompt:
                print("❌ --prompt (code to optimize) required for optimize command")
                sys.exit(1)
            result = await jarvis.optimize_code(args.prompt, args.optimization_type)
            print("\n⚡ Optimized Code:")
            print(result)

        elif args.command == "search":
            if not args.prompt:
                print("❌ --prompt (problem description) required for search command")
                sys.exit(1)
            results = await jarvis.search_solutions(args.prompt, args.max_iterations)
            print(f"\n🔍 Found {len(results)} solutions:")
            for i, solution in enumerate(results[:5], 1):
                print(f"  {i}. Score: {solution.get('score', 0):.3f}")
                print(f"     {solution.get('description', 'No description')}")

        elif args.command == "status":
            status = jarvis.get_system_status()
            print("\n🏁 System Status:")
            for key, value in status.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")

        elif args.command == "interactive":
            print("\n🚀 Interactive Jarvis2 Mode")
            print(
                "Commands: generate <prompt>, analyze [path], optimize <code>, search <problem>, status, quit"
            )

            while True:
                try:
                    user_input = input("\njarvis2> ").strip()
                    if user_input.lower() in ["quit", "exit", "q"]:
                        break

                    parts = user_input.split(" ", 1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""

                    if cmd == "generate" and arg:
                        result = await jarvis.generate_code(arg)
                        print(f"Generated: {result[:200]}...")
                    elif cmd == "analyze":
                        target = arg or "."
                        result = await jarvis.analyze_codebase(target)
                        print(f"Analysis: {len(result)} metrics found")
                    elif cmd == "optimize" and arg:
                        result = await jarvis.optimize_code(arg)
                        print(f"Optimized: {result[:200]}...")
                    elif cmd == "search" and arg:
                        results = await jarvis.search_solutions(arg, 100)
                        print(f"Found {len(results)} solutions")
                    elif cmd == "status":
                        status = jarvis.get_system_status()
                        print(
                            f"Status: {status['neural_backend']} backend, {status['cpu_cores_allocated']} cores"
                        )
                    else:
                        print(
                            "Invalid command. Use: generate, analyze, optimize, search, status, quit"
                        )

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    finally:
        # Cleanup
        jarvis.db_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
