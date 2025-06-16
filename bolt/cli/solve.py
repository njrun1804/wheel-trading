#!/usr/bin/env python3
"""
Problem solving CLI for Bolt system.

Handles the main problem-solving functionality with 8-agent orchestration.
"""

import asyncio
import sys
from typing import Any

import click

from ..core.integration import BoltIntegration


async def solve_query(
    query: str, analyze_only: bool = False, num_agents: int = 8
) -> dict[str, Any]:
    """Solve a query using the integrated Bolt system."""

    # Create integrated system
    integration = BoltIntegration(num_agents=num_agents)

    try:
        # Initialize all components
        await integration.initialize()

        if analyze_only:
            # Just analyze the query
            result = await integration.analyze_query(query)

            click.echo("=== Query Analysis ===")
            click.echo(f"Query: {result['query']}")
            click.echo(f"Relevant files: {len(result.get('relevant_files', []))}")
            click.echo(f"Planned tasks: {len(result.get('tasks', []))}")
            click.echo(f"Estimated agents: {result.get('estimated_agents', 0)}")

            click.echo("\n=== Planned Tasks ===")
            for i, task in enumerate(result.get("tasks", []), 1):
                click.echo(
                    f"{i}. {task['description']} (Priority: {task['priority'].name})"
                )

        else:
            # Execute the full query
            result = await integration.execute_query(query)

            click.echo("=== Execution Results ===")
            click.echo(f"Query: {result['query']}")
            click.echo(f"Total duration: {result['total_duration']:.2f}s")

            if result.get("system_state"):
                state = result["system_state"]
                click.echo("\nSystem state:")
                click.echo(f"  CPU: {state['cpu_percent']:.1f}%")
                click.echo(f"  Memory: {state['memory_percent']:.1f}%")
                click.echo(f"  GPU Memory: {state['gpu_memory_used_gb']:.1f}GB")

            click.echo("\n=== Task Results ===")
            for i, task_result in enumerate(result.get("results", []), 1):
                click.echo(f"\n{i}. {task_result['task']}")
                click.echo(f"   Status: {task_result['status']}")
                if task_result.get("duration"):
                    click.echo(f"   Duration: {task_result['duration']:.2f}s")
                if task_result.get("error"):
                    click.echo(f"   Error: {task_result['error']}")

        return result

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return {"error": str(e)}

    finally:
        # Ensure clean shutdown
        await integration.shutdown()


def solve_main(
    query: str, analyze_only: bool = False, num_agents: int = 8, verbose: bool = False
):
    """Main entry point for solve command."""

    if verbose:
        click.echo(f"Starting Bolt solver with {num_agents} agents...")
        if analyze_only:
            click.echo("Running in analysis-only mode")

    # Run the async solve function
    result = asyncio.run(solve_query(query, analyze_only, num_agents))

    # Return appropriate exit code
    if "error" in result:
        sys.exit(1)
    else:
        sys.exit(0)


@click.command()
@click.argument("query")
@click.option("--analyze-only", is_flag=True, help="Only analyze, don't execute")
@click.option("--agents", type=int, default=8, help="Number of agents to use")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(query: str, analyze_only: bool, agents: int, verbose: bool):
    """
    Solve problems using integrated 8-agent system with M4 Pro acceleration.

    Features:
    - 8 parallel Claude Code agents
    - Hardware-accelerated tools (MLX GPU, Metal)
    - Real-time system monitoring
    - Memory safety enforcement
    - Intelligent task orchestration

    Examples:
        bolt-solve "optimize database queries"
        bolt-solve "fix memory leak in trading module" --analyze-only
        bolt-solve "refactor wheel strategy code" --agents 4
    """
    solve_main(query, analyze_only, agents, verbose)


if __name__ == "__main__":
    main()
