#!/usr/bin/env python3
"""
Main CLI entry point for Bolt system.

Provides a unified interface to all Bolt functionality:
- Problem solving with 8-agent system
- Hardware monitoring and benchmarking
- System configuration and management
- Comprehensive error handling and diagnostics
"""

import logging
import sys
import traceback

import click

from ..error_handling import BoltException, ErrorSeverity
from .benchmark import benchmark_main
from .diagnostics import diagnostics_main
from .monitor import monitor_main
from .solve import solve_main


# Configure logging for CLI
def setup_cli_logging(verbose: bool = False) -> None:
    """Setup logging configuration for CLI."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[console_handler],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Reduce noise from some loggers in non-verbose mode
    if not verbose:
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def handle_cli_exception(func):
    """Decorator to handle CLI exceptions gracefully."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BoltException as e:
            logger = logging.getLogger("bolt.cli")

            # Handle different severity levels
            if e.severity == ErrorSeverity.CRITICAL:
                click.echo(
                    click.style(f"CRITICAL ERROR: {e.message}", fg="red", bold=True),
                    err=True,
                )
            elif e.severity == ErrorSeverity.HIGH:
                click.echo(click.style(f"ERROR: {e.message}", fg="red"), err=True)
            elif e.severity == ErrorSeverity.MEDIUM:
                click.echo(click.style(f"Warning: {e.message}", fg="yellow"), err=True)
            else:
                click.echo(f"Info: {e.message}")

            # Show recovery hints if available
            if e.recovery_hints:
                click.echo("\nRecovery suggestions:", err=True)
                for hint in e.recovery_hints:
                    click.echo(f"  • {hint}", err=True)

            # Show diagnostic data in verbose mode
            if e.diagnostic_data and logging.getLogger().level == logging.DEBUG:
                click.echo("\nDiagnostic information:", err=True)
                for key, value in e.diagnostic_data.items():
                    click.echo(f"  {key}: {value}", err=True)

            logger.error(f"CLI command failed: {e}")
            sys.exit(
                1 if e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else 0
            )

        except Exception as e:
            logger = logging.getLogger("bolt.cli")
            click.echo(
                click.style(f"Unexpected error: {e}", fg="red", bold=True), err=True
            )

            if logging.getLogger().level == logging.DEBUG:
                click.echo("\nFull traceback:", err=True)
                click.echo(traceback.format_exc(), err=True)
            else:
                click.echo("Run with --verbose for more details", err=True)

            logger.error(f"Unexpected CLI error: {e}", exc_info=True)
            sys.exit(1)

    return wrapper


@click.group()
@click.version_option()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--config", type=click.Path(exists=True), help="Path to configuration file"
)
@click.option("--log-file", type=click.Path(), help="Path to log file")
@click.pass_context
@handle_cli_exception
def main(ctx: click.Context, verbose: bool, config: str | None, log_file: str | None):
    """
    Bolt - 8-Agent Hardware-Accelerated Problem Solver

    A complete system for solving complex programming problems using 8 parallel
    Claude Code agents with M4 Pro hardware acceleration.

    Features:
    • 8 parallel agents with hardware acceleration
    • MLX GPU acceleration for Apple Silicon
    • Real-time system monitoring and error handling
    • Memory safety enforcement with recovery
    • Intelligent task orchestration
    • Comprehensive diagnostics and troubleshooting
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Setup logging first
    setup_cli_logging(verbose)
    logger = logging.getLogger("bolt.cli")

    # Add file logging if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    # Store global options
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = config
    ctx.obj["log_file"] = log_file

    logger.info("Bolt CLI initialized")


@main.command()
@click.argument("query")
@click.option("--analyze-only", is_flag=True, help="Only analyze, don't execute")
@click.option("--agents", type=int, default=8, help="Number of agents to use")
@click.pass_context
def solve(ctx: click.Context, query: str, analyze_only: bool, agents: int):
    """Solve problems using integrated 8-agent system with M4 Pro acceleration."""
    # Call the solve command with context
    solve_main(query, analyze_only, agents, ctx.obj.get("verbose", False))


@main.command()
@click.option("--duration", type=int, default=60, help="Monitoring duration in seconds")
@click.option("--interval", type=float, default=1.0, help="Update interval in seconds")
@click.pass_context
def monitor(ctx: click.Context, duration: int, interval: float):
    """Monitor system performance and hardware utilization."""
    monitor_main(duration, interval, ctx.obj.get("verbose", False))


@main.command()
@click.option("--quick", is_flag=True, help="Run quick benchmark")
@click.option("--full", is_flag=True, help="Run comprehensive benchmark")
@click.pass_context
def benchmark(ctx: click.Context, quick: bool, full: bool):
    """Run performance benchmarks for hardware acceleration."""
    benchmark_main(quick, full, ctx.obj.get("verbose", False))


# Add diagnostics commands
main.add_command(diagnostics_main)


@main.command()
@handle_cli_exception
def status():
    """Show system status and configuration."""
    from ..core.system_info import get_system_status

    status_info = get_system_status()

    click.echo("=== Bolt System Status ===")
    click.echo(f"Hardware: {status_info['hardware']}")
    click.echo(f"GPU Backend: {status_info['gpu_backend']}")
    click.echo(f"Memory: {status_info['memory_gb']:.1f}GB available")
    click.echo(f"CPU Cores: {status_info['cpu_cores']}")

    if status_info.get("gpu_memory_gb"):
        click.echo(f"GPU Memory: {status_info['gpu_memory_gb']:.1f}GB")

    # Show accelerated tools status
    tools_status = status_info.get("accelerated_tools", {})
    click.echo("\n=== Accelerated Tools ===")
    for tool, available in tools_status.items():
        status = "✓" if available else "✗"
        click.echo(f"{status} {tool}")


if __name__ == "__main__":
    main()
