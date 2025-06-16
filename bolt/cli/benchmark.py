#!/usr/bin/env python3
"""
Benchmarking CLI for Bolt system.

Provides performance benchmarking for hardware acceleration and system components.
"""

import asyncio
from typing import Any

import click

from ..hardware.benchmarks import (
    run_comprehensive_benchmark,
    run_gpu_benchmark,
    run_memory_benchmark,
    run_quick_benchmark,
    run_storage_benchmark,
)
from ..utils.display import format_benchmark_results


async def run_benchmarks(
    quick: bool = False, full: bool = False, verbose: bool = False
) -> dict[str, Any]:
    """Run performance benchmarks."""

    results = {}

    try:
        if quick:
            click.echo("=== Quick Benchmark ===")
            results = await run_quick_benchmark()

        elif full:
            click.echo("=== Comprehensive Benchmark ===")
            results = await run_comprehensive_benchmark()

        else:
            # Default: run standard benchmark suite
            click.echo("=== Standard Benchmark Suite ===")

            # CPU and Memory
            click.echo("Running CPU benchmark...")
            results["cpu"] = await run_quick_benchmark()

            # GPU acceleration
            click.echo("Running GPU benchmark...")
            results["gpu"] = await run_gpu_benchmark()

            # Memory performance
            click.echo("Running memory benchmark...")
            results["memory"] = await run_memory_benchmark()

            # Storage performance
            click.echo("Running storage benchmark...")
            results["storage"] = await run_storage_benchmark()

        # Display results
        formatted_results = format_benchmark_results(results)
        click.echo("\n" + formatted_results)

        # Show recommendations
        recommendations = generate_recommendations(results)
        if recommendations:
            click.echo("\n=== Recommendations ===")
            for rec in recommendations:
                click.echo(f"• {rec}")

        return results

    except Exception as e:
        click.echo(f"Benchmark error: {str(e)}", err=True)
        return {"error": str(e)}


def generate_recommendations(results: dict[str, Any]) -> list:
    """Generate performance recommendations based on benchmark results."""
    recommendations = []

    # CPU recommendations
    if "cpu" in results:
        cpu_score = results["cpu"].get("score", 0)
        if cpu_score < 50:
            recommendations.append(
                "Consider closing other applications to free up CPU resources"
            )
        elif cpu_score > 90:
            recommendations.append(
                "Excellent CPU performance - system is well optimized"
            )

    # GPU recommendations
    if "gpu" in results:
        gpu_available = results["gpu"].get("available", False)
        if not gpu_available:
            recommendations.append(
                "GPU acceleration not available - install MLX for Apple Silicon optimization"
            )
        else:
            gpu_score = results["gpu"].get("score", 0)
            if gpu_score > 80:
                recommendations.append("GPU acceleration is working well")

    # Memory recommendations
    if "memory" in results:
        memory_score = results["memory"].get("score", 0)
        if memory_score < 60:
            recommendations.append(
                "Consider increasing available memory or closing memory-intensive applications"
            )

    # Storage recommendations
    if "storage" in results:
        storage_score = results["storage"].get("score", 0)
        if storage_score < 70:
            recommendations.append("Consider using SSD storage for better performance")

    return recommendations


def benchmark_main(quick: bool = False, full: bool = False, verbose: bool = False):
    """Main entry point for benchmark command."""

    if verbose:
        mode = "quick" if quick else "comprehensive" if full else "standard"
        click.echo(f"Starting {mode} benchmark...")

    # Run the async benchmarks
    results = asyncio.run(run_benchmarks(quick, full, verbose))

    # Return appropriate exit code
    if "error" in results:
        return 1
    else:
        return 0


@click.command()
@click.option("--quick", is_flag=True, help="Run quick benchmark")
@click.option("--full", is_flag=True, help="Run comprehensive benchmark")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(quick: bool, full: bool, verbose: bool):
    """
    Run performance benchmarks for hardware acceleration.

    Benchmarks include:
    - CPU performance and core utilization
    - GPU acceleration (MLX, Metal)
    - Memory bandwidth and latency
    - Storage I/O performance
    - Accelerated tool performance

    Modes:
    --quick: Fast benchmark (30 seconds)
    --full: Comprehensive benchmark (5+ minutes)
    Default: Standard benchmark suite (2-3 minutes)

    Examples:
        bolt-bench --quick
        bolt-bench --full --verbose
        bolt-bench  # Run standard benchmark
    """
    exit_code = benchmark_main(quick, full, verbose)
    if exit_code != 0:
        click.get_current_context().exit(exit_code)


if __name__ == "__main__":
    main()
