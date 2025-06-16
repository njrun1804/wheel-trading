#!/usr/bin/env python3
"""
System monitoring CLI for Bolt system.

Provides real-time monitoring of hardware utilization, memory usage, and performance.
"""

import asyncio
import time

import click

from ..hardware.hardware_state import get_hardware_state
from ..hardware.performance_monitor import get_performance_monitor
from ..utils.display import create_performance_display


async def monitor_system(duration: int, interval: float, verbose: bool = False) -> None:
    """Monitor system performance for specified duration."""

    # Get monitoring components
    perf_monitor = get_performance_monitor()
    hardware_state = get_hardware_state()
    display = create_performance_display()

    try:
        # Initialize monitoring
        perf_monitor.start_monitoring(interval)

        click.echo("=== Bolt System Monitor ===")
        click.echo(f"Monitoring for {duration} seconds (interval: {interval}s)")
        click.echo("Press Ctrl+C to stop early\n")

        start_time = time.time()
        iteration = 0

        while (time.time() - start_time) < duration:
            iteration += 1

            # Get current system state
            if hasattr(hardware_state, "get_current_state"):
                current_state = await hardware_state.get_current_state()
            else:
                current_state = {
                    "cpu_percent": 0,
                    "memory_percent": 0,
                    "memory_used_gb": 0,
                }

            # Get performance metrics
            metrics = perf_monitor.get_real_time_dashboard()

            # Update display if available
            if display and hasattr(display, "update"):
                display.update(current_state, metrics)

            if verbose or iteration % 10 == 0:  # Show detailed info every 10 iterations
                click.echo(f"\n=== Iteration {iteration} ===")
                click.echo(f"CPU: {current_state.get('cpu_percent', 0):.1f}%")
                click.echo(
                    f"Memory: {current_state.get('memory_percent', 0):.1f}% ({current_state.get('memory_used_gb', 0):.1f}GB used)"
                )

                if current_state.get("gpu_memory_used_gb"):
                    click.echo(
                        f"GPU Memory: {current_state['gpu_memory_used_gb']:.1f}GB"
                    )

                # Show metrics summary if available
                if "error" not in metrics and "summary" in metrics:
                    summary = metrics["summary"]
                    click.echo(f"Health: {summary.get('health', 'unknown')}")
                    click.echo(f"Active agents: {summary.get('agents_working', 0)}")
                    if summary.get("bottleneck"):
                        click.echo(f"Bottleneck: {summary['bottleneck']}")

            # Wait for next interval
            await asyncio.sleep(interval)

        # Show final summary
        click.echo("\n=== Monitoring Complete ===")
        final_metrics = perf_monitor.get_performance_report()

        if "error" not in final_metrics:
            click.echo(f"Average CPU: {final_metrics['cpu']['average']:.1f}%")
            click.echo(f"Peak Memory: {final_metrics['memory']['peak']:.1f}%")
            click.echo(f"Total samples: {final_metrics['total_samples']}")
        else:
            click.echo("No performance data collected")

    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped by user")

    except Exception as e:
        click.echo(f"Monitoring error: {str(e)}", err=True)

    finally:
        # Clean shutdown
        perf_monitor.stop_monitoring()


def monitor_main(duration: int = 60, interval: float = 1.0, verbose: bool = False):
    """Main entry point for monitor command."""

    if verbose:
        click.echo(
            f"Starting system monitor (duration: {duration}s, interval: {interval}s)"
        )

    # Run the async monitoring
    asyncio.run(monitor_system(duration, interval, verbose))


@click.command()
@click.option("--duration", type=int, default=60, help="Monitoring duration in seconds")
@click.option("--interval", type=float, default=1.0, help="Update interval in seconds")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(duration: int, interval: float, verbose: bool):
    """
    Monitor system performance and hardware utilization.

    Provides real-time monitoring of:
    - CPU usage and core utilization
    - Memory consumption and availability
    - GPU memory usage (if available)
    - Process-level resource usage
    - Hardware acceleration status

    Examples:
        bolt-monitor --duration 120
        bolt-monitor --interval 0.5 --verbose
        bolt-monitor --duration 300 --interval 2.0
    """
    monitor_main(duration, interval, verbose)


if __name__ == "__main__":
    main()
