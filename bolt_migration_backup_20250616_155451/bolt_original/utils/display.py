"""
Display and formatting utilities for Bolt system.

Provides formatted output for system monitoring, benchmarks, and status.
"""

import time
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text


def create_console() -> Console:
    """Create a rich console instance for output."""
    return Console()


class PerformanceDisplay:
    """Real-time performance display for monitoring."""

    def __init__(self):
        self.console = Console()
        self.start_time = time.time()
        self.update_count = 0

    def update(self, system_state: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Update the display with current system state and metrics."""
        self.update_count += 1

        # Clear screen and show header
        self.console.clear()
        self.console.print(
            f"[bold blue]Bolt System Monitor[/bold blue] - Updates: {self.update_count}"
        )
        self.console.print(f"Runtime: {time.time() - self.start_time:.1f}s\n")

        # System state table
        state_table = Table(title="System State")
        state_table.add_column("Metric", style="cyan")
        state_table.add_column("Value", style="magenta")
        state_table.add_column("Status", style="green")

        # CPU
        cpu_percent = system_state.get("cpu_percent", 0)
        cpu_status = "🔥" if cpu_percent > 80 else "⚡" if cpu_percent > 50 else "✅"
        state_table.add_row("CPU Usage", f"{cpu_percent:.1f}%", cpu_status)

        # Memory
        memory_percent = system_state.get("memory_percent", 0)
        memory_gb = system_state.get("memory_used_gb", 0)
        memory_status = (
            "🔥" if memory_percent > 90 else "⚡" if memory_percent > 70 else "✅"
        )
        state_table.add_row(
            "Memory", f"{memory_percent:.1f}% ({memory_gb:.1f}GB)", memory_status
        )

        # GPU if available
        if system_state.get("gpu_memory_used_gb"):
            gpu_memory = system_state["gpu_memory_used_gb"]
            gpu_status = "⚡" if gpu_memory > 5 else "✅"
            state_table.add_row("GPU Memory", f"{gpu_memory:.1f}GB", gpu_status)

        self.console.print(state_table)

        # Performance metrics if available
        if metrics:
            self.console.print("\n")
            metrics_table = Table(title="Performance Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="yellow")

            for key, value in metrics.items():
                if isinstance(value, int | float):
                    metrics_table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
                else:
                    metrics_table.add_row(key.replace("_", " ").title(), str(value))

            self.console.print(metrics_table)


def create_performance_display() -> PerformanceDisplay:
    """Create a new performance display instance."""
    return PerformanceDisplay()


def format_benchmark_results(results: dict[str, Any]) -> str:
    """Format benchmark results for display."""

    if not results:
        return "No benchmark results available"

    output = []
    output.append("=== Benchmark Results ===")

    # Header info
    benchmark_type = results.get("benchmark_type", "unknown")
    duration = results.get("duration", 0)
    output.append(f"Type: {benchmark_type.title()}")
    output.append(f"Duration: {duration:.1f}s")
    output.append("")

    # CPU results
    if "cpu" in results:
        cpu_data = results["cpu"]
        output.append("CPU Performance:")
        if "gflops" in cpu_data:
            output.append(f"  Performance: {cpu_data['gflops']:.1f} GFLOPS")
        if "score" in cpu_data:
            output.append(f"  Score: {cpu_data['score']:.0f}/100")
        output.append("")

    # GPU results
    if "gpu" in results:
        gpu_data = results["gpu"]
        output.append("GPU Performance:")
        output.append(f"  Available: {'Yes' if gpu_data.get('available') else 'No'}")
        if gpu_data.get("available"):
            output.append(f"  Backend: {gpu_data.get('backend', 'unknown')}")
            if "overall_score" in gpu_data:
                output.append(f"  Score: {gpu_data['overall_score']:.0f}/100")
        output.append("")

    # Memory results
    if "memory" in results:
        memory_data = results["memory"]
        output.append("Memory Performance:")
        if "bandwidth_gbps" in memory_data:
            output.append(f"  Bandwidth: {memory_data['bandwidth_gbps']:.1f} GB/s")
        if "score" in memory_data:
            output.append(f"  Score: {memory_data['score']:.0f}/100")
        output.append("")

    # Storage results
    if "storage" in results:
        storage_data = results["storage"]
        output.append("Storage Performance:")
        if "overall_score" in storage_data:
            output.append(f"  Score: {storage_data['overall_score']:.0f}/100")
        output.append("")

    # Accelerated tools results
    if "accelerated_tools" in results:
        tools_data = results["accelerated_tools"]
        output.append("Accelerated Tools:")
        if "overall_score" in tools_data:
            output.append(f"  Overall Score: {tools_data['overall_score']:.0f}/100")

        if "tests" in tools_data:
            for tool_name, tool_data in tools_data["tests"].items():
                available = tool_data.get("available", False)
                status = "✓" if available else "✗"
                output.append(f"  {status} {tool_name}")
        output.append("")

    return "\n".join(output)


def format_system_status(status: dict[str, Any]) -> str:
    """Format system status for display."""

    output = []
    output.append("=== System Status ===")

    # Hardware info
    output.append(f"Hardware: {status.get('hardware', 'Unknown')}")
    output.append(f"Platform: {status.get('platform', 'Unknown')}")
    output.append(f"Python: {status.get('python_version', 'Unknown')}")
    output.append("")

    # Resources
    output.append("Resources:")
    output.append(f"  Memory: {status.get('memory_gb', 0):.1f}GB")
    output.append(f"  CPU Cores: {status.get('cpu_cores', 0)}")
    output.append(f"  CPU Threads: {status.get('cpu_threads', 0)}")

    if status.get("gpu_memory_gb"):
        output.append(f"  GPU Memory: {status.get('gpu_memory_gb', 0):.1f}GB")
    output.append("")

    # GPU backend
    output.append(f"GPU Backend: {status.get('gpu_backend', 'None')}")
    output.append("")

    # Accelerated tools
    tools = status.get("accelerated_tools", {})
    if tools:
        output.append("Accelerated Tools:")
        for tool_name, available in tools.items():
            status_icon = "✓" if available else "✗"
            output.append(f"  {status_icon} {tool_name}")

    return "\n".join(output)


def create_progress_display(description: str) -> Progress:
    """Create a progress display for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=Console(),
    )


def display_error(message: str, details: str | None = None) -> None:
    """Display an error message with optional details."""
    console = Console()

    error_text = Text(f"Error: {message}", style="bold red")
    console.print(error_text)

    if details:
        console.print(f"Details: {details}", style="red")


def display_success(message: str) -> None:
    """Display a success message."""
    console = Console()
    success_text = Text(f"✓ {message}", style="bold green")
    console.print(success_text)


def display_info(message: str) -> None:
    """Display an info message."""
    console = Console()
    info_text = Text(f"ℹ {message}", style="bold blue")
    console.print(info_text)
