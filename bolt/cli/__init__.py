"""
Bolt CLI module - Command line interfaces for the Bolt system.

Provides entry points for:
- bolt: Main CLI interface
- bolt-solve: Problem solving interface
- bolt-monitor: System monitoring interface
- bolt-bench: Benchmarking interface
"""

from .benchmark import benchmark_main
from .main import main
from .monitor import monitor_main
from .solve import solve_main

__all__ = ["main", "solve_main", "monitor_main", "benchmark_main"]
