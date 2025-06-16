"""
Bolt - 8-Agent Hardware-Accelerated Problem Solver

A complete system for solving complex programming problems using 8 parallel
Claude Code agents with M4 Pro hardware acceleration.
"""

__version__ = "1.0.0"
__author__ = "Generated with Claude Code"

# Core components
from .core.integration import BoltIntegration

# Error handling
from .error_handling import BoltException, ErrorRecoveryManager
from .hardware.hardware_state import get_hardware_state
from .hardware.performance_monitor import get_performance_monitor

__all__ = [
    "get_hardware_state",
    "BoltIntegration",
    "get_performance_monitor",
    "BoltException",
    "ErrorRecoveryManager",
]
