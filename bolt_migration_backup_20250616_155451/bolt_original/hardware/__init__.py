"""
Bolt hardware module - Hardware acceleration and monitoring.

Provides hardware-specific functionality including:
- Hardware state monitoring
- Performance monitoring
- GPU acceleration
- Memory management
- Benchmarking tools
"""

from .hardware_state import HardwareState, get_hardware_state
from .memory_manager import BoltMemoryManager, get_memory_manager
from .performance_monitor import PerformanceMonitor, get_performance_monitor

__all__ = [
    "get_hardware_state",
    "HardwareState",
    "get_performance_monitor",
    "PerformanceMonitor",
    "get_memory_manager",
    "BoltMemoryManager",
]
