#!/usr/bin/env python3
"""
Memory-Optimized Bolt CLI Integration
Integrates memory optimizations into the main bolt system

Usage:
    from bolt.memory_optimized_bolt import initialize_memory_optimization, get_memory_report
    
    # Initialize memory optimization
    await initialize_memory_optimization()
    
    # Get memory status
    report = get_memory_report()
    print(f"Memory usage: {report['current_mb']:.1f}MB / {report['target_mb']:.1f}MB")
"""

import asyncio
import atexit
import logging
from typing import Any

# Import memory optimization components
from .memory_optimization_integration import (
    get_optimization_system,
    initialize_memory_optimization,
    shutdown_memory_optimization,
    validate_memory_optimization,
)

logger = logging.getLogger(__name__)

# Global state
_memory_optimization_initialized = False
_optimization_system = None


async def init_bolt_memory_optimization():
    """Initialize memory optimization for bolt system."""
    global _memory_optimization_initialized, _optimization_system

    if _memory_optimization_initialized:
        return

    try:
        logger.info("Initializing Bolt memory optimization...")
        _optimization_system = await initialize_memory_optimization()
        _memory_optimization_initialized = True

        # Register cleanup on exit
        atexit.register(cleanup_on_exit)

        logger.info("Bolt memory optimization initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize memory optimization: {e}")
        raise


def cleanup_on_exit():
    """Cleanup function called on exit."""
    if _memory_optimization_initialized and _optimization_system:
        try:
            asyncio.run(shutdown_memory_optimization())
        except Exception as e:
            logger.error(f"Error during memory optimization cleanup: {e}")


def get_memory_report() -> dict[str, Any]:
    """Get current memory optimization report."""
    if not _memory_optimization_initialized:
        return {
            "error": "Memory optimization not initialized",
            "current_mb": 0,
            "target_mb": 4096,
            "optimization_active": False,
        }

    try:
        system = get_optimization_system()
        full_report = system.get_comprehensive_report()

        # Return simplified report for CLI use
        return {
            "current_mb": full_report["system_memory"]["current_mb"],
            "target_mb": full_report["system_memory"]["target_mb"],
            "usage_percent": full_report["system_memory"]["usage_percent"],
            "under_target": full_report["system_memory"]["under_target"],
            "pressure_level": full_report["component_reports"]["pressure_handler"][
                "current_pressure_level"
            ],
            "optimization_active": full_report["optimization_system"][
                "monitoring_active"
            ],
            "recommendations": full_report["summary"]["recommendations"][
                :3
            ],  # Top 3 recommendations
            "component_summary": {
                name: {
                    "usage_mb": stats["current_mb"],
                    "budget_mb": stats["budget_mb"],
                    "usage_percent": stats["usage_percent"],
                }
                for name, stats in full_report["component_reports"]["pressure_handler"][
                    "components"
                ].items()
                if stats["current_mb"] > 0
            },
        }

    except Exception as e:
        logger.error(f"Error getting memory report: {e}")
        return {
            "error": str(e),
            "current_mb": 0,
            "target_mb": 4096,
            "optimization_active": False,
        }


def check_memory_status() -> bool:
    """Check if memory usage is within acceptable limits."""
    if not _memory_optimization_initialized:
        return False

    try:
        return validate_memory_optimization()
    except Exception as e:
        logger.error(f"Error checking memory status: {e}")
        return False


def print_memory_status_summary():
    """Print a concise memory status summary."""
    if not _memory_optimization_initialized:
        print("❌ Memory optimization not initialized")
        return

    try:
        report = get_memory_report()

        if "error" in report:
            print(f"❌ Memory report error: {report['error']}")
            return

        # Status icon based on usage
        if report["under_target"]:
            if report["usage_percent"] < 70:
                status_icon = "✅"
                status_text = "OPTIMAL"
            elif report["usage_percent"] < 85:
                status_icon = "⚠️"
                status_text = "MODERATE"
            else:
                status_icon = "🔶"
                status_text = "HIGH"
        else:
            status_icon = "🔴"
            status_text = "OVER TARGET"

        print(
            f"{status_icon} Memory: {report['current_mb']:.1f}MB / {report['target_mb']:.1f}MB "
            f"({report['usage_percent']:.1f}%) - {status_text}"
        )

        if report["pressure_level"] != "low":
            print(f"   Pressure: {report['pressure_level'].upper()}")

        if report["recommendations"]:
            print(f"   💡 {report['recommendations'][0]}")

    except Exception as e:
        print(f"❌ Error displaying memory status: {e}")


def force_memory_cleanup() -> float:
    """Force immediate memory cleanup and return amount freed."""
    if not _memory_optimization_initialized:
        logger.warning("Memory optimization not initialized")
        return 0.0

    try:
        from .unified_memory_pressure_handler import force_emergency_cleanup

        freed_mb = force_emergency_cleanup()
        logger.info(f"Forced memory cleanup freed {freed_mb:.1f}MB")
        return freed_mb
    except Exception as e:
        logger.error(f"Error during forced cleanup: {e}")
        return 0.0


# Decorator for memory-aware functions
def memory_aware(target_mb: float | None = None):
    """Decorator to make functions memory-aware."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Check memory before operation
            if _memory_optimization_initialized:
                report = get_memory_report()
                if not report.get("under_target", True):
                    logger.warning(
                        f"Memory usage high before {func.__name__}: {report['current_mb']:.1f}MB"
                    )
                    force_memory_cleanup()

            # Execute function
            result = await func(*args, **kwargs)

            # Check memory after operation
            if _memory_optimization_initialized:
                report = get_memory_report()
                if not report.get("under_target", True):
                    logger.warning(
                        f"Memory usage high after {func.__name__}: {report['current_mb']:.1f}MB"
                    )

            return result

        def sync_wrapper(*args, **kwargs):
            # Check memory before operation
            if _memory_optimization_initialized:
                report = get_memory_report()
                if not report.get("under_target", True):
                    logger.warning(
                        f"Memory usage high before {func.__name__}: {report['current_mb']:.1f}MB"
                    )
                    force_memory_cleanup()

            # Execute function
            result = func(*args, **kwargs)

            # Check memory after operation
            if _memory_optimization_initialized:
                report = get_memory_report()
                if not report.get("under_target", True):
                    logger.warning(
                        f"Memory usage high after {func.__name__}: {report['current_mb']:.1f}MB"
                    )

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Context manager for memory-intensive operations
class memory_context:
    """Context manager for memory-intensive operations."""

    def __init__(self, operation_name: str, expected_mb: float | None = None):
        self.operation_name = operation_name
        self.expected_mb = expected_mb
        self.initial_memory = 0

    def __enter__(self):
        if _memory_optimization_initialized:
            report = get_memory_report()
            self.initial_memory = report["current_mb"]

            # Check if we have enough memory
            if (
                self.expected_mb
                and report["current_mb"] + self.expected_mb > report["target_mb"]
            ):
                logger.warning(
                    f"Operation {self.operation_name} may exceed memory target"
                )
                force_memory_cleanup()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _memory_optimization_initialized:
            report = get_memory_report()
            final_memory = report["current_mb"]
            memory_used = final_memory - self.initial_memory

            logger.debug(
                f"Operation {self.operation_name} used {memory_used:.1f}MB memory"
            )

            # Cleanup if memory usage is high
            if not report.get("under_target", True):
                logger.info(f"Memory cleanup after {self.operation_name}")
                force_memory_cleanup()


# Convenience functions for common use cases
def ensure_memory_available(required_mb: float) -> bool:
    """Ensure enough memory is available for an operation."""
    if not _memory_optimization_initialized:
        return True

    report = get_memory_report()
    available_mb = report["target_mb"] - report["current_mb"]

    if available_mb < required_mb:
        logger.info(f"Cleaning up memory for operation requiring {required_mb:.1f}MB")
        force_memory_cleanup()

        # Check again
        report = get_memory_report()
        available_mb = report["target_mb"] - report["current_mb"]

        if available_mb < required_mb:
            logger.warning(
                f"Insufficient memory: need {required_mb:.1f}MB, have {available_mb:.1f}MB"
            )
            return False

    return True


def get_available_memory_mb() -> float:
    """Get available memory in MB."""
    if not _memory_optimization_initialized:
        return 4096.0  # Default target

    report = get_memory_report()
    return max(0, report["target_mb"] - report["current_mb"])


# Export main functions
__all__ = [
    "init_bolt_memory_optimization",
    "get_memory_report",
    "check_memory_status",
    "print_memory_status_summary",
    "force_memory_cleanup",
    "memory_aware",
    "memory_context",
    "ensure_memory_available",
    "get_available_memory_mb",
]


if __name__ == "__main__":
    # Simple test
    async def test_memory_integration():
        print("Testing Bolt Memory Optimization Integration...")

        # Initialize
        await init_bolt_memory_optimization()

        # Print status
        print_memory_status_summary()

        # Get report
        report = get_memory_report()
        print("\nMemory Report:")
        print(f"  Current: {report['current_mb']:.1f}MB")
        print(f"  Target:  {report['target_mb']:.1f}MB")
        print(f"  Status:  {'GOOD' if report['under_target'] else 'HIGH'}")

        # Test memory context
        with memory_context("test_operation", 100):
            print("  Running test operation...")
            # Simulate work
            await asyncio.sleep(0.1)

        print("✅ Test completed successfully!")

    asyncio.run(test_memory_integration())
