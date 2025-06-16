#!/usr/bin/env python3
"""
Memory Optimization Integration for Bolt System
Integrates all memory optimization components and validates <4GB target

This script:
1. Integrates all memory managers
2. Validates memory usage stays under 4GB
3. Provides monitoring and reporting
4. Sets up automatic memory pressure handling
5. Tests system under load
"""

import asyncio
import logging
import time
from typing import Any

import psutil

from .database_memory_optimizer import (
    get_database_memory_manager,
    get_optimized_connection,
)
from .gpu_memory_optimizer import allocate_gpu_operation, get_gpu_memory_manager

# Import all memory optimization components
from .optimized_memory_manager import (
    get_optimized_memory_manager,
    register_component_memory,
)
from .unified_memory_pressure_handler import (
    get_memory_pressure_handler,
    get_memory_pressure_report,
    start_memory_monitoring,
)

logger = logging.getLogger(__name__)


class BoltMemoryOptimizationSystem:
    """Complete memory optimization system for Bolt."""

    def __init__(self):
        self.target_memory_mb = 4096  # 4GB target

        # Initialize all managers
        self.memory_manager = get_optimized_memory_manager()
        self.gpu_manager = get_gpu_memory_manager()
        self.database_manager = get_database_memory_manager()
        self.pressure_handler = get_memory_pressure_handler()

        # System state
        self.monitoring_active = False
        self.optimization_active = True

        logger.info("Bolt Memory Optimization System initialized")

    async def initialize(self):
        """Initialize the complete memory optimization system."""
        logger.info("Initializing Bolt Memory Optimization System...")

        # Start monitoring for all components
        await self.gpu_manager.start_monitoring(interval_seconds=10)
        await self.database_manager.start_monitoring(interval_seconds=30)
        await start_memory_monitoring()

        self.monitoring_active = True
        logger.info("Memory optimization system fully initialized")

    async def shutdown(self):
        """Shutdown the memory optimization system."""
        logger.info("Shutting down Bolt Memory Optimization System...")

        self.optimization_active = False
        self.monitoring_active = False

        # Stop monitoring
        await self.gpu_manager.stop_monitoring()
        await self.database_manager.stop_monitoring()

        # Cleanup all components
        self.memory_manager.shutdown()
        self.gpu_manager.shutdown()
        self.database_manager.shutdown()
        self.pressure_handler.shutdown()

        logger.info("Memory optimization system shutdown complete")

    def get_system_memory_usage(self) -> float:
        """Get current system memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def validate_memory_target(self) -> bool:
        """Validate that memory usage is under target."""
        current_usage = self.get_system_memory_usage()
        return current_usage <= self.target_memory_mb

    def get_comprehensive_report(self) -> dict[str, Any]:
        """Get comprehensive memory report from all components."""
        current_memory = self.get_system_memory_usage()

        # Get reports from all managers
        memory_report = self.memory_manager.get_memory_report()
        gpu_report = self.gpu_manager.get_memory_report()
        db_report = self.database_manager.get_memory_report()
        pressure_report = get_memory_pressure_report()

        return {
            "timestamp": time.time(),
            "system_memory": {
                "current_mb": current_memory,
                "target_mb": self.target_memory_mb,
                "usage_percent": (current_memory / self.target_memory_mb) * 100,
                "under_target": current_memory <= self.target_memory_mb,
                "available_system_mb": psutil.virtual_memory().available
                / (1024 * 1024),
                "system_usage_percent": psutil.virtual_memory().percent,
            },
            "optimization_system": {
                "monitoring_active": self.monitoring_active,
                "optimization_active": self.optimization_active,
                "memory_manager_active": True,
                "gpu_manager_active": gpu_report["gpu_available"],
                "database_manager_active": True,
                "pressure_handler_active": True,
            },
            "component_reports": {
                "memory_manager": memory_report,
                "gpu_manager": gpu_report,
                "database_manager": db_report,
                "pressure_handler": pressure_report,
            },
            "summary": {
                "total_allocated_mb": current_memory,
                "target_compliance": current_memory <= self.target_memory_mb,
                "pressure_level": pressure_report["current_pressure_level"],
                "active_components": len(
                    [
                        name
                        for name, stats in pressure_report["components"].items()
                        if stats["current_mb"] > 0
                    ]
                ),
                "recommendations": self._get_system_recommendations(
                    current_memory, pressure_report
                ),
            },
        }

    def _get_system_recommendations(
        self, current_memory: float, pressure_report: dict
    ) -> list:
        """Get system-wide recommendations."""
        recommendations = []

        # Memory target compliance
        if current_memory > self.target_memory_mb:
            over_mb = current_memory - self.target_memory_mb
            recommendations.append(
                f"Memory usage exceeds target by {over_mb:.1f}MB - immediate optimization required"
            )
        elif current_memory > self.target_memory_mb * 0.9:
            recommendations.append(
                f"Memory usage near target ({(current_memory/self.target_memory_mb)*100:.1f}%) - consider cleanup"
            )

        # Pressure level recommendations
        pressure_level = pressure_report["current_pressure_level"]
        if pressure_level == "critical":
            recommendations.append(
                "CRITICAL memory pressure detected - emergency cleanup triggered"
            )
        elif pressure_level == "high":
            recommendations.append(
                "HIGH memory pressure - aggressive optimization recommended"
            )
        elif pressure_level == "moderate":
            recommendations.append(
                "MODERATE memory pressure - preventive optimization suggested"
            )

        # Component-specific recommendations
        problem_components = []
        for name, stats in pressure_report["components"].items():
            if stats["over_threshold"]:
                problem_components.append(name)

        if problem_components:
            recommendations.append(
                f"Components over threshold: {', '.join(problem_components)}"
            )

        # System health
        system_memory = psutil.virtual_memory()
        if system_memory.percent > 90:
            recommendations.append(
                f"System memory critically high: {system_memory.percent:.1f}%"
            )
        elif system_memory.percent > 80:
            recommendations.append(f"System memory high: {system_memory.percent:.1f}%")

        if not recommendations:
            recommendations.append("All memory optimization targets are being met")

        return recommendations

    async def run_memory_stress_test(
        self, duration_seconds: int = 60
    ) -> dict[str, Any]:
        """Run memory stress test to validate optimization under load."""
        logger.info(f"Running memory stress test for {duration_seconds} seconds...")

        test_results = {
            "duration_seconds": duration_seconds,
            "start_time": time.time(),
            "initial_memory_mb": self.get_system_memory_usage(),
            "peak_memory_mb": 0,
            "final_memory_mb": 0,
            "target_violations": 0,
            "pressure_events": 0,
            "cleanup_events": 0,
            "test_passed": False,
        }

        start_time = time.time()
        test_results["initial_memory_mb"]

        # Simulate memory-intensive operations
        test_operations = []

        try:
            while time.time() - start_time < duration_seconds:
                current_memory = self.get_system_memory_usage()
                test_results["peak_memory_mb"] = max(
                    test_results["peak_memory_mb"], current_memory
                )

                # Check for target violations
                if current_memory > self.target_memory_mb:
                    test_results["target_violations"] += 1
                    logger.warning(
                        f"Memory target violation: {current_memory:.1f}MB > {self.target_memory_mb:.1f}MB"
                    )

                # Simulate agent operations
                agent_cache = register_component_memory("test_agents", 20)
                for i in range(10):
                    test_data = {"result": f"test_data_{i}" * 100}
                    agent_cache.put(f"test_key_{i}", test_data)
                test_operations.append(("agent_cache", agent_cache))

                # Simulate GPU operations
                if gpu_report := self.gpu_manager.get_memory_report():
                    if gpu_report["gpu_available"]:
                        with allocate_gpu_operation("stress_test", 10.0):
                            # Simulate GPU work
                            await asyncio.sleep(0.1)

                # Simulate database operations
                try:
                    with get_optimized_connection() as conn:
                        # Simulate database work
                        if hasattr(conn, "execute"):
                            conn.execute("SELECT 1 as stress_test")
                except Exception as e:
                    logger.debug(f"Database stress test operation failed: {e}")

                # Check memory pressure
                pressure_report = get_memory_pressure_report()
                if pressure_report["current_pressure_level"] != "low":
                    test_results["pressure_events"] += 1

                await asyncio.sleep(1)  # Check every second

            # Final measurements
            test_results["final_memory_mb"] = self.get_system_memory_usage()
            test_results["end_time"] = time.time()

            # Determine if test passed
            test_results["test_passed"] = (
                test_results["target_violations"] == 0
                and test_results["peak_memory_mb"] <= self.target_memory_mb
                and test_results["final_memory_mb"]
                <= self.target_memory_mb * 1.1  # 10% tolerance
            )

            logger.info(
                f"Memory stress test completed: "
                f"Peak: {test_results['peak_memory_mb']:.1f}MB, "
                f"Final: {test_results['final_memory_mb']:.1f}MB, "
                f"Violations: {test_results['target_violations']}, "
                f"Passed: {test_results['test_passed']}"
            )

        except Exception as e:
            logger.error(f"Memory stress test failed: {e}")
            test_results["error"] = str(e)
            test_results["test_passed"] = False

        finally:
            # Cleanup test operations
            for _op_type, op_obj in test_operations:
                try:
                    if hasattr(op_obj, "clear"):
                        op_obj.clear()
                except AttributeError as e:
                    logger.debug(f"Could not clear operation object: {e}")

        return test_results

    def print_status_report(self):
        """Print comprehensive status report."""
        report = self.get_comprehensive_report()

        print("\n" + "=" * 80)
        print("BOLT MEMORY OPTIMIZATION SYSTEM STATUS")
        print("=" * 80)

        # System memory
        sys_mem = report["system_memory"]
        status_icon = "✅" if sys_mem["under_target"] else "⚠️"
        print(f"\n{status_icon} SYSTEM MEMORY:")
        print(
            f"   Current: {sys_mem['current_mb']:.1f}MB / {sys_mem['target_mb']:.1f}MB ({sys_mem['usage_percent']:.1f}%)"
        )
        print(
            f"   System:  {sys_mem['system_usage_percent']:.1f}% used, {sys_mem['available_system_mb']:.1f}MB available"
        )
        print(
            f"   Status:  {'UNDER TARGET' if sys_mem['under_target'] else 'OVER TARGET'}"
        )

        # Pressure status
        pressure = report["component_reports"]["pressure_handler"]
        print("\n🔥 MEMORY PRESSURE:")
        print(f"   Level:   {pressure['current_pressure_level'].upper()}")
        print(f"   Trend:   {pressure['memory_trend']}")
        print(f"   Events:  {pressure['recent_pressure_events']} in last hour")

        # Component status
        print("\n📊 COMPONENT USAGE:")
        for name, stats in pressure["components"].items():
            status_icon = "⚠️" if stats["over_threshold"] else "✅"
            print(
                f"   {status_icon} {name.capitalize():12} "
                f"{stats['current_mb']:6.1f}MB / {stats['budget_mb']:6.1f}MB "
                f"({stats['usage_percent']:5.1f}%) "
                f"[Priority: {stats['priority']}]"
            )

        # Optimization system status
        opt_sys = report["optimization_system"]
        print("\n⚙️  OPTIMIZATION SYSTEM:")
        print(
            f"   Monitoring:    {'ACTIVE' if opt_sys['monitoring_active'] else 'INACTIVE'}"
        )
        print(
            f"   Memory Mgr:    {'ACTIVE' if opt_sys['memory_manager_active'] else 'INACTIVE'}"
        )
        print(
            f"   GPU Mgr:       {'ACTIVE' if opt_sys['gpu_manager_active'] else 'INACTIVE'}"
        )
        print(
            f"   Database Mgr:  {'ACTIVE' if opt_sys['database_manager_active'] else 'INACTIVE'}"
        )
        print(
            f"   Pressure Mgr:  {'ACTIVE' if opt_sys['pressure_handler_active'] else 'INACTIVE'}"
        )

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        for rec in report["summary"]["recommendations"]:
            print(f"   • {rec}")

        print("=" * 80)


# Global system instance
_optimization_system: BoltMemoryOptimizationSystem | None = None


def get_optimization_system() -> BoltMemoryOptimizationSystem:
    """Get the global optimization system."""
    global _optimization_system
    if _optimization_system is None:
        _optimization_system = BoltMemoryOptimizationSystem()
    return _optimization_system


async def initialize_memory_optimization():
    """Initialize the complete memory optimization system."""
    system = get_optimization_system()
    await system.initialize()
    return system


async def shutdown_memory_optimization():
    """Shutdown the memory optimization system."""
    system = get_optimization_system()
    await system.shutdown()


def validate_memory_optimization() -> bool:
    """Validate that memory optimization is working."""
    system = get_optimization_system()
    return system.validate_memory_target()


def print_memory_status():
    """Print current memory status."""
    system = get_optimization_system()
    system.print_status_report()


async def run_memory_validation_test(duration_seconds: int = 60) -> bool:
    """Run comprehensive memory validation test."""
    system = get_optimization_system()

    logger.info("Starting memory validation test...")

    # Initialize system
    await system.initialize()

    try:
        # Run stress test
        test_results = await system.run_memory_stress_test(duration_seconds)

        # Print results
        print("\n" + "=" * 80)
        print("MEMORY VALIDATION TEST RESULTS")
        print("=" * 80)
        print(f"Duration:         {test_results['duration_seconds']} seconds")
        print(f"Initial Memory:   {test_results['initial_memory_mb']:.1f}MB")
        print(f"Peak Memory:      {test_results['peak_memory_mb']:.1f}MB")
        print(f"Final Memory:     {test_results['final_memory_mb']:.1f}MB")
        print(f"Target:           {system.target_memory_mb:.1f}MB")
        print(f"Target Violations: {test_results['target_violations']}")
        print(f"Pressure Events:  {test_results['pressure_events']}")
        print(
            f"Test Result:      {'PASSED' if test_results['test_passed'] else 'FAILED'}"
        )
        print("=" * 80)

        # Print final status
        system.print_status_report()

        return test_results["test_passed"]

    finally:
        # Shutdown system
        await system.shutdown()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        print("Bolt Memory Optimization System - Integration Test")
        print("=" * 60)

        # Run validation test
        success = await run_memory_validation_test(duration_seconds=30)

        if success:
            print("\n✅ Memory optimization validation PASSED!")
            print("   System successfully maintains <4GB memory usage under load")
        else:
            print("\n❌ Memory optimization validation FAILED!")
            print("   System exceeded memory targets or encountered errors")

        return success

    # Run the test
    result = asyncio.run(main())
    exit(0 if result else 1)
