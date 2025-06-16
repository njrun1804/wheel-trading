#!/usr/bin/env python3
"""
Bolt System Validation Suite
===========================

Comprehensive validation that proves the Bolt 8-agent system is production-ready:
- Hardware acceleration is working (MLX, Metal)
- Memory limits are enforced correctly  
- All 8 agents can run in parallel
- Einstein search performance (<50ms)
- Complete workflows from CLI to results
- GPU utilization and safety
- System health checks and diagnostics

This is the final validation gate before production deployment.
"""

import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core Bolt imports
from bolt.agents.types import TaskPriority
from bolt.core.integration import BoltIntegration, SystemState
from bolt.hardware.hardware_state import HardwareState
from bolt.hardware.memory_manager import BoltMemoryManager
from bolt.hardware.performance_monitor import PerformanceMonitor

# Hardware detection
try:
    import mlx.core as mx

    HAS_MLX = mx.metal.is_available()
except ImportError:
    HAS_MLX = False

try:
    import torch

    HAS_TORCH_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH_MPS = False

# Einstein integration
try:
    import importlib

    importlib.import_module("einstein.einstein_config")
    importlib.import_module("einstein.unified_index")
    HAS_EINSTEIN = True
except ImportError:
    HAS_EINSTEIN = False

# Accelerated tools
try:
    import importlib

    tools = [
        "src.unity_wheel.accelerated_tools.dependency_graph_turbo",
        "src.unity_wheel.accelerated_tools.duckdb_turbo",
        "src.unity_wheel.accelerated_tools.python_analysis_turbo",
        "src.unity_wheel.accelerated_tools.ripgrep_turbo",
        "src.unity_wheel.accelerated_tools.trace_simple",
    ]
    for tool in tools:
        importlib.import_module(tool)
    HAS_ACCELERATED_TOOLS = True
except ImportError:
    HAS_ACCELERATED_TOOLS = False


@dataclass
class ValidationResult:
    """Result of a validation test."""

    name: str
    passed: bool
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class SystemValidationReport:
    """Complete system validation report."""

    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    duration_seconds: float
    hardware_info: dict[str, Any]
    results: list[ValidationResult]
    production_ready: bool
    critical_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class BoltSystemValidator:
    """Comprehensive Bolt system validator."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[ValidationResult] = []
        self.start_time = time.time()

    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            colors = {
                "INFO": "\033[94m",  # Blue
                "SUCCESS": "\033[92m",  # Green
                "WARNING": "\033[93m",  # Yellow
                "ERROR": "\033[91m",  # Red
                "RESET": "\033[0m",
            }
            color = colors.get(level, colors["INFO"])
            reset = colors["RESET"]
            print(f"{color}[{timestamp}] {level}: {message}{reset}")

    async def run_test(self, name: str, test_func, *args, **kwargs) -> ValidationResult:
        """Run a single validation test with timing and error handling."""
        start_time = time.time()
        self.log(f"Running test: {name}")

        try:
            result = await test_func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            if isinstance(result, dict):
                passed = result.get("passed", True)
                details = result.get("details", {})
                warnings = result.get("warnings", [])
                error = result.get("error")
            else:
                passed = bool(result)
                details = {}
                warnings = []
                error = None

            validation_result = ValidationResult(
                name=name,
                passed=passed,
                duration_ms=duration_ms,
                details=details,
                warnings=warnings,
                error=error,
            )

            if passed:
                self.log(f"✅ {name} - PASSED ({duration_ms:.1f}ms)", "SUCCESS")
            else:
                self.log(f"❌ {name} - FAILED ({duration_ms:.1f}ms)", "ERROR")
                if error:
                    self.log(f"   Error: {error}", "ERROR")

            for warning in warnings:
                self.log(f"⚠️  {warning}", "WARNING")

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.log(
                f"❌ {name} - EXCEPTION ({duration_ms:.1f}ms): {error_msg}", "ERROR"
            )

            validation_result = ValidationResult(
                name=name, passed=False, duration_ms=duration_ms, error=error_msg
            )

        self.results.append(validation_result)
        return validation_result

    async def validate_hardware_detection(self) -> dict[str, Any]:
        """Validate hardware detection and capabilities."""
        details = {}
        warnings = []

        # CPU detection
        details["cpu_cores"] = psutil.cpu_count()
        details["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        details["cpu_percent"] = psutil.cpu_percent(interval=1)

        # Memory detection
        memory = psutil.virtual_memory()
        details["memory_total_gb"] = memory.total / (1024**3)
        details["memory_available_gb"] = memory.available / (1024**3)
        details["memory_percent"] = memory.percent

        # GPU detection
        details["has_mlx"] = HAS_MLX
        details["has_torch_mps"] = HAS_TORCH_MPS

        if HAS_MLX:
            try:
                details["mlx_devices"] = len(mx.metal.get_devices())
                details["mlx_memory_limit"] = mx.metal.get_memory_limit()
            except Exception as e:
                warnings.append(f"MLX device info failed: {e}")

        # M4 Pro validation
        is_m4_pro = (
            details["cpu_physical_cores"] >= 10
            and details["memory_total_gb"] >= 16  # 8P + 4E cores
            and (HAS_MLX or HAS_TORCH_MPS)  # Minimum for M4 Pro  # GPU acceleration
        )
        details["is_m4_pro_compatible"] = is_m4_pro

        if not is_m4_pro:
            warnings.append("System may not be M4 Pro - performance may be reduced")

        return {"passed": True, "details": details, "warnings": warnings}

    async def validate_memory_management(self) -> dict[str, Any]:
        """Validate memory management and safety limits."""
        details = {}
        warnings = []

        try:
            # Test memory manager initialization
            memory_manager = BoltMemoryManager()
            details["memory_manager_initialized"] = True

            # Check memory limits
            current_usage = memory_manager.get_current_usage_gb()
            details["current_memory_usage_gb"] = current_usage
            details["memory_limit_gb"] = memory_manager.limit_gb
            details["memory_usage_percent"] = (
                current_usage / memory_manager.limit_gb
            ) * 100

            # Test memory pressure detection
            is_under_pressure = memory_manager.is_under_pressure()
            details["under_memory_pressure"] = is_under_pressure

            if is_under_pressure:
                warnings.append("System currently under memory pressure")

            # Test safety mechanisms
            if current_usage > memory_manager.limit_gb * 0.9:
                warnings.append(
                    "Memory usage above 90% - safety mechanisms should engage"
                )

            return {"passed": True, "details": details, "warnings": warnings}

        except Exception as e:
            return {"passed": False, "error": str(e), "details": details}

    async def validate_gpu_acceleration(self) -> dict[str, Any]:
        """Validate GPU acceleration is working."""
        details = {}
        warnings = []

        # MLX validation
        if HAS_MLX:
            try:
                # Test basic MLX operations
                start_time = time.time()
                a = mx.random.normal((1000, 1000))
                b = mx.random.normal((1000, 1000))
                c = mx.matmul(a, b)
                mx.eval(c)
                mlx_duration = (time.time() - start_time) * 1000

                details["mlx_matmul_ms"] = mlx_duration
                details["mlx_working"] = True

                if mlx_duration > 100:  # Should be much faster on GPU
                    warnings.append(
                        f"MLX matrix multiplication slow ({mlx_duration:.1f}ms)"
                    )

            except Exception as e:
                details["mlx_working"] = False
                details["mlx_error"] = str(e)
                warnings.append(f"MLX acceleration failed: {e}")
        else:
            warnings.append("MLX not available - GPU acceleration limited")

        # PyTorch MPS validation
        if HAS_TORCH_MPS:
            try:
                import torch

                device = torch.device("mps")

                # Test basic MPS operations
                start_time = time.time()
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                c = torch.matmul(a, b)
                torch.mps.synchronize()
                mps_duration = (time.time() - start_time) * 1000

                details["mps_matmul_ms"] = mps_duration
                details["mps_working"] = True

                if mps_duration > 100:
                    warnings.append(f"PyTorch MPS slow ({mps_duration:.1f}ms)")

            except Exception as e:
                details["mps_working"] = False
                details["mps_error"] = str(e)
                warnings.append(f"PyTorch MPS failed: {e}")
        else:
            warnings.append("PyTorch MPS not available")

        # Overall GPU status
        gpu_available = details.get("mlx_working", False) or details.get(
            "mps_working", False
        )
        details["gpu_acceleration_available"] = gpu_available

        return {"passed": gpu_available, "details": details, "warnings": warnings}

    async def validate_accelerated_tools(self) -> dict[str, Any]:
        """Validate accelerated tools performance."""
        details = {}
        warnings = []

        if not HAS_ACCELERATED_TOOLS:
            return {
                "passed": False,
                "error": "Accelerated tools not available",
                "details": details,
            }

        try:
            # Import functions for testing
            from src.unity_wheel.accelerated_tools.dependency_graph_turbo import (
                get_dependency_graph,
            )
            from src.unity_wheel.accelerated_tools.python_analysis_turbo import (
                get_python_analyzer,
            )
            from src.unity_wheel.accelerated_tools.ripgrep_turbo import (
                get_ripgrep_turbo,
            )

            # Test ripgrep turbo
            start_time = time.time()
            rg = get_ripgrep_turbo()
            search_results = await rg.search("TODO", ".")
            ripgrep_duration = (time.time() - start_time) * 1000

            details["ripgrep_duration_ms"] = ripgrep_duration
            details["ripgrep_results_count"] = len(search_results)

            if ripgrep_duration > 50:  # Should be very fast
                warnings.append(f"Ripgrep search slow ({ripgrep_duration:.1f}ms)")

            # Test dependency graph
            start_time = time.time()
            dep_graph = get_dependency_graph()
            await dep_graph.build_graph()
            dep_graph_duration = (time.time() - start_time) * 1000

            details["dependency_graph_duration_ms"] = dep_graph_duration

            if dep_graph_duration > 1000:  # Should be under 1 second
                warnings.append(
                    f"Dependency graph build slow ({dep_graph_duration:.1f}ms)"
                )

            # Test python analyzer
            start_time = time.time()
            analyzer = get_python_analyzer()
            analysis = await analyzer.analyze_file(__file__)
            python_analysis_duration = (time.time() - start_time) * 1000

            details["python_analysis_duration_ms"] = python_analysis_duration
            details["python_analysis_functions"] = len(analysis.get("functions", []))

            if python_analysis_duration > 100:  # Should be very fast
                warnings.append(
                    f"Python analysis slow ({python_analysis_duration:.1f}ms)"
                )

            return {"passed": True, "details": details, "warnings": warnings}

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": details,
                "warnings": warnings,
            }

    async def validate_einstein_performance(self) -> dict[str, Any]:
        """Validate Einstein search performance (<50ms requirement)."""
        details = {}
        warnings = []

        if not HAS_EINSTEIN:
            return {
                "passed": False,
                "error": "Einstein not available",
                "details": details,
            }

        try:
            # Import Einstein components
            from einstein.unified_index import EinsteinIndexHub

            # Initialize Einstein
            hub = EinsteinIndexHub(".")
            await hub.initialize()

            # Test search performance
            search_queries = [
                "validate system",
                "async function",
                "error handling",
                "performance test",
            ]

            search_times = []
            for query in search_queries:
                start_time = time.time()
                results = await hub.search(query, limit=10)
                duration_ms = (time.time() - start_time) * 1000
                search_times.append(duration_ms)

                details[f'search_{query.replace(" ", "_")}_ms'] = duration_ms
                details[f'search_{query.replace(" ", "_")}_results'] = len(results)

            avg_search_time = sum(search_times) / len(search_times)
            max_search_time = max(search_times)

            details["average_search_time_ms"] = avg_search_time
            details["max_search_time_ms"] = max_search_time
            details["search_performance_target_ms"] = 50

            # Performance validation
            performance_ok = max_search_time < 50

            if avg_search_time > 25:
                warnings.append(f"Average search time high ({avg_search_time:.1f}ms)")

            if max_search_time > 50:
                warnings.append(
                    f"Max search time exceeds target ({max_search_time:.1f}ms > 50ms)"
                )

            return {"passed": performance_ok, "details": details, "warnings": warnings}

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": details,
                "warnings": warnings,
            }

    async def validate_8_agent_parallel_execution(self) -> dict[str, Any]:
        """Validate all 8 agents can run in parallel under load."""
        details = {}
        warnings = []

        try:
            # Initialize Bolt with 8 agents
            bolt = BoltIntegration(num_agents=8)
            await bolt.initialize()

            details["agents_initialized"] = len(bolt.agents)
            details["target_agent_count"] = 8

            # Create parallel tasks for all agents
            tasks = []
            for i in range(8):
                task_desc = f"Search for pattern_{i} in codebase"
                task = bolt.submit_task(task_desc, TaskPriority.NORMAL)
                tasks.append(task)

            details["tasks_submitted"] = len(tasks)

            # Execute tasks in parallel
            start_time = time.time()

            # Simulate parallel execution by creating task futures
            task_futures = []
            for i, agent in enumerate(bolt.agents):
                future = asyncio.create_task(agent.execute_task(tasks[i]))
                task_futures.append(future)

            # Wait for all tasks with timeout
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*task_futures, return_exceptions=True), timeout=30.0
            )

            execution_duration = (time.time() - start_time) * 1000
            details["parallel_execution_duration_ms"] = execution_duration

            # Analyze results
            successful_tasks = sum(
                1 for result in completed_tasks if not isinstance(result, Exception)
            )
            details["successful_tasks"] = successful_tasks
            details["failed_tasks"] = len(completed_tasks) - successful_tasks

            # Check agent health after parallel execution
            healthy_agents = sum(1 for agent in bolt.agents if agent.is_healthy)
            details["healthy_agents_after_test"] = healthy_agents

            await bolt.shutdown()

            # Validation criteria
            parallel_success = (
                successful_tasks >= 6
                and healthy_agents >= 7  # At least 75% success rate
                and execution_duration  # At least 87.5% agents healthy
                < 10000  # Under 10 seconds
            )

            if successful_tasks < 8:
                warnings.append(f"Some tasks failed ({8 - successful_tasks}/8)")

            if healthy_agents < 8:
                warnings.append(
                    f"Some agents unhealthy after test ({8 - healthy_agents}/8)"
                )

            if execution_duration > 5000:
                warnings.append(f"Parallel execution slow ({execution_duration:.1f}ms)")

            return {
                "passed": parallel_success,
                "details": details,
                "warnings": warnings,
            }

        except TimeoutError:
            return {
                "passed": False,
                "error": "Parallel execution timeout (>30s)",
                "details": details,
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": details,
                "warnings": warnings,
            }

    async def validate_system_health_monitoring(self) -> dict[str, Any]:
        """Validate system health checks and diagnostics."""
        details = {}
        warnings = []

        try:
            # Test system state capture
            state = SystemState.capture()

            details["system_state_captured"] = True
            details["cpu_percent"] = state.cpu_percent
            details["memory_percent"] = state.memory_percent
            details["gpu_backend"] = state.gpu_backend
            details["is_healthy"] = state.is_healthy
            details["warnings_count"] = len(state.warnings)

            if state.warnings:
                details["system_warnings"] = state.warnings
                for warning in state.warnings[:3]:  # Limit warnings shown
                    warnings.append(f"System warning: {warning}")

            # Test performance monitoring
            try:
                perf_monitor = PerformanceMonitor()
                perf_data = perf_monitor.get_current_metrics()

                details["performance_monitoring_available"] = True
                details["performance_metrics"] = perf_data

            except Exception as e:
                warnings.append(f"Performance monitoring failed: {e}")

            # Test hardware state monitoring
            try:
                hw_state = HardwareState()
                hw_data = hw_state.get_state()

                details["hardware_monitoring_available"] = True
                details["hardware_state"] = hw_data

            except Exception as e:
                warnings.append(f"Hardware monitoring failed: {e}")

            return {
                "passed": state.is_healthy,
                "details": details,
                "warnings": warnings,
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": details,
                "warnings": warnings,
            }

    async def validate_complete_workflow(self) -> dict[str, Any]:
        """Validate complete workflow from CLI input to results."""
        details = {}
        warnings = []

        try:
            # Initialize Bolt system
            bolt = BoltIntegration(num_agents=4)  # Use fewer agents for workflow test
            await bolt.initialize()

            # Test query analysis
            test_query = "optimize database performance and check for memory leaks"
            start_time = time.time()

            analysis = await bolt.analyze_query(test_query)
            analysis_duration = (time.time() - start_time) * 1000

            details["query_analysis_duration_ms"] = analysis_duration
            details["planned_tasks_count"] = len(analysis["tasks"])
            details["query_analyzed"] = analysis["query"]

            # Test task execution
            start_time = time.time()
            execution_result = await bolt.execute_query(test_query)
            execution_duration = (time.time() - start_time) * 1000

            details["query_execution_duration_ms"] = execution_duration
            details["execution_status"] = execution_result["status"]
            details["completed_tasks"] = len(
                [r for r in execution_result["results"] if r["status"] == "completed"]
            )
            details["total_tasks_executed"] = len(execution_result["results"])

            await bolt.shutdown()

            # Validation criteria
            workflow_success = (
                analysis_duration < 1000
                and execution_duration < 15000  # Analysis under 1s
                and execution_result["status"] == "completed"  # Execution under 15s
                and details["completed_tasks"] > 0  # At least some tasks completed
            )

            if analysis_duration > 500:
                warnings.append(f"Query analysis slow ({analysis_duration:.1f}ms)")

            if execution_duration > 10000:
                warnings.append(f"Query execution slow ({execution_duration:.1f}ms)")

            completion_rate = (
                details["completed_tasks"] / details["total_tasks_executed"]
            )
            if completion_rate < 0.8:
                warnings.append(f"Low task completion rate ({completion_rate:.1%})")

            return {
                "passed": workflow_success,
                "details": details,
                "warnings": warnings,
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": details,
                "warnings": warnings,
            }

    async def run_full_validation(self) -> SystemValidationReport:
        """Run complete system validation suite."""
        self.log("🚀 Starting Bolt System Validation Suite", "INFO")
        self.log("=" * 60, "INFO")

        # Define all validation tests
        validation_tests = [
            ("Hardware Detection", self.validate_hardware_detection),
            ("Memory Management", self.validate_memory_management),
            ("GPU Acceleration", self.validate_gpu_acceleration),
            ("Accelerated Tools", self.validate_accelerated_tools),
            ("Einstein Performance", self.validate_einstein_performance),
            ("8-Agent Parallel Execution", self.validate_8_agent_parallel_execution),
            ("System Health Monitoring", self.validate_system_health_monitoring),
            ("Complete Workflow", self.validate_complete_workflow),
        ]

        # Run all tests
        for test_name, test_func in validation_tests:
            await self.run_test(test_name, test_func)

        # Generate report
        total_duration = time.time() - self.start_time
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = len(self.results) - passed_tests

        # Collect hardware info
        hardware_info = {
            "cpu_cores": psutil.cpu_count(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "has_mlx": HAS_MLX,
            "has_torch_mps": HAS_TORCH_MPS,
            "has_accelerated_tools": HAS_ACCELERATED_TOOLS,
            "has_einstein": HAS_EINSTEIN,
        }

        # Determine production readiness
        critical_failures = [
            r
            for r in self.results
            if not r.passed
            and r.name
            in [
                "Hardware Detection",
                "Memory Management",
                "8-Agent Parallel Execution",
                "Complete Workflow",
            ]
        ]

        production_ready = len(critical_failures) == 0 and passed_tests >= 6

        # Collect critical issues and recommendations
        critical_issues = []
        recommendations = []

        for result in self.results:
            if not result.passed:
                critical_issues.append(f"{result.name}: {result.error or 'Failed'}")

            for warning in result.warnings:
                recommendations.append(f"{result.name}: {warning}")

        if not HAS_MLX and not HAS_TORCH_MPS:
            critical_issues.append("No GPU acceleration available")
            recommendations.append("Install MLX or ensure PyTorch MPS is available")

        if hardware_info["memory_total_gb"] < 16:
            recommendations.append("Increase system memory for optimal performance")

        report = SystemValidationReport(
            timestamp=datetime.now(),
            total_tests=len(self.results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            duration_seconds=total_duration,
            hardware_info=hardware_info,
            results=self.results,
            production_ready=production_ready,
            critical_issues=critical_issues,
            recommendations=recommendations,
        )

        # Log summary
        self.log("=" * 60, "INFO")
        self.log("📊 VALIDATION SUMMARY", "INFO")
        self.log(f"Tests Run: {report.total_tests}", "INFO")
        self.log(
            f"Passed: {report.passed_tests}",
            "SUCCESS" if report.passed_tests > 0 else "INFO",
        )
        self.log(
            f"Failed: {report.failed_tests}",
            "ERROR" if report.failed_tests > 0 else "INFO",
        )
        self.log(f"Duration: {report.duration_seconds:.1f}s", "INFO")

        if report.production_ready:
            self.log("🎉 SYSTEM IS PRODUCTION READY! 🎉", "SUCCESS")
        else:
            self.log("⚠️  SYSTEM NOT READY FOR PRODUCTION", "ERROR")
            self.log("Critical Issues:", "ERROR")
            for issue in report.critical_issues:
                self.log(f"  - {issue}", "ERROR")

        if report.recommendations:
            self.log("💡 Recommendations:", "WARNING")
            for rec in report.recommendations[:5]:  # Show top 5
                self.log(f"  - {rec}", "WARNING")

        return report


async def main():
    """Main validation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Bolt System Validation Suite")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Run in quiet mode (less output)"
    )
    parser.add_argument("--output", "-o", type=str, help="Output report to JSON file")
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)",
    )

    args = parser.parse_args()

    # Create validator
    validator = BoltSystemValidator(verbose=not args.quiet)

    try:
        # Run validation with timeout
        report = await asyncio.wait_for(
            validator.run_full_validation(), timeout=args.timeout
        )

        # Save report if requested
        if args.output:
            report_data = {
                "timestamp": report.timestamp.isoformat(),
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "duration_seconds": report.duration_seconds,
                "hardware_info": report.hardware_info,
                "production_ready": report.production_ready,
                "critical_issues": report.critical_issues,
                "recommendations": report.recommendations,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "duration_ms": r.duration_ms,
                        "details": r.details,
                        "error": r.error,
                        "warnings": r.warnings,
                    }
                    for r in report.results
                ],
            }

            with open(args.output, "w") as f:
                json.dump(report_data, f, indent=2)

            print(f"\n📄 Report saved to: {args.output}")

        # Exit with appropriate code
        sys.exit(0 if report.production_ready else 1)

    except TimeoutError:
        print(f"\n⏰ Validation timeout after {args.timeout}s")
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(3)
    except Exception as e:
        print(f"\n💥 Validation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(4)


if __name__ == "__main__":
    asyncio.run(main())
