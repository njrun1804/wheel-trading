"""
Bolt Diagnostic System

Comprehensive diagnostic and troubleshooting tools for the Bolt system.
Provides system health checks, performance analysis, and debugging utilities.
"""

import asyncio
import json
import logging
import os
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


@dataclass
class SystemInfo:
    """System information for diagnostics."""

    platform: str
    python_version: str
    cpu_count: int
    memory_total_gb: float
    disk_total_gb: float
    gpu_backend: str
    gpu_available: bool
    bolt_version: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report."""

    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    system_info: SystemInfo | None = None
    health_checks: list[HealthCheckResult] = field(default_factory=list)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    error_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SystemHealthChecker:
    """Performs comprehensive system health checks."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SystemHealthChecker")
        self.health_checks = {
            "system_resources": self._check_system_resources,
            "python_environment": self._check_python_environment,
            "gpu_availability": self._check_gpu_availability,
            "disk_space": self._check_disk_space,
            "network_connectivity": self._check_network_connectivity,
            "bolt_components": self._check_bolt_components,
            "dependencies": self._check_dependencies,
            "configuration": self._check_configuration,
        }

    async def run_all_checks(self) -> list[HealthCheckResult]:
        """Run all health checks."""
        results = []

        for check_name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = await self._run_check(check_name, check_func)
                result.duration = time.time() - start_time
                results.append(result)
            except Exception as e:
                self.logger.error(
                    f"Health check {check_name} failed: {e}", exc_info=True
                )
                results.append(
                    HealthCheckResult(
                        name=check_name,
                        status="unknown",
                        message=f"Check failed: {e}",
                        duration=time.time() - start_time
                        if "start_time" in locals()
                        else 0.0,
                    )
                )

        return results

    async def _run_check(self, name: str, check_func) -> HealthCheckResult:
        """Run a single health check."""
        try:
            if asyncio.iscoroutinefunction(check_func):
                return await check_func()
            else:
                return check_func()
        except Exception as e:
            return HealthCheckResult(
                name=name, status="critical", message=f"Check execution failed: {e}"
            )

    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability."""
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1.0)

            # Determine status
            if memory_percent > 90 or cpu_percent > 90:
                status = "critical"
                message = f"High resource usage: Memory {memory_percent:.1f}%, CPU {cpu_percent:.1f}%"
            elif memory_percent > 80 or cpu_percent > 80:
                status = "warning"
                message = f"Elevated resource usage: Memory {memory_percent:.1f}%, CPU {cpu_percent:.1f}%"
            else:
                status = "healthy"
                message = f"Resources normal: Memory {memory_percent:.1f}%, CPU {cpu_percent:.1f}%"

            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "memory_percent": memory_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count(logical=True),
                    "load_average": os.getloadavg()
                    if hasattr(os, "getloadavg")
                    else None,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status="unknown",
                message=f"Failed to check system resources: {e}",
            )

    def _check_python_environment(self) -> HealthCheckResult:
        """Check Python environment health."""
        try:
            python_version = sys.version
            python_major, python_minor = sys.version_info[:2]

            # Check Python version compatibility
            if python_major < 3 or (python_major == 3 and python_minor < 8):
                status = "critical"
                message = f"Python version {python_major}.{python_minor} is too old (requires 3.8+)"
            elif python_major == 3 and python_minor < 10:
                status = "warning"
                message = (
                    f"Python {python_major}.{python_minor} works but 3.10+ recommended"
                )
            else:
                status = "healthy"
                message = f"Python {python_major}.{python_minor} is compatible"

            # Check virtual environment
            in_venv = hasattr(sys, "real_prefix") or (
                hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
            )

            return HealthCheckResult(
                name="python_environment",
                status=status,
                message=message,
                details={
                    "python_version": python_version,
                    "python_executable": sys.executable,
                    "in_virtual_env": in_venv,
                    "sys_path_length": len(sys.path),
                    "platform": platform.platform(),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="python_environment",
                status="unknown",
                message=f"Failed to check Python environment: {e}",
            )

    def _check_gpu_availability(self) -> HealthCheckResult:
        """Check GPU availability and configuration."""
        try:
            gpu_backends = []
            gpu_available = False
            gpu_memory = 0

            # Check MLX
            try:
                import mlx.core as mx

                if mx.metal.is_available():
                    gpu_backends.append("mlx")
                    gpu_available = True
            except ImportError:
                pass

            # Check PyTorch MPS
            try:
                import torch

                if torch.backends.mps.is_available():
                    gpu_backends.append("mps")
                    gpu_available = True
                    if hasattr(torch.mps, "current_allocated_memory"):
                        gpu_memory = torch.mps.current_allocated_memory()
            except ImportError:
                pass

            # Determine status
            if gpu_available:
                status = "healthy"
                message = f"GPU available with backends: {', '.join(gpu_backends)}"
            else:
                # On M4 Pro, GPU should be available
                if (
                    platform.machine() == "arm64"
                    and "darwin" in platform.system().lower()
                ):
                    status = "warning"
                    message = "GPU acceleration not available on Apple Silicon (check MLX/PyTorch installation)"
                else:
                    status = "healthy"
                    message = "No GPU acceleration (CPU-only operation)"

            return HealthCheckResult(
                name="gpu_availability",
                status=status,
                message=message,
                details={
                    "backends_available": gpu_backends,
                    "gpu_available": gpu_available,
                    "gpu_memory_bytes": gpu_memory,
                    "platform": platform.machine(),
                    "metal_workspace_limit": os.environ.get(
                        "PYTORCH_METAL_WORKSPACE_LIMIT_BYTES"
                    ),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="gpu_availability",
                status="unknown",
                message=f"Failed to check GPU availability: {e}",
            )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage("/")
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024**3)

            # Determine status
            if disk_percent > 95:
                status = "critical"
                message = f"Disk space critical: {disk_percent:.1f}% used, {free_gb:.1f}GB free"
            elif disk_percent > 85:
                status = "warning"
                message = (
                    f"Disk space low: {disk_percent:.1f}% used, {free_gb:.1f}GB free"
                )
            else:
                status = "healthy"
                message = f"Disk space adequate: {disk_percent:.1f}% used, {free_gb:.1f}GB free"

            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "disk_percent_used": disk_percent,
                    "disk_free_gb": free_gb,
                    "disk_total_gb": disk_usage.total / (1024**3),
                    "disk_used_gb": disk_usage.used / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status="unknown",
                message=f"Failed to check disk space: {e}",
            )

    async def _check_network_connectivity(self) -> HealthCheckResult:
        """Check network connectivity."""
        try:
            # Simple connectivity test
            import socket

            test_hosts = [
                ("8.8.8.8", 53),  # Google DNS
                ("1.1.1.1", 53),  # Cloudflare DNS
            ]

            connectivity_results = []
            for host, port in test_hosts:
                try:
                    sock = socket.create_connection((host, port), timeout=5)
                    sock.close()
                    connectivity_results.append(f"{host}:✓")
                except Exception:
                    connectivity_results.append(f"{host}:✗")

            # Check localhost
            localhost_ok = True
            try:
                sock = socket.create_connection(("127.0.0.1", 22), timeout=1)
                sock.close()
            except Exception:
                localhost_ok = False

            # Determine status
            successful_connections = sum(
                1 for result in connectivity_results if "✓" in result
            )
            if successful_connections == 0:
                status = "critical"
                message = "No network connectivity"
            elif successful_connections < len(test_hosts):
                status = "warning"
                message = f"Limited connectivity: {', '.join(connectivity_results)}"
            else:
                status = "healthy"
                message = f"Network connectivity OK: {', '.join(connectivity_results)}"

            return HealthCheckResult(
                name="network_connectivity",
                status=status,
                message=message,
                details={
                    "connectivity_tests": connectivity_results,
                    "localhost_accessible": localhost_ok,
                    "successful_connections": successful_connections,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="network_connectivity",
                status="unknown",
                message=f"Failed to check network connectivity: {e}",
            )

    def _check_bolt_components(self) -> HealthCheckResult:
        """Check Bolt system components."""
        try:
            components_status = {}

            # Check if main modules can be imported using importlib
            import importlib

            modules_to_check = {
                "integration": "bolt.core.integration",
                "memory_manager": "bolt.hardware.memory_manager",
                "error_handling": "bolt.error_handling.recovery",
                "circuit_breaker": "bolt.error_handling.circuit_breaker",
                "resource_guards": "bolt.error_handling.resource_guards",
                "accelerated_tools": "src.unity_wheel.accelerated_tools.ripgrep_turbo",
            }

            for component, module_name in modules_to_check.items():
                try:
                    module = importlib.import_module(module_name)
                    components_status[component] = "available"

                    # Additional component health checks
                    if component == "error_handling":
                        # Check if recovery manager can be instantiated
                        try:
                            from bolt.error_handling.recovery import (
                                ErrorRecoveryManager,
                            )

                            manager = ErrorRecoveryManager()
                            components_status[component] = "healthy"
                        except Exception:
                            components_status[component] = "available_but_unhealthy"

                    elif component == "circuit_breaker":
                        # Check if circuit breakers work
                        try:
                            from bolt.error_handling.circuit_breaker import (
                                CircuitBreaker,
                                CircuitBreakerConfig,
                            )

                            test_cb = CircuitBreaker("test", CircuitBreakerConfig())
                            components_status[component] = "healthy"
                        except Exception:
                            components_status[component] = "available_but_unhealthy"

                    elif component == "resource_guards":
                        # Check if resource guards can be created
                        try:
                            from bolt.error_handling.resource_guards import (
                                CPUGuard,
                                MemoryGuard,
                            )

                            memory_guard = MemoryGuard(enable_monitoring=False)
                            cpu_guard = CPUGuard(enable_monitoring=False)
                            components_status[component] = "healthy"
                        except Exception:
                            components_status[component] = "available_but_unhealthy"

                except ImportError as e:
                    components_status[component] = f"import_error: {e}"
                except Exception as e:
                    components_status[component] = f"error: {e}"

            # Check for active error handling systems
            active_systems = {}
            try:
                from bolt.error_handling.circuit_breaker import _circuit_breaker_manager

                active_systems["circuit_breaker_manager"] = len(
                    _circuit_breaker_manager.circuit_breakers
                )
            except Exception:
                active_systems["circuit_breaker_manager"] = "unavailable"

            try:
                from bolt.error_handling.resource_guards import _resource_guard_manager

                active_systems["resource_guard_manager"] = len(
                    _resource_guard_manager.guards
                )
            except Exception:
                active_systems["resource_guard_manager"] = "unavailable"

            # Determine overall status
            healthy_count = sum(
                1 for status in components_status.values() if status == "healthy"
            )
            available_count = sum(
                1
                for status in components_status.values()
                if "available" in status or status == "healthy"
            )
            total_count = len(components_status)

            if healthy_count == total_count:
                status = "healthy"
                message = "All Bolt components healthy and functional"
            elif available_count == total_count:
                status = "warning"
                message = f"All components available, {healthy_count}/{total_count} fully healthy"
            elif available_count > total_count // 2:
                status = "warning"
                message = f"{available_count}/{total_count} components available, {healthy_count} healthy"
            else:
                status = "critical"
                message = f"Only {available_count}/{total_count} components available"

            return HealthCheckResult(
                name="bolt_components",
                status=status,
                message=message,
                details={
                    "components": components_status,
                    "active_systems": active_systems,
                    "summary": {
                        "total": total_count,
                        "available": available_count,
                        "healthy": healthy_count,
                    },
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="bolt_components",
                status="unknown",
                message=f"Failed to check Bolt components: {e}",
            )

    def _check_dependencies(self) -> HealthCheckResult:
        """Check critical dependencies."""
        try:
            dependencies = {
                "psutil": "System monitoring",
                "asyncio": "Async operations",
                "logging": "Logging system",
                "pathlib": "Path operations",
                "json": "JSON serialization",
                "time": "Time operations",
                "threading": "Threading support",
            }

            dependency_status = {}
            for dep_name, _description in dependencies.items():
                try:
                    __import__(dep_name)
                    dependency_status[dep_name] = "available"
                except ImportError as e:
                    dependency_status[dep_name] = f"missing: {e}"

            # Check optional dependencies
            optional_deps = {
                "mlx": "MLX GPU acceleration",
                "torch": "PyTorch GPU support",
                "numpy": "Numerical computing",
            }

            for dep_name, _description in optional_deps.items():
                try:
                    __import__(dep_name)
                    dependency_status[f"{dep_name} (optional)"] = "available"
                except ImportError:
                    dependency_status[f"{dep_name} (optional)"] = "not installed"

            # Determine status
            missing_critical = [
                dep
                for dep, status in dependency_status.items()
                if "missing" in status and "(optional)" not in dep
            ]

            if missing_critical:
                status = "critical"
                message = (
                    f"Missing critical dependencies: {', '.join(missing_critical)}"
                )
            else:
                status = "healthy"
                message = "All critical dependencies available"

            return HealthCheckResult(
                name="dependencies",
                status=status,
                message=message,
                details=dependency_status,
            )

        except Exception as e:
            return HealthCheckResult(
                name="dependencies",
                status="unknown",
                message=f"Failed to check dependencies: {e}",
            )

    def _check_configuration(self) -> HealthCheckResult:
        """Check system configuration."""
        try:
            config_issues = []
            config_info = {}

            # Check environment variables
            recommended_env_vars = {
                "PYTORCH_METAL_WORKSPACE_LIMIT_BYTES": "18GB limit for GPU",
                "KMP_DUPLICATE_LIB_OK": "OpenMP compatibility",
            }

            for var, description in recommended_env_vars.items():
                value = os.environ.get(var)
                config_info[var] = value or "not set"
                if not value:
                    config_issues.append(
                        f"Recommended environment variable {var} not set ({description})"
                    )

            # Check file permissions in current directory
            current_dir = Path.cwd()
            if not os.access(current_dir, os.W_OK):
                config_issues.append("Current directory not writable")

            # Check temp directory
            temp_dir = Path(
                "/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:\\temp")
            )
            if not temp_dir.exists() or not os.access(temp_dir, os.W_OK):
                config_issues.append("Temp directory not accessible")

            # Determine status
            if len(config_issues) > 3:
                status = "critical"
                message = (
                    f"Multiple configuration issues: {len(config_issues)} problems"
                )
            elif config_issues:
                status = "warning"
                message = f"Configuration issues found: {len(config_issues)} warnings"
            else:
                status = "healthy"
                message = "Configuration looks good"

            return HealthCheckResult(
                name="configuration",
                status=status,
                message=message,
                details={
                    "issues": config_issues,
                    "environment_variables": config_info,
                    "current_directory": str(current_dir),
                    "current_dir_writable": os.access(current_dir, os.W_OK),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="configuration",
                status="unknown",
                message=f"Failed to check configuration: {e}",
            )


class DiagnosticCollector:
    """Collects comprehensive diagnostic information."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DiagnosticCollector")
        self.health_checker = SystemHealthChecker()

    async def collect_full_diagnostic(
        self,
        include_performance: bool = True,
        include_logs: bool = True,
        include_system_info: bool = True,
    ) -> DiagnosticReport:
        """Collect comprehensive diagnostic information."""

        report = DiagnosticReport()

        try:
            # Collect system information
            if include_system_info:
                report.system_info = await self._collect_system_info()

            # Run health checks
            report.health_checks = await self.health_checker.run_all_checks()

            # Collect performance metrics
            if include_performance:
                report.performance_metrics = await self._collect_performance_metrics()

            # Collect error information
            report.error_summary = await self._collect_error_summary()

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

            # Add metadata
            report.metadata = {
                "collection_duration": time.time() - report.timestamp,
                "diagnostic_version": "1.0",
                "collection_time_iso": datetime.fromtimestamp(
                    report.timestamp
                ).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to collect diagnostics: {e}", exc_info=True)
            report.metadata["collection_error"] = str(e)

        return report

    async def _collect_system_info(self) -> SystemInfo:
        """Collect basic system information."""
        try:
            # Detect GPU backend
            gpu_backend = "none"
            gpu_available = False

            try:
                import mlx.core as mx

                if mx.metal.is_available():
                    gpu_backend = "mlx"
                    gpu_available = True
            except ImportError:
                pass

            if not gpu_available:
                try:
                    import torch

                    if torch.backends.mps.is_available():
                        gpu_backend = "mps"
                        gpu_available = True
                except ImportError:
                    pass

            # Get system specs
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return SystemInfo(
                platform=platform.platform(),
                python_version=sys.version,
                cpu_count=psutil.cpu_count(logical=True),
                memory_total_gb=memory.total / (1024**3),
                disk_total_gb=disk.total / (1024**3),
                gpu_backend=gpu_backend,
                gpu_available=gpu_available,
                bolt_version="1.0.0",  # Would come from package info
            )

        except Exception as e:
            self.logger.error(f"Failed to collect system info: {e}")
            return SystemInfo(
                platform="unknown",
                python_version="unknown",
                cpu_count=0,
                memory_total_gb=0,
                disk_total_gb=0,
                gpu_backend="unknown",
                gpu_available=False,
                bolt_version="unknown",
            )

    async def _collect_performance_metrics(self) -> dict[str, Any]:
        """Collect performance metrics."""
        try:
            metrics = {}

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1.0)
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

            metrics["cpu"] = {
                "overall_percent": cpu_percent,
                "per_core_percent": cpu_per_core,
                "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
                "context_switches": psutil.cpu_stats().ctx_switches
                if hasattr(psutil.cpu_stats(), "ctx_switches")
                else None,
            }

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            metrics["memory"] = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent,
                "swap_total_gb": swap.total / (1024**3) if swap.total > 0 else 0,
                "swap_used_gb": swap.used / (1024**3) if swap.total > 0 else 0,
                "swap_percent": swap.percent if swap.total > 0 else 0,
            }

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_io = psutil.disk_io_counters()

            metrics["disk"] = {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent": (disk.used / disk.total) * 100,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0,
            }

            # Network metrics
            network_io = psutil.net_io_counters()

            metrics["network"] = {
                "bytes_sent": network_io.bytes_sent if network_io else 0,
                "bytes_recv": network_io.bytes_recv if network_io else 0,
                "packets_sent": network_io.packets_sent if network_io else 0,
                "packets_recv": network_io.packets_recv if network_io else 0,
                "errin": network_io.errin if network_io else 0,
                "errout": network_io.errout if network_io else 0,
            }

            # Process metrics
            process = psutil.Process()

            metrics["process"] = {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_info": process.memory_info()._asdict(),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time(),
                "status": process.status(),
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
            return {"error": str(e)}

    async def _collect_error_summary(self) -> dict[str, Any]:
        """Collect error and exception information."""
        try:
            error_summary = {
                "recent_exceptions": [],
                "error_counts": {},
                "last_error_time": None,
                "total_errors": 0,
            }

            # This would integrate with actual error tracking when available
            # For now, return placeholder structure

            return error_summary

        except Exception as e:
            self.logger.error(f"Failed to collect error summary: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, report: DiagnosticReport) -> list[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []

        try:
            # Analyze health check results
            for health_check in report.health_checks:
                if health_check.status == "critical":
                    if health_check.name == "system_resources":
                        memory_percent = health_check.details.get("memory_percent", 0)
                        cpu_percent = health_check.details.get("cpu_percent", 0)

                        if memory_percent > 95:
                            recommendations.append(
                                "URGENT: Memory critically low - restart system or kill memory-intensive processes"
                            )
                        elif cpu_percent > 95:
                            recommendations.append(
                                "URGENT: CPU critically overloaded - reduce concurrent operations immediately"
                            )
                        else:
                            recommendations.append(
                                "System resources are critically low - consider restarting services or adding more resources"
                            )

                    elif health_check.name == "disk_space":
                        free_gb = health_check.details.get("disk_free_gb", 0)
                        recommendations.append(
                            f"URGENT: Disk space critically low ({free_gb:.1f}GB free) - clean up files immediately"
                        )

                    elif health_check.name == "dependencies":
                        missing = [
                            dep
                            for dep, status in health_check.details.items()
                            if "missing" in str(status)
                        ]
                        recommendations.append(
                            f"Critical dependencies missing: {', '.join(missing)} - reinstall required packages"
                        )

                    elif health_check.name == "bolt_components":
                        summary = health_check.details.get("summary", {})
                        available = summary.get("available", 0)
                        total = summary.get("total", 1)
                        recommendations.append(
                            f"Bolt system components failing ({available}/{total} available) - check installation and imports"
                        )

                elif health_check.status == "warning":
                    if health_check.name == "system_resources":
                        memory_percent = health_check.details.get("memory_percent", 0)
                        cpu_percent = health_check.details.get("cpu_percent", 0)

                        if memory_percent > 85:
                            recommendations.append(
                                f"Memory usage high ({memory_percent:.1f}%) - enable memory conservation mode"
                            )
                        if cpu_percent > 85:
                            recommendations.append(
                                f"CPU usage high ({cpu_percent:.1f}%) - reduce concurrent operations"
                            )

                    elif health_check.name == "gpu_availability":
                        gpu_backends = health_check.details.get(
                            "backends_available", []
                        )
                        if not gpu_backends:
                            recommendations.append(
                                "GPU acceleration not available - install MLX or PyTorch with MPS support for better performance"
                            )
                        else:
                            recommendations.append(
                                f"GPU available but may have issues - check {', '.join(gpu_backends)} configuration"
                            )

                    elif health_check.name == "configuration":
                        issues = health_check.details.get("issues", [])
                        if issues:
                            recommendations.append(
                                f"Configuration issues: {'; '.join(issues[:3])}"
                            )
                        else:
                            recommendations.append(
                                "Configuration issues detected - review environment variables and file permissions"
                            )

                    elif health_check.name == "bolt_components":
                        summary = health_check.details.get("summary", {})
                        healthy = summary.get("healthy", 0)
                        total = summary.get("total", 1)
                        recommendations.append(
                            f"Some Bolt components need attention ({healthy}/{total} fully healthy) - check error logs"
                        )

            # Performance-based recommendations
            if report.performance_metrics:
                cpu_metrics = report.performance_metrics.get("cpu", {})
                memory_metrics = report.performance_metrics.get("memory", {})
                disk_metrics = report.performance_metrics.get("disk", {})

                cpu_percent = cpu_metrics.get("overall_percent", 0)
                memory_percent = memory_metrics.get("percent", 0)
                disk_percent = disk_metrics.get("percent", 0)

                # Prioritize most critical resource issues
                if memory_percent > 90:
                    recommendations.append(
                        "Critical memory pressure - enable emergency memory reclamation"
                    )
                elif memory_percent > 80:
                    recommendations.append(
                        "High memory usage - consider reducing batch sizes and clearing caches"
                    )

                if cpu_percent > 90:
                    recommendations.append(
                        "Critical CPU load - reduce worker count and disable non-essential features"
                    )
                elif cpu_percent > 80:
                    recommendations.append(
                        "High CPU usage - optimize algorithms and reduce concurrent operations"
                    )

                if disk_percent > 95:
                    recommendations.append(
                        "Critical disk usage - clean up logs, temporary files, and old data"
                    )
                elif disk_percent > 85:
                    recommendations.append(
                        "High disk usage - monitor disk space and plan cleanup"
                    )

                # Cross-resource recommendations
                if memory_percent > 80 and cpu_percent > 80:
                    recommendations.append(
                        "Both memory and CPU stressed - consider system restart or resource upgrade"
                    )

            # System-specific recommendations
            if report.system_info:
                if (
                    report.system_info.gpu_available
                    and report.system_info.gpu_backend == "none"
                ):
                    recommendations.append(
                        "GPU hardware detected but not utilized - check GPU acceleration setup"
                    )

                if report.system_info.memory_total_gb < 8:
                    recommendations.append(
                        "Low system memory (< 8GB) - consider RAM upgrade for better performance"
                    )
                elif report.system_info.memory_total_gb < 16:
                    recommendations.append(
                        "Moderate system memory - consider RAM upgrade for optimal performance"
                    )

                if report.system_info.cpu_count < 4:
                    recommendations.append(
                        "Limited CPU cores - optimize for single-threaded performance"
                    )

            # Add error handling recommendations
            error_handling_health = next(
                (hc for hc in report.health_checks if hc.name == "bolt_components"),
                None,
            )

            if error_handling_health:
                if error_handling_health.status in ["warning", "critical"]:
                    recommendations.append(
                        "Error handling system needs attention - this affects system reliability"
                    )

                    # Check specific components
                    components = error_handling_health.details.get("components", {})
                    if "circuit_breaker" in components and "error" in str(
                        components["circuit_breaker"]
                    ):
                        recommendations.append(
                            "Circuit breaker system unavailable - implement basic retry logic"
                        )
                    if "resource_guards" in components and "error" in str(
                        components["resource_guards"]
                    ):
                        recommendations.append(
                            "Resource guards unavailable - monitor system resources manually"
                        )

            # General recommendations if no specific issues found
            if not recommendations:
                recommendations.extend(
                    [
                        "System appears healthy - continue regular monitoring",
                        "Consider implementing automated health checks",
                        "Review logs periodically for early warning signs",
                    ]
                )

            # Add preventive recommendations based on trends
            if len(recommendations) > 3:
                recommendations.append(
                    "Multiple issues detected - consider systematic review and potential system restart"
                )

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append(
                "Unable to generate recommendations due to analysis error - check diagnostic logs"
            )

        return recommendations

    def save_diagnostic_report(
        self, report: DiagnosticReport, file_path: Path | None = None
    ) -> Path:
        """Save diagnostic report to file."""

        if file_path is None:
            timestamp = datetime.fromtimestamp(report.timestamp).strftime(
                "%Y%m%d_%H%M%S"
            )
            file_path = Path(f"bolt_diagnostic_{timestamp}_{report.report_id}.json")

        try:
            # Convert report to dictionary
            report_dict = {
                "report_id": report.report_id,
                "timestamp": report.timestamp,
                "system_info": report.system_info.__dict__
                if report.system_info
                else None,
                "health_checks": [
                    {
                        "name": hc.name,
                        "status": hc.status,
                        "message": hc.message,
                        "details": hc.details,
                        "timestamp": hc.timestamp,
                        "duration": hc.duration,
                    }
                    for hc in report.health_checks
                ],
                "performance_metrics": report.performance_metrics,
                "error_summary": report.error_summary,
                "recommendations": report.recommendations,
                "metadata": report.metadata,
            }

            # Write to file
            with open(file_path, "w") as f:
                json.dump(report_dict, f, indent=2, default=str)

            self.logger.info(f"Diagnostic report saved to {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Failed to save diagnostic report: {e}")
            raise
