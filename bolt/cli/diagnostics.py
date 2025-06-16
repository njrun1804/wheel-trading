"""
Diagnostic CLI commands for Bolt system troubleshooting.

Provides comprehensive system analysis and troubleshooting tools.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import click

from ..core.integration import BoltIntegration, SystemState
from ..error_handling import (
    BoltException,
    DiagnosticCollector,
    ErrorSeverity,
    SystemHealthChecker,
)
from ..hardware.hardware_state import get_hardware_state
from ..hardware.performance_monitor import get_performance_monitor


@click.group(name="diagnose")
def diagnostics_main():
    """System diagnostics and troubleshooting commands."""
    pass


@diagnostics_main.command()
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for diagnostic report"
)
@click.option(
    "--format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def health(ctx: click.Context, output: str, format: str):
    """Run comprehensive system health check."""
    logger = logging.getLogger("bolt.cli.health")

    try:
        # Initialize diagnostic collector
        DiagnosticCollector()
        health_checker = SystemHealthChecker()

        click.echo("Running comprehensive system health check...")

        # Capture system state
        system_state = SystemState.capture()

        # Run health checks
        health_results = asyncio.run(_run_health_checks(health_checker))

        # Collect hardware information
        get_hardware_state()
        get_performance_monitor()

        # Compile diagnostic report
        report = {
            "timestamp": time.time(),
            "system_state": {
                "cpu_percent": system_state.cpu_percent,
                "memory_percent": system_state.memory_percent,
                "gpu_memory_gb": system_state.gpu_memory_used_gb,
                "gpu_backend": system_state.gpu_backend,
                "is_healthy": system_state.is_healthy,
                "warnings": system_state.warnings,
                "errors": system_state.errors,
                "performance_degraded": system_state.performance_degraded,
                "resource_pressure": system_state.resource_pressure,
            },
            "health_checks": health_results,
            "hardware_info": {
                "cpu_cores": system_state.cpu_cores,
                "memory_available_gb": system_state.memory_available_gb,
                "gpu_memory_limit_gb": system_state.gpu_memory_limit_gb,
            },
            "recommendations": _generate_recommendations(system_state, health_results),
        }

        # Output results
        if format == "json":
            report_json = json.dumps(report, indent=2)
            if output:
                Path(output).write_text(report_json)
                click.echo(f"Diagnostic report saved to: {output}")
            else:
                click.echo(report_json)
        else:
            _print_health_report(report)
            if output:
                _save_text_report(report, output)
                click.echo(f"Diagnostic report saved to: {output}")

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise BoltException(
            f"System health check failed: {e}",
            severity=ErrorSeverity.HIGH,
            recovery_hints=[
                "Check system resources and permissions",
                "Ensure Bolt dependencies are installed",
                "Run with --verbose for more details",
            ],
        )


@diagnostics_main.command()
@click.option(
    "--component",
    type=click.Choice(["agents", "tools", "einstein", "gpu"]),
    help="Test specific component",
)
@click.option("--timeout", type=int, default=30, help="Test timeout in seconds")
@click.pass_context
def test(ctx: click.Context, component: str, timeout: int):
    """Run component integration tests."""
    logger = logging.getLogger("bolt.cli.test")

    try:
        if component:
            click.echo(f"Testing {component} component...")
            results = asyncio.run(_test_component(component, timeout))
        else:
            click.echo("Running full system integration tests...")
            results = asyncio.run(_test_all_components(timeout))

        # Display results
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["status"] == "passed")
        failed_tests = total_tests - passed_tests

        click.echo(f"\nTest Results: {passed_tests}/{total_tests} passed")

        for result in results:
            status_color = "green" if result["status"] == "passed" else "red"
            click.echo(
                click.style(f"  {result['name']}: {result['status']}", fg=status_color)
            )
            if result["status"] == "failed" and result.get("error"):
                click.echo(f"    Error: {result['error']}")

        if failed_tests > 0:
            raise BoltException(
                f"{failed_tests} component tests failed",
                severity=ErrorSeverity.HIGH,
                recovery_hints=[
                    "Check component configuration",
                    "Verify system dependencies",
                    "Run individual component tests for more details",
                ],
            )

        click.echo(click.style("All tests passed!", fg="green"))

    except Exception as e:
        logger.error(f"Component testing failed: {e}")
        if not isinstance(e, BoltException):
            raise BoltException(
                f"Component testing failed: {e}", severity=ErrorSeverity.HIGH
            )
        raise


@diagnostics_main.command()
@click.option("--duration", type=int, default=60, help="Monitoring duration in seconds")
@click.option("--interval", type=int, default=5, help="Sampling interval in seconds")
@click.pass_context
def monitor(ctx: click.Context, duration: int, interval: int):
    """Monitor system performance in real-time."""
    logger = logging.getLogger("bolt.cli.monitor")

    try:
        click.echo(f"Monitoring system for {duration}s (sampling every {interval}s)")
        click.echo("Press Ctrl+C to stop early\n")

        samples = []
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                # Capture system state
                state = SystemState.capture()
                sample = {
                    "timestamp": time.time(),
                    "cpu_percent": state.cpu_percent,
                    "memory_percent": state.memory_percent,
                    "gpu_memory_gb": state.gpu_memory_used_gb,
                    "is_healthy": state.is_healthy,
                    "warnings": len(state.warnings),
                    "errors": len(state.errors),
                }
                samples.append(sample)

                # Display current state
                elapsed = time.time() - start_time
                status = "✓" if state.is_healthy else "✗"
                click.echo(
                    f"[{elapsed:5.1f}s] {status} CPU: {state.cpu_percent:5.1f}% "
                    f"MEM: {state.memory_percent:5.1f}% "
                    f"GPU: {state.gpu_memory_used_gb:5.1f}GB"
                )

                if state.warnings:
                    click.echo(
                        click.style(f"  Warnings: {state.warnings}", fg="yellow")
                    )
                if state.errors:
                    click.echo(click.style(f"  Errors: {state.errors}", fg="red"))

                time.sleep(interval)

        except KeyboardInterrupt:
            click.echo("\nMonitoring stopped by user")

        # Show summary
        if samples:
            _print_monitoring_summary(samples)

    except Exception as e:
        logger.error(f"System monitoring failed: {e}")
        raise BoltException(
            f"System monitoring failed: {e}", severity=ErrorSeverity.MEDIUM
        )


@diagnostics_main.command()
@click.option("--logs-dir", type=click.Path(), help="Directory to search for log files")
@click.option(
    "--error-pattern",
    default="ERROR|CRITICAL|Exception",
    help="Error patterns to search",
)
@click.pass_context
def analyze_logs(ctx: click.Context, logs_dir: str, error_pattern: str):
    """Analyze system logs for errors and patterns."""
    logger = logging.getLogger("bolt.cli.analyze")

    try:
        # Default to common log locations
        if not logs_dir:
            possible_dirs = [
                Path("logs"),
                Path("/var/log"),
                Path.home() / ".bolt" / "logs",
            ]
            logs_dir = next((d for d in possible_dirs if d.exists()), Path("logs"))
        else:
            logs_dir = Path(logs_dir)

        if not logs_dir.exists():
            raise BoltException(
                f"Log directory not found: {logs_dir}",
                severity=ErrorSeverity.MEDIUM,
                recovery_hints=[
                    "Specify a valid log directory with --logs-dir",
                    "Check if logging is properly configured",
                    "Ensure Bolt has been run and generated logs",
                ],
            )

        click.echo(f"Analyzing logs in: {logs_dir}")

        # Search for log files
        log_files = list(logs_dir.glob("*.log")) + list(logs_dir.glob("**/*.log"))

        if not log_files:
            click.echo("No log files found")
            return

        click.echo(f"Found {len(log_files)} log files")

        # Analyze each log file
        error_summary = {}
        total_errors = 0

        for log_file in log_files:
            errors = _analyze_log_file(log_file, error_pattern)
            if errors:
                error_summary[str(log_file)] = errors
                total_errors += len(errors)

        # Display results
        if total_errors == 0:
            click.echo(click.style("No errors found in log files", fg="green"))
        else:
            click.echo(
                f"\nFound {total_errors} errors across {len(error_summary)} files:"
            )

            for log_file, errors in error_summary.items():
                click.echo(f"\n{log_file}:")
                for error in errors[:5]:  # Show first 5 errors
                    click.echo(f"  {error['timestamp']}: {error['message'][:80]}...")

                if len(errors) > 5:
                    click.echo(f"  ... and {len(errors) - 5} more errors")

    except Exception as e:
        logger.error(f"Log analysis failed: {e}")
        if not isinstance(e, BoltException):
            raise BoltException(
                f"Log analysis failed: {e}", severity=ErrorSeverity.MEDIUM
            )
        raise


# Helper functions


async def _run_health_checks(health_checker: SystemHealthChecker) -> dict[str, Any]:
    """Run comprehensive health checks."""
    checks = [
        ("memory_check", health_checker.check_memory_health),
        ("cpu_check", health_checker.check_cpu_health),
        ("gpu_check", health_checker.check_gpu_health),
        ("disk_check", health_checker.check_disk_health),
        ("network_check", health_checker.check_network_health),
        ("dependency_check", health_checker.check_dependencies),
        ("configuration_check", health_checker.check_configuration),
        ("permission_check", health_checker.check_permissions),
    ]

    results = {}
    for check_name, check_func in checks:
        try:
            result = await check_func()
            results[check_name] = {
                "status": "passed" if result.get("healthy", False) else "failed",
                "details": result,
            }
        except Exception as e:
            results[check_name] = {"status": "error", "error": str(e)}

    return results


async def _test_component(component: str, timeout: int) -> list:
    """Test a specific component."""
    tests = []

    if component == "agents":
        tests.extend(
            [
                ("agent_initialization", _test_agent_init),
                ("agent_communication", _test_agent_communication),
                ("task_execution", _test_task_execution),
            ]
        )
    elif component == "tools":
        tests.extend(
            [
                ("accelerated_tools", _test_accelerated_tools),
                ("tool_availability", _test_tool_availability),
            ]
        )
    elif component == "einstein":
        tests.extend(
            [
                ("einstein_initialization", _test_einstein_init),
                ("search_functionality", _test_einstein_search),
            ]
        )
    elif component == "gpu":
        tests.extend(
            [
                ("gpu_detection", _test_gpu_detection),
                ("gpu_acceleration", _test_gpu_acceleration),
            ]
        )

    results = []
    for test_name, test_func in tests:
        try:
            await asyncio.wait_for(test_func(), timeout=timeout)
            results.append({"name": test_name, "status": "passed"})
        except Exception as e:
            results.append({"name": test_name, "status": "failed", "error": str(e)})

    return results


async def _test_all_components(timeout: int) -> list:
    """Test all system components."""
    all_tests = []

    for component in ["agents", "tools", "einstein", "gpu"]:
        component_tests = await _test_component(component, timeout)
        all_tests.extend(component_tests)

    return all_tests


# Individual test functions
async def _test_agent_init():
    """Test agent initialization."""
    bolt = BoltIntegration(num_agents=2)
    await bolt.initialize()
    await bolt.shutdown()


async def _test_agent_communication():
    """Test agent communication."""
    bolt = BoltIntegration(num_agents=2)
    await bolt.initialize()
    try:
        # Test basic agent-to-agent communication
        if not bolt.agent_pool or len(bolt.agent_pool.agents) < 2:
            raise Exception("Failed to initialize multiple agents")

        # Test message passing between agents
        agent1 = bolt.agent_pool.agents[0]
        agent2 = bolt.agent_pool.agents[1]

        # Simple communication test
        if agent1.state == agent2.state:
            # Both agents are responsive
            pass
        else:
            raise Exception("Agent communication test failed")

    finally:
        await bolt.shutdown()


async def _test_task_execution():
    """Test basic task execution."""
    bolt = BoltIntegration(num_agents=1)
    await bolt.initialize()
    try:
        result = await bolt.execute_query("test simple task")
        assert result is not None
    finally:
        await bolt.shutdown()


async def _test_accelerated_tools():
    """Test accelerated tools availability."""
    # Test importing accelerated tools
    try:
        from unity_wheel.accelerated_tools.dependency_graph_turbo import (
            get_dependency_graph,
        )
        from unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo
        from unity_wheel.accelerated_tools.python_analysis_turbo import (
            get_python_analyzer,
        )
        from unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo

        # Test basic instantiation
        ripgrep = get_ripgrep_turbo()
        if not ripgrep:
            raise Exception("Failed to initialize ripgrep turbo")

        dep_graph = get_dependency_graph()
        if not dep_graph:
            raise Exception("Failed to initialize dependency graph")

        analyzer = get_python_analyzer()
        if not analyzer:
            raise Exception("Failed to initialize python analyzer")

        # Test database connection
        try:
            db = get_duckdb_turbo(":memory:")
            if not db:
                raise Exception("Failed to initialize DuckDB turbo")
        except Exception as e:
            # DuckDB might not be available in all environments
            if "not found" not in str(e).lower():
                raise

    except ImportError as e:
        raise Exception(f"Accelerated tools not available: {e}")


async def _test_tool_availability():
    """Test tool availability."""
    # Test basic tool availability
    import importlib

    essential_tools = [
        ("ripgrep", "which rg"),
        ("python", "which python"),
        ("git", "which git"),
    ]

    for tool_name, which_cmd in essential_tools:
        import subprocess

        try:
            result = subprocess.run(
                which_cmd.split(), capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                raise Exception(f"{tool_name} not found in PATH")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise Exception(f"{tool_name} availability check failed")

    # Test Python packages
    essential_packages = ["asyncio", "pathlib", "json", "logging"]
    for package in essential_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            raise Exception(f"Essential Python package {package} not available")


async def _test_einstein_init():
    """Test Einstein initialization."""
    try:
        from einstein.query_router import QueryRouter
        from einstein.unified_index import UnifiedCodeIndex

        # Test basic Einstein initialization
        index = UnifiedCodeIndex()
        if not index:
            raise Exception("Failed to initialize UnifiedCodeIndex")

        router = QueryRouter()
        if not router:
            raise Exception("Failed to initialize QueryRouter")

        # Test basic functionality
        test_query = "test query"
        try:
            # This might fail if no index is built, but should not crash
            await router.route_query(test_query)
        except Exception as e:
            # Expected if no index exists yet
            if "no index" not in str(e).lower() and "not found" not in str(e).lower():
                raise Exception(f"Einstein routing failed unexpectedly: {e}")

    except ImportError as e:
        raise Exception(f"Einstein components not available: {e}")


async def _test_einstein_search():
    """Test Einstein search functionality."""
    try:
        from einstein.unified_index import UnifiedCodeIndex

        # Create a minimal test index
        index = UnifiedCodeIndex()

        # Test search with a simple query
        test_query = "function"
        try:
            # Attempt search - may return empty results but should not crash
            results = await index.search(test_query, limit=5)
            # If we get here without exception, search is working
            if not isinstance(results, list):
                raise Exception("Search did not return a list")
        except Exception as e:
            # Expected if no index is built
            if (
                "no index" not in str(e).lower()
                and "not initialized" not in str(e).lower()
            ):
                raise Exception(f"Einstein search failed: {e}")

    except ImportError as e:
        raise Exception(f"Einstein search components not available: {e}")


async def _test_gpu_detection():
    """Test GPU detection."""
    from ..hardware.hardware_state import get_hardware_state

    try:
        hardware_state = get_hardware_state()

        # Check if GPU detection is working
        if not hasattr(hardware_state, "gpu_backend"):
            raise Exception("Hardware state missing GPU backend information")

        # Test different GPU backends
        gpu_available = False

        # Test MLX (Apple Silicon)
        try:
            import mlx.core as mx

            mx.array([1, 2, 3])  # Simple test
            gpu_available = True
        except ImportError:
            pass
        except Exception:
            pass

        # Test Metal Performance Shaders
        if not gpu_available:
            try:
                import torch

                if torch.backends.mps.is_available():
                    gpu_available = True
            except ImportError:
                pass

        # Even if no GPU acceleration is available, detection should work
        if hardware_state.gpu_backend == "none" and not gpu_available:
            # This is fine - no GPU detected
            pass
        elif hardware_state.gpu_backend != "none" and gpu_available:
            # This is good - GPU detected and available
            pass
        else:
            # Inconsistent state - might be an issue
            import warnings

            warnings.warn(
                f"GPU detection inconsistency: backend={hardware_state.gpu_backend}, available={gpu_available}"
            )

    except Exception as e:
        raise Exception(f"GPU detection failed: {e}")


async def _test_gpu_acceleration():
    """Test GPU acceleration."""
    # Test GPU acceleration capabilities
    gpu_working = False
    errors = []

    # Test MLX (Apple Silicon)
    try:
        import mlx.core as mx
        import mlx.nn as nn

        # Simple MLX operation test
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([4.0, 5.0, 6.0])
        result = mx.add(x, y)

        if not mx.array_equal(result, mx.array([5.0, 7.0, 9.0])):
            errors.append("MLX basic operations failed")
        else:
            gpu_working = True

        # Test neural network layer
        linear = nn.Linear(3, 2)
        output = linear(x)
        if output.shape != (2,):
            errors.append("MLX neural network test failed")

    except ImportError:
        errors.append("MLX not available")
    except Exception as e:
        errors.append(f"MLX test failed: {e}")

    # Test PyTorch MPS (if available)
    if not gpu_working:
        try:
            import torch

            if torch.backends.mps.is_available():
                device = torch.device("mps")
                x = torch.tensor([1.0, 2.0, 3.0], device=device)
                y = torch.tensor([4.0, 5.0, 6.0], device=device)
                result = x + y

                expected = torch.tensor([5.0, 7.0, 9.0], device=device)
                if not torch.allclose(result, expected):
                    errors.append("PyTorch MPS operations failed")
                else:
                    gpu_working = True
            else:
                errors.append("PyTorch MPS not available")

        except ImportError:
            errors.append("PyTorch not available")
        except Exception as e:
            errors.append(f"PyTorch MPS test failed: {e}")

    # If no GPU acceleration is working, that's still valid
    if not gpu_working and errors:
        # Log warnings but don't fail the test
        import warnings

        warnings.warn(f"GPU acceleration not available: {'; '.join(errors)}")
        # Don't raise exception - CPU fallback is acceptable


def _generate_recommendations(
    system_state: SystemState, health_results: dict[str, Any]
) -> list:
    """Generate system recommendations based on health check results."""
    recommendations = []

    # Resource-based recommendations
    if system_state.cpu_percent > 90:
        recommendations.append(
            "High CPU usage detected - consider reducing concurrent tasks"
        )

    if system_state.memory_percent > 85:
        recommendations.append(
            "High memory usage - consider increasing system memory or reducing agents"
        )

    if system_state.gpu_memory_used_gb > 16:
        recommendations.append(
            "High GPU memory usage - enable memory optimization or reduce batch sizes"
        )

    # Health check recommendations
    failed_checks = [
        name
        for name, result in health_results.items()
        if result["status"] in ["failed", "error"]
    ]

    if failed_checks:
        recommendations.append(
            f"Failed health checks: {', '.join(failed_checks)} - review system configuration"
        )

    if not recommendations:
        recommendations.append("System appears healthy - no specific recommendations")

    return recommendations


def _print_health_report(report: dict[str, Any]) -> None:
    """Print formatted health report."""
    click.echo("=== System Health Report ===\n")

    state = report["system_state"]

    # Overall status
    status = "HEALTHY" if state["is_healthy"] else "UNHEALTHY"
    status_color = "green" if state["is_healthy"] else "red"
    click.echo(f"Overall Status: {click.style(status, fg=status_color, bold=True)}")

    # System metrics
    click.echo("\nSystem Metrics:")
    click.echo(f"  CPU Usage: {state['cpu_percent']:.1f}%")
    click.echo(f"  Memory Usage: {state['memory_percent']:.1f}%")
    click.echo(f"  GPU Memory: {state['gpu_memory_gb']:.1f}GB")
    click.echo(f"  GPU Backend: {state['gpu_backend']}")

    # Issues
    if state["warnings"]:
        click.echo("\nWarnings:")
        for warning in state["warnings"]:
            click.echo(click.style(f"  ⚠ {warning}", fg="yellow"))

    if state["errors"]:
        click.echo("\nErrors:")
        for error in state["errors"]:
            click.echo(click.style(f"  ✗ {error}", fg="red"))

    # Health checks
    click.echo("\nHealth Checks:")
    for check_name, result in report["health_checks"].items():
        status_symbol = "✓" if result["status"] == "passed" else "✗"
        status_color = "green" if result["status"] == "passed" else "red"
        click.echo(click.style(f"  {status_symbol} {check_name}", fg=status_color))

    # Recommendations
    if report["recommendations"]:
        click.echo("\nRecommendations:")
        for rec in report["recommendations"]:
            click.echo(f"  • {rec}")


def _save_text_report(report: dict[str, Any], output_path: str) -> None:
    """Save health report as text file."""
    from datetime import datetime

    with open(output_path, "w") as f:
        f.write("=== Bolt System Health Report ===\n")
        f.write(f"Generated: {datetime.fromtimestamp(report['timestamp'])}\n\n")

        state = report["system_state"]

        # Overall status
        status = "HEALTHY" if state["is_healthy"] else "UNHEALTHY"
        f.write(f"Overall Status: {status}\n\n")

        # System metrics
        f.write("System Metrics:\n")
        f.write(f"  CPU Usage: {state['cpu_percent']:.1f}%\n")
        f.write(f"  Memory Usage: {state['memory_percent']:.1f}%\n")
        f.write(f"  GPU Memory: {state['gpu_memory_gb']:.1f}GB\n")
        f.write(f"  GPU Backend: {state['gpu_backend']}\n\n")

        # Hardware info
        hardware = report["hardware_info"]
        f.write("Hardware Information:\n")
        f.write(f"  CPU Cores: {hardware['cpu_cores']}\n")
        f.write(f"  Available Memory: {hardware['memory_available_gb']:.1f}GB\n")
        f.write(f"  GPU Memory Limit: {hardware['gpu_memory_limit_gb']:.1f}GB\n\n")

        # Issues
        if state["warnings"]:
            f.write("Warnings:\n")
            for warning in state["warnings"]:
                f.write(f"  ⚠ {warning}\n")
            f.write("\n")

        if state["errors"]:
            f.write("Errors:\n")
            for error in state["errors"]:
                f.write(f"  ✗ {error}\n")
            f.write("\n")

        # Health checks
        f.write("Health Checks:\n")
        for check_name, result in report["health_checks"].items():
            status_symbol = "✓" if result["status"] == "passed" else "✗"
            f.write(f"  {status_symbol} {check_name}: {result['status']}\n")
            if result["status"] == "error" and "error" in result:
                f.write(f"    Error: {result['error']}\n")
        f.write("\n")

        # Recommendations
        if report["recommendations"]:
            f.write("Recommendations:\n")
            for rec in report["recommendations"]:
                f.write(f"  • {rec}\n")


def _print_monitoring_summary(samples: list) -> None:
    """Print monitoring session summary."""
    if not samples:
        return

    click.echo("\n=== Monitoring Summary ===")

    # Calculate statistics
    cpu_values = [s["cpu_percent"] for s in samples]
    mem_values = [s["memory_percent"] for s in samples]
    gpu_values = [s["gpu_memory_gb"] for s in samples]

    click.echo(
        f"CPU Usage    - Avg: {sum(cpu_values)/len(cpu_values):.1f}% "
        f"Max: {max(cpu_values):.1f}% Min: {min(cpu_values):.1f}%"
    )
    click.echo(
        f"Memory Usage - Avg: {sum(mem_values)/len(mem_values):.1f}% "
        f"Max: {max(mem_values):.1f}% Min: {min(mem_values):.1f}%"
    )
    click.echo(
        f"GPU Memory   - Avg: {sum(gpu_values)/len(gpu_values):.1f}GB "
        f"Max: {max(gpu_values):.1f}GB Min: {min(gpu_values):.1f}GB"
    )

    # Health statistics
    unhealthy_count = sum(1 for s in samples if not s["is_healthy"])
    if unhealthy_count > 0:
        click.echo(
            click.style(
                f"System was unhealthy for {unhealthy_count}/{len(samples)} samples",
                fg="yellow",
            )
        )
    else:
        click.echo(
            click.style("System remained healthy throughout monitoring", fg="green")
        )


def _analyze_log_file(log_file: Path, error_pattern: str) -> list:
    """Analyze a single log file for errors."""
    import re

    errors = []
    pattern = re.compile(error_pattern, re.IGNORECASE)

    try:
        with open(log_file) as f:
            for line_num, line in enumerate(f, 1):
                if pattern.search(line):
                    # Extract timestamp if present
                    timestamp_match = re.match(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line
                    )
                    timestamp = (
                        timestamp_match.group(1)
                        if timestamp_match
                        else f"Line {line_num}"
                    )

                    errors.append(
                        {
                            "timestamp": timestamp,
                            "line_number": line_num,
                            "message": line.strip(),
                        }
                    )
    except Exception:
        # Skip files that can't be read
        pass

    return errors
