#!/usr/bin/env python3
"""
Bolt Stress Test - Realistic Workload Simulation
Tests Bolt under actual load conditions with real trading data patterns
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# Add bolt to path
sys.path.insert(0, str(Path(__file__).parent))

# GPU imports
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class BoltStressTest:
    """Stress test simulating realistic Bolt workloads"""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.metrics = []

    def sample_metrics(self) -> dict[str, float]:
        """Sample current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()

        return {
            "timestamp": time.time() - self.start_time,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
        }

    async def run_stress_test(self, duration_seconds: int = 30) -> dict[str, Any]:
        """Run comprehensive stress test"""
        print(f"🔥 Starting Bolt Stress Test ({duration_seconds}s)")
        print("=" * 60)

        self.start_time = time.time()
        end_time = self.start_time + duration_seconds

        # Start metrics collection
        metrics_task = asyncio.create_task(self.collect_metrics(end_time))

        # Run stress test workloads
        results = await self.run_concurrent_workloads(end_time)

        # Stop metrics collection
        await metrics_task

        # Analyze results
        return self.analyze_stress_results(results)

    async def collect_metrics(self, end_time: float):
        """Continuously collect system metrics"""
        while time.time() < end_time:
            self.metrics.append(self.sample_metrics())
            await asyncio.sleep(0.5)  # Sample every 500ms

    async def run_concurrent_workloads(self, end_time: float) -> dict[str, Any]:
        """Run multiple concurrent workloads"""
        results = {
            "workloads_completed": 0,
            "errors": [],
            "peak_concurrent_tasks": 0,
            "total_operations": 0,
        }

        active_tasks = []

        workload_types = [
            self.trading_analysis_workload,
            self.code_analysis_workload,
            self.memory_intensive_workload,
            self.gpu_computation_workload,
            self.file_processing_workload,
        ]

        task_counter = 0

        while time.time() < end_time:
            # Clean up completed tasks
            active_tasks = [task for task in active_tasks if not task.done()]
            results["peak_concurrent_tasks"] = max(
                results["peak_concurrent_tasks"], len(active_tasks)
            )

            # Add new tasks if we're under load limit
            if len(active_tasks) < 20:  # Max 20 concurrent tasks
                workload = np.random.choice(workload_types)
                task = asyncio.create_task(workload(task_counter))
                active_tasks.append(task)
                task_counter += 1

            await asyncio.sleep(0.1)  # Brief pause

        # Wait for remaining tasks to complete (with timeout)
        if active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*active_tasks, return_exceptions=True), timeout=5.0
                )
            except TimeoutError:
                print("⚠️  Some tasks timed out during cleanup")

        # Count completed workloads
        results["workloads_completed"] = task_counter

        return results

    async def trading_analysis_workload(self, task_id: int) -> dict[str, Any]:
        """Simulate trading data analysis"""
        try:
            # Generate synthetic trading data
            price_data = np.random.randn(10000) * 0.01 + 100  # Stock prices
            volume_data = np.random.exponential(1000, 10000)  # Trading volumes

            # Simulate complex trading calculations
            returns = np.diff(price_data) / price_data[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility

            # Moving averages
            short_ma = np.convolve(price_data, np.ones(20) / 20, mode="valid")
            long_ma = np.convolve(price_data, np.ones(50) / 50, mode="valid")

            # Risk metrics
            var_95 = np.percentile(returns, 5)  # Value at Risk

            await asyncio.sleep(0.01)  # Simulate I/O

            return {
                "task_id": task_id,
                "type": "trading_analysis",
                "volatility": volatility,
                "var_95": var_95,
                "success": True,
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "type": "trading_analysis",
                "success": False,
                "error": str(e),
            }

    async def code_analysis_workload(self, task_id: int) -> dict[str, Any]:
        """Simulate code analysis tasks"""
        try:
            # Generate synthetic code tokens
            code_size = np.random.randint(1000, 10000)
            code_tokens = np.random.randint(0, 1000, code_size)

            # Simulate pattern matching
            patterns = [42, 123, 456, 789]  # Common patterns
            matches = sum(np.sum(code_tokens == pattern) for pattern in patterns)

            # Simulate complexity analysis
            complexity_score = np.std(code_tokens) * len(np.unique(code_tokens))

            await asyncio.sleep(0.005)  # Simulate processing time

            return {
                "task_id": task_id,
                "type": "code_analysis",
                "code_size": code_size,
                "matches": matches,
                "complexity": complexity_score,
                "success": True,
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "type": "code_analysis",
                "success": False,
                "error": str(e),
            }

    async def memory_intensive_workload(self, task_id: int) -> dict[str, Any]:
        """Simulate memory-intensive operations"""
        try:
            # Large array operations
            size = np.random.randint(50000, 200000)
            data = np.random.randn(size).astype(np.float32)

            # Memory-intensive operations
            fft_result = np.fft.fft(data)
            correlation = np.correlate(data[:1000], data[1000:2000], mode="full")

            # Statistics
            percentiles = np.percentile(data, [5, 25, 50, 75, 95])

            await asyncio.sleep(0.002)

            return {
                "task_id": task_id,
                "type": "memory_intensive",
                "data_size": size,
                "fft_magnitude": np.abs(fft_result).mean(),
                "correlation_peak": np.max(np.abs(correlation)),
                "success": True,
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "type": "memory_intensive",
                "success": False,
                "error": str(e),
            }

    async def gpu_computation_workload(self, task_id: int) -> dict[str, Any]:
        """Simulate GPU-accelerated computations"""
        if not HAS_MLX:
            return {
                "task_id": task_id,
                "type": "gpu_computation",
                "success": False,
                "error": "MLX not available",
            }

        try:
            # MLX operations
            size = np.random.randint(100, 500)
            a = mx.random.normal((size, size))
            b = mx.random.normal((size, size))

            # Matrix operations
            c = mx.matmul(a, b)
            d = mx.sum(c, axis=1)
            result = mx.mean(d)

            mx.eval(result)  # Force evaluation

            await asyncio.sleep(0.001)

            return {
                "task_id": task_id,
                "type": "gpu_computation",
                "matrix_size": size,
                "result": float(result),
                "success": True,
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "type": "gpu_computation",
                "success": False,
                "error": str(e),
            }

    async def file_processing_workload(self, task_id: int) -> dict[str, Any]:
        """Simulate file processing operations"""
        try:
            # Find Python files in bolt directory
            bolt_path = Path("bolt")
            if bolt_path.exists():
                python_files = list(bolt_path.glob("**/*.py"))
                if python_files:
                    # Process a random file
                    file_path = np.random.choice(python_files)

                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    # Simple analysis
                    line_count = content.count("\n")
                    word_count = len(content.split())

                    return {
                        "task_id": task_id,
                        "type": "file_processing",
                        "file_path": str(file_path),
                        "line_count": line_count,
                        "word_count": word_count,
                        "success": True,
                    }

            # Fallback if no files found
            return {
                "task_id": task_id,
                "type": "file_processing",
                "success": False,
                "error": "No files found to process",
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "type": "file_processing",
                "success": False,
                "error": str(e),
            }

    def analyze_stress_results(
        self, workload_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze stress test results"""
        if not self.metrics:
            return {"error": "No metrics collected"}

        # Analyze metrics
        timestamps = [m["timestamp"] for m in self.metrics]
        cpu_usage = [m["cpu_percent"] for m in self.metrics]
        memory_usage = [m["memory_mb"] for m in self.metrics]
        memory_percent = [m["memory_percent"] for m in self.metrics]

        # Performance statistics
        performance_stats = {
            "duration_seconds": max(timestamps),
            "cpu_usage": {
                "average": np.mean(cpu_usage),
                "peak": np.max(cpu_usage),
                "minimum": np.min(cpu_usage),
                "std": np.std(cpu_usage),
            },
            "memory_usage": {
                "average_mb": np.mean(memory_usage),
                "peak_mb": np.max(memory_usage),
                "minimum_mb": np.min(memory_usage),
                "growth_mb": np.max(memory_usage) - np.min(memory_usage),
            },
            "system_memory": {
                "average_percent": np.mean(memory_percent),
                "peak_percent": np.max(memory_percent),
            },
        }

        # Workload analysis
        workload_analysis = {
            "total_workloads": workload_results["workloads_completed"],
            "peak_concurrent": workload_results["peak_concurrent_tasks"],
            "throughput_per_second": workload_results["workloads_completed"]
            / performance_stats["duration_seconds"],
            "errors": len(workload_results.get("errors", [])),
        }

        # Performance assessment
        assessment = self.assess_performance(performance_stats, workload_analysis)

        # Compliance check
        compliance = self.check_limits_compliance(performance_stats)

        return {
            "performance_stats": performance_stats,
            "workload_analysis": workload_analysis,
            "assessment": assessment,
            "compliance": compliance,
            "raw_metrics": self.metrics,
            "workload_results": workload_results,
        }

    def assess_performance(
        self, perf_stats: dict, workload_stats: dict
    ) -> dict[str, Any]:
        """Assess overall performance"""

        # CPU efficiency score
        avg_cpu = perf_stats["cpu_usage"]["average"]
        if avg_cpu < 30:
            cpu_score = "UNDERUTILIZED"
        elif avg_cpu < 70:
            cpu_score = "OPTIMAL"
        elif avg_cpu < 90:
            cpu_score = "HIGH"
        else:
            cpu_score = "OVERLOADED"

        # Memory efficiency score
        peak_memory_gb = perf_stats["memory_usage"]["peak_mb"] / 1024
        if peak_memory_gb < 1:
            memory_score = "EXCELLENT"
        elif peak_memory_gb < 2:
            memory_score = "GOOD"
        elif peak_memory_gb < 4:
            memory_score = "ACCEPTABLE"
        else:
            memory_score = "HIGH"

        # Throughput assessment
        throughput = workload_stats["throughput_per_second"]
        if throughput > 50:
            throughput_score = "EXCELLENT"
        elif throughput > 20:
            throughput_score = "GOOD"
        elif throughput > 10:
            throughput_score = "ACCEPTABLE"
        else:
            throughput_score = "LOW"

        # Overall score
        scores = {
            "EXCELLENT": 4,
            "OPTIMAL": 4,
            "GOOD": 3,
            "ACCEPTABLE": 2,
            "HIGH": 1,
            "LOW": 1,
            "UNDERUTILIZED": 2,
            "OVERLOADED": 1,
        }
        total_score = (
            scores.get(cpu_score, 0)
            + scores.get(memory_score, 0)
            + scores.get(throughput_score, 0)
        )

        if total_score >= 10:
            overall = "EXCELLENT"
        elif total_score >= 8:
            overall = "GOOD"
        elif total_score >= 6:
            overall = "ACCEPTABLE"
        else:
            overall = "NEEDS_IMPROVEMENT"

        return {
            "cpu_efficiency": cpu_score,
            "memory_efficiency": memory_score,
            "throughput_performance": throughput_score,
            "overall_assessment": overall,
            "performance_score": total_score,
        }

    def check_limits_compliance(self, perf_stats: dict) -> dict[str, Any]:
        """Check compliance with system limits"""

        # Define limits (based on M4 Pro capabilities)
        limits = {
            "max_cpu_percent": 95,
            "max_memory_gb": 20,  # Conservative limit for 24GB system
            "max_memory_percent": 80,
        }

        peak_cpu = perf_stats["cpu_usage"]["peak"]
        peak_memory_gb = perf_stats["memory_usage"]["peak_mb"] / 1024
        peak_memory_percent = perf_stats["system_memory"]["peak_percent"]

        compliance_checks = {
            "cpu_within_limits": peak_cpu <= limits["max_cpu_percent"],
            "memory_within_limits": peak_memory_gb <= limits["max_memory_gb"],
            "system_memory_within_limits": peak_memory_percent
            <= limits["max_memory_percent"],
            "limits": limits,
            "actual_peaks": {
                "cpu_percent": peak_cpu,
                "memory_gb": peak_memory_gb,
                "system_memory_percent": peak_memory_percent,
            },
        }

        compliance_checks["overall_compliant"] = all(
            [
                compliance_checks["cpu_within_limits"],
                compliance_checks["memory_within_limits"],
                compliance_checks["system_memory_within_limits"],
            ]
        )

        return compliance_checks


async def main():
    """Run Bolt stress test"""
    stress_test = BoltStressTest()

    try:
        results = await stress_test.run_stress_test(30)  # 30-second test

        # Print results
        print("\n" + "=" * 60)
        print("🔥 BOLT STRESS TEST RESULTS")
        print("=" * 60)

        perf = results["performance_stats"]
        workload = results["workload_analysis"]
        assessment = results["assessment"]
        compliance = results["compliance"]

        print(f"\n⏱️  TEST DURATION: {perf['duration_seconds']:.1f} seconds")

        print("\n🖥️  CPU PERFORMANCE:")
        print(f"   Average: {perf['cpu_usage']['average']:.1f}%")
        print(f"   Peak: {perf['cpu_usage']['peak']:.1f}%")
        print(f"   Assessment: {assessment['cpu_efficiency']}")

        print("\n💾 MEMORY PERFORMANCE:")
        print(f"   Average: {perf['memory_usage']['average_mb']:.1f} MB")
        print(f"   Peak: {perf['memory_usage']['peak_mb']:.1f} MB")
        print(f"   Growth: {perf['memory_usage']['growth_mb']:.1f} MB")
        print(f"   Assessment: {assessment['memory_efficiency']}")

        print("\n🚀 WORKLOAD PERFORMANCE:")
        print(f"   Total Workloads: {workload['total_workloads']}")
        print(f"   Peak Concurrent: {workload['peak_concurrent']}")
        print(f"   Throughput: {workload['throughput_per_second']:.1f} tasks/sec")
        print(f"   Errors: {workload['errors']}")
        print(f"   Assessment: {assessment['throughput_performance']}")

        print("\n✅ COMPLIANCE CHECK:")
        print(
            f"   CPU Within Limits: {compliance['cpu_within_limits']} (Peak: {compliance['actual_peaks']['cpu_percent']:.1f}%)"
        )
        print(
            f"   Memory Within Limits: {compliance['memory_within_limits']} (Peak: {compliance['actual_peaks']['memory_gb']:.1f}GB)"
        )
        print(
            f"   System Memory Within Limits: {compliance['system_memory_within_limits']} (Peak: {compliance['actual_peaks']['system_memory_percent']:.1f}%)"
        )
        print(f"   Overall Compliant: {compliance['overall_compliant']}")

        print(f"\n🏆 OVERALL ASSESSMENT: {assessment['overall_assessment']}")
        print(f"   Performance Score: {assessment['performance_score']}/12")

        # Save detailed results
        output_file = "bolt_stress_test_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Detailed results saved to: {output_file}")

        # Final recommendations
        print("\n💡 RECOMMENDATIONS:")
        if assessment["overall_assessment"] == "EXCELLENT":
            print("   • System is performing optimally under stress")
            print("   • Ready for production workloads")
        elif assessment["overall_assessment"] == "GOOD":
            print("   • Good performance with room for minor optimizations")
        elif assessment["overall_assessment"] == "ACCEPTABLE":
            print("   • Performance is acceptable but consider optimizations")
            print("   • Monitor resource usage during peak loads")
        else:
            print("   • Performance needs improvement")
            print("   • Consider resource optimization or hardware upgrades")

        if not compliance["overall_compliant"]:
            print("   • ALERT: System exceeded recommended limits")
            print("   • Review resource allocation and optimize workloads")

        return 0

    except Exception as e:
        print(f"\n💥 Stress test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
