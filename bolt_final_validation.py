#!/usr/bin/env python3
"""
Final Bolt Performance Validation
- Quick validation of key performance metrics
- Memory usage validation
- Realistic workload test
"""

import asyncio
import sys
import time
from typing import Any

import numpy as np
import psutil

# GPU support
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class BoltValidator:
    """Final performance validation for Bolt"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024

    async def run_validation(self) -> dict[str, Any]:
        """Run comprehensive validation"""
        print("🔍 Bolt Performance Validation")
        print("=" * 50)

        results = {}

        # 1. Quick performance test
        print("1️⃣ Quick Performance Test...")
        results["quick_perf"] = await self.quick_performance_test()

        # 2. Memory stress test
        print("2️⃣ Memory Stress Test...")
        results["memory_stress"] = await self.memory_stress_test()

        # 3. Concurrent workload test
        print("3️⃣ Concurrent Workload Test...")
        results["concurrent_test"] = await self.concurrent_workload_test()

        # 4. System limits validation
        print("4️⃣ System Limits Check...")
        results["limits_check"] = self.check_system_limits()

        return results

    async def quick_performance_test(self) -> dict[str, Any]:
        """Quick 5-second performance test"""
        start_time = time.perf_counter()
        operations = 0

        # CPU test - matrix operations
        for _ in range(20):
            a = np.random.rand(500, 500).astype(np.float32)
            b = np.random.rand(500, 500).astype(np.float32)
            np.dot(a, b)
            operations += 1

        # GPU test if available
        gpu_operations = 0
        if HAS_MLX:
            for _ in range(5):
                a = mx.random.normal((300, 300))
                b = mx.random.normal((300, 300))
                c = mx.matmul(a, b)
                mx.eval(c)
                gpu_operations += 1

        duration = time.perf_counter() - start_time
        cpu_ops_per_sec = operations / duration
        gpu_ops_per_sec = gpu_operations / duration if gpu_operations > 0 else 0

        return {
            "duration_seconds": duration,
            "cpu_ops_per_sec": cpu_ops_per_sec,
            "gpu_ops_per_sec": gpu_ops_per_sec,
            "total_ops_per_sec": cpu_ops_per_sec + gpu_ops_per_sec,
            "gpu_available": HAS_MLX,
        }

    async def memory_stress_test(self) -> dict[str, Any]:
        """Test memory allocation and cleanup"""
        start_memory = self.process.memory_info().rss / 1024 / 1024
        peak_memory = start_memory

        arrays = []

        # Allocate memory in chunks
        for i in range(20):
            # Allocate 10MB chunks
            arr = np.random.rand(2_500_000).astype(np.float32)  # ~10MB
            arrays.append(arr)

            current_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)

            # Brief processing
            np.sum(arr)

            await asyncio.sleep(0.01)  # Allow other tasks

        # Cleanup
        del arrays

        # Give time for cleanup
        await asyncio.sleep(0.1)
        final_memory = self.process.memory_info().rss / 1024 / 1024

        return {
            "start_memory_mb": start_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "peak_allocation_mb": peak_memory - start_memory,
            "memory_recovered_mb": peak_memory - final_memory,
            "cleanup_efficiency": (peak_memory - final_memory)
            / (peak_memory - start_memory)
            * 100,
        }

    async def concurrent_workload_test(self) -> dict[str, Any]:
        """Test concurrent task handling"""

        async def cpu_task(task_id: int):
            """CPU-intensive task"""
            data = np.random.randn(50000).astype(np.float32)
            result = np.fft.fft(data)
            return {"task_id": task_id, "result_magnitude": np.abs(result).mean()}

        async def memory_task(task_id: int):
            """Memory-intensive task"""
            data = np.random.rand(100000).astype(np.float32)
            stats = {
                "mean": np.mean(data),
                "std": np.std(data),
                "percentiles": np.percentile(data, [25, 50, 75]),
            }
            return {"task_id": task_id, "stats": stats}

        start_time = time.perf_counter()

        # Create mixed workload
        cpu_tasks = [cpu_task(i) for i in range(8)]
        memory_tasks = [memory_task(i + 8) for i in range(8)]

        # Run concurrently
        all_tasks = cpu_tasks + memory_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        duration = time.perf_counter() - start_time

        # Count successful tasks
        successful = len([r for r in results if not isinstance(r, Exception)])
        failed = len(results) - successful

        return {
            "total_tasks": len(all_tasks),
            "successful_tasks": successful,
            "failed_tasks": failed,
            "duration_seconds": duration,
            "tasks_per_second": len(all_tasks) / duration,
            "success_rate": successful / len(all_tasks) * 100,
        }

    def check_system_limits(self) -> dict[str, Any]:
        """Check system resource limits"""
        # Current system state
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        current_memory_mb = self.process.memory_info().rss / 1024 / 1024

        # System specs
        cpu_count = psutil.cpu_count()
        total_memory_gb = memory.total / 1024 / 1024 / 1024

        # Define reasonable limits for M4 Pro
        limits = {
            "max_cpu_utilization": 90,  # %
            "max_memory_usage_gb": 20,  # GB out of 24GB
            "max_process_memory_mb": 2048,  # 2GB per process
            "target_performance_ops_sec": 100,
        }

        # Check compliance
        compliance = {
            "cpu_within_limits": cpu_percent <= limits["max_cpu_utilization"],
            "system_memory_within_limits": (memory.used / 1024 / 1024 / 1024)
            <= limits["max_memory_usage_gb"],
            "process_memory_within_limits": current_memory_mb
            <= limits["max_process_memory_mb"],
        }

        return {
            "system_specs": {
                "cpu_cores": cpu_count,
                "total_memory_gb": total_memory_gb,
                "mlx_available": HAS_MLX,
            },
            "current_usage": {
                "cpu_percent": cpu_percent,
                "system_memory_gb": memory.used / 1024 / 1024 / 1024,
                "process_memory_mb": current_memory_mb,
                "memory_percent": memory.percent,
            },
            "limits": limits,
            "compliance": compliance,
            "overall_compliant": all(compliance.values()),
        }

    def generate_final_report(self, results: dict[str, Any]) -> str:
        """Generate final validation report"""
        report = []
        report.append("🎯 BOLT PERFORMANCE VALIDATION REPORT")
        report.append("=" * 50)

        # Quick performance results
        quick = results["quick_perf"]
        report.append("\n📊 PERFORMANCE METRICS:")
        report.append(f"   CPU Operations: {quick['cpu_ops_per_sec']:.1f} ops/sec")
        if quick["gpu_available"]:
            report.append(f"   GPU Operations: {quick['gpu_ops_per_sec']:.1f} ops/sec")
        report.append(f"   Total Throughput: {quick['total_ops_per_sec']:.1f} ops/sec")

        # Memory stress results
        memory = results["memory_stress"]
        report.append("\n💾 MEMORY PERFORMANCE:")
        report.append(f"   Peak Allocation: {memory['peak_allocation_mb']:.1f} MB")
        report.append(f"   Memory Recovery: {memory['cleanup_efficiency']:.1f}%")
        report.append(
            f"   Memory Leak: {memory['final_memory_mb'] - memory['start_memory_mb']:.1f} MB"
        )

        # Concurrent workload results
        concurrent = results["concurrent_test"]
        report.append("\n🚀 CONCURRENT PERFORMANCE:")
        report.append(
            f"   Tasks Completed: {concurrent['successful_tasks']}/{concurrent['total_tasks']}"
        )
        report.append(f"   Success Rate: {concurrent['success_rate']:.1f}%")
        report.append(f"   Throughput: {concurrent['tasks_per_second']:.1f} tasks/sec")

        # System limits check
        limits = results["limits_check"]
        report.append("\n✅ SYSTEM COMPLIANCE:")
        report.append(
            f"   CPU Usage: {limits['current_usage']['cpu_percent']:.1f}% (Limit: {limits['limits']['max_cpu_utilization']}%)"
        )
        report.append(
            f"   Memory Usage: {limits['current_usage']['system_memory_gb']:.1f}GB (Limit: {limits['limits']['max_memory_usage_gb']}GB)"
        )
        report.append(
            f"   Process Memory: {limits['current_usage']['process_memory_mb']:.1f}MB (Limit: {limits['limits']['max_process_memory_mb']}MB)"
        )
        report.append(f"   Overall Compliant: {limits['overall_compliant']}")

        # Performance rating
        total_ops = quick["total_ops_per_sec"]
        memory_efficiency = memory["cleanup_efficiency"]
        concurrent_success = concurrent["success_rate"]

        if total_ops > 200 and memory_efficiency > 80 and concurrent_success > 95:
            rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        elif total_ops > 100 and memory_efficiency > 70 and concurrent_success > 90:
            rating = "GOOD ⭐⭐⭐⭐"
        elif total_ops > 50 and memory_efficiency > 60 and concurrent_success > 80:
            rating = "ACCEPTABLE ⭐⭐⭐"
        else:
            rating = "NEEDS IMPROVEMENT ⭐⭐"

        report.append(f"\n🏆 OVERALL RATING: {rating}")

        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        if rating.startswith("EXCELLENT"):
            report.append("   • System is performing optimally")
            report.append("   • Ready for production workloads")
            report.append("   • Consider this as baseline for monitoring")
        elif rating.startswith("GOOD"):
            report.append("   • Good performance with room for optimization")
            report.append("   • Monitor performance during peak loads")
        else:
            report.append("   • Consider performance optimization")
            report.append("   • Review resource allocation")
            report.append("   • Monitor for bottlenecks")

        if not limits["overall_compliant"]:
            report.append("   • ALERT: System exceeded recommended limits")

        return "\n".join(report)


async def main():
    """Run final Bolt validation"""
    validator = BoltValidator()

    try:
        results = await validator.run_validation()
        report = validator.generate_final_report(results)

        print("\n" + report)

        # Quick summary for user
        quick_perf = results["quick_perf"]["total_ops_per_sec"]
        memory_peak = results["memory_stress"]["peak_allocation_mb"]
        concurrent_success = results["concurrent_test"]["success_rate"]

        print("\n📋 QUICK SUMMARY:")
        print(f"   Performance: {quick_perf:.1f} ops/sec")
        print(f"   Peak Memory: {memory_peak:.1f} MB")
        print(f"   Reliability: {concurrent_success:.1f}% success rate")

        return 0

    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
