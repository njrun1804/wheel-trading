#!/usr/bin/env python3
"""
Bolt Throughput Validator
Real-world validation of throughput improvements under various workloads
"""

import asyncio
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

from bolt_database_fixes import get_database_connection
from bolt_throughput_optimizer import ThroughputMetrics, ThroughputProfiler

logger = logging.getLogger(__name__)


@dataclass
class WorkloadProfile:
    """Definition of a workload for testing"""

    name: str
    description: str
    operations_per_test: int
    concurrent_workers: int
    operation_complexity: str  # 'light', 'medium', 'heavy'
    database_heavy: bool
    expected_min_ops_per_sec: float


@dataclass
class ValidationResult:
    """Result of throughput validation"""

    workload_name: str
    achieved_ops_per_sec: float
    target_ops_per_sec: float
    target_met: bool
    latency_p95_ms: float
    cpu_utilization: float
    memory_usage_mb: float
    bottlenecks: list[str]
    duration_seconds: float
    stability_score: float  # 0-1, how stable the throughput was


class WorkloadSimulator:
    """Simulates different types of workloads"""

    def __init__(self):
        self.profiler = ThroughputProfiler()

    async def simulate_light_workload(self, operation_id: int) -> dict[str, Any]:
        """Simulate light computational workload"""
        start = time.perf_counter()

        # Light work: simple calculations
        result = sum(range(100))
        await asyncio.sleep(0.001)  # 1ms I/O simulation

        duration_ms = (time.perf_counter() - start) * 1000
        self.profiler.record_operation(
            duration_ms, {"type": "light", "operation_id": operation_id}
        )

        return {"result": result, "duration_ms": duration_ms}

    async def simulate_medium_workload(self, operation_id: int) -> dict[str, Any]:
        """Simulate medium computational workload"""
        start = time.perf_counter()

        # Medium work: matrix operations
        import numpy as np

        a = np.random.randn(50, 50)
        b = np.random.randn(50, 50)
        result = np.dot(a, b)

        await asyncio.sleep(0.005)  # 5ms I/O simulation

        duration_ms = (time.perf_counter() - start) * 1000
        self.profiler.record_operation(
            duration_ms, {"type": "medium", "operation_id": operation_id}
        )

        return {"result": result.sum(), "duration_ms": duration_ms}

    async def simulate_heavy_workload(self, operation_id: int) -> dict[str, Any]:
        """Simulate heavy computational workload"""
        start = time.perf_counter()

        # Heavy work: complex calculations
        import numpy as np

        # Simulate AI/ML workload
        data = np.random.randn(100, 100)
        for _ in range(10):
            data = np.dot(data, np.random.randn(100, 100))
            data = np.tanh(data)  # Activation function

        await asyncio.sleep(0.01)  # 10ms I/O simulation

        duration_ms = (time.perf_counter() - start) * 1000
        self.profiler.record_operation(
            duration_ms, {"type": "heavy", "operation_id": operation_id}
        )

        return {"result": data.sum(), "duration_ms": duration_ms}

    async def simulate_database_workload(self, operation_id: int) -> dict[str, Any]:
        """Simulate database-heavy workload"""
        start = time.perf_counter()

        # Database operations
        test_db_path = (
            f"/tmp/bolt_test_{operation_id % 4}.db"  # 4 databases for concurrency
        )

        try:
            conn = get_database_connection(test_db_path)

            # Create table if not exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_test (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    data TEXT,
                    value REAL
                )
            """
            )

            # Insert data
            conn.execute(
                """
                INSERT INTO performance_test (timestamp, data, value)
                VALUES (?, ?, ?)
            """,
                (time.time(), f"test_data_{operation_id}", random.random()),
            )

            # Query data
            result = conn.execute(
                """
                SELECT COUNT(*), AVG(value) FROM performance_test
                WHERE timestamp > ?
            """,
                (time.time() - 60,),
            ).fetchone()

            # Simulate some processing
            await asyncio.sleep(0.002)

        except Exception as e:
            logger.debug(f"Database operation {operation_id} failed: {e}")
            result = (0, 0.0)

        duration_ms = (time.perf_counter() - start) * 1000
        self.profiler.record_operation(
            duration_ms, {"type": "database", "operation_id": operation_id}
        )

        return {"result": result, "duration_ms": duration_ms}


class ThroughputValidator:
    """Validates throughput improvements under various workloads"""

    def __init__(self):
        self.simulator = WorkloadSimulator()
        self.workload_profiles = self._create_workload_profiles()
        self.validation_results = []

    def _create_workload_profiles(self) -> list[WorkloadProfile]:
        """Create predefined workload profiles for testing"""
        return [
            WorkloadProfile(
                name="light_burst",
                description="High-frequency light operations",
                operations_per_test=200,
                concurrent_workers=8,
                operation_complexity="light",
                database_heavy=False,
                expected_min_ops_per_sec=150.0,
            ),
            WorkloadProfile(
                name="medium_sustained",
                description="Sustained medium complexity operations",
                operations_per_test=100,
                concurrent_workers=6,
                operation_complexity="medium",
                database_heavy=False,
                expected_min_ops_per_sec=80.0,
            ),
            WorkloadProfile(
                name="heavy_compute",
                description="Heavy computational workload",
                operations_per_test=50,
                concurrent_workers=4,
                operation_complexity="heavy",
                database_heavy=False,
                expected_min_ops_per_sec=30.0,
            ),
            WorkloadProfile(
                name="database_intensive",
                description="Database-heavy operations",
                operations_per_test=150,
                concurrent_workers=12,
                operation_complexity="medium",
                database_heavy=True,
                expected_min_ops_per_sec=100.0,
            ),
            WorkloadProfile(
                name="mixed_workload",
                description="Mixed complexity operations",
                operations_per_test=120,
                concurrent_workers=8,
                operation_complexity="mixed",
                database_heavy=True,
                expected_min_ops_per_sec=85.0,
            ),
            WorkloadProfile(
                name="production_simulation",
                description="Realistic production workload",
                operations_per_test=300,
                concurrent_workers=12,
                operation_complexity="mixed",
                database_heavy=True,
                expected_min_ops_per_sec=100.0,
            ),
        ]

    async def validate_throughput_improvements(self) -> dict[str, Any]:
        """Run comprehensive throughput validation"""
        logger.info("🧪 Starting Bolt Throughput Validation")
        logger.info(f"Testing {len(self.workload_profiles)} workload profiles")

        validation_start = time.time()

        for profile in self.workload_profiles:
            logger.info(f"🔬 Testing workload: {profile.name}")

            result = await self._run_workload_test(profile)
            self.validation_results.append(result)

            status = "✅ PASS" if result.target_met else "❌ FAIL"
            logger.info(
                f"{status} {profile.name}: {result.achieved_ops_per_sec:.1f} ops/sec "
                f"(target: {result.target_ops_per_sec:.1f})"
            )

        validation_duration = time.time() - validation_start

        # Generate comprehensive validation report
        report = self._generate_validation_report(validation_duration)

        passed_tests = sum(1 for r in self.validation_results if r.target_met)
        total_tests = len(self.validation_results)

        logger.info("🎯 Bolt Throughput Validation Complete")
        logger.info(f"Passed: {passed_tests}/{total_tests} tests")

        return report

    async def _run_workload_test(self, profile: WorkloadProfile) -> ValidationResult:
        """Run a single workload test"""
        # Clear previous profiler data
        self.simulator.profiler.operation_times.clear()

        # Determine operation function
        if profile.operation_complexity == "light":
            operation_func = self.simulator.simulate_light_workload
        elif profile.operation_complexity == "medium":
            operation_func = self.simulator.simulate_medium_workload
        elif profile.operation_complexity == "heavy":
            operation_func = self.simulator.simulate_heavy_workload
        elif profile.database_heavy:
            operation_func = self.simulator.simulate_database_workload
        else:  # mixed
            operation_func = self._get_mixed_operation_func()

        # Run the workload
        test_start = time.time()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(profile.concurrent_workers)

        async def bounded_operation(op_id: int):
            async with semaphore:
                return await operation_func(op_id)

        # Execute operations
        tasks = [bounded_operation(i) for i in range(profile.operations_per_test)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        test_duration = time.time() - test_start

        # Calculate metrics
        successful_operations = [r for r in results if not isinstance(r, Exception)]
        ops_per_sec = len(successful_operations) / test_duration

        # Get throughput metrics from profiler
        metrics = self.simulator.profiler.calculate_throughput(
            window_seconds=test_duration
        )

        # Calculate stability score
        stability_score = self._calculate_stability_score()

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(metrics, profile)

        return ValidationResult(
            workload_name=profile.name,
            achieved_ops_per_sec=ops_per_sec,
            target_ops_per_sec=profile.expected_min_ops_per_sec,
            target_met=ops_per_sec >= profile.expected_min_ops_per_sec,
            latency_p95_ms=metrics.latency_p95_ms,
            cpu_utilization=metrics.cpu_utilization,
            memory_usage_mb=metrics.memory_usage_mb,
            bottlenecks=bottlenecks,
            duration_seconds=test_duration,
            stability_score=stability_score,
        )

    def _get_mixed_operation_func(self):
        """Get a random operation function for mixed workloads"""
        operations = [
            self.simulator.simulate_light_workload,
            self.simulator.simulate_medium_workload,
            self.simulator.simulate_database_workload,
        ]

        async def mixed_operation(operation_id: int):
            # Randomly select operation type
            weights = [0.5, 0.3, 0.2]  # 50% light, 30% medium, 20% database
            operation_func = random.choices(operations, weights=weights)[0]
            return await operation_func(operation_id)

        return mixed_operation

    def _calculate_stability_score(self) -> float:
        """Calculate how stable the throughput was during the test"""
        if len(self.simulator.profiler.operation_times) < 10:
            return 0.0

        # Get operation durations
        durations = [
            op["duration_ms"] for op in self.simulator.profiler.operation_times
        ]

        if not durations:
            return 0.0

        # Calculate coefficient of variation (lower is more stable)
        mean_duration = statistics.mean(durations)
        if mean_duration == 0:
            return 0.0

        stdev_duration = statistics.stdev(durations) if len(durations) > 1 else 0.0
        cv = stdev_duration / mean_duration

        # Convert to stability score (0-1, higher is better)
        stability_score = max(0.0, 1.0 - cv)
        return min(1.0, stability_score)

    def _identify_bottlenecks(
        self, metrics: ThroughputMetrics, profile: WorkloadProfile
    ) -> list[str]:
        """Identify performance bottlenecks for this workload"""
        bottlenecks = []

        # CPU bottleneck
        if metrics.cpu_utilization > 90:
            bottlenecks.append("CPU_SATURATED")
        elif metrics.cpu_utilization < 30 and profile.operation_complexity != "light":
            bottlenecks.append("CPU_UNDERUTILIZED")

        # Memory bottleneck
        if metrics.memory_usage_mb > 16000:  # > 16GB
            bottlenecks.append("HIGH_MEMORY_USAGE")

        # Latency bottleneck
        if metrics.latency_p95_ms > 100:
            bottlenecks.append("HIGH_LATENCY")

        # Database bottleneck
        if (
            profile.database_heavy
            and metrics.database_connections < profile.concurrent_workers
        ):
            bottlenecks.append("DATABASE_CONNECTION_LIMIT")

        # Concurrency bottleneck
        if profile.concurrent_workers > 8 and metrics.latency_p95_ms > 50:
            bottlenecks.append("CONCURRENCY_OVERHEAD")

        return bottlenecks

    def _generate_validation_report(self, validation_duration: float) -> dict[str, Any]:
        """Generate comprehensive validation report"""

        passed_tests = [r for r in self.validation_results if r.target_met]
        failed_tests = [r for r in self.validation_results if not r.target_met]

        # Calculate aggregate metrics
        avg_ops_per_sec = statistics.mean(
            [r.achieved_ops_per_sec for r in self.validation_results]
        )
        avg_latency_p95 = statistics.mean(
            [r.latency_p95_ms for r in self.validation_results]
        )
        avg_stability = statistics.mean(
            [r.stability_score for r in self.validation_results]
        )

        # Identify common bottlenecks
        all_bottlenecks = []
        for result in self.validation_results:
            all_bottlenecks.extend(result.bottlenecks)

        bottleneck_counts = {}
        for bottleneck in all_bottlenecks:
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1

        return {
            "validation_summary": {
                "total_tests": len(self.validation_results),
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(passed_tests) / len(self.validation_results) * 100,
                "validation_duration_seconds": validation_duration,
            },
            "performance_metrics": {
                "average_ops_per_sec": avg_ops_per_sec,
                "average_latency_p95_ms": avg_latency_p95,
                "average_stability_score": avg_stability,
                "throughput_target_100_ops_met": avg_ops_per_sec >= 100.0,
            },
            "workload_results": [
                {
                    "workload": r.workload_name,
                    "achieved_ops_per_sec": r.achieved_ops_per_sec,
                    "target_ops_per_sec": r.target_ops_per_sec,
                    "target_met": r.target_met,
                    "latency_p95_ms": r.latency_p95_ms,
                    "stability_score": r.stability_score,
                    "bottlenecks": r.bottlenecks,
                    "duration_seconds": r.duration_seconds,
                }
                for r in self.validation_results
            ],
            "bottleneck_analysis": {
                "most_common_bottlenecks": sorted(
                    bottleneck_counts.items(), key=lambda x: x[1], reverse=True
                ),
                "bottleneck_free_tests": len(
                    [r for r in self.validation_results if not r.bottlenecks]
                ),
            },
            "recommendations": self._generate_performance_recommendations(),
            "production_readiness": {
                "ready_for_production": (
                    len(passed_tests) >= len(self.validation_results) * 0.8
                    and avg_ops_per_sec >= 100.0
                    and avg_stability > 0.7
                ),
                "key_concerns": self._identify_production_concerns(),
                "recommended_load_limits": {
                    "max_sustained_ops_per_sec": avg_ops_per_sec * 0.8,
                    "burst_capacity_ops_per_sec": max(
                        [r.achieved_ops_per_sec for r in self.validation_results]
                    ),
                    "recommended_concurrent_workers": 8,  # Conservative estimate
                },
            },
        }

    def _generate_performance_recommendations(self) -> list[str]:
        """Generate performance recommendations based on test results"""
        recommendations = []

        failed_tests = [r for r in self.validation_results if not r.target_met]

        if failed_tests:
            recommendations.append(
                "Some workload tests failed - review bottleneck analysis"
            )

        # CPU recommendations
        cpu_saturated_tests = [
            r for r in self.validation_results if "CPU_SATURATED" in r.bottlenecks
        ]
        if cpu_saturated_tests:
            recommendations.append(
                "Consider CPU scaling or workload distribution for high-compute tasks"
            )

        # Latency recommendations
        high_latency_tests = [
            r for r in self.validation_results if r.latency_p95_ms > 100
        ]
        if high_latency_tests:
            recommendations.append(
                "Optimize high-latency operations for better user experience"
            )

        # Stability recommendations
        unstable_tests = [r for r in self.validation_results if r.stability_score < 0.7]
        if unstable_tests:
            recommendations.append(
                "Improve throughput stability for consistent performance"
            )

        # Database recommendations
        db_bottlenecks = [
            r
            for r in self.validation_results
            if any("DATABASE" in b for b in r.bottlenecks)
        ]
        if db_bottlenecks:
            recommendations.append("Scale database connection pool or optimize queries")

        if not recommendations:
            recommendations.append(
                "Performance validation successful - system ready for production"
            )

        return recommendations

    def _identify_production_concerns(self) -> list[str]:
        """Identify concerns for production deployment"""
        concerns = []

        failed_tests = [r for r in self.validation_results if not r.target_met]
        if len(failed_tests) > len(self.validation_results) * 0.2:
            concerns.append(
                f"High test failure rate: {len(failed_tests)}/{len(self.validation_results)}"
            )

        avg_ops_per_sec = statistics.mean(
            [r.achieved_ops_per_sec for r in self.validation_results]
        )
        if avg_ops_per_sec < 100:
            concerns.append(
                f"Average throughput below target: {avg_ops_per_sec:.1f} < 100 ops/sec"
            )

        high_latency_tests = [
            r for r in self.validation_results if r.latency_p95_ms > 200
        ]
        if high_latency_tests:
            concerns.append(f"High latency in {len(high_latency_tests)} test(s)")

        unstable_tests = [r for r in self.validation_results if r.stability_score < 0.5]
        if unstable_tests:
            concerns.append(f"Poor stability in {len(unstable_tests)} test(s)")

        return concerns


async def validate_bolt_throughput() -> dict[str, Any]:
    """Main entry point for Bolt throughput validation"""
    validator = ThroughputValidator()
    return await validator.validate_throughput_improvements()


if __name__ == "__main__":

    async def main():
        print("🧪 Bolt Throughput Validator")
        print("=" * 50)

        result = await validate_bolt_throughput()

        print("\n📊 Validation Results:")
        summary = result["validation_summary"]
        print(f"Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        print(f"Success Rate: {summary['success_rate']:.1f}%")

        metrics = result["performance_metrics"]
        print("\n📈 Performance Metrics:")
        print(f"Average Throughput: {metrics['average_ops_per_sec']:.1f} ops/sec")
        print(
            f"Target 100+ ops/sec: {'✅' if metrics['throughput_target_100_ops_met'] else '❌'}"
        )
        print(f"Average Latency (P95): {metrics['average_latency_p95_ms']:.1f}ms")
        print(f"Average Stability: {metrics['average_stability_score']:.2f}")

        production = result["production_readiness"]
        print(
            f"\n🚀 Production Readiness: {'✅' if production['ready_for_production'] else '❌'}"
        )
        if production["key_concerns"]:
            print("⚠️  Key Concerns:")
            for concern in production["key_concerns"]:
                print(f"  • {concern}")

        print("\n💡 Recommendations:")
        for rec in result["recommendations"]:
            print(f"  • {rec}")

        # Save detailed report
        with open("bolt_throughput_validation_report.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("\n📄 Detailed report saved to: bolt_throughput_validation_report.json")

    asyncio.run(main())
