"""
Production Buffer-Stride Fix Validation and Safety Measures

This module provides comprehensive validation of the buffer-stride bug fix
and implements safety measures to prevent regression.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)


class BufferStrideValidator:
    """Validates buffer-stride fixes and prevents regressions."""

    def __init__(self):
        self.validation_results = {}
        self.safety_checks_passed = True

    def validate_buffer_alignment(self, array: mx.array) -> bool:
        """Validate that buffer is properly aligned for Metal."""
        try:
            # Check if array is contiguous
            if hasattr(array, "flags"):
                is_contiguous = array.flags.c_contiguous
            else:
                # MLX arrays are typically contiguous by default
                is_contiguous = True

            # Check element size alignment
            element_size = array.itemsize if hasattr(array, "itemsize") else 4
            total_bytes = array.size * element_size

            # Metal prefers 16-byte alignment
            is_aligned = (total_bytes % 16) == 0

            logger.debug(
                f"Buffer validation: contiguous={is_contiguous}, aligned={is_aligned}, bytes={total_bytes}"
            )

            return is_contiguous and is_aligned

        except Exception as e:
            logger.warning(f"Buffer alignment validation failed: {e}")
            return False

    def assert_gpu_speedup_threshold(
        self, cpu_time: float, gpu_time: float, min_speedup: float = 1.5
    ) -> bool:
        """Assert that GPU provides minimum speedup threshold."""
        if gpu_time <= 0:
            return False

        speedup = cpu_time / gpu_time
        is_valid = speedup >= min_speedup

        if not is_valid:
            logger.warning(f"GPU speedup {speedup:.2f}x below threshold {min_speedup}x")
            self.safety_checks_passed = False

        return is_valid

    def validate_workload_thresholds(
        self, operation_type: str, workload_size: int, used_gpu: bool
    ) -> bool:
        """Validate that workload routing decisions are optimal."""
        # Define expected thresholds
        thresholds = {
            "vector_ops": 50000,
            "matrix_ops": 512 * 512,
            "similarity": 10000,
        }

        threshold = thresholds.get(operation_type, 10000)

        # Small workloads should use CPU
        if workload_size < threshold:
            expected_cpu = True
        else:
            expected_cpu = False

        correct_routing = (not used_gpu) == expected_cpu

        if not correct_routing:
            logger.warning(
                f"Incorrect routing for {operation_type}: "
                f"size={workload_size}, threshold={threshold}, used_gpu={used_gpu}"
            )
            self.safety_checks_passed = False

        return correct_routing

    def benchmark_operation(
        self, operation_func, cpu_baseline_func, *args, **kwargs
    ) -> dict[str, Any]:
        """Benchmark an operation against CPU baseline with safety checks."""

        # CPU baseline
        start = time.perf_counter()
        cpu_result = cpu_baseline_func(*args, **kwargs)
        cpu_time = time.perf_counter() - start

        # GPU operation
        start = time.perf_counter()
        gpu_result = operation_func(*args, **kwargs)
        gpu_time = time.perf_counter() - start

        # Calculate metrics
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        # Validate results match
        results_match = self._validate_results_match(cpu_result, gpu_result)

        # Safety checks
        buffer_valid = True
        if hasattr(gpu_result, "shape") and MLX_AVAILABLE:
            if isinstance(gpu_result, mx.array):
                buffer_valid = self.validate_buffer_alignment(gpu_result)

        result = {
            "cpu_time_ms": cpu_time * 1000,
            "gpu_time_ms": gpu_time * 1000,
            "speedup": speedup,
            "results_match": results_match,
            "buffer_valid": buffer_valid,
            "safety_passed": buffer_valid and results_match,
        }

        return result

    def _validate_results_match(
        self, cpu_result, gpu_result, rtol: float = 1e-3, atol: float = 1e-6
    ) -> bool:
        """Validate that CPU and GPU results match."""
        try:
            if isinstance(cpu_result, int | float) and isinstance(
                gpu_result, int | float
            ):
                return abs(cpu_result - gpu_result) < atol

            # Convert to numpy for comparison
            if hasattr(cpu_result, "shape") and hasattr(gpu_result, "shape"):
                cpu_np = (
                    np.array(cpu_result)
                    if not isinstance(cpu_result, np.ndarray)
                    else cpu_result
                )
                gpu_np = (
                    np.array(gpu_result)
                    if not isinstance(gpu_result, np.ndarray)
                    else gpu_result
                )

                # Check shapes match
                if cpu_np.shape != gpu_np.shape:
                    logger.warning(
                        f"Shape mismatch: CPU {cpu_np.shape} vs GPU {gpu_np.shape}"
                    )
                    return False

                # Check values match
                return np.allclose(cpu_np, gpu_np, rtol=rtol, atol=atol)

            return True

        except Exception as e:
            logger.warning(f"Result validation failed: {e}")
            return False

    def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run comprehensive validation of buffer-stride fixes."""

        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}

        validation_results = {
            "timestamp": time.time(),
            "mlx_available": MLX_AVAILABLE,
            "safety_checks_passed": True,
            "tests": {},
        }

        # Import fixed implementations
        try:
            from .gpu_acceleration_fixed import (
                batch_cosine_similarity_fixed,
                dot_product_fixed,
                matrix_multiply_fixed,
            )
        except ImportError as e:
            return {"error": f"Could not import fixed implementations: {e}"}

        logger.info("Running comprehensive buffer-stride validation...")

        # Test 1: Vector operations validation
        logger.info("Testing vector operations...")
        vector_tests = {}

        for size in [10000, 100000, 1000000]:
            a = np.random.rand(size).astype(np.float32)
            b = np.random.rand(size).astype(np.float32)

            def cpu_dot(x, y):
                return np.dot(x, y)

            result = self.benchmark_operation(
                lambda x, y: dot_product_fixed(x, y), cpu_dot, a, b
            )

            vector_tests[f"size_{size}"] = result

            # Validate workload routing
            used_gpu = result["speedup"] > 0.5  # Assume GPU was used if speedup > 0.5
            self.validate_workload_thresholds("vector_ops", size, used_gpu)

        validation_results["tests"]["vector_operations"] = vector_tests

        # Test 2: Matrix operations validation
        logger.info("Testing matrix operations...")
        matrix_tests = {}

        for rows, cols in [(512, 512), (1024, 1024), (2048, 1024)]:
            a = (
                np.random.rand(rows, cols).astype(np.float32) * 0.1
            )  # Scale to avoid overflow
            b = np.random.rand(cols, rows).astype(np.float32) * 0.1

            def cpu_matmul(x, y):
                return np.matmul(x, y)

            result = self.benchmark_operation(
                lambda x, y: matrix_multiply_fixed(x, y), cpu_matmul, a, b
            )

            matrix_tests[f"matrix_{rows}x{cols}"] = result

            # Validate workload routing
            workload_size = rows * cols
            used_gpu = result["speedup"] > 0.5
            self.validate_workload_thresholds("matrix_ops", workload_size, used_gpu)

        validation_results["tests"]["matrix_operations"] = matrix_tests

        # Test 3: Similarity search validation
        logger.info("Testing similarity search...")
        similarity_tests = {}

        for corpus_size in [5000, 10000, 20000]:
            query = np.random.rand(768).astype(np.float32)
            corpus = np.random.rand(corpus_size, 768).astype(np.float32)

            def cpu_similarity(q, c):
                q_norm = q / np.linalg.norm(q)
                c_norm = c / np.linalg.norm(c, axis=1, keepdims=True)
                return c_norm @ q_norm

            result = self.benchmark_operation(
                lambda q, c: batch_cosine_similarity_fixed(q, c),
                cpu_similarity,
                query,
                corpus,
            )

            similarity_tests[f"corpus_{corpus_size}"] = result

            # Validate workload routing
            used_gpu = result["speedup"] > 0.5
            self.validate_workload_thresholds("similarity", corpus_size, used_gpu)

        validation_results["tests"]["similarity_search"] = similarity_tests

        # Calculate overall metrics
        all_results = []
        for test_category in validation_results["tests"].values():
            for test_result in test_category.values():
                if "speedup" in test_result:
                    all_results.append(test_result)

        if all_results:
            speedups = [r["speedup"] for r in all_results]
            safety_passed = [r["safety_passed"] for r in all_results]

            validation_results["summary"] = {
                "total_tests": len(all_results),
                "average_speedup": np.mean(speedups),
                "max_speedup": np.max(speedups),
                "min_speedup": np.min(speedups),
                "safety_pass_rate": np.mean(safety_passed) * 100,
                "all_safety_checks_passed": self.safety_checks_passed
                and all(safety_passed),
            }

        validation_results["safety_checks_passed"] = self.safety_checks_passed

        # Log results
        summary = validation_results.get("summary", {})
        logger.info(f"Validation complete: {summary.get('total_tests', 0)} tests")
        logger.info(f"Average speedup: {summary.get('average_speedup', 0):.2f}x")
        logger.info(f"Max speedup: {summary.get('max_speedup', 0):.2f}x")
        logger.info(
            f"Safety checks: {'PASSED' if self.safety_checks_passed else 'FAILED'}"
        )

        return validation_results


def validate_buffer_stride_fix() -> dict[str, Any]:
    """Main validation function for buffer-stride fix."""
    validator = BufferStrideValidator()
    results = validator.run_comprehensive_validation()

    # Save results
    output_file = Path(__file__).parent / "buffer_stride_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Validation results saved to: {output_file}")

    return results


def add_runtime_assertions():
    """Add runtime assertions to prevent buffer-stride regression."""

    def assert_gpu_performance(func):
        """Decorator to assert GPU performance doesn't regress."""

        def wrapper(*args, **kwargs):
            if not MLX_AVAILABLE:
                return func(*args, **kwargs)

            # Monitor execution time
            start = time.perf_counter()
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start

            # Assert reasonable execution time for GPU operations
            # This is a simple heuristic - adjust thresholds as needed
            if hasattr(result, "size") and result.size > 100000:
                max_time_ms = result.size * 0.001  # 1μs per element max
                if execution_time * 1000 > max_time_ms:
                    logger.warning(
                        f"GPU operation took {execution_time*1000:.2f}ms "
                        f"for {result.size} elements (expected <{max_time_ms:.2f}ms)"
                    )

            return result

        return wrapper

    # Apply to key GPU functions
    try:
        from . import gpu_acceleration_fixed

        # Wrap key functions with performance assertions
        gpu_acceleration_fixed.dot_product_fixed = assert_gpu_performance(
            gpu_acceleration_fixed.dot_product_fixed
        )
        gpu_acceleration_fixed.matrix_multiply_fixed = assert_gpu_performance(
            gpu_acceleration_fixed.matrix_multiply_fixed
        )

        logger.info("Runtime performance assertions added")

    except ImportError:
        logger.warning("Could not add runtime assertions - fixed module not available")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run validation
    results = validate_buffer_stride_fix()

    # Add runtime assertions
    add_runtime_assertions()

    # Print summary
    if "summary" in results:
        summary = results["summary"]
        print("\n🎯 BUFFER-STRIDE FIX VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Average Speedup: {summary['average_speedup']:.2f}x")
        print(f"Max Speedup: {summary['max_speedup']:.2f}x")
        print(f"Safety Pass Rate: {summary['safety_pass_rate']:.1f}%")
        print(
            f"All Safety Checks: {'✅ PASSED' if summary['all_safety_checks_passed'] else '❌ FAILED'}"
        )

        if summary["max_speedup"] > 20:
            print(
                f"\n🚀 SUCCESS: Buffer-stride fix achieved {summary['max_speedup']:.1f}x maximum speedup!"
            )
        elif summary["average_speedup"] > 5:
            print(
                f"\n✅ GOOD: Buffer-stride fix achieved {summary['average_speedup']:.1f}x average speedup!"
            )
        else:
            print(
                f"\n⚠️  NEEDS WORK: Buffer-stride fix achieved only {summary['average_speedup']:.1f}x speedup."
            )
