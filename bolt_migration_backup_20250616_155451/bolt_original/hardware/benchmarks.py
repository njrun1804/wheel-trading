"""
Benchmarking tools for Bolt hardware acceleration.

Provides comprehensive benchmarking for:
- CPU performance
- GPU acceleration (MLX, Metal)
- Memory bandwidth
- Storage I/O
- Accelerated tools
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# GPU backends
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


def run_quick_benchmark() -> dict[str, Any]:
    """Run a quick performance benchmark (~30 seconds)."""

    results = {
        "benchmark_type": "quick",
        "duration": 0,
        "timestamp": time.time(),
    }

    start_time = time.time()

    # CPU benchmark - matrix operations
    cpu_start = time.time()
    matrix_size = 1000
    a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    b = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    # Perform matrix multiplication
    np.dot(a, b)
    cpu_time = time.time() - cpu_start

    # Calculate performance score (operations per second)
    flops = 2 * matrix_size**3  # Approximate FLOPs for matrix multiplication
    cpu_gflops = flops / cpu_time / 1e9

    results["cpu"] = {
        "matrix_size": matrix_size,
        "time_seconds": cpu_time,
        "gflops": cpu_gflops,
        "score": min(100, cpu_gflops * 2),  # Scale to 0-100
    }

    # Memory benchmark
    memory_start = time.time()
    memory_size = 100_000_000  # 100M floats
    memory_array = np.random.rand(memory_size).astype(np.float32)
    np.sum(memory_array)
    memory_time = time.time() - memory_start

    memory_bandwidth_gb = (memory_size * 4) / memory_time / 1e9  # GB/s

    results["memory"] = {
        "size_mb": memory_size * 4 / 1e6,
        "time_seconds": memory_time,
        "bandwidth_gbps": memory_bandwidth_gb,
        "score": min(100, memory_bandwidth_gb * 5),  # Scale to 0-100
    }

    results["duration"] = time.time() - start_time
    return results


async def run_comprehensive_benchmark() -> dict[str, Any]:
    """Run comprehensive benchmark suite (~5+ minutes)."""

    results = {
        "benchmark_type": "comprehensive",
        "duration": 0,
        "timestamp": time.time(),
    }

    start_time = time.time()

    # Run all individual benchmarks
    results["cpu"] = run_cpu_benchmark()
    results["gpu"] = run_gpu_benchmark()
    results["memory"] = run_memory_benchmark()
    results["storage"] = run_storage_benchmark()
    results["accelerated_tools"] = await run_accelerated_tools_benchmark()

    results["duration"] = time.time() - start_time
    return results


def run_cpu_benchmark() -> dict[str, Any]:
    """Comprehensive CPU benchmark."""

    results = {"tests": {}, "overall_score": 0}

    # Matrix multiplication test
    for size in [500, 1000, 2000]:
        start_time = time.time()
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        np.dot(a, b)
        duration = time.time() - start_time

        flops = 2 * size**3
        gflops = flops / duration / 1e9

        results["tests"][f"matrix_{size}x{size}"] = {
            "duration": duration,
            "gflops": gflops,
        }

    # FFT test
    start_time = time.time()
    signal = np.random.rand(1000000).astype(np.complex64)
    np.fft.fft(signal)
    fft_duration = time.time() - start_time

    results["tests"]["fft_1m"] = {
        "duration": fft_duration,
        "samples_per_second": len(signal) / fft_duration,
    }

    # Calculate overall score
    avg_gflops = np.mean(
        [test["gflops"] for test in results["tests"].values() if "gflops" in test]
    )
    results["overall_score"] = min(100, avg_gflops * 2)

    return results


def run_gpu_benchmark() -> dict[str, Any]:
    """GPU acceleration benchmark."""

    results = {"available": False, "backend": "none", "tests": {}, "overall_score": 0}

    if HAS_MLX:
        results["available"] = True
        results["backend"] = "mlx"

        # MLX matrix multiplication
        start_time = time.time()
        size = 2000
        a = mx.random.normal((size, size))
        b = mx.random.normal((size, size))
        c = mx.matmul(a, b)
        mx.eval(c)  # Force evaluation
        duration = time.time() - start_time

        flops = 2 * size**3
        gflops = flops / duration / 1e9

        results["tests"]["mlx_matrix_2000x2000"] = {
            "duration": duration,
            "gflops": gflops,
        }

        results["overall_score"] = min(100, gflops)

    elif HAS_TORCH_MPS:
        results["available"] = True
        results["backend"] = "torch_mps"

        # PyTorch MPS matrix multiplication
        device = torch.device("mps")
        start_time = time.time()
        size = 2000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        torch.mps.synchronize()  # Wait for completion
        duration = time.time() - start_time

        flops = 2 * size**3
        gflops = flops / duration / 1e9

        results["tests"]["torch_mps_matrix_2000x2000"] = {
            "duration": duration,
            "gflops": gflops,
        }

        results["overall_score"] = min(100, gflops)

    return results


def run_memory_benchmark() -> dict[str, Any]:
    """Memory performance benchmark."""

    results = {"tests": {}, "overall_score": 0}

    # Sequential read test
    sizes = [10_000_000, 100_000_000, 1_000_000_000]  # 10M, 100M, 1B floats

    for size in sizes:
        # Allocation test
        start_time = time.time()
        array = np.random.rand(size).astype(np.float32)
        alloc_time = time.time() - start_time

        # Sequential read test
        start_time = time.time()
        np.sum(array)
        read_time = time.time() - start_time

        # Sequential write test
        start_time = time.time()
        array[:] = 1.0
        write_time = time.time() - start_time

        size_mb = size * 4 / 1e6
        read_bandwidth = size_mb / read_time / 1000  # GB/s
        write_bandwidth = size_mb / write_time / 1000  # GB/s

        results["tests"][f"sequential_{size_mb:.0f}mb"] = {
            "alloc_time": alloc_time,
            "read_time": read_time,
            "write_time": write_time,
            "read_bandwidth_gbps": read_bandwidth,
            "write_bandwidth_gbps": write_bandwidth,
        }

        # Clean up memory
        del array

    # Calculate overall score based on average bandwidth
    bandwidths = []
    for test in results["tests"].values():
        bandwidths.extend([test["read_bandwidth_gbps"], test["write_bandwidth_gbps"]])

    avg_bandwidth = np.mean(bandwidths)
    results["overall_score"] = min(100, avg_bandwidth * 5)

    return results


def run_storage_benchmark() -> dict[str, Any]:
    """Storage I/O performance benchmark."""

    results = {"tests": {}, "overall_score": 0}

    # Create temporary directory for tests
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # File sizes to test (in MB)
        file_sizes = [1, 10, 100]

        for size_mb in file_sizes:
            file_path = temp_path / f"test_{size_mb}mb.dat"
            size_bytes = size_mb * 1024 * 1024

            # Generate test data
            test_data = np.random.bytes(size_bytes)

            # Write test
            start_time = time.time()
            with open(file_path, "wb") as f:
                f.write(test_data)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            write_time = time.time() - start_time

            # Read test
            start_time = time.time()
            with open(file_path, "rb") as f:
                f.read()
            read_time = time.time() - start_time

            write_mbps = size_mb / write_time
            read_mbps = size_mb / read_time

            results["tests"][f"file_{size_mb}mb"] = {
                "write_time": write_time,
                "read_time": read_time,
                "write_mbps": write_mbps,
                "read_mbps": read_mbps,
            }

    # Calculate overall score
    speeds = []
    for test in results["tests"].values():
        speeds.extend([test["write_mbps"], test["read_mbps"]])

    avg_speed = np.mean(speeds)
    results["overall_score"] = min(
        100, avg_speed / 10
    )  # Scale based on expected SSD performance

    return results


async def run_accelerated_tools_benchmark() -> dict[str, Any]:
    """Benchmark accelerated tools performance."""

    results = {"tests": {}, "overall_score": 0}

    # Test each accelerated tool if available
    tools_to_test = [
        "ripgrep_turbo",
        "dependency_graph",
        "python_analysis",
        "duckdb_turbo",
        "trace_turbo",
        "python_helpers",
    ]

    for tool_name in tools_to_test:
        try:
            # Dynamic import and basic benchmark
            if tool_name == "ripgrep_turbo":
                from src.unity_wheel.accelerated_tools.ripgrep_turbo import (
                    get_ripgrep_turbo,
                )

                start_time = time.time()
                rg = get_ripgrep_turbo()
                # Simple test search
                await rg.search("import", ".", max_results=100)
                duration = time.time() - start_time

                results["tests"][tool_name] = {
                    "available": True,
                    "duration": duration,
                    "performance_score": max(0, 100 - duration * 10),  # Lower is better
                }

            # Add other tool benchmarks as needed...

        except ImportError:
            results["tests"][tool_name] = {"available": False, "performance_score": 0}
        except Exception as e:
            results["tests"][tool_name] = {
                "available": True,
                "error": str(e),
                "performance_score": 0,
            }

    # Calculate overall score
    available_tools = [
        test for test in results["tests"].values() if test.get("available", False)
    ]
    if available_tools:
        avg_score = np.mean([test["performance_score"] for test in available_tools])
        results["overall_score"] = avg_score
    else:
        results["overall_score"] = 0

    return results
