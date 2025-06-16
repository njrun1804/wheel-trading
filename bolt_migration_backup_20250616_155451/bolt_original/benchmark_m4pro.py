#!/usr/bin/env python3
"""Comprehensive benchmark suite for M4 Pro hardware capabilities."""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import contextlib

from bolt.agents.types import TaskPriority
from bolt.core.integration import BoltIntegration, SystemState


class M4ProBenchmark:
    """Benchmark suite for M4 Pro hardware."""

    def __init__(self):
        self.results = {}

    async def run_all(self):
        """Run all benchmarks."""
        print("=== M4 Pro Hardware Benchmark Suite ===\n")

        await self.benchmark_system_state()
        await self.benchmark_parallel_agents()
        await self.benchmark_gpu_acceleration()
        await self.benchmark_memory_pressure()
        await self.benchmark_task_throughput()

        self.print_summary()

    async def benchmark_system_state(self):
        """Benchmark system state monitoring."""
        print("1. System State Monitoring Performance")

        times = []
        for _i in range(100):
            start = time.perf_counter()
            state = SystemState.capture()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"   Average capture time: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"   CPU cores detected: {state.cpu_cores}")
        print(f"   GPU backend: {state.gpu_backend}")
        print()

        self.results["system_state"] = {
            "avg_ms": avg_time,
            "std_ms": std_time,
            "cpu_cores": state.cpu_cores,
            "gpu_backend": state.gpu_backend,
        }

    async def benchmark_parallel_agents(self):
        """Benchmark parallel agent execution."""
        print("2. Parallel Agent Execution")

        integration = BoltIntegration(num_agents=8)
        await integration.initialize()

        # Submit tasks for all agents
        tasks = []
        num_tasks = 32  # 4 tasks per agent

        start = time.perf_counter()
        for i in range(num_tasks):
            task = integration.submit_task(f"Benchmark task {i}", TaskPriority.NORMAL)
            tasks.append(task)

        # Mock fast execution
        for agent in integration.agents:
            agent._execute_task_logic = lambda t: {"benchmark": True, "task_id": t.id}

        # Run agents
        run_task = asyncio.create_task(integration.run())

        # Wait for completion
        while len(integration.completed_tasks) < num_tasks:
            await asyncio.sleep(0.01)

        elapsed = time.perf_counter() - start

        # Cancel run task
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task

        await integration.shutdown()

        throughput = num_tasks / elapsed
        print(f"   Processed {num_tasks} tasks in {elapsed:.2f}s")
        print(f"   Throughput: {throughput:.1f} tasks/second")
        print(f"   Average per task: {elapsed/num_tasks*1000:.1f}ms")
        print()

        self.results["parallel_agents"] = {
            "num_tasks": num_tasks,
            "total_seconds": elapsed,
            "throughput": throughput,
            "avg_ms_per_task": elapsed / num_tasks * 1000,
        }

    async def benchmark_gpu_acceleration(self):
        """Benchmark GPU acceleration capabilities."""
        print("3. GPU Acceleration Performance")

        # Test MLX if available
        try:
            import mlx.core as mx

            # Matrix multiplication benchmark
            size = (2048, 2048)
            a = mx.random.normal(size)
            b = mx.random.normal(size)

            # Warmup
            _ = a @ b
            mx.eval(_)

            # Benchmark
            start = time.perf_counter()
            for _ in range(10):
                c = a @ b
                mx.eval(c)
            mlx_time = time.perf_counter() - start

            gflops = (2 * size[0] ** 3 * 10) / (mlx_time * 1e9)
            print("   MLX Matrix Multiply (2048x2048):")
            print(f"     Time: {mlx_time:.2f}s for 10 iterations")
            print(f"     Performance: {gflops:.1f} GFLOPS")

            self.results["mlx_gpu"] = {
                "available": True,
                "matmul_seconds": mlx_time,
                "gflops": gflops,
            }
        except ImportError:
            print("   MLX not available")
            self.results["mlx_gpu"] = {"available": False}

        # Test PyTorch MPS if available
        try:
            import torch

            if torch.backends.mps.is_available():
                size = (2048, 2048)
                a = torch.randn(size, device="mps")
                b = torch.randn(size, device="mps")

                # Warmup
                _ = a @ b
                torch.mps.synchronize()

                # Benchmark
                start = time.perf_counter()
                for _ in range(10):
                    c = a @ b
                    torch.mps.synchronize()
                mps_time = time.perf_counter() - start

                gflops = (2 * size[0] ** 3 * 10) / (mps_time * 1e9)
                print("   PyTorch MPS Matrix Multiply (2048x2048):")
                print(f"     Time: {mps_time:.2f}s for 10 iterations")
                print(f"     Performance: {gflops:.1f} GFLOPS")

                self.results["pytorch_mps"] = {
                    "available": True,
                    "matmul_seconds": mps_time,
                    "gflops": gflops,
                }
            else:
                print("   PyTorch MPS not available")
                self.results["pytorch_mps"] = {"available": False}
        except ImportError:
            print("   PyTorch not installed")
            self.results["pytorch_mps"] = {"available": False}

        print()

    async def benchmark_memory_pressure(self):
        """Benchmark behavior under memory pressure."""
        print("4. Memory Pressure Handling")

        integration = BoltIntegration(num_agents=4)

        # Allocate some memory to create pressure
        mem = psutil.virtual_memory()
        baseline_percent = mem.percent

        # Try to allocate arrays that push memory usage
        arrays = []
        target_mb = 1024  # 1GB chunks

        try:
            while psutil.virtual_memory().percent < 80:
                arr = np.zeros((target_mb * 1024 * 1024 // 8,), dtype=np.float64)
                arrays.append(arr)

            # Now test task execution under pressure
            high_pressure_percent = psutil.virtual_memory().percent

            # Submit a task
            integration.submit_task("Memory pressure test")

            # Check if system detects pressure
            state = SystemState.capture()

            print(f"   Baseline memory: {baseline_percent:.1f}%")
            print(f"   Under pressure: {high_pressure_percent:.1f}%")
            print(f"   System healthy: {state.is_healthy}")
            print(f"   Warnings: {len(state.warnings)}")

            self.results["memory_pressure"] = {
                "baseline_percent": baseline_percent,
                "pressure_percent": high_pressure_percent,
                "is_healthy": state.is_healthy,
                "num_warnings": len(state.warnings),
            }

        finally:
            # Clean up arrays
            arrays.clear()

        print()

    async def benchmark_task_throughput(self):
        """Benchmark task creation and dependency resolution."""
        print("5. Task Management Performance")

        integration = BoltIntegration(num_agents=8)

        # Create tasks with complex dependencies
        num_layers = 5
        tasks_per_layer = 10
        all_tasks = []

        start = time.perf_counter()

        # Layer 0: Independent tasks
        layer_0 = []
        for i in range(tasks_per_layer):
            task = integration.submit_task(f"Layer 0 Task {i}")
            layer_0.append(task)
            all_tasks.append(task)

        # Subsequent layers depend on previous
        prev_layer = layer_0
        for layer in range(1, num_layers):
            current_layer = []
            for i in range(tasks_per_layer):
                # Each task depends on 2 tasks from previous layer
                deps = {
                    prev_layer[i % len(prev_layer)].id,
                    prev_layer[(i + 1) % len(prev_layer)].id,
                }
                task = integration.submit_task(
                    f"Layer {layer} Task {i}", dependencies=deps
                )
                current_layer.append(task)
                all_tasks.append(task)
            prev_layer = current_layer

        creation_time = time.perf_counter() - start
        total_tasks = len(all_tasks)

        print(
            f"   Created {total_tasks} tasks with dependencies in {creation_time:.3f}s"
        )
        print(
            f"   Average creation time: {creation_time/total_tasks*1000:.2f}ms per task"
        )
        print(f"   Dependency graph layers: {num_layers}")
        print()

        self.results["task_throughput"] = {
            "total_tasks": total_tasks,
            "creation_seconds": creation_time,
            "avg_ms_per_task": creation_time / total_tasks * 1000,
            "num_layers": num_layers,
        }

    def print_summary(self):
        """Print benchmark summary."""
        print("=== Benchmark Summary ===")
        print()
        print("System Capabilities:")
        print(f"  CPU Cores: {self.results['system_state']['cpu_cores']}")
        print(f"  GPU Backend: {self.results['system_state']['gpu_backend']}")
        print()

        print("Performance Metrics:")
        print(f"  System State Capture: {self.results['system_state']['avg_ms']:.2f}ms")
        print(
            f"  Task Throughput: {self.results['parallel_agents']['throughput']:.1f} tasks/sec"
        )
        print(
            f"  Task Creation: {self.results['task_throughput']['avg_ms_per_task']:.2f}ms per task"
        )

        if self.results["mlx_gpu"]["available"]:
            print(
                f"  MLX GPU Performance: {self.results['mlx_gpu']['gflops']:.1f} GFLOPS"
            )

        if self.results["pytorch_mps"]["available"]:
            print(
                f"  PyTorch MPS Performance: {self.results['pytorch_mps']['gflops']:.1f} GFLOPS"
            )

        print()
        print("Memory Management:")
        print(
            f"  Handles pressure up to: {self.results['memory_pressure']['pressure_percent']:.1f}%"
        )
        print(
            f"  Health detection working: {'Yes' if not self.results['memory_pressure']['is_healthy'] else 'No'}"
        )


async def main():
    """Run the benchmark suite."""
    benchmark = M4ProBenchmark()
    await benchmark.run_all()


if __name__ == "__main__":
    asyncio.run(main())
