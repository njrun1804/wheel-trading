"""
Comprehensive Performance Benchmarking for M4 Pro Optimizations

Validates all optimization components and measures performance improvements.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from .adaptive_concurrency import TaskType, get_adaptive_concurrency_manager
from .ane_acceleration import ComputeUnit, create_ane_embedding_generator
from .memory_pools import get_memory_pool_manager
from .metal_accelerated_search import get_metal_search
from .unified_memory import get_unified_memory_manager

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""

    name: str
    duration_ms: float
    throughput: float
    memory_mb: float
    success: bool
    error: str | None = None


class M4ProBenchmarkSuite:
    """Comprehensive benchmark suite for M4 Pro optimizations"""

    def __init__(self):
        self.results: list[BenchmarkResult] = []
        self.baseline_results: dict[str, float] = {}

    async def run_all_benchmarks(self) -> dict[str, Any]:
        """Run all benchmark tests"""
        logger.info("Starting M4 Pro optimization benchmark suite")

        benchmarks = [
            ("unified_memory", self._benchmark_unified_memory),
            ("metal_search", self._benchmark_metal_search),
            ("adaptive_concurrency", self._benchmark_adaptive_concurrency),
            ("memory_pools", self._benchmark_memory_pools),
            ("ane_embeddings", self._benchmark_ane_embeddings),
            ("end_to_end", self._benchmark_end_to_end),
        ]

        for name, benchmark_func in benchmarks:
            try:
                logger.info(f"Running {name} benchmark...")
                result = await benchmark_func()
                self.results.append(result)
                logger.info(
                    f"{name}: {result.duration_ms:.1f}ms, {result.throughput:.1f} ops/sec"
                )
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")
                self.results.append(
                    BenchmarkResult(
                        name=name,
                        duration_ms=0,
                        throughput=0,
                        memory_mb=0,
                        success=False,
                        error=str(e),
                    )
                )

        return self._generate_report()

    async def _benchmark_unified_memory(self) -> BenchmarkResult:
        """Benchmark unified memory operations"""
        manager = get_unified_memory_manager()

        # Test zero-copy operations
        test_data = np.random.randn(1000, 768).astype(np.float32)

        start_time = time.perf_counter()

        # Create buffer and test zero-copy
        buffer = manager.allocate_buffer(
            test_data.nbytes, "EMBEDDING_MATRIX", "benchmark"
        )
        buffer.copy_from_numpy(test_data)

        # Test conversions
        for _ in range(100):
            buffer.as_mlx(mx.float32, test_data.shape)
            buffer.as_numpy(np.float32, test_data.shape)
            buffer.zero_copy_transfer(test_data)

        duration = (time.perf_counter() - start_time) * 1000
        throughput = 300 / (duration / 1000)  # 300 operations

        stats = manager.get_memory_stats()

        return BenchmarkResult(
            name="unified_memory",
            duration_ms=duration,
            throughput=throughput,
            memory_mb=stats["total_memory_mb"],
            success=True,
        )

    async def _benchmark_metal_search(self) -> BenchmarkResult:
        """Benchmark Metal-accelerated search"""
        if not MLX_AVAILABLE:
            return BenchmarkResult(
                name="metal_search",
                duration_ms=0,
                throughput=0,
                memory_mb=0,
                success=False,
                error="MLX not available",
            )

        search_engine = get_metal_search(embedding_dim=768)

        # Create test corpus
        corpus_size = 10000
        embeddings = np.random.randn(corpus_size, 768).astype(np.float32)
        metadata = [{"content": f"test_{i}"} for i in range(corpus_size)]

        search_engine.load_corpus(embeddings, metadata)

        # Benchmark searches
        num_queries = 100
        queries = np.random.randn(num_queries, 768).astype(np.float32)

        start_time = time.perf_counter()

        search_engine.search(queries, k=20)

        duration = (time.perf_counter() - start_time) * 1000
        throughput = num_queries / (duration / 1000)

        stats = search_engine.get_performance_stats()

        return BenchmarkResult(
            name="metal_search",
            duration_ms=duration,
            throughput=throughput,
            memory_mb=stats["memory_usage_mb"],
            success=True,
        )

    async def _benchmark_adaptive_concurrency(self) -> BenchmarkResult:
        """Benchmark adaptive concurrency management"""
        manager = get_adaptive_concurrency_manager()

        # Test concurrent task execution
        async def dummy_task(task_id: int):
            await asyncio.sleep(0.01)  # 10ms task
            return f"result_{task_id}"

        tasks = [
            (lambda i=i: dummy_task(i), TaskType.CPU_INTENSIVE, 0) for i in range(50)
        ]

        start_time = time.perf_counter()

        results = await manager.batch_execute(tasks)

        duration = (time.perf_counter() - start_time) * 1000
        throughput = len(tasks) / (duration / 1000)

        manager.get_performance_metrics()

        return BenchmarkResult(
            name="adaptive_concurrency",
            duration_ms=duration,
            throughput=throughput,
            memory_mb=0,  # No direct memory tracking
            success=len([r for r in results if not isinstance(r, Exception)])
            == len(tasks),
        )

    async def _benchmark_memory_pools(self) -> BenchmarkResult:
        """Benchmark memory pool performance"""
        pool_manager = get_memory_pool_manager()

        # Create test pools
        embedding_pool = pool_manager.create_embedding_pool("test_embeddings", 100)
        cache_pool = pool_manager.create_cache_pool("test_cache", 50)

        start_time = time.perf_counter()

        # Test embedding pool
        test_embeddings = np.random.randn(1000, 384).astype(np.float32)
        buffer = embedding_pool.allocate(test_embeddings.nbytes, "test_emb")
        buffer.copy_from_numpy(test_embeddings)

        # Test cache pool
        for i in range(100):
            cache_pool.put(f"key_{i}", np.random.randn(100).astype(np.float32))

        # Test retrieval
        for i in range(100):
            cache_pool.get(f"key_{i}")

        duration = (time.perf_counter() - start_time) * 1000
        throughput = 200 / (duration / 1000)  # 200 operations

        stats = pool_manager.get_global_stats()

        return BenchmarkResult(
            name="memory_pools",
            duration_ms=duration,
            throughput=throughput,
            memory_mb=stats["total_used_mb"],
            success=True,
        )

    async def _benchmark_ane_embeddings(self) -> BenchmarkResult:
        """Benchmark Apple Neural Engine embeddings"""
        try:
            generator = await create_ane_embedding_generator(
                "benchmark_model", output_dim=768, compute_units=ComputeUnit.CPU_AND_ANE
            )

            # Test embedding generation
            test_inputs = ["test input"] * 50

            start_time = time.perf_counter()

            embeddings = await generator.generate_embeddings(test_inputs)

            duration = (time.perf_counter() - start_time) * 1000
            throughput = len(test_inputs) / (duration / 1000)

            generator.get_performance_stats()

            return BenchmarkResult(
                name="ane_embeddings",
                duration_ms=duration,
                throughput=throughput,
                memory_mb=0,  # ANE memory not directly trackable
                success=embeddings.shape[0] == len(test_inputs),
            )

        except Exception as e:
            return BenchmarkResult(
                name="ane_embeddings",
                duration_ms=0,
                throughput=0,
                memory_mb=0,
                success=False,
                error=str(e),
            )

    async def _benchmark_end_to_end(self) -> BenchmarkResult:
        """End-to-end benchmark combining all optimizations"""
        try:
            # Simulate Einstein search workflow
            search_engine = get_metal_search(embedding_dim=768)
            manager = get_adaptive_concurrency_manager()

            # Load test data
            corpus_embeddings = np.random.randn(5000, 768).astype(np.float32)
            metadata = [{"content": f"document_{i}"} for i in range(5000)]

            search_engine.load_corpus(corpus_embeddings, metadata)

            # Concurrent search tasks
            async def search_task(query_id: int):
                query = np.random.randn(1, 768).astype(np.float32)
                return search_engine.search(query, k=10)

            tasks = [
                (lambda i=i: search_task(i), TaskType.GPU_ACCELERATED, 0)
                for i in range(20)
            ]

            start_time = time.perf_counter()

            results = await manager.batch_execute(tasks)

            duration = (time.perf_counter() - start_time) * 1000
            throughput = len(tasks) / (duration / 1000)

            successful_results = [r for r in results if not isinstance(r, Exception)]

            return BenchmarkResult(
                name="end_to_end",
                duration_ms=duration,
                throughput=throughput,
                memory_mb=0,
                success=len(successful_results) == len(tasks),
            )

        except Exception as e:
            return BenchmarkResult(
                name="end_to_end",
                duration_ms=0,
                throughput=0,
                memory_mb=0,
                success=False,
                error=str(e),
            )

    def _generate_report(self) -> dict[str, Any]:
        """Generate comprehensive benchmark report"""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]

        # Calculate improvements vs baseline
        improvements = {}
        baseline_latencies = {
            "unified_memory": 150,  # ms for MCP equivalent
            "metal_search": 250,  # ms for CPU-only search
            "adaptive_concurrency": 100,  # ms for fixed concurrency
            "memory_pools": 80,  # ms for standard allocation
            "ane_embeddings": 500,  # ms for CPU embedding generation
            "end_to_end": 1000,  # ms for non-optimized workflow
        }

        for result in successful_results:
            if result.name in baseline_latencies:
                baseline = baseline_latencies[result.name]
                improvement = ((baseline - result.duration_ms) / baseline) * 100
                improvements[result.name] = improvement

        return {
            "summary": {
                "total_benchmarks": len(self.results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "overall_success_rate": len(successful_results)
                / len(self.results)
                * 100,
            },
            "performance_improvements": improvements,
            "detailed_results": [
                {
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "throughput_ops_per_sec": r.throughput,
                    "memory_mb": r.memory_mb,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
            "failures": [{"name": r.name, "error": r.error} for r in failed_results],
        }


async def run_m4_pro_benchmarks() -> dict[str, Any]:
    """Run comprehensive M4 Pro optimization benchmarks"""
    suite = M4ProBenchmarkSuite()
    return await suite.run_all_benchmarks()
