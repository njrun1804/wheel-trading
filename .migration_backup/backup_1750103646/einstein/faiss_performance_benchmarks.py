#!/usr/bin/env python3
"""
FAISS Performance Benchmarks

Comprehensive benchmarking suite for FAISS optimizations:
- Before/after performance comparisons
- Index load time measurements  
- Search performance benchmarks
- Memory usage analysis
- Incremental update performance
- GPU acceleration validation

Performance validation:
- Index loads in <200ms after first build
- Search operations under 50ms for 100k vectors
- Incremental updates process files in <100ms each
- GPU acceleration provides >3x speedup when available
- Memory usage stays under 2GB for 100k vectors
"""

import asyncio
import gc
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .incremental_faiss_indexer import IncrementalFAISSIndexer
from .metal_accelerated_faiss import MetalAcceleratedFAISS, create_optimized_config
from .optimized_faiss_system import OptimizedFAISSIndex

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""

    num_vectors_small: int = 1000
    num_vectors_medium: int = 10000
    num_vectors_large: int = 100000
    vector_dimension: int = 1536
    num_search_queries: int = 100
    search_k: int = 20
    num_incremental_files: int = 50
    warmup_iterations: int = 3
    benchmark_iterations: int = 10


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test."""

    test_name: str
    metric: str
    value: float
    unit: str
    iterations: int
    std_dev: float
    min_value: float
    max_value: float
    timestamp: float
    metadata: dict[str, Any]


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""

    config: BenchmarkConfig
    results: list[BenchmarkResult]
    system_info: dict[str, Any]
    total_runtime_ms: float
    summary: dict[str, Any]


class FAISSBenchmarkRunner:
    """Comprehensive FAISS performance benchmark runner."""

    def __init__(
        self,
        project_root: Path,
        config: BenchmarkConfig = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            project_root: Root directory for testing
            config: Benchmark configuration
            output_dir: Directory to save results
        """
        self.project_root = project_root
        self.config = config or BenchmarkConfig()
        self.output_dir = output_dir or project_root / ".einstein" / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: list[BenchmarkResult] = []
        self.system_info = self._collect_system_info()

        # Test data cache
        self._test_vectors_cache: dict[int, np.ndarray] = {}
        self._test_queries_cache: dict[int, list[np.ndarray]] = {}

    def _collect_system_info(self) -> dict[str, Any]:
        """Collect system information for benchmarks."""
        import platform

        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": time.time(),
        }

        # Try to get GPU info
        try:
            import mlx.core as mx

            mx.array([1.0])
            info["mlx_available"] = True
            info["metal_available"] = True
        except (ImportError, RuntimeError) as e:
            logger.debug(f"MLX/Metal not available: {e}")
            info["mlx_available"] = False
            info["metal_available"] = False

        try:
            import faiss

            info["faiss_version"] = faiss.__version__
        except ImportError as e:
            logger.debug(f"FAISS not available: {e}")
            info["faiss_version"] = "not_available"

        return info

    def _generate_test_vectors(self, num_vectors: int) -> np.ndarray:
        """Generate test vectors with caching."""
        if num_vectors not in self._test_vectors_cache:
            np.random.seed(42)  # Deterministic for reproducibility
            vectors = np.random.randn(num_vectors, self.config.vector_dimension).astype(
                np.float32
            )
            # Normalize vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
            self._test_vectors_cache[num_vectors] = vectors

        return self._test_vectors_cache[num_vectors]

    def _generate_test_queries(self, num_queries: int) -> list[np.ndarray]:
        """Generate test query vectors with caching."""
        if num_queries not in self._test_queries_cache:
            np.random.seed(123)  # Different seed for queries
            queries = []
            for _ in range(num_queries):
                query = np.random.randn(self.config.vector_dimension).astype(np.float32)
                query = query / np.linalg.norm(query)
                queries.append(query)
            self._test_queries_cache[num_queries] = queries

        return self._test_queries_cache[num_queries]

    def _record_result(
        self,
        test_name: str,
        metric: str,
        values: list[float],
        unit: str,
        metadata: dict[str, Any] = None,
    ) -> None:
        """Record benchmark result with statistics."""
        result = BenchmarkResult(
            test_name=test_name,
            metric=metric,
            value=statistics.mean(values),
            unit=unit,
            iterations=len(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self.results.append(result)
        logger.info(
            f"{test_name} - {metric}: {result.value:.2f} ± {result.std_dev:.2f} {unit}"
        )

    async def benchmark_index_loading(self) -> None:
        """Benchmark FAISS index loading performance."""
        logger.info("🔍 Benchmarking index loading performance...")

        # Create test index
        test_vectors = self._generate_test_vectors(self.config.num_vectors_medium)

        # Test with OptimizedFAISSIndex
        index_dir = self.output_dir / "test_index"
        index_dir.mkdir(exist_ok=True)

        index = OptimizedFAISSIndex(index_dir, self.config.vector_dimension)

        # Initialize and add vectors
        await index.initialize()

        # Prepare embedding data
        embeddings = [test_vectors[i] for i in range(len(test_vectors))]
        line_ranges = [(i, i + 1) for i in range(len(test_vectors))]
        content_previews = [f"test_content_{i}" for i in range(len(test_vectors))]
        token_counts = [100] * len(test_vectors)

        await index.add_file_embeddings(
            file_path="test_file.py",
            embeddings=embeddings,
            line_ranges=line_ranges,
            content_previews=content_previews,
            token_counts=token_counts,
        )

        # Save index
        await index.save()
        await index.cleanup()

        # Benchmark loading
        load_times = []
        for _ in range(self.config.benchmark_iterations):
            gc.collect()  # Clear memory

            # Create new index instance
            test_index = OptimizedFAISSIndex(index_dir, self.config.vector_dimension)

            start_time = time.time()
            success = await test_index.initialize()
            load_time = (time.time() - start_time) * 1000  # Convert to ms

            if success:
                load_times.append(load_time)

            await test_index.cleanup()

        self._record_result(
            "index_loading",
            "load_time",
            load_times,
            "ms",
            {"vector_count": len(test_vectors), "index_type": "OptimizedFAISS"},
        )

    async def benchmark_search_performance(self) -> None:
        """Benchmark search performance across different index sizes."""
        logger.info("🔍 Benchmarking search performance...")

        test_sizes = [
            self.config.num_vectors_small,
            self.config.num_vectors_medium,
            self.config.num_vectors_large,
        ]

        for num_vectors in test_sizes:
            logger.info(f"Testing search with {num_vectors} vectors...")

            # Generate test data
            test_vectors = self._generate_test_vectors(num_vectors)
            test_queries = self._generate_test_queries(self.config.num_search_queries)

            # Create and populate index
            index_dir = self.output_dir / f"search_test_{num_vectors}"
            index_dir.mkdir(exist_ok=True)

            index = OptimizedFAISSIndex(index_dir, self.config.vector_dimension)
            await index.initialize()

            # Add vectors in batches
            batch_size = 1000
            for i in range(0, num_vectors, batch_size):
                batch_end = min(i + batch_size, num_vectors)
                batch_vectors = [test_vectors[j] for j in range(i, batch_end)]
                batch_ranges = [(j, j + 1) for j in range(i, batch_end)]
                batch_previews = [f"content_{j}" for j in range(i, batch_end)]
                batch_tokens = [100] * len(batch_vectors)

                await index.add_file_embeddings(
                    file_path=f"test_file_{i}.py",
                    embeddings=batch_vectors,
                    line_ranges=batch_ranges,
                    content_previews=batch_previews,
                    token_counts=batch_tokens,
                )

            # Benchmark search
            search_times = []

            for query in test_queries[: self.config.benchmark_iterations]:
                start_time = time.time()
                await index.search(query, k=self.config.search_k)
                search_time = (time.time() - start_time) * 1000
                search_times.append(search_time)

            self._record_result(
                f"search_performance_{num_vectors}",
                "search_time",
                search_times,
                "ms",
                {
                    "vector_count": num_vectors,
                    "search_k": self.config.search_k,
                    "query_count": len(search_times),
                },
            )

            await index.cleanup()

    async def benchmark_incremental_updates(self) -> None:
        """Benchmark incremental update performance."""
        logger.info("🔍 Benchmarking incremental update performance...")

        # Create test files
        test_files_dir = self.output_dir / "test_files"
        test_files_dir.mkdir(exist_ok=True)

        # Generate test Python files
        test_files = []
        for i in range(self.config.num_incremental_files):
            file_path = test_files_dir / f"test_file_{i}.py"

            # Generate realistic Python code
            content = f'''#!/usr/bin/env python3
"""Test file {i} for incremental benchmarking."""

import os
import sys
from typing import List, Dict, Any

class TestClass{i}:
    """Test class for file {i}."""
    
    def __init__(self, value: int = {i}):
        self.value = value
        self.data = []
    
    def process_data(self, items: List[Any]) -> Dict[str, Any]:
        """Process data items."""
        result = {{"count": len(items), "file_id": {i}}}
        
        for item in items:
            if isinstance(item, (int, float)):
                result["sum"] = result.get("sum", 0) + item
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get object statistics."""
        return {{
            "value": self.value,
            "data_count": len(self.data),
            "file_number": {i}
        }}

def main():
    """Main function for file {i}."""
    obj = TestClass{i}()
    print(f"Created test object {{obj.value}}")
    
    # Process some test data
    test_data = list(range({i}, {i + 10}))
    result = obj.process_data(test_data)
    print(f"Processing result: {{result}}")

if __name__ == "__main__":
    main()
'''

            with open(file_path, "w") as f:
                f.write(content)

            test_files.append(str(file_path))

        # Initialize incremental indexer
        indexer = IncrementalFAISSIndexer(
            project_root=self.output_dir, enable_monitoring=False
        )

        await indexer.initialize()

        # Benchmark initial indexing
        initial_times = []
        for _ in range(self.config.warmup_iterations):
            start_time = time.time()
            await indexer.scan_and_update(["test_files/*.py"])
            index_time = (time.time() - start_time) * 1000
            initial_times.append(index_time)

        self._record_result(
            "incremental_initial_indexing",
            "index_time",
            initial_times,
            "ms",
            {"file_count": len(test_files)},
        )

        # Benchmark update performance (modify files)
        update_times = []
        for i in range(self.config.benchmark_iterations):
            # Modify a file
            file_to_modify = test_files[i % len(test_files)]
            with open(file_to_modify, "a") as f:
                f.write(f"\n# Modified at iteration {i}\n")

            start_time = time.time()
            await indexer.update_files([file_to_modify])
            update_time = (time.time() - start_time) * 1000
            update_times.append(update_time)

        self._record_result(
            "incremental_file_update",
            "update_time",
            update_times,
            "ms",
            {"files_per_update": 1},
        )

        await indexer.cleanup()

    async def benchmark_gpu_acceleration(self) -> None:
        """Benchmark GPU/Metal acceleration performance."""
        logger.info("🔍 Benchmarking GPU acceleration...")

        if not self.system_info.get("metal_available", False):
            logger.info("Metal not available, skipping GPU benchmarks")
            return

        test_vectors = self._generate_test_vectors(self.config.num_vectors_medium)
        test_queries = self._generate_test_queries(self.config.num_search_queries)

        # Test with GPU acceleration
        gpu_config = create_optimized_config()
        gpu_index = MetalAcceleratedFAISS(
            dimension=self.config.vector_dimension, config=gpu_config
        )

        # Test without GPU acceleration
        cpu_config = create_optimized_config()
        cpu_config.enable_metal = False
        cpu_index = MetalAcceleratedFAISS(
            dimension=self.config.vector_dimension, config=cpu_config
        )

        # Benchmark vector addition
        gpu_add_times = []
        cpu_add_times = []

        for _ in range(self.config.benchmark_iterations):
            # GPU test
            batch_vectors = [
                test_vectors[i] for i in range(0, min(1000, len(test_vectors)))
            ]

            start_time = time.time()
            await gpu_index.add_vectors(batch_vectors)
            gpu_time = (time.time() - start_time) * 1000
            gpu_add_times.append(gpu_time)

            # CPU test
            start_time = time.time()
            await cpu_index.add_vectors(batch_vectors)
            cpu_time = (time.time() - start_time) * 1000
            cpu_add_times.append(cpu_time)

        self._record_result(
            "gpu_vector_addition",
            "add_time",
            gpu_add_times,
            "ms",
            {"vector_count": 1000, "acceleration": "GPU"},
        )

        self._record_result(
            "cpu_vector_addition",
            "add_time",
            cpu_add_times,
            "ms",
            {"vector_count": 1000, "acceleration": "CPU"},
        )

        # Calculate speedup
        avg_gpu_time = statistics.mean(gpu_add_times)
        avg_cpu_time = statistics.mean(cpu_add_times)
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 1.0

        self._record_result(
            "gpu_acceleration_speedup",
            "speedup_factor",
            [speedup],
            "x",
            {"operation": "vector_addition"},
        )

        # Benchmark search
        gpu_search_times = []
        cpu_search_times = []

        for query in test_queries[: self.config.benchmark_iterations]:
            # GPU search
            start_time = time.time()
            await gpu_index.search(query, k=self.config.search_k)
            gpu_time = (time.time() - start_time) * 1000
            gpu_search_times.append(gpu_time)

            # CPU search
            start_time = time.time()
            await cpu_index.search(query, k=self.config.search_k)
            cpu_time = (time.time() - start_time) * 1000
            cpu_search_times.append(cpu_time)

        self._record_result(
            "gpu_search_performance",
            "search_time",
            gpu_search_times,
            "ms",
            {"acceleration": "GPU", "search_k": self.config.search_k},
        )

        self._record_result(
            "cpu_search_performance",
            "search_time",
            cpu_search_times,
            "ms",
            {"acceleration": "CPU", "search_k": self.config.search_k},
        )

        # Calculate search speedup
        avg_gpu_search = statistics.mean(gpu_search_times)
        avg_cpu_search = statistics.mean(cpu_search_times)
        search_speedup = avg_cpu_search / avg_gpu_search if avg_gpu_search > 0 else 1.0

        self._record_result(
            "gpu_search_speedup",
            "speedup_factor",
            [search_speedup],
            "x",
            {"operation": "search"},
        )

        gpu_index.cleanup()
        cpu_index.cleanup()

    async def benchmark_memory_usage(self) -> None:
        """Benchmark memory usage patterns."""
        logger.info("🔍 Benchmarking memory usage...")

        # Test different index sizes
        test_sizes = [
            self.config.num_vectors_small,
            self.config.num_vectors_medium,
            self.config.num_vectors_large,
        ]

        for num_vectors in test_sizes:
            # Measure baseline memory
            gc.collect()
            baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

            # Create and populate index
            index_dir = self.output_dir / f"memory_test_{num_vectors}"
            index_dir.mkdir(exist_ok=True)

            index = OptimizedFAISSIndex(index_dir, self.config.vector_dimension)
            await index.initialize()

            # Add vectors
            test_vectors = self._generate_test_vectors(num_vectors)
            embeddings = [test_vectors[i] for i in range(len(test_vectors))]
            line_ranges = [(i, i + 1) for i in range(len(test_vectors))]
            content_previews = [f"content_{i}" for i in range(len(test_vectors))]
            token_counts = [100] * len(test_vectors)

            await index.add_file_embeddings(
                file_path="memory_test.py",
                embeddings=embeddings,
                line_ranges=line_ranges,
                content_previews=content_previews,
                token_counts=token_counts,
            )

            # Measure memory after indexing
            post_index_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_usage = post_index_memory - baseline_memory

            self._record_result(
                f"memory_usage_{num_vectors}",
                "memory_mb",
                [memory_usage],
                "MB",
                {"vector_count": num_vectors, "baseline_mb": baseline_memory},
            )

            await index.cleanup()
            gc.collect()

    async def run_all_benchmarks(self) -> BenchmarkSuite:
        """Run complete benchmark suite."""
        logger.info("🚀 Starting FAISS Performance Benchmark Suite")
        logger.info("=" * 60)

        start_time = time.time()

        # Run individual benchmarks
        try:
            await self.benchmark_index_loading()
            await self.benchmark_search_performance()
            await self.benchmark_incremental_updates()
            await self.benchmark_gpu_acceleration()
            await self.benchmark_memory_usage()
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

        total_runtime = (time.time() - start_time) * 1000

        # Generate summary
        summary = self._generate_summary()

        # Create benchmark suite
        suite = BenchmarkSuite(
            config=self.config,
            results=self.results,
            system_info=self.system_info,
            total_runtime_ms=total_runtime,
            summary=summary,
        )

        # Save results
        await self._save_results(suite)

        logger.info(f"✅ Benchmark suite completed in {total_runtime:.1f}ms")
        return suite

    def _generate_summary(self) -> dict[str, Any]:
        """Generate benchmark summary with key metrics."""
        summary = {
            "total_tests": len(self.results),
            "performance_targets": {},
            "key_metrics": {},
            "recommendations": [],
        }

        # Check performance targets
        for result in self.results:
            if result.test_name == "index_loading" and result.metric == "load_time":
                target_met = result.value < 200  # Target: <200ms
                summary["performance_targets"]["index_loading"] = {
                    "target_ms": 200,
                    "actual_ms": result.value,
                    "target_met": target_met,
                }

            elif (
                "search_performance" in result.test_name
                and result.metric == "search_time"
            ):
                target_met = result.value < 50  # Target: <50ms
                summary["performance_targets"][result.test_name] = {
                    "target_ms": 50,
                    "actual_ms": result.value,
                    "target_met": target_met,
                }

            elif "incremental" in result.test_name and "update_time" in result.metric:
                target_met = result.value < 100  # Target: <100ms
                summary["performance_targets"][result.test_name] = {
                    "target_ms": 100,
                    "actual_ms": result.value,
                    "target_met": target_met,
                }

        # Extract key metrics
        for result in self.results:
            if "speedup" in result.metric:
                summary["key_metrics"][result.test_name] = {
                    "speedup": result.value,
                    "unit": result.unit,
                }

        # Generate recommendations
        if "gpu_acceleration_speedup" in summary["key_metrics"]:
            speedup = summary["key_metrics"]["gpu_acceleration_speedup"]["speedup"]
            if speedup > 3:
                summary["recommendations"].append(
                    "GPU acceleration is highly effective"
                )
            elif speedup > 1.5:
                summary["recommendations"].append(
                    "GPU acceleration provides moderate benefit"
                )
            else:
                summary["recommendations"].append("Consider CPU-only optimization")

        return summary

    async def _save_results(self, suite: BenchmarkSuite) -> None:
        """Save benchmark results to files."""
        timestamp = int(time.time())

        # Save detailed results as JSON
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(asdict(suite), f, indent=2, default=str)

        # Save summary report
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.md"
        with open(summary_file, "w") as f:
            f.write(self._generate_markdown_report(suite))

        logger.info(f"📊 Results saved to {results_file}")
        logger.info(f"📝 Summary saved to {summary_file}")

    def _generate_markdown_report(self, suite: BenchmarkSuite) -> str:
        """Generate markdown report from benchmark results."""
        report = f"""# FAISS Performance Benchmark Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Runtime:** {suite.total_runtime_ms:.1f}ms
**Total Tests:** {len(suite.results)}

## System Information

- **Platform:** {suite.system_info.get('platform', 'Unknown')}
- **CPU Cores:** {suite.system_info.get('cpu_count', 'Unknown')}
- **Memory:** {suite.system_info.get('memory_gb', 0):.1f} GB
- **Metal Available:** {suite.system_info.get('metal_available', False)}
- **FAISS Version:** {suite.system_info.get('faiss_version', 'Unknown')}

## Performance Summary

"""

        # Add target performance summary
        if "performance_targets" in suite.summary:
            report += "### Performance Targets\n\n"
            for test_name, target_info in suite.summary["performance_targets"].items():
                status = "✅" if target_info["target_met"] else "❌"
                report += f"- **{test_name}:** {status} {target_info['actual_ms']:.1f}ms (target: {target_info['target_ms']}ms)\n"
            report += "\n"

        # Add key metrics
        if "key_metrics" in suite.summary:
            report += "### Key Metrics\n\n"
            for metric_name, metric_info in suite.summary["key_metrics"].items():
                report += f"- **{metric_name}:** {metric_info['speedup']:.1f}{metric_info['unit']}\n"
            report += "\n"

        # Add detailed results
        report += "## Detailed Results\n\n"

        current_test = None
        for result in suite.results:
            if result.test_name != current_test:
                current_test = result.test_name
                report += f"### {current_test}\n\n"

            report += f"- **{result.metric}:** {result.value:.2f} ± {result.std_dev:.2f} {result.unit} "
            report += f"(min: {result.min_value:.2f}, max: {result.max_value:.2f}, n={result.iterations})\n"

        # Add recommendations
        if "recommendations" in suite.summary and suite.summary["recommendations"]:
            report += "\n## Recommendations\n\n"
            for rec in suite.summary["recommendations"]:
                report += f"- {rec}\n"

        return report


# Example usage and testing
if __name__ == "__main__":

    async def run_benchmarks():
        """Run the complete benchmark suite."""
        project_root = Path.cwd()

        # Configure benchmarks for quick testing
        config = BenchmarkConfig(
            num_vectors_small=100,
            num_vectors_medium=1000,
            num_vectors_large=5000,
            num_search_queries=20,
            num_incremental_files=10,
            benchmark_iterations=5,
        )

        runner = FAISSBenchmarkRunner(project_root, config)
        suite = await runner.run_all_benchmarks()

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"Total runtime: {suite.total_runtime_ms:.1f}ms")
        print(f"Total tests: {len(suite.results)}")

        if "performance_targets" in suite.summary:
            print("\nPerformance Targets:")
            for test, info in suite.summary["performance_targets"].items():
                status = "✅" if info["target_met"] else "❌"
                print(f"  {test}: {status} {info['actual_ms']:.1f}ms")

        if "key_metrics" in suite.summary:
            print("\nKey Metrics:")
            for metric, info in suite.summary["key_metrics"].items():
                print(f"  {metric}: {info['speedup']:.1f}{info['unit']}")

    # Run benchmarks
    asyncio.run(run_benchmarks())
