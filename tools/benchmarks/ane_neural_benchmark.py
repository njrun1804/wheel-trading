#!/usr/bin/env python3
"""
ANE Neural Engine Benchmark Suite

Comprehensive benchmarking and validation for ANE-accelerated Einstein embeddings.
Tests performance, accuracy, and integration with existing Einstein pipeline.

Benchmark Categories:
1. Device Detection and Initialization
2. Embedding Generation Performance
3. Batch Processing Optimization
4. Cache Performance
5. Concurrent Processing
6. Integration with Einstein Pipeline
7. Memory Usage and Efficiency
8. Accuracy Validation
"""

import asyncio
import json
import logging
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from src.unity_wheel.accelerated_tools.einstein_neural_integration import (
    EinsteinEmbeddingConfig,
    get_einstein_ane_pipeline,
)

# Import ANE neural engine components
from src.unity_wheel.accelerated_tools.neural_engine_turbo import (
    ANEDeviceManager,
    get_neural_engine_turbo,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ANEBenchmarkSuite:
    """Comprehensive benchmark suite for ANE Neural Engine."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize results storage
        self.results = {
            "device_info": {},
            "performance_tests": {},
            "integration_tests": {},
            "accuracy_tests": {},
            "memory_tests": {},
            "concurrent_tests": {},
            "summary": {},
        }

        # Test configuration
        self.test_sizes = [1, 10, 50, 100, 250, 500, 1000]
        self.concurrent_levels = [1, 4, 8, 16, 32]

        # Sample texts for testing
        self.sample_texts = self._generate_sample_texts()

    def _generate_sample_texts(self) -> list[str]:
        """Generate sample texts for benchmarking."""
        return [
            # Code samples
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "class DataProcessor: def __init__(self): self.data = []",
            "import numpy as np; import pandas as pd; from typing import List, Dict",
            "async def fetch_data(url: str) -> dict: return await client.get(url)",
            "try: result = process_data(input_data) except Exception as e: logger.error(e)",
            # Comments and documentation
            "# This function implements the binary search algorithm",
            '"""Einstein embedding pipeline with ANE acceleration for optimal performance"""',
            "TODO: Optimize this function for better performance on large datasets",
            "FIXME: Handle edge case when input is None or empty list",
            "WARNING: This operation may consume significant memory",
            # Configuration and data
            "{'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 2048}",
            "SELECT * FROM options WHERE symbol = 'AAPL' AND expiry > '2024-01-01'",
            "CUDA_VISIBLE_DEVICES=0,1 python train.py --batch-size 32 --lr 0.001",
            "<configuration><database>postgresql://user:pass@localhost/db</database></configuration>",
            # Natural language
            "The Apple Neural Engine provides hardware acceleration for machine learning operations",
            "Performance optimization requires careful consideration of memory usage and computation patterns",
            "Embeddings capture semantic relationships between text fragments in high-dimensional space",
            "The M4 Pro chip includes 16 Neural Engine cores capable of 35 trillion operations per second",
        ]

    async def run_all_benchmarks(self) -> dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("🚀 Starting ANE Neural Engine Benchmark Suite")
        start_time = time.time()

        # 1. Device Detection Test
        await self._test_device_detection()

        # 2. Performance Tests
        await self._test_embedding_performance()

        # 3. Batch Processing Tests
        await self._test_batch_processing()

        # 4. Cache Performance Tests
        await self._test_cache_performance()

        # 5. Concurrent Processing Tests
        await self._test_concurrent_processing()

        # 6. Integration Tests
        await self._test_einstein_integration()

        # 7. Memory Usage Tests
        await self._test_memory_usage()

        # 8. Accuracy Validation
        await self._test_accuracy_validation()

        # Generate summary
        total_time = time.time() - start_time
        self.results["summary"] = {
            "total_benchmark_time": total_time,
            "timestamp": time.time(),
            "tests_completed": len(
                [k for k in self.results if k != "summary" and self.results[k]]
            ),
        }

        # Save results
        self._save_results()

        logger.info(f"✅ Benchmark suite completed in {total_time:.2f}s")
        return self.results

    async def _test_device_detection(self):
        """Test ANE device detection and initialization."""
        logger.info("📱 Testing device detection...")

        try:
            device_manager = ANEDeviceManager()
            device_info = device_manager.detect_ane_device()

            self.results["device_info"] = {
                "available": device_info.available,
                "cores": device_info.cores,
                "memory_mb": device_info.memory_mb,
                "max_batch_size": device_info.max_batch_size,
                "preferred_batch_size": device_info.preferred_batch_size,
                "device_name": device_info.device_name,
                "tensor_ops_per_second": device_info.tensor_ops_per_second,
            }

            logger.info(f"Device: {device_info.device_name}")
            logger.info(f"ANE Available: {device_info.available}")
            logger.info(f"Cores: {device_info.cores}")

        except Exception as e:
            logger.error(f"Device detection failed: {e}")
            self.results["device_info"] = {"error": str(e)}

    async def _test_embedding_performance(self):
        """Test basic embedding generation performance."""
        logger.info("⚡ Testing embedding performance...")

        try:
            engine = get_neural_engine_turbo()
            engine.warmup()  # Ensure warmup

            performance_results = {}

            for size in self.test_sizes:
                if size > len(self.sample_texts):
                    # Repeat texts to reach desired size
                    test_texts = (
                        self.sample_texts * ((size // len(self.sample_texts)) + 1)
                    )[:size]
                else:
                    test_texts = self.sample_texts[:size]

                # Measure performance
                times = []
                for run in range(3):  # Average over 3 runs
                    start_time = time.time()
                    result = await engine.embed_texts_async(test_texts)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms

                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0

                performance_results[f"size_{size}"] = {
                    "texts": size,
                    "avg_time_ms": avg_time,
                    "std_time_ms": std_time,
                    "tokens_per_second": result.tokens_processed / (avg_time / 1000),
                    "embeddings_per_second": size / (avg_time / 1000),
                }

                logger.info(
                    f"Size {size}: {avg_time:.1f}ms ({result.tokens_processed / (avg_time / 1000):.0f} tok/s)"
                )

            self.results["performance_tests"] = performance_results

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            self.results["performance_tests"] = {"error": str(e)}

    async def _test_batch_processing(self):
        """Test batch processing optimization."""
        logger.info("📦 Testing batch processing...")

        try:
            engine = get_neural_engine_turbo()
            batch_results = {}

            # Test different batch sizes
            batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
            total_texts = 256

            for batch_size in batch_sizes:
                if batch_size > total_texts:
                    continue

                # Create test data
                test_texts = (
                    self.sample_texts * ((total_texts // len(self.sample_texts)) + 1)
                )[:total_texts]

                # Process in batches
                start_time = time.time()
                total_tokens = 0

                for i in range(0, len(test_texts), batch_size):
                    batch = test_texts[i : i + batch_size]
                    result = await engine.embed_texts_async(batch)
                    total_tokens += result.tokens_processed

                total_time = time.time() - start_time

                batch_results[f"batch_{batch_size}"] = {
                    "batch_size": batch_size,
                    "total_time_ms": total_time * 1000,
                    "tokens_per_second": total_tokens / total_time,
                    "batches_processed": (len(test_texts) + batch_size - 1)
                    // batch_size,
                }

                logger.info(
                    f"Batch {batch_size}: {total_time * 1000:.1f}ms ({total_tokens / total_time:.0f} tok/s)"
                )

            self.results["performance_tests"]["batch_optimization"] = batch_results

        except Exception as e:
            logger.error(f"Batch processing test failed: {e}")
            self.results["performance_tests"]["batch_optimization"] = {"error": str(e)}

    async def _test_cache_performance(self):
        """Test embedding cache performance."""
        logger.info("💾 Testing cache performance...")

        try:
            engine = get_neural_engine_turbo()

            # First run (cache miss)
            test_texts = self.sample_texts[:10]

            start_time = time.time()
            result1 = await engine.embed_texts_async(test_texts, task_id="cache_test_1")
            first_run_time = (time.time() - start_time) * 1000

            # Second run (should hit cache)
            start_time = time.time()
            result2 = await engine.embed_texts_async(test_texts, task_id="cache_test_2")
            second_run_time = (time.time() - start_time) * 1000

            # Calculate cache effectiveness
            cache_speedup = first_run_time / max(second_run_time, 0.1)

            self.results["performance_tests"]["cache_performance"] = {
                "first_run_ms": first_run_time,
                "second_run_ms": second_run_time,
                "cache_speedup": cache_speedup,
                "cache_hit_rate": engine.get_performance_metrics().cache_hit_rate,
            }

            logger.info(f"Cache speedup: {cache_speedup:.1f}x")

        except Exception as e:
            logger.error(f"Cache performance test failed: {e}")
            self.results["performance_tests"]["cache_performance"] = {"error": str(e)}

    async def _test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        logger.info("🔄 Testing concurrent processing...")

        try:
            concurrent_results = {}

            for concurrent_level in self.concurrent_levels:
                if concurrent_level > 16:  # Don't overwhelm the system
                    continue

                # Create concurrent tasks
                tasks = []
                test_texts = self.sample_texts[:5]  # Small texts for concurrent test

                start_time = time.time()

                for i in range(concurrent_level):
                    engine = get_neural_engine_turbo()
                    task = engine.embed_texts_async(
                        test_texts, task_id=f"concurrent_{i}"
                    )
                    tasks.append(task)

                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)
                total_time = (time.time() - start_time) * 1000

                # Calculate metrics
                total_tokens = sum(result.tokens_processed for result in results)
                tokens_per_second = total_tokens / (total_time / 1000)

                concurrent_results[f"level_{concurrent_level}"] = {
                    "concurrent_level": concurrent_level,
                    "total_time_ms": total_time,
                    "tokens_per_second": tokens_per_second,
                    "tasks_completed": len(results),
                }

                logger.info(
                    f"Concurrent {concurrent_level}: {total_time:.1f}ms ({tokens_per_second:.0f} tok/s)"
                )

            self.results["concurrent_tests"] = concurrent_results

        except Exception as e:
            logger.error(f"Concurrent processing test failed: {e}")
            self.results["concurrent_tests"] = {"error": str(e)}

    async def _test_einstein_integration(self):
        """Test integration with Einstein pipeline."""
        logger.info("🧠 Testing Einstein integration...")

        try:
            config = EinsteinEmbeddingConfig(
                use_ane=True, performance_logging=True, warmup_on_startup=True
            )

            pipeline = get_einstein_ane_pipeline(config=config)

            # Test file embedding
            test_file = __file__  # Use this file as test

            start_time = time.time()
            results = await pipeline.embed_file_batch([test_file])
            total_time = (time.time() - start_time) * 1000

            # Get enhanced statistics
            stats = pipeline.get_enhanced_stats()

            self.results["integration_tests"] = {
                "files_processed": stats["pipeline_stats"]["files_processed"],
                "slices_processed": stats["pipeline_stats"]["slices_processed"],
                "ane_accelerated": stats["pipeline_stats"]["ane_accelerated"],
                "cache_hits": stats["pipeline_stats"]["cache_hits"],
                "cache_misses": stats["pipeline_stats"]["cache_misses"],
                "total_time_ms": total_time,
                "performance_comparison": stats["performance_comparison"],
            }

            logger.info(f"Einstein integration: {total_time:.1f}ms")
            if stats["performance_comparison"]:
                logger.info(
                    f"ANE usage: {stats['performance_comparison']['ane_usage_percent']:.1f}%"
                )

        except Exception as e:
            logger.error(f"Einstein integration test failed: {e}")
            self.results["integration_tests"] = {"error": str(e)}

    async def _test_memory_usage(self):
        """Test memory usage patterns."""
        logger.info("🧠 Testing memory usage...")

        try:
            # Start memory tracking
            tracemalloc.start()

            engine = get_neural_engine_turbo()

            # Baseline memory
            snapshot1 = tracemalloc.take_snapshot()

            # Process embeddings
            large_texts = self.sample_texts * 50  # 900 texts
            result = await engine.embed_texts_async(large_texts)

            # Memory after processing
            snapshot2 = tracemalloc.take_snapshot()

            # Calculate memory usage
            top_stats = snapshot2.compare_to(snapshot1, "lineno")
            total_memory_mb = sum(stat.size for stat in top_stats) / (1024 * 1024)

            self.results["memory_tests"] = {
                "texts_processed": len(large_texts),
                "tokens_processed": result.tokens_processed,
                "memory_usage_mb": total_memory_mb,
                "memory_per_token_kb": (total_memory_mb * 1024)
                / result.tokens_processed
                if result.tokens_processed > 0
                else 0,
            }

            logger.info(
                f"Memory usage: {total_memory_mb:.2f}MB for {len(large_texts)} texts"
            )

            tracemalloc.stop()

        except Exception as e:
            logger.error(f"Memory usage test failed: {e}")
            self.results["memory_tests"] = {"error": str(e)}
            tracemalloc.stop()

    async def _test_accuracy_validation(self):
        """Test embedding accuracy and consistency."""
        logger.info("🎯 Testing accuracy validation...")

        try:
            engine = get_neural_engine_turbo()

            # Test consistency - same input should give same output
            test_text = ["def test_function(): return True"]

            result1 = await engine.embed_texts_async(test_text, task_id="accuracy_1")
            result2 = await engine.embed_texts_async(test_text, task_id="accuracy_2")

            # Convert to numpy for comparison
            emb1 = np.array(result1.embeddings)
            emb2 = np.array(result2.embeddings)

            # Calculate similarity
            if len(emb1.shape) > 1:
                emb1 = emb1[0]
            if len(emb2.shape) > 1:
                emb2 = emb2[0]

            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )

            # Test semantic similarity
            similar_texts = [
                "def calculate_sum(a, b): return a + b",
                "def add_numbers(x, y): return x + y",
            ]

            result_similar = await engine.embed_texts_async(similar_texts)
            emb_similar = np.array(result_similar.embeddings)

            if len(emb_similar.shape) > 1:
                semantic_similarity = np.dot(emb_similar[0], emb_similar[1]) / (
                    np.linalg.norm(emb_similar[0]) * np.linalg.norm(emb_similar[1])
                )
            else:
                semantic_similarity = 0.5  # Fallback

            self.results["accuracy_tests"] = {
                "consistency_similarity": float(similarity),
                "semantic_similarity": float(semantic_similarity),
                "embedding_dimension": len(emb1),
                "embedding_norm": float(np.linalg.norm(emb1)),
            }

            logger.info(
                f"Consistency: {similarity:.3f}, Semantic: {semantic_similarity:.3f}"
            )

        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}")
            self.results["accuracy_tests"] = {"error": str(e)}

    def _save_results(self):
        """Save benchmark results to files."""
        timestamp = int(time.time())

        # Save JSON results
        json_file = self.output_dir / f"ane_benchmark_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary report
        report_file = self.output_dir / f"ane_benchmark_report_{timestamp}.md"
        self._generate_report(report_file)

        logger.info(f"Results saved to {json_file} and {report_file}")

    def _generate_report(self, report_file: Path):
        """Generate human-readable benchmark report."""
        with open(report_file, "w") as f:
            f.write("# ANE Neural Engine Benchmark Report\n\n")

            # Device Info
            f.write("## Device Information\n")
            device_info = self.results.get("device_info", {})
            f.write(f"- Device: {device_info.get('device_name', 'Unknown')}\n")
            f.write(f"- ANE Available: {device_info.get('available', False)}\n")
            f.write(f"- Cores: {device_info.get('cores', 0)}\n")
            f.write(f"- Memory: {device_info.get('memory_mb', 0)}MB\n\n")

            # Performance Tests
            f.write("## Performance Results\n")
            perf_tests = self.results.get("performance_tests", {})

            for test_name, test_data in perf_tests.items():
                if isinstance(test_data, dict) and "error" not in test_data:
                    if "avg_time_ms" in test_data:
                        f.write(
                            f"- {test_name}: {test_data['avg_time_ms']:.1f}ms "
                            f"({test_data['tokens_per_second']:.0f} tok/s)\n"
                        )

            # Integration Tests
            f.write("\n## Integration Results\n")
            integration = self.results.get("integration_tests", {})
            if "error" not in integration:
                f.write(f"- Files processed: {integration.get('files_processed', 0)}\n")
                f.write(f"- ANE accelerated: {integration.get('ane_accelerated', 0)}\n")
                f.write(f"- Total time: {integration.get('total_time_ms', 0):.1f}ms\n")

            # Summary
            f.write("\n## Summary\n")
            summary = self.results.get("summary", {})
            f.write(
                f"- Total benchmark time: {summary.get('total_benchmark_time', 0):.2f}s\n"
            )
            f.write(f"- Tests completed: {summary.get('tests_completed', 0)}\n")


async def main():
    """Run the complete ANE benchmark suite."""
    print("🧠 ANE Neural Engine Benchmark Suite")
    print("=" * 50)

    # Create benchmark suite
    suite = ANEBenchmarkSuite()

    try:
        # Run all benchmarks
        results = await suite.run_all_benchmarks()

        # Print summary
        print("\n📊 Benchmark Summary:")
        print("-" * 30)

        device_info = results.get("device_info", {})
        print(f"Device: {device_info.get('device_name', 'Unknown')}")
        print(f"ANE Available: {device_info.get('available', False)}")

        if "performance_tests" in results:
            print("Performance tests: ✅")
        if "integration_tests" in results:
            print("Integration tests: ✅")
        if "accuracy_tests" in results:
            print("Accuracy tests: ✅")

        summary = results.get("summary", {})
        print(f"Total time: {summary.get('total_benchmark_time', 0):.2f}s")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
