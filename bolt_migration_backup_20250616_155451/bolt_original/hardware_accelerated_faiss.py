#!/usr/bin/env python3
"""
MLX Metal-Accelerated FAISS Vector Operations for M4 Pro

Integrates MLX Metal acceleration with FAISS for optimal vector operations:
- Uses Metal GPU for vector computation and similarity search
- Intelligent CPU/GPU workload distribution
- Memory-optimized vector storage and retrieval
- Hardware-aware batch processing

Optimized for:
- M4 Pro: 8 P-cores + 4 E-cores + 20-core GPU
- 24GB unified memory
- Metal performance shaders
"""

import asyncio
import logging
import time
from typing import Any

import mlx.core as mx
import numpy as np

# Optional FAISS import with fallback
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .gpu_acceleration import gpuify

logger = logging.getLogger(__name__)


class MetalFAISSAccelerator:
    """MLX Metal-accelerated FAISS operations optimized for M4 Pro."""

    def __init__(self, embedding_dim: int = 768):
        """Initialize Metal-accelerated FAISS system.

        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.device = mx.gpu if mx.metal.is_available() else mx.cpu

        # FAISS index (fallback to brute force if FAISS unavailable)
        self.faiss_index = None
        self.fallback_vectors = None  # For non-FAISS systems

        # Metal GPU optimization settings
        self.metal_available = mx.metal.is_available()
        self.batch_size = 512 if self.metal_available else 128
        self.max_vectors = 1000000  # 1M vectors max

        # Performance tracking
        self.stats = {
            "vectors_indexed": 0,
            "searches_performed": 0,
            "total_search_time_ms": 0.0,
            "average_search_time_ms": 0.0,
            "metal_gpu_used": self.metal_available,
            "index_build_time_ms": 0.0,
        }

        logger.info(
            f"🔧 MetalFAISS initialized: dim={embedding_dim}, Metal={'✅' if self.metal_available else '❌'}, FAISS={'✅' if HAS_FAISS else '❌'}"
        )

    def _create_faiss_index(self, use_gpu: bool = True) -> Any | None:
        """Create optimized FAISS index.

        Args:
            use_gpu: Whether to use GPU acceleration

        Returns:
            FAISS index or None if FAISS unavailable
        """
        if not HAS_FAISS:
            logger.info("FAISS not available, using fallback vector storage")
            return None

        try:
            # Use IVF for large datasets, Flat for smaller ones
            if self.embedding_dim <= 512:
                # Flat index for high accuracy with smaller dimensions
                index = faiss.IndexFlatIP(
                    self.embedding_dim
                )  # Inner product (cosine similarity)
            else:
                # IVF index for larger dimensions with good performance/accuracy tradeoff
                nlist = min(
                    100, max(10, int(np.sqrt(self.max_vectors)))
                )  # Adaptive cluster count
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

            # GPU acceleration if available and beneficial
            if use_gpu and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info("🚀 FAISS GPU acceleration enabled")
                except Exception as e:
                    logger.warning(f"FAISS GPU acceleration failed, using CPU: {e}")

            return index

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return None

    @gpuify(operation_type="vector_ops", fallback=True)
    async def normalize_vectors(self, vectors: mx.array | np.ndarray) -> mx.array:
        """Normalize vectors for cosine similarity using Metal GPU.

        Args:
            vectors: Input vectors to normalize

        Returns:
            Normalized vectors
        """
        if isinstance(vectors, np.ndarray):
            vectors = mx.array(vectors)

        # L2 normalization
        norms = mx.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = mx.maximum(norms, mx.array(1e-8))
        normalized = vectors / norms

        mx.eval(normalized)
        return normalized

    async def add_vectors(
        self, vectors: np.ndarray, ids: list[int] | None = None
    ) -> bool:
        """Add vectors to the index with Metal GPU acceleration.

        Args:
            vectors: Vectors to add (N x embedding_dim)
            ids: Optional vector IDs

        Returns:
            Success status
        """
        start_time = time.perf_counter()

        try:
            # Normalize vectors using Metal GPU
            normalized_vectors = await self.normalize_vectors(vectors)

            # Convert back to numpy for FAISS
            np_vectors = np.array(normalized_vectors).astype(np.float32)

            if self.faiss_index is None:
                self.faiss_index = self._create_faiss_index()

            if self.faiss_index is not None:
                # Use FAISS
                if (
                    hasattr(self.faiss_index, "is_trained")
                    and not self.faiss_index.is_trained
                ):
                    # Train IVF index if needed
                    train_size = min(len(np_vectors), 10000)
                    self.faiss_index.train(np_vectors[:train_size])

                if ids is not None:
                    self.faiss_index.add_with_ids(
                        np_vectors, np.array(ids, dtype=np.int64)
                    )
                else:
                    self.faiss_index.add(np_vectors)
            else:
                # Fallback storage
                if self.fallback_vectors is None:
                    self.fallback_vectors = []
                self.fallback_vectors.extend(np_vectors)

            # Update stats
            self.stats["vectors_indexed"] += len(vectors)
            build_time = (time.perf_counter() - start_time) * 1000
            self.stats["index_build_time_ms"] += build_time

            logger.debug(f"Added {len(vectors)} vectors in {build_time:.1f}ms")
            return True

        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            return False

    @gpuify(operation_type="similarity", fallback=True)
    async def compute_similarities_metal(
        self, query: mx.array, database: mx.array
    ) -> mx.array:
        """Compute similarities using Metal GPU acceleration.

        Args:
            query: Query vector
            database: Database vectors

        Returns:
            Similarity scores
        """
        # Normalize inputs
        query_norm = query / mx.linalg.norm(query)
        db_norms = mx.linalg.norm(database, axis=1, keepdims=True)
        database_norm = database / mx.maximum(db_norms, mx.array(1e-8))

        # Compute cosine similarities
        similarities = database_norm @ query_norm
        mx.eval(similarities)

        return similarities

    async def search(
        self, query_vector: np.ndarray, k: int = 10, use_metal: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for k most similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return
            use_metal: Whether to use Metal GPU acceleration

        Returns:
            Tuple of (distances, indices)
        """
        start_time = time.perf_counter()

        try:
            # Normalize query vector using Metal if available
            if use_metal and self.metal_available:
                query_mx = mx.array(query_vector.reshape(1, -1))
                normalized_query = await self.normalize_vectors(query_mx)
                query_normalized = np.array(normalized_query[0]).astype(np.float32)
            else:
                # CPU normalization fallback
                norm = np.linalg.norm(query_vector)
                query_normalized = (query_vector / max(norm, 1e-8)).astype(np.float32)

            if self.faiss_index is not None and self.faiss_index.ntotal > 0:
                # Use FAISS search
                distances, indices = self.faiss_index.search(
                    query_normalized.reshape(1, -1), k
                )
                distances = distances[0]
                indices = indices[0]
            elif self.fallback_vectors:
                # Fallback search using Metal GPU
                database = mx.array(np.array(self.fallback_vectors))
                query_mx = mx.array(query_normalized)

                similarities = await self.compute_similarities_metal(query_mx, database)
                similarities_np = np.array(similarities)

                # Get top k
                top_k_indices = np.argsort(similarities_np)[-k:][::-1]
                distances = similarities_np[top_k_indices]
                indices = top_k_indices
            else:
                # No vectors indexed
                distances = np.array([])
                indices = np.array([])

            # Update performance stats
            search_time = (time.perf_counter() - start_time) * 1000
            self.stats["searches_performed"] += 1
            self.stats["total_search_time_ms"] += search_time
            self.stats["average_search_time_ms"] = (
                self.stats["total_search_time_ms"] / self.stats["searches_performed"]
            )

            logger.debug(
                f"Search completed in {search_time:.1f}ms, found {len(indices)} results"
            )
            return distances, indices

        except Exception as e:
            logger.error(f"Search error: {e}")
            return np.array([]), np.array([])

    @gpuify(operation_type="batch_ops", fallback=True)
    async def batch_search_metal(
        self, queries: mx.array, k: int = 10
    ) -> tuple[mx.array, mx.array]:
        """Batch search using Metal GPU optimization.

        Args:
            queries: Batch of query vectors (N x embedding_dim)
            k: Number of results per query

        Returns:
            Tuple of (batch_distances, batch_indices)
        """
        if self.fallback_vectors is None or len(self.fallback_vectors) == 0:
            return mx.array([]), mx.array([])

        # Normalize queries
        normalized_queries = await self.normalize_vectors(queries)

        # Normalize database
        database = mx.array(np.array(self.fallback_vectors))
        normalized_database = await self.normalize_vectors(database)

        # Batch similarity computation
        similarities = normalized_queries @ normalized_database.T
        mx.eval(similarities)

        # Get top k for each query
        # Note: MLX doesn't have direct topk, so we use argsort
        sorted_indices = mx.argsort(similarities, axis=1)[:, -k:]
        sorted_indices = sorted_indices[:, ::-1]  # Reverse to get descending order

        # Gather the corresponding scores
        batch_size = similarities.shape[0]
        row_indices = mx.arange(batch_size).reshape(-1, 1)
        batch_distances = similarities[row_indices, sorted_indices]

        mx.eval(sorted_indices)
        mx.eval(batch_distances)

        return batch_distances, sorted_indices

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics.

        Returns:
            Performance statistics dictionary
        """
        total_vectors = (
            self.faiss_index.ntotal
            if self.faiss_index is not None
            else len(self.fallback_vectors)
            if self.fallback_vectors
            else 0
        )

        return {
            **self.stats,
            "total_vectors_stored": total_vectors,
            "index_type": "FAISS" if self.faiss_index is not None else "Fallback",
            "memory_usage_estimate_mb": total_vectors
            * self.embedding_dim
            * 4
            / (1024 * 1024),
            "throughput_vectors_per_sec": (
                self.stats["vectors_indexed"]
                / (self.stats["index_build_time_ms"] / 1000)
                if self.stats["index_build_time_ms"] > 0
                else 0
            ),
            "search_throughput_per_sec": (
                1000 / self.stats["average_search_time_ms"]
                if self.stats["average_search_time_ms"] > 0
                else 0
            ),
        }

    def optimize_for_workload(self, expected_vectors: int, search_frequency: float):
        """Optimize index configuration for expected workload.

        Args:
            expected_vectors: Expected number of vectors
            search_frequency: Expected searches per second
        """
        # Adjust batch size based on workload
        if search_frequency > 100:  # High search frequency
            self.batch_size = min(1024, max(256, expected_vectors // 1000))
        elif search_frequency > 10:  # Medium search frequency
            self.batch_size = min(512, max(128, expected_vectors // 2000))
        else:  # Low search frequency
            self.batch_size = min(256, max(64, expected_vectors // 5000))

        # Recreate index with optimized settings if needed
        if expected_vectors > 50000 and HAS_FAISS:
            self.faiss_index = self._create_faiss_index(use_gpu=True)

        logger.info(
            f"🎯 Optimized for {expected_vectors} vectors, {search_frequency} searches/sec, batch_size={self.batch_size}"
        )

    async def benchmark(
        self, num_vectors: int = 1000, num_queries: int = 100
    ) -> dict[str, float]:
        """Benchmark vector operations.

        Args:
            num_vectors: Number of test vectors to index
            num_queries: Number of test queries to perform

        Returns:
            Benchmark results
        """
        logger.info(f"🏃 Benchmarking with {num_vectors} vectors, {num_queries} queries")

        # Generate test data
        np.random.seed(42)  # Reproducible results
        test_vectors = np.random.randn(num_vectors, self.embedding_dim).astype(
            np.float32
        )
        test_queries = np.random.randn(num_queries, self.embedding_dim).astype(
            np.float32
        )

        # Benchmark indexing
        start_time = time.perf_counter()
        await self.add_vectors(test_vectors)
        index_time = (time.perf_counter() - start_time) * 1000

        # Benchmark individual searches
        start_time = time.perf_counter()
        for query in test_queries:
            await self.search(query, k=10)
        single_search_time = (time.perf_counter() - start_time) * 1000

        # Benchmark batch search if using fallback
        batch_search_time = 0.0
        if self.fallback_vectors:
            start_time = time.perf_counter()
            query_batch = mx.array(test_queries)
            await self.batch_search_metal(query_batch, k=10)
            batch_search_time = (time.perf_counter() - start_time) * 1000

        results = {
            "index_time_ms": index_time,
            "index_throughput_vectors_per_sec": num_vectors / (index_time / 1000),
            "single_search_time_ms": single_search_time,
            "single_search_avg_ms": single_search_time / num_queries,
            "single_search_throughput_per_sec": num_queries
            / (single_search_time / 1000),
            "batch_search_time_ms": batch_search_time,
            "batch_search_throughput_per_sec": (
                num_queries / (batch_search_time / 1000) if batch_search_time > 0 else 0
            ),
            "metal_gpu_used": self.metal_available,
            "faiss_available": HAS_FAISS,
            "vectors_indexed": num_vectors,
            "embedding_dim": self.embedding_dim,
        }

        logger.info("📊 Benchmark Results:")
        logger.info(
            f"   Index: {results['index_throughput_vectors_per_sec']:.0f} vectors/sec"
        )
        logger.info(
            f"   Search: {results['single_search_avg_ms']:.1f}ms avg, {results['single_search_throughput_per_sec']:.0f} searches/sec"
        )
        if batch_search_time > 0:
            logger.info(
                f"   Batch: {results['batch_search_throughput_per_sec']:.0f} queries/sec"
            )

        return results


# Global instance
_metal_faiss_accelerator = None


def get_metal_faiss_accelerator(embedding_dim: int = 768) -> MetalFAISSAccelerator:
    """Get global Metal FAISS accelerator instance.

    Args:
        embedding_dim: Embedding dimension

    Returns:
        MetalFAISSAccelerator instance
    """
    global _metal_faiss_accelerator
    if (
        _metal_faiss_accelerator is None
        or _metal_faiss_accelerator.embedding_dim != embedding_dim
    ):
        _metal_faiss_accelerator = MetalFAISSAccelerator(embedding_dim)
    return _metal_faiss_accelerator


# Example usage and testing
if __name__ == "__main__":

    async def test_metal_faiss():
        """Test Metal FAISS acceleration."""
        print("🧪 Testing Metal FAISS Acceleration")

        # Initialize accelerator
        accelerator = get_metal_faiss_accelerator(embedding_dim=384)

        # Run benchmark
        results = await accelerator.benchmark(num_vectors=5000, num_queries=100)

        # Print results
        print("\n📊 Performance Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        # Get performance stats
        stats = accelerator.get_performance_stats()
        print("\n📈 System Stats:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    asyncio.run(test_metal_faiss())
