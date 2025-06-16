import re
#!/usr/bin/env python3
"""
Metal-Accelerated FAISS System

GPU/Metal acceleration for FAISS operations on Apple Silicon:
- Metal Performance Shaders integration
- Accelerated vector operations using MLX
- Optimized batch processing for embeddings
- Hardware-aware memory management
- Fallback to CPU when GPU unavailable

Performance targets:
- 5-10x speedup for vector operations
- Efficient GPU memory usage
- Optimized batch sizes for M4 Pro
- Automatic hardware detection
- Graceful degradation to CPU
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Metal/GPU acceleration libraries
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
    logger.info("✅ MLX (Metal) acceleration available")
except ImportError:
    MLX_AVAILABLE = False
    logger.info("ℹ️ MLX not available, using CPU only")

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ FAISS not available")


@dataclass
class GPUAccelerationConfig:
    """Configuration for GPU acceleration."""

    enable_metal: bool = True
    batch_size: int = 256
    max_gpu_memory_mb: int = 2048
    enable_mixed_precision: bool = True
    fallback_to_cpu: bool = True
    auto_detect_batch_size: bool = True


@dataclass
class AccelerationStats:
    """Statistics for acceleration performance."""

    operations_gpu: int = 0
    operations_cpu: int = 0
    gpu_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    speedup_factor: float = 1.0


class MetalVectorProcessor:
    """Metal-accelerated vector processing using MLX."""

    def __init__(self, config: GPUAccelerationConfig):
        """Initialize Metal vector processor."""
        self.config = config
        self.is_available = MLX_AVAILABLE and config.enable_metal
        self.stats = AccelerationStats()

        if self.is_available:
            # Set up Metal device
            self._setup_metal()
        else:
            logger.info("Metal acceleration disabled or unavailable")

    def _setup_metal(self) -> None:
        """Set up Metal Performance Shaders."""
        try:
            # Test Metal availability
            test_array = mx.array([1.0, 2.0, 3.0])
            _ = mx.sum(test_array)
            logger.info("✅ Metal Performance Shaders initialized")
        except Exception as e:
            logger.warning(f"Metal setup failed: {e}")
            self.is_available = False

    def normalize_vectors_gpu(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors using Metal acceleration."""
        if not self.is_available or vectors.size == 0:
            return self._normalize_vectors_cpu(vectors)

        start_time = time.time()

        try:
            # Convert to MLX array
            mx_vectors = mx.array(vectors.astype(np.float32))

            # Compute L2 norms
            norms = mx.sqrt(mx.sum(mx_vectors * mx_vectors, axis=1, keepdims=True))

            # Avoid division by zero
            norms = mx.maximum(norms, mx.array(1e-12))

            # Normalize
            normalized = mx_vectors / norms

            # Convert back to numpy
            result = np.array(normalized)

            self.stats.operations_gpu += 1
            self.stats.gpu_time_ms += (time.time() - start_time) * 1000

            return result

        except Exception as e:
            logger.debug(f"GPU normalization failed, falling back to CPU: {e}")
            return self._normalize_vectors_cpu(vectors)

    def _normalize_vectors_cpu(self, vectors: np.ndarray) -> np.ndarray:
        """CPU fallback for vector normalization."""
        start_time = time.time()

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        result = vectors / norms

        self.stats.operations_cpu += 1
        self.stats.cpu_time_ms += (time.time() - start_time) * 1000

        return result

    def compute_similarities_gpu(
        self, query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarities using Metal acceleration."""
        if not self.is_available or vectors.size == 0:
            return self._compute_similarities_cpu(query, vectors)

        start_time = time.time()

        try:
            # Convert to MLX arrays
            mx_query = mx.array(query.astype(np.float32))
            mx_vectors = mx.array(vectors.astype(np.float32))

            # Ensure query is 2D
            if mx_query.ndim == 1:
                mx_query = mx.expand_dims(mx_query, 0)

            # Compute dot products (cosine similarity for normalized vectors)
            similarities = mx.matmul(mx_query, mx_vectors.T)

            # Convert back to numpy
            result = np.array(similarities).flatten()

            self.stats.operations_gpu += 1
            self.stats.gpu_time_ms += (time.time() - start_time) * 1000

            return result

        except Exception as e:
            logger.debug(f"GPU similarity computation failed, falling back to CPU: {e}")
            return self._compute_similarities_cpu(query, vectors)

    def _compute_similarities_cpu(
        self, query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """CPU fallback for similarity computation."""
        start_time = time.time()

        if query.ndim == 1:
            query = query.reshape(1, -1)

        similarities = np.dot(query, vectors.T).flatten()

        self.stats.operations_cpu += 1
        self.stats.cpu_time_ms += (time.time() - start_time) * 1000

        return similarities

    def batch_process_embeddings(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Process embeddings in optimized batches."""
        if not embeddings:
            return np.array([])

        # Convert to numpy array
        embedding_matrix = np.array(embeddings, dtype=np.float32)

        if not self.is_available:
            return self._batch_process_cpu(embedding_matrix)

        # Determine optimal batch size
        batch_size = self.config.batch_size
        if self.config.auto_detect_batch_size:
            batch_size = self._calculate_optimal_batch_size(embedding_matrix.shape)

        start_time = time.time()
        processed_batches = []

        try:
            for i in range(0, len(embedding_matrix), batch_size):
                batch = embedding_matrix[i : i + batch_size]

                # Process batch on GPU
                mx_batch = mx.array(batch)

                # Apply any GPU-accelerated preprocessing (e.g., normalization)
                if self.config.enable_mixed_precision:
                    # Use float16 for memory efficiency where possible
                    mx_batch = mx_batch.astype(mx.float16)

                # Normalize
                norms = mx.sqrt(mx.sum(mx_batch * mx_batch, axis=1, keepdims=True))
                norms = mx.maximum(norms, mx.array(1e-12))
                normalized_batch = mx_batch / norms

                # Convert back to float32 and numpy
                result_batch = np.array(normalized_batch.astype(mx.float32))
                processed_batches.append(result_batch)

            result = np.vstack(processed_batches)

            self.stats.operations_gpu += 1
            self.stats.gpu_time_ms += (time.time() - start_time) * 1000

            return result

        except Exception as e:
            logger.debug(f"GPU batch processing failed, falling back to CPU: {e}")
            return self._batch_process_cpu(embedding_matrix)

    def _batch_process_cpu(self, embeddings: np.ndarray) -> np.ndarray:
        """CPU fallback for batch processing."""
        start_time = time.time()

        # Simple normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        result = embeddings / norms

        self.stats.operations_cpu += 1
        self.stats.cpu_time_ms += (time.time() - start_time) * 1000

        return result

    def _calculate_optimal_batch_size(self, shape: tuple[int, int]) -> int:
        """Calculate optimal batch size based on data size and available memory."""
        num_vectors, vector_dim = shape

        # Estimate memory usage per vector (in MB)
        bytes_per_vector = vector_dim * 4  # 4 bytes per float32
        mb_per_vector = bytes_per_vector / (1024 * 1024)

        # Use at most half of configured GPU memory
        max_memory_mb = self.config.max_gpu_memory_mb / 2
        max_batch_size = int(max_memory_mb / mb_per_vector)

        # Clamp to reasonable bounds
        optimal_batch_size = min(
            max(max_batch_size, 32), min(self.config.batch_size, num_vectors)
        )

        logger.debug(
            f"Calculated optimal batch size: {optimal_batch_size} for {num_vectors} vectors"
        )
        return optimal_batch_size

    def get_acceleration_stats(self) -> AccelerationStats:
        """Get acceleration performance statistics."""
        total_ops = self.stats.operations_gpu + self.stats.operations_cpu
        if total_ops > 0:
            self.stats.operations_gpu / total_ops
            self.stats.gpu_time_ms + self.stats.cpu_time_ms

            if self.stats.cpu_time_ms > 0 and self.stats.gpu_time_ms > 0:
                # Calculate speedup factor
                avg_gpu_time = self.stats.gpu_time_ms / max(
                    self.stats.operations_gpu, 1
                )
                avg_cpu_time = self.stats.cpu_time_ms / max(
                    self.stats.operations_cpu, 1
                )
                self.stats.speedup_factor = (
                    avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 1.0
                )

        return self.stats


class MetalAcceleratedFAISS:
    """FAISS index with Metal acceleration integration."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "HNSW",
        config: GPUAccelerationConfig = None,
    ):
        """
        Initialize Metal-accelerated FAISS index.

        Args:
            dimension: Vector dimension
            index_type: FAISS index type ("HNSW", "IVF", "Flat")
            config: GPU acceleration configuration
        """
        self.dimension = dimension
        self.index_type = index_type
        self.config = config or GPUAccelerationConfig()

        # Initialize components
        self.metal_processor = MetalVectorProcessor(self.config)
        self.faiss_index: faiss.Index | None = None
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Initialize FAISS index
        self._create_faiss_index()

    def _create_faiss_index(self) -> None:
        """Create optimized FAISS index."""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return

        try:
            if self.index_type == "HNSW":
                # Use HNSW for best search performance
                self.faiss_index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.faiss_index.hnsw.efConstruction = 200
                self.faiss_index.hnsw.efSearch = 100

            elif self.index_type == "IVF":
                # Use IVF for large datasets
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)

            else:  # Flat
                self.faiss_index = faiss.IndexFlatL2(self.dimension)

            # Configure threading
            faiss.omp_set_num_threads(min(os.cpu_count(), 8))

            logger.info(
                f"✅ Created {self.index_type} FAISS index (dimension={self.dimension})"
            )

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            self.faiss_index = None

    async def add_vectors(self, vectors: list[np.ndarray]) -> bool:
        """Add vectors to the index with Metal acceleration."""
        if not self.faiss_index or not vectors:
            return False

        try:
            # Convert to numpy array
            np.array(vectors, dtype=np.float32)

            # Preprocess with Metal acceleration
            processed_vectors = self.metal_processor.batch_process_embeddings(vectors)

            # Add to FAISS index
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self.faiss_index.add, processed_vectors
            )

            logger.debug(f"Added {len(vectors)} vectors to FAISS index")
            return True

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False

    async def search(
        self, query_vector: np.ndarray, k: int = 20
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search with Metal-accelerated preprocessing."""
        if not self.faiss_index:
            return np.array([]), np.array([])

        try:
            # Preprocess query with Metal acceleration
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            processed_query = self.metal_processor.normalize_vectors_gpu(query_vector)

            # Perform FAISS search
            distances, indices = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.faiss_index.search, processed_query, k
            )

            return distances, indices

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return np.array([]), np.array([])

    async def similarity_search(
        self, query_vector: np.ndarray, vectors: np.ndarray, k: int = 20
    ) -> np.ndarray:
        """Direct similarity search using Metal acceleration."""
        try:
            # Normalize vectors
            normalized_query = self.metal_processor.normalize_vectors_gpu(
                query_vector.reshape(1, -1)
            )
            normalized_vectors = self.metal_processor.normalize_vectors_gpu(vectors)

            # Compute similarities
            similarities = self.metal_processor.compute_similarities_gpu(
                normalized_query[0], normalized_vectors
            )

            # Get top k
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            return top_k_indices

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return np.array([])

    def save(self, filepath: str) -> bool:
        """Save FAISS index to disk."""
        if not self.faiss_index:
            return False

        try:
            faiss.write_index(self.faiss_index, filepath)
            logger.info(f"Saved FAISS index to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Load FAISS index from disk."""
        try:
            self.faiss_index = faiss.read_index(filepath)
            logger.info(f"Loaded FAISS index from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "metal_available": self.metal_processor.is_available,
            "acceleration_stats": self.metal_processor.get_acceleration_stats(),
        }

        return stats

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        self.faiss_index = None
        logger.info("🧹 Metal-accelerated FAISS cleanup completed")


# Hardware detection and optimization
def detect_metal_capabilities() -> dict[str, Any]:
    """Detect Metal/GPU capabilities on the system."""
    capabilities = {
        "mlx_available": MLX_AVAILABLE,
        "metal_available": False,
        "recommended_batch_size": 256,
        "max_memory_mb": 2048,
        "cpu_cores": os.cpu_count(),
    }

    if MLX_AVAILABLE:
        try:
            # Test Metal functionality
            test_array = mx.array([1.0, 2.0, 3.0])
            _ = mx.sum(test_array)
            capabilities["metal_available"] = True

            # Try to estimate GPU memory (rough approximation)
            try:
                # Test with larger arrays to estimate memory
                test_size = 1024 * 1024  # 1M elements
                large_array = mx.ones((test_size,), dtype=mx.float32)
                _ = mx.sum(large_array)
                capabilities["max_memory_mb"] = 4096  # Assume more memory available
                capabilities["recommended_batch_size"] = 512
            except (RuntimeError, MemoryError, ImportError) as e:
                logger.debug(f"Metal memory test failed: {e}")
                capabilities["max_memory_mb"] = 1024  # Conservative estimate
                capabilities["recommended_batch_size"] = 128

        except Exception as e:
            logger.debug(f"Metal capability test failed: {e}")

    return capabilities


def create_optimized_config() -> GPUAccelerationConfig:
    """Create optimized configuration based on hardware."""
    capabilities = detect_metal_capabilities()

    return GPUAccelerationConfig(
        enable_metal=capabilities["metal_available"],
        batch_size=capabilities["recommended_batch_size"],
        max_gpu_memory_mb=capabilities["max_memory_mb"],
        enable_mixed_precision=capabilities["metal_available"],
        fallback_to_cpu=True,
        auto_detect_batch_size=True,
    )


# Example usage and testing
if __name__ == "__main__":

    async def test_metal_acceleration():
        """Test Metal-accelerated FAISS system."""
        print("Testing Metal-Accelerated FAISS System")
        print("=" * 50)

        # Detect capabilities
        capabilities = detect_metal_capabilities()
        print(f"Metal capabilities: {capabilities}")

        # Create optimized config
        config = create_optimized_config()
        print(
            f"Optimized config: batch_size={config.batch_size}, metal={config.enable_metal}"
        )

        # Initialize system
        faiss_system = MetalAcceleratedFAISS(
            dimension=1536, index_type="HNSW", config=config
        )

        # Generate test vectors
        num_vectors = 1000
        test_vectors = [
            np.random.randn(1536).astype(np.float32) for _ in range(num_vectors)
        ]
        query_vector = np.random.randn(1536).astype(np.float32)

        print(f"\nTesting with {num_vectors} vectors...")

        # Test vector addition
        start_time = time.time()
        success = await faiss_system.add_vectors(test_vectors)
        add_time = (time.time() - start_time) * 1000
        print(f"Vector addition: {'✅' if success else '❌'} ({add_time:.1f}ms)")

        # Test search
        start_time = time.time()
        distances, indices = await faiss_system.search(query_vector, k=10)
        search_time = (time.time() - start_time) * 1000
        print(f"Search: {'✅' if len(indices) > 0 else '❌'} ({search_time:.1f}ms)")

        # Get statistics
        stats = faiss_system.get_stats()
        print("\nSystem Statistics:")
        print(f"  Total vectors: {stats['total_vectors']}")
        print(f"  Metal available: {stats['metal_available']}")

        acceleration_stats = stats["acceleration_stats"]
        print(f"  GPU operations: {acceleration_stats.operations_gpu}")
        print(f"  CPU operations: {acceleration_stats.operations_cpu}")
        print(f"  GPU time: {acceleration_stats.gpu_time_ms:.1f}ms")
        print(f"  CPU time: {acceleration_stats.cpu_time_ms:.1f}ms")
        print(f"  Speedup factor: {acceleration_stats.speedup_factor:.1f}x")

        # Cleanup
        faiss_system.cleanup()

    # Run test
    asyncio.run(test_metal_acceleration())
