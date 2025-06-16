"""
Buffer Size Calculator for Real-World M4 Pro Workloads

Calculates optimal buffer sizes based on actual workload characteristics
and available system memory.
"""

import logging
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)


@dataclass
class WorkloadProfile:
    """Workload characteristics for buffer sizing"""

    max_concurrent_searches: int = 10
    typical_corpus_size: int = 50000
    embedding_dimension: int = 768
    max_query_batch_size: int = 32
    cache_hit_ratio: float = 0.7
    memory_safety_margin: float = 0.2  # 20% safety margin


@dataclass
class SystemResources:
    """Available system resources"""

    total_memory_gb: float
    available_memory_gb: float
    gpu_cores: int = 20
    cpu_cores: int = 12
    storage_type: str = "ssd"  # ssd or hdd


class BufferSizeCalculator:
    """
    Calculates optimal buffer sizes for M4 Pro hardware based on
    real workload requirements and available system resources.
    """

    def __init__(self):
        self.system_resources = self._detect_system_resources()
        self.logger = logging.getLogger(__name__)

    def _detect_system_resources(self) -> SystemResources:
        """Detect available system resources"""
        try:
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)

            # Detect M4 Pro specifically
            cpu_cores = psutil.cpu_count(logical=False)  # Physical cores

            return SystemResources(
                total_memory_gb=total_gb,
                available_memory_gb=available_gb,
                cpu_cores=cpu_cores,
                gpu_cores=20 if cpu_cores >= 12 else 10,  # M4 Pro vs M4
            )
        except Exception as e:
            logger.warning(f"Could not detect system resources: {e}")
            # Default to M4 Pro specs
            return SystemResources(
                total_memory_gb=24.0,
                available_memory_gb=18.0,
                cpu_cores=12,
                gpu_cores=20,
            )

    def calculate_optimal_buffer_sizes(
        self, workload: WorkloadProfile
    ) -> dict[str, int]:
        """
        Calculate optimal buffer sizes for the given workload.

        Returns buffer sizes in bytes.
        """
        logger.info("Calculating optimal buffer sizes for M4 Pro workload")

        # Available memory for our application (excluding safety margin)
        usable_memory_gb = self.system_resources.available_memory_gb * (
            1 - workload.memory_safety_margin
        )
        usable_memory_bytes = int(usable_memory_gb * 1024**3)

        logger.info(
            f"Usable memory: {usable_memory_gb:.1f}GB ({usable_memory_bytes/1024**3:.1f}GB)"
        )

        # Calculate component memory requirements
        sizes = {}

        # 1. Embedding Matrix Buffer
        # Size = corpus_size * embedding_dim * 4 bytes (float32)
        embedding_matrix_size = (
            workload.typical_corpus_size * workload.embedding_dimension * 4  # float32
        )

        # Add buffer for growth (50% more than typical)
        embedding_matrix_size = int(embedding_matrix_size * 1.5)
        sizes["embedding_matrix"] = embedding_matrix_size

        # 2. Search Results Buffer
        # Size = max_concurrent_searches * max_results_per_query * result_size
        max_results_per_query = 100  # Conservative estimate
        result_entry_size = 1024  # 1KB per result (with metadata)
        search_results_size = (
            workload.max_concurrent_searches * max_results_per_query * result_entry_size
        )
        sizes["search_results"] = search_results_size

        # 3. Query Buffer
        # Size = max_batch_size * embedding_dim * 4 bytes
        query_buffer_size = (
            workload.max_query_batch_size * workload.embedding_dimension * 4
        )
        sizes["query_buffer"] = query_buffer_size

        # 4. Cache Buffer (based on cache hit ratio and working set)
        # Estimate working set as 10% of corpus for good cache hit ratio
        working_set_size = int(workload.typical_corpus_size * 0.1)
        cache_entry_size = (
            workload.embedding_dimension * 4 + 512
        )  # embedding + metadata
        cache_buffer_size = working_set_size * cache_entry_size

        # Adjust cache size based on desired hit ratio
        if workload.cache_hit_ratio > 0.8:
            cache_buffer_size = int(
                cache_buffer_size * 1.5
            )  # Larger cache for high hit ratio

        sizes["cache_buffer"] = cache_buffer_size

        # 5. Temporary Buffers (for intermediate operations)
        temporary_buffer_size = max(
            embedding_matrix_size // 10,  # 10% of embedding matrix
            64 * 1024 * 1024,  # Minimum 64MB
        )
        sizes["temporary_buffer"] = temporary_buffer_size

        # 6. Memory Pool Overhead (management structures)
        total_data_size = sum(sizes.values())
        overhead_size = int(total_data_size * 0.05)  # 5% overhead
        sizes["memory_pool_overhead"] = overhead_size

        # Validate total size fits in available memory
        total_size = sum(sizes.values())
        if total_size > usable_memory_bytes:
            logger.warning(
                f"Calculated buffer sizes ({total_size/1024**3:.1f}GB) exceed "
                f"available memory ({usable_memory_gb:.1f}GB)"
            )

            # Scale down all buffers proportionally
            scale_factor = usable_memory_bytes / total_size
            sizes = {name: int(size * scale_factor) for name, size in sizes.items()}
            logger.info(f"Scaled down buffer sizes by {scale_factor:.2f}x")

        # Log calculated sizes
        total_size = sum(sizes.values())
        logger.info(f"Calculated buffer sizes (total: {total_size/1024**3:.1f}GB):")
        for name, size in sizes.items():
            logger.info(f"  {name}: {size/1024**2:.1f}MB")

        return sizes

    def calculate_pool_distribution(
        self, total_memory_mb: float, workload: WorkloadProfile
    ) -> dict[str, float]:
        """
        Calculate optimal distribution of memory across different pool types.

        Returns percentages that sum to 1.0.
        """
        # Base distribution for typical workloads
        distribution = {
            "embedding_pools": 0.60,  # 60% for embedding data
            "cache_pools": 0.25,  # 25% for caching
            "temporary_pools": 0.10,  # 10% for temporary operations
            "overhead": 0.05,  # 5% for management overhead
        }

        # Adjust based on workload characteristics
        if workload.cache_hit_ratio > 0.8:
            # High cache hit ratio - allocate more to cache
            distribution["cache_pools"] = 0.35
            distribution["embedding_pools"] = 0.50
        elif workload.max_concurrent_searches > 20:
            # High concurrency - need more temporary space
            distribution["temporary_pools"] = 0.15
            distribution["embedding_pools"] = 0.55

        # Ensure distribution sums to 1.0
        total = sum(distribution.values())
        distribution = {k: v / total for k, v in distribution.items()}

        logger.info("Memory pool distribution:")
        for pool_type, percentage in distribution.items():
            mb_allocation = total_memory_mb * percentage
            logger.info(f"  {pool_type}: {percentage:.1%} ({mb_allocation:.1f}MB)")

        return distribution

    def recommend_embedding_pool_settings(
        self, corpus_size: int, embedding_dim: int
    ) -> dict[str, any]:
        """Recommend embedding pool configuration"""

        # Calculate base size needed
        base_size_mb = (corpus_size * embedding_dim * 4) / (1024**2)

        # Add growth buffer (corpus might grow)
        recommended_size_mb = base_size_mb * 1.8  # 80% growth buffer

        # Determine eviction policy based on size
        if recommended_size_mb > 1000:  # Large corpus
            eviction_policy = "LRU"  # Simple and efficient for large data
        else:
            eviction_policy = "ADAPTIVE"  # More sophisticated for smaller data

        # Calculate optimal block size for mmap
        if corpus_size > 100000:
            mmap_block_size_mb = 256  # 256MB blocks for large corpus
        else:
            mmap_block_size_mb = 64  # 64MB blocks for smaller corpus

        return {
            "pool_size_mb": recommended_size_mb,
            "eviction_policy": eviction_policy,
            "enable_mmap": corpus_size > 50000,  # Use mmap for large corpus
            "mmap_block_size_mb": mmap_block_size_mb,
            "preload_percentage": 0.1 if corpus_size > 100000 else 0.5,
        }

    def validate_buffer_configuration(
        self, buffer_sizes: dict[str, int], workload: WorkloadProfile
    ) -> tuple[bool, list[str]]:
        """
        Validate that buffer configuration will work for the given workload.

        Returns (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True

        # Check total memory usage
        total_size = sum(buffer_sizes.values())
        usable_memory = (
            self.system_resources.available_memory_gb
            * (1 - workload.memory_safety_margin)
            * 1024**3
        )

        if total_size > usable_memory:
            warnings.append(
                f"Total buffer size ({total_size/1024**3:.1f}GB) exceeds "
                f"available memory ({usable_memory/1024**3:.1f}GB)"
            )
            is_valid = False

        # Check embedding matrix can hold corpus
        embedding_size = buffer_sizes.get("embedding_matrix", 0)
        required_embedding_size = (
            workload.typical_corpus_size * workload.embedding_dimension * 4
        )

        if embedding_size < required_embedding_size:
            warnings.append(
                f"Embedding matrix buffer too small: {embedding_size/1024**2:.1f}MB "
                f"< required {required_embedding_size/1024**2:.1f}MB"
            )
            is_valid = False

        # Check search results buffer can handle concurrent searches
        search_buffer_size = buffer_sizes.get("search_results", 0)
        required_search_size = (
            workload.max_concurrent_searches * 100 * 1024
        )  # 100KB per search

        if search_buffer_size < required_search_size:
            warnings.append(
                f"Search results buffer may be too small for {workload.max_concurrent_searches} "
                f"concurrent searches"
            )

        return is_valid, warnings


def calculate_buffers_for_workload(
    corpus_size: int = 50000,
    embedding_dim: int = 768,
    max_concurrent_searches: int = 10,
    cache_hit_ratio: float = 0.7,
) -> dict[str, int]:
    """
    Convenience function to calculate optimal buffer sizes for a workload.

    Returns buffer sizes in bytes.
    """
    workload = WorkloadProfile(
        typical_corpus_size=corpus_size,
        embedding_dimension=embedding_dim,
        max_concurrent_searches=max_concurrent_searches,
        cache_hit_ratio=cache_hit_ratio,
    )

    calculator = BufferSizeCalculator()
    return calculator.calculate_optimal_buffer_sizes(workload)


def get_recommended_pool_settings(
    corpus_size: int, embedding_dim: int
) -> dict[str, any]:
    """Get recommended pool settings for a given corpus"""
    calculator = BufferSizeCalculator()
    return calculator.recommend_embedding_pool_settings(corpus_size, embedding_dim)
