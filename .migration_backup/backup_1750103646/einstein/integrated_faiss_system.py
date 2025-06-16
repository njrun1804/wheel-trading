#!/usr/bin/env python3
"""
Integrated FAISS System

Production-ready FAISS system combining all optimizations:
- Persistent FAISS index with <200ms loading
- Code-optimized embeddings for programming structures  
- Incremental indexing for changed files only
- Metal/GPU acceleration for vector operations
- Real-time file monitoring and updates
- Comprehensive performance monitoring

This is the main entry point for the optimized FAISS indexing system.
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .code_optimized_embeddings import CodeEmbeddingResult, CodeOptimizedEmbeddingSystem
from .faiss_performance_benchmarks import BenchmarkConfig, FAISSBenchmarkRunner
from .incremental_faiss_indexer import IncrementalFAISSIndexer
from .metal_accelerated_faiss import (
    MetalAcceleratedFAISS,
    create_optimized_config,
    detect_metal_capabilities,
)
from .optimized_faiss_system import FAISSManager

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSystemConfig:
    """Configuration for the integrated FAISS system."""

    # Embedding configuration
    embedding_dimension: int = 1536
    embedding_model: str | None = None  # Path to custom model if needed
    max_chunk_size: int = 1000

    # Index configuration
    index_type: str = "HNSW"  # "HNSW", "IVF", "Flat"
    enable_persistence: bool = True
    enable_incremental: bool = True
    enable_monitoring: bool = True

    # GPU acceleration
    enable_gpu: bool = True
    auto_detect_gpu: bool = True
    gpu_batch_size: int = 256

    # Performance
    max_concurrent_embeddings: int = 4
    debounce_interval: float = 1.0

    # File patterns
    file_patterns: list[str] = None
    exclude_patterns: list[str] = None


@dataclass
class SystemStats:
    """Statistics for the integrated system."""

    total_files: int = 0
    total_vectors: int = 0
    files_indexed_today: int = 0
    vectors_added_today: int = 0
    average_search_time_ms: float = 0.0
    average_embedding_time_ms: float = 0.0
    index_load_time_ms: float = 0.0
    gpu_acceleration_active: bool = False
    memory_usage_mb: float = 0.0
    last_update_time: float = 0.0


class IntegratedFAISSSystem:
    """
    Production-ready integrated FAISS system.

    Combines all optimizations into a single, easy-to-use interface:
    - Persistent, fast-loading indexes
    - Code-optimized embeddings
    - Incremental updates
    - GPU acceleration
    - Real-time monitoring
    """

    def __init__(self, project_root: Path, config: IntegratedSystemConfig = None):
        """
        Initialize the integrated FAISS system.

        Args:
            project_root: Root directory of the project
            config: System configuration
        """
        self.project_root = project_root
        self.config = config or IntegratedSystemConfig()

        # Apply defaults
        if self.config.file_patterns is None:
            self.config.file_patterns = ["**/*.py"]
        if self.config.exclude_patterns is None:
            self.config.exclude_patterns = [
                "**/__pycache__/**",
                "**/.git/**",
                "**/node_modules/**",
            ]

        # System directories
        self.system_dir = project_root / ".einstein" / "integrated"
        self.system_dir.mkdir(parents=True, exist_ok=True)

        # Core components (initialized in initialize())
        self.faiss_manager: FAISSManager | None = None
        self.embedding_system: CodeOptimizedEmbeddingSystem | None = None
        self.incremental_indexer: IncrementalFAISSIndexer | None = None
        self.metal_accelerator: MetalAcceleratedFAISS | None = None

        # System state
        self.is_initialized = False
        self.stats = SystemStats()
        self.performance_history = []

        # Auto-configure GPU if enabled
        if self.config.enable_gpu and self.config.auto_detect_gpu:
            self._auto_configure_gpu()

    def _auto_configure_gpu(self) -> None:
        """Auto-configure GPU settings based on hardware."""
        capabilities = detect_metal_capabilities()

        if capabilities["metal_available"]:
            self.config.gpu_batch_size = capabilities["recommended_batch_size"]
            logger.info(
                f"✅ GPU auto-configured: batch_size={self.config.gpu_batch_size}"
            )
        else:
            self.config.enable_gpu = False
            logger.info("ℹ️ GPU not available, disabled acceleration")

    async def initialize(self) -> bool:
        """
        Initialize the integrated FAISS system.

        Returns:
            bool: True if initialization successful
        """
        logger.info("🚀 Initializing Integrated FAISS System...")
        start_time = time.time()

        try:
            # Initialize embedding system
            self.embedding_system = CodeOptimizedEmbeddingSystem(
                embedding_model=None,  # Use default for now
                max_chunk_size=self.config.max_chunk_size,
            )

            # Initialize FAISS manager
            if self.config.enable_persistence:
                self.faiss_manager = FAISSManager(self.project_root)
                success = await self.faiss_manager.initialize()
                if not success:
                    logger.error("Failed to initialize FAISS manager")
                    return False

            # Initialize incremental indexer if enabled
            if self.config.enable_incremental:
                self.incremental_indexer = IncrementalFAISSIndexer(
                    project_root=self.project_root,
                    embedding_model=None,  # Use default
                    enable_monitoring=self.config.enable_monitoring,
                )
                success = await self.incremental_indexer.initialize()
                if not success:
                    logger.error("Failed to initialize incremental indexer")
                    return False

            # Initialize Metal acceleration if enabled
            if self.config.enable_gpu:
                gpu_config = create_optimized_config()
                gpu_config.batch_size = self.config.gpu_batch_size

                self.metal_accelerator = MetalAcceleratedFAISS(
                    dimension=self.config.embedding_dimension,
                    index_type=self.config.index_type,
                    config=gpu_config,
                )

                self.stats.gpu_acceleration_active = True
                logger.info("✅ GPU acceleration initialized")

            # Record initialization time
            self.stats.index_load_time_ms = (time.time() - start_time) * 1000
            self.is_initialized = True

            logger.info(
                f"✅ Integrated FAISS system initialized in {self.stats.index_load_time_ms:.1f}ms"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize integrated system: {e}")
            return False

    async def index_project(self, force_rebuild: bool = False) -> dict[str, Any]:
        """
        Index the entire project with optimized processing.

        Args:
            force_rebuild: Force complete rebuild instead of incremental

        Returns:
            Dict with indexing results and statistics
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")

        logger.info("📚 Starting project indexing...")
        start_time = time.time()

        if self.config.enable_incremental and not force_rebuild:
            # Use incremental indexing
            stats = await self.incremental_indexer.scan_and_update(
                self.config.file_patterns
            )

            result = {
                "method": "incremental",
                "files_checked": stats.files_checked,
                "files_changed": stats.files_changed,
                "files_added": stats.files_added,
                "vectors_added": stats.vectors_added,
                "processing_time_ms": stats.processing_time_ms,
                "embedding_time_ms": stats.embedding_time_ms,
                "index_update_time_ms": stats.index_update_time_ms,
            }

            # Update system stats
            self.stats.files_indexed_today += stats.files_changed + stats.files_added
            self.stats.vectors_added_today += stats.vectors_added
            self.stats.average_embedding_time_ms = stats.embedding_time_ms / max(
                stats.files_changed + stats.files_added, 1
            )

        else:
            # Full rebuild using FAISS manager
            all_files = []
            for pattern in self.config.file_patterns:
                files = list(self.project_root.rglob(pattern))
                # Apply exclusion patterns
                for exclude in self.config.exclude_patterns:
                    files = [f for f in files if not f.match(exclude)]
                all_files.extend([str(f) for f in files])

            # Generate embeddings for all files
            file_embeddings = {}
            embedding_start = time.time()

            for file_path in all_files:
                try:
                    results = await self.embedding_system.embed_file(file_path)
                    if results:
                        file_embeddings[file_path] = self._prepare_embedding_data(
                            results
                        )
                except Exception as e:
                    logger.error(f"Failed to embed {file_path}: {e}")

            embedding_time = (time.time() - embedding_start) * 1000

            # Index embeddings
            index_start = time.time()
            index_results = await self.faiss_manager.index_files(
                file_embeddings, force_update=True
            )
            index_time = (time.time() - index_start) * 1000

            successful_files = sum(1 for success in index_results.values() if success)
            total_vectors = sum(
                len(data["embeddings"]) for data in file_embeddings.values()
            )

            result = {
                "method": "full_rebuild",
                "files_processed": len(all_files),
                "files_successful": successful_files,
                "total_vectors": total_vectors,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "embedding_time_ms": embedding_time,
                "index_update_time_ms": index_time,
            }

            # Update system stats
            self.stats.total_files = successful_files
            self.stats.total_vectors = total_vectors
            self.stats.files_indexed_today = successful_files
            self.stats.vectors_added_today = total_vectors

        self.stats.last_update_time = time.time()

        logger.info(
            f"✅ Project indexing completed in {result['processing_time_ms']:.1f}ms"
        )
        logger.info(f"   Method: {result['method']}")
        if "files_changed" in result:
            logger.info(
                f"   Files: {result['files_changed']} changed, {result['files_added']} added"
            )
        else:
            logger.info(
                f"   Files: {result['files_successful']}/{result['files_processed']} successful"
            )
        logger.info(
            f"   Vectors: {result.get('vectors_added', result.get('total_vectors', 0))}"
        )

        return result

    async def search(
        self,
        query: str,
        k: int = 20,
        min_score: float = 0.0,
        file_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search the index with the query.

        Args:
            query: Search query text
            k: Number of results to return
            min_score: Minimum similarity score threshold
            file_filter: Optional file path filter pattern

        Returns:
            List of search results with metadata
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")

        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding, _ = self.embedding_system.embedding_function(query)

            # Use FAISS manager for search
            if self.faiss_manager:
                results = await self.faiss_manager.search(query_embedding, k)

                # Convert to dictionary format
                search_results = []
                for result in results:
                    # Apply file filter if specified
                    if file_filter and file_filter not in result.file_path:
                        continue

                    # Apply score threshold
                    if result.score < min_score:
                        continue

                    search_results.append(
                        {
                            "content": result.content,
                            "file_path": result.file_path,
                            "absolute_path": result.absolute_path,
                            "line_start": result.line_start,
                            "line_end": result.line_end,
                            "score": result.score,
                            "metadata": asdict(result.metadata),
                        }
                    )

                # Update performance stats
                search_time = (time.time() - start_time) * 1000
                self._update_search_performance(search_time)

                logger.debug(
                    f"Search completed in {search_time:.1f}ms: {len(search_results)} results"
                )
                return search_results

            else:
                logger.error("FAISS manager not available")
                return []

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _prepare_embedding_data(
        self, results: list[CodeEmbeddingResult]
    ) -> dict[str, Any]:
        """Prepare embedding data for indexing."""
        embeddings = []
        line_ranges = []
        content_previews = []
        token_counts = []

        for result in results:
            embeddings.append(result.embedding)
            chunk = result.chunk
            line_ranges.append((chunk.start_line, chunk.end_line))
            content_previews.append(chunk.content[:200])
            token_counts.append(chunk.tokens)

        return {
            "embeddings": embeddings,
            "line_ranges": line_ranges,
            "content_previews": content_previews,
            "token_counts": token_counts,
        }

    def _update_search_performance(self, search_time_ms: float) -> None:
        """Update search performance statistics."""
        self.performance_history.append(search_time_ms)

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Update average
        self.stats.average_search_time_ms = sum(self.performance_history) / len(
            self.performance_history
        )

    async def get_system_stats(self) -> SystemStats:
        """Get current system statistics."""
        if self.is_initialized and self.faiss_manager:
            # Update stats from FAISS manager
            faiss_stats = await self.faiss_manager.get_stats()
            self.stats.total_vectors = faiss_stats.total_vectors

            # Update memory usage
            import psutil

            self.stats.memory_usage_mb = psutil.Process().memory_info().rss / (
                1024 * 1024
            )

        return self.stats

    async def run_benchmarks(self, quick: bool = True) -> dict[str, Any]:
        """
        Run performance benchmarks on the system.

        Args:
            quick: Run quick benchmarks (fewer iterations)

        Returns:
            Benchmark results
        """
        logger.info("🏃 Running performance benchmarks...")

        # Configure benchmark parameters
        if quick:
            config = BenchmarkConfig(
                num_vectors_small=100,
                num_vectors_medium=1000,
                num_vectors_large=5000,
                num_search_queries=20,
                num_incremental_files=10,
                benchmark_iterations=3,
            )
        else:
            config = BenchmarkConfig()

        # Run benchmarks
        runner = FAISSBenchmarkRunner(self.project_root, config)
        suite = await runner.run_all_benchmarks()

        logger.info(f"✅ Benchmarks completed in {suite.total_runtime_ms:.1f}ms")

        return {
            "runtime_ms": suite.total_runtime_ms,
            "total_tests": len(suite.results),
            "summary": suite.summary,
            "performance_targets_met": all(
                target.get("target_met", False)
                for target in suite.summary.get("performance_targets", {}).values()
            ),
        }

    async def update_file(self, file_path: str) -> bool:
        """
        Update a specific file in the index.

        Args:
            file_path: Path to the file to update

        Returns:
            bool: True if update successful
        """
        if not self.is_initialized:
            return False

        try:
            if self.incremental_indexer:
                await self.incremental_indexer.update_files([file_path])
                logger.debug(f"Updated index for {file_path}")
                return True
            else:
                logger.warning("Incremental indexer not available")
                return False
        except Exception as e:
            logger.error(f"Failed to update {file_path}: {e}")
            return False

    async def remove_file(self, file_path: str) -> bool:
        """
        Remove a file from the index.

        Args:
            file_path: Path to the file to remove

        Returns:
            bool: True if removal successful
        """
        if not self.is_initialized:
            return False

        try:
            if self.incremental_indexer:
                await self.incremental_indexer.remove_file(file_path)
                logger.debug(f"Removed {file_path} from index")
                return True
            else:
                logger.warning("Incremental indexer not available")
                return False
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")
            return False

    async def cleanup(self) -> None:
        """Clean up system resources."""
        logger.info("🧹 Cleaning up integrated FAISS system...")

        if self.faiss_manager:
            await self.faiss_manager.cleanup()

        if self.incremental_indexer:
            await self.incremental_indexer.cleanup()

        if self.metal_accelerator:
            self.metal_accelerator.cleanup()

        self.is_initialized = False
        logger.info("✅ Cleanup completed")


# Convenience functions for easy usage
async def create_faiss_system(
    project_root: Path, config: IntegratedSystemConfig = None
) -> IntegratedFAISSSystem:
    """
    Create and initialize an integrated FAISS system.

    Args:
        project_root: Root directory of the project
        config: System configuration

    Returns:
        Initialized IntegratedFAISSSystem
    """
    system = IntegratedFAISSSystem(project_root, config)
    success = await system.initialize()

    if not success:
        raise RuntimeError("Failed to initialize FAISS system")

    return system


# Example usage and testing
if __name__ == "__main__":

    async def test_integrated_system():
        """Test the integrated FAISS system."""
        project_root = Path.cwd()

        # Create system with default configuration
        config = IntegratedSystemConfig(
            enable_gpu=True,
            enable_incremental=True,
            enable_monitoring=False,  # Disable for testing
            file_patterns=["*.py"],  # Just test Python files in root
        )

        system = await create_faiss_system(project_root, config)

        # Index project
        print("Indexing project...")
        index_result = await system.index_project()
        print(f"Indexing result: {index_result}")

        # Test search
        print("\nTesting search...")
        results = await system.search("async function", k=5)
        print(f"Search results: {len(results)} found")

        for i, result in enumerate(results[:3]):
            print(
                f"  {i+1}. {result['file_path']}:{result['line_start']} (score: {result['score']:.3f})"
            )

        # Get stats
        stats = await system.get_system_stats()
        print("\nSystem stats:")
        print(f"  Total files: {stats.total_files}")
        print(f"  Total vectors: {stats.total_vectors}")
        print(f"  Average search time: {stats.average_search_time_ms:.1f}ms")
        print(f"  GPU acceleration: {stats.gpu_acceleration_active}")
        print(f"  Memory usage: {stats.memory_usage_mb:.1f}MB")

        # Run quick benchmarks
        print("\nRunning benchmarks...")
        benchmark_result = await system.run_benchmarks(quick=True)
        print(
            f"Benchmarks: {benchmark_result['total_tests']} tests in {benchmark_result['runtime_ms']:.1f}ms"
        )
        print(f"Performance targets met: {benchmark_result['performance_targets_met']}")

        # Cleanup
        await system.cleanup()
        print("\n✅ Test completed successfully")

    # Run test
    asyncio.run(test_integrated_system())
