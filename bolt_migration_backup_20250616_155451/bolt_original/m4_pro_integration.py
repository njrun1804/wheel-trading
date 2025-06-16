"""
M4 Pro Integration Module - Central Deployment and Orchestration

Connects all optimization components and ensures proper initialization and deployment.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adaptive_concurrency import (
    AdaptiveConcurrencyManager,
    get_adaptive_concurrency_manager,
)
from .ane_acceleration import ANEEmbeddingGenerator, create_ane_embedding_generator
from .core.integration import BoltIntegration
from .memory_pools import MemoryPoolManager, get_memory_pool_manager
from .metal_accelerated_search import MetalAcceleratedSearch, get_metal_search
from .performance_benchmark import run_m4_pro_benchmarks
from .production_error_recovery import production_error_handling

# Import all optimization components
from .unified_memory import BufferType, UnifiedMemoryManager, get_unified_memory_manager

logger = logging.getLogger(__name__)


@dataclass
class M4ProSystemStatus:
    """System status for M4 Pro optimizations"""

    unified_memory_active: bool = False
    metal_search_active: bool = False
    adaptive_concurrency_active: bool = False
    memory_pools_active: bool = False
    ane_acceleration_active: bool = False
    benchmark_validation_passed: bool = False
    total_memory_mb: float = 0.0
    active_components: int = 0
    initialization_time_ms: float = 0.0


class M4ProOptimizedSystem:
    """
    Central system that orchestrates all M4 Pro optimizations.

    Provides unified interface for Einstein/Bolt integration with full
    hardware acceleration and optimization capabilities.
    """

    def __init__(self, enable_all_optimizations: bool = True):
        self.enable_all_optimizations = enable_all_optimizations
        self.status = M4ProSystemStatus()
        self.initialization_start_time: float | None = None

        # Component references
        self.unified_memory: UnifiedMemoryManager | None = None
        self.metal_search: MetalAcceleratedSearch | None = None
        self.adaptive_concurrency: AdaptiveConcurrencyManager | None = None
        self.memory_pools: MemoryPoolManager | None = None
        self.ane_generator: ANEEmbeddingGenerator | None = None
        self.bolt_integration: BoltIntegration | None = None

        # Performance tracking
        self.performance_metrics: dict[str, Any] = {}
        self.last_benchmark_results: dict[str, Any] | None = None

        logger.info("M4 Pro Optimized System initialized")

    async def initialize_all_components(self) -> M4ProSystemStatus:
        """Initialize all optimization components"""
        async with production_error_handling(
            "M4ProOptimizedSystem", "initialize_all_components"
        ):
            self.initialization_start_time = time.perf_counter()
            logger.info("Initializing M4 Pro optimization components...")

            try:
                # 1. Initialize Unified Memory Management
                await self._initialize_unified_memory()

                # 2. Initialize Metal-Accelerated Search
                await self._initialize_metal_search()

                # 3. Initialize Adaptive Concurrency
                await self._initialize_adaptive_concurrency()

                # 4. Initialize Memory Pools
                await self._initialize_memory_pools()

                # 5. Initialize ANE Acceleration
                await self._initialize_ane_acceleration()

                # 6. Initialize Bolt Integration
                await self._initialize_bolt_integration()

                # 7. Run validation benchmarks
                await self._run_validation_benchmarks()

                # Finalize initialization
                self.status.initialization_time_ms = (
                    time.perf_counter() - self.initialization_start_time
                ) * 1000

                logger.info(
                    f"M4 Pro system initialized in {self.status.initialization_time_ms:.1f}ms"
                )
                logger.info(f"Active components: {self.status.active_components}/5")

                return self.status

            except Exception as e:
                logger.error(f"Failed to initialize M4 Pro system: {e}")
                raise

    async def _initialize_unified_memory(self):
        """Initialize unified memory management"""
        async with production_error_handling(
            "M4ProOptimizedSystem", "initialize_unified_memory"
        ):
            try:
                self.unified_memory = get_unified_memory_manager()

                # Create default buffers for common use cases
                test_buffer = await self.unified_memory.allocate_buffer(
                    1024 * 1024,  # 1MB test buffer
                    BufferType.TEMPORARY,
                    "initialization_test",
                )

                # Test zero-copy operations
                import numpy as np

                test_data = np.random.randn(1000, 256).astype(np.float32)
                await test_buffer.copy_from_numpy(test_data)

                # Verify functionality
                numpy_view = await test_buffer.as_numpy(np.float32, test_data.shape)
                assert numpy_view.shape == test_data.shape

                self.unified_memory.release_buffer("initialization_test")

                self.status.unified_memory_active = True
                self.status.active_components += 1

                logger.info("✅ Unified Memory Management initialized and validated")

            except Exception as e:
                logger.error(f"❌ Unified Memory initialization failed: {e}")
                self.status.unified_memory_active = False

    async def _initialize_metal_search(self):
        """Initialize Metal-accelerated search"""
        async with production_error_handling(
            "M4ProOptimizedSystem", "initialize_metal_search"
        ):
            try:
                self.metal_search = await get_metal_search(embedding_dim=768)

                # Load test corpus for validation
                import numpy as np

                test_embeddings = np.random.randn(1000, 768).astype(np.float32)
                test_metadata = [
                    {"content": f"test_doc_{i}", "id": i} for i in range(1000)
                ]

                await self.metal_search.load_corpus(test_embeddings, test_metadata)

                # Test search functionality
                test_query = np.random.randn(1, 768).astype(np.float32)
                search_results = await self.metal_search.search(test_query, k=10)

                assert len(search_results) == 1
                assert len(search_results[0]) <= 10

                self.status.metal_search_active = True
                self.status.active_components += 1

                logger.info("✅ Metal-Accelerated Search initialized and validated")

            except Exception as e:
                logger.error(f"❌ Metal Search initialization failed: {e}")
                self.status.metal_search_active = False

    async def _initialize_adaptive_concurrency(self):
        """Initialize adaptive concurrency management"""
        async with production_error_handling(
            "M4ProOptimizedSystem", "initialize_adaptive_concurrency"
        ):
            try:
                self.adaptive_concurrency = get_adaptive_concurrency_manager()

                # Test concurrency functionality
                from .adaptive_concurrency import TaskType

                async def test_task():
                    await asyncio.sleep(0.01)
                    return "test_result"

                # Create the coroutine function, not the coroutine itself
                result = await self.adaptive_concurrency.execute_task(
                    test_task, TaskType.CPU_INTENSIVE, timeout=5.0
                )

                assert result == "test_result"

                self.status.adaptive_concurrency_active = True
                self.status.active_components += 1

                logger.info(
                    "✅ Adaptive Concurrency Management initialized and validated"
                )

            except Exception as e:
                logger.error(f"❌ Adaptive Concurrency initialization failed: {e}")
                self.status.adaptive_concurrency_active = False

    async def _initialize_memory_pools(self):
        """Initialize memory pool management"""
        async with production_error_handling(
            "M4ProOptimizedSystem", "initialize_memory_pools"
        ):
            try:
                self.memory_pools = get_memory_pool_manager()

                # Create optimized pools
                embedding_pool = self.memory_pools.create_embedding_pool(
                    "main_embeddings", 512
                )
                cache_pool = self.memory_pools.create_cache_pool("main_cache", 256)

                # Test pool functionality
                import numpy as np

                test_data = np.random.randn(100, 384).astype(np.float32)

                buffer = embedding_pool.allocate(test_data.nbytes, "test_embedding")
                await buffer.copy_from_numpy(test_data)

                # Test cache
                cache_pool.put("test_key", test_data)
                retrieved = cache_pool.get("test_key")
                assert retrieved is not None

                # Cleanup test data
                embedding_pool.deallocate("test_embedding")
                cache_pool.deallocate("test_key")

                self.status.memory_pools_active = True
                self.status.active_components += 1

                logger.info("✅ Memory Pool Management initialized and validated")

            except Exception as e:
                logger.error(f"❌ Memory Pools initialization failed: {e}")
                self.status.memory_pools_active = False

    async def _initialize_ane_acceleration(self):
        """Initialize Apple Neural Engine acceleration"""
        async with production_error_handling(
            "M4ProOptimizedSystem", "initialize_ane_acceleration"
        ):
            try:
                from .ane_acceleration import ComputeUnit

                self.ane_generator = await create_ane_embedding_generator(
                    "production_embeddings",
                    output_dim=768,
                    compute_units=ComputeUnit.CPU_AND_ANE,
                )

                # Test embedding generation
                test_inputs = ["test input for embedding generation"]
                embeddings = await self.ane_generator.generate_embeddings(test_inputs)

                assert embeddings.shape[0] == len(test_inputs)
                assert embeddings.shape[1] == 768

                self.status.ane_acceleration_active = True
                self.status.active_components += 1

                logger.info(
                    "✅ Apple Neural Engine acceleration initialized and validated"
                )

            except Exception as e:
                logger.warning(
                    f"⚠️ ANE acceleration initialization failed (optional): {e}"
                )
                self.status.ane_acceleration_active = False

    async def _initialize_bolt_integration(self):
        """Initialize Bolt integration with optimizations"""
        async with production_error_handling(
            "M4ProOptimizedSystem", "initialize_bolt_integration"
        ):
            try:
                self.bolt_integration = BoltIntegration(
                    num_agents=8, enable_error_handling=True
                )
                await self.bolt_integration.initialize()

                logger.info("✅ Bolt Integration initialized with M4 Pro optimizations")

            except Exception as e:
                logger.error(f"❌ Bolt Integration initialization failed: {e}")
                raise

    async def _run_validation_benchmarks(self):
        """Run validation benchmarks to ensure optimizations work"""
        async with production_error_handling(
            "M4ProOptimizedSystem", "run_validation_benchmarks"
        ):
            try:
                logger.info("Running validation benchmarks...")

                self.last_benchmark_results = await run_m4_pro_benchmarks()

                # Check if benchmarks passed
                summary = self.last_benchmark_results.get("summary", {})
                success_rate = summary.get("overall_success_rate", 0)

                if success_rate >= 80:  # 80% success threshold
                    self.status.benchmark_validation_passed = True
                    logger.info(
                        f"✅ Validation benchmarks passed ({success_rate:.1f}% success rate)"
                    )
                else:
                    logger.warning(
                        f"⚠️ Validation benchmarks partially failed ({success_rate:.1f}% success rate)"
                    )

            except Exception as e:
                logger.error(f"❌ Validation benchmarks failed: {e}")
                self.status.benchmark_validation_passed = False

    async def get_system_performance_report(self) -> dict[str, Any]:
        """Get comprehensive system performance report"""
        report = {
            "system_status": {
                "unified_memory_active": self.status.unified_memory_active,
                "metal_search_active": self.status.metal_search_active,
                "adaptive_concurrency_active": self.status.adaptive_concurrency_active,
                "memory_pools_active": self.status.memory_pools_active,
                "ane_acceleration_active": self.status.ane_acceleration_active,
                "benchmark_validation_passed": self.status.benchmark_validation_passed,
                "active_components": self.status.active_components,
                "initialization_time_ms": self.status.initialization_time_ms,
            },
            "component_stats": {},
        }

        # Collect component statistics
        try:
            if self.unified_memory:
                report["component_stats"][
                    "unified_memory"
                ] = self.unified_memory.get_memory_stats()

            if self.metal_search:
                report["component_stats"][
                    "metal_search"
                ] = self.metal_search.get_performance_stats()

            if self.adaptive_concurrency:
                report["component_stats"][
                    "adaptive_concurrency"
                ] = self.adaptive_concurrency.get_performance_metrics()

            if self.memory_pools:
                report["component_stats"][
                    "memory_pools"
                ] = self.memory_pools.get_global_stats()

            if self.ane_generator:
                report["component_stats"][
                    "ane_acceleration"
                ] = self.ane_generator.get_performance_stats()

        except Exception as e:
            logger.warning(f"Error collecting component stats: {e}")

        # Include latest benchmark results
        if self.last_benchmark_results:
            report["latest_benchmarks"] = self.last_benchmark_results

        return report

    async def optimize_for_workload(self, workload_profile: dict[str, Any]):
        """Optimize system for specific workload characteristics"""
        async with production_error_handling(
            "M4ProOptimizedSystem", "optimize_for_workload"
        ):
            logger.info(f"Optimizing system for workload: {workload_profile}")

            # Extract workload parameters
            typical_batch_size = workload_profile.get("batch_size", 32)
            typical_query_size = workload_profile.get("query_size", 20)
            embedding_dim = workload_profile.get("embedding_dim", 768)
            workload_profile.get("concurrent_requests", 10)

            try:
                # Optimize Metal search
                if self.metal_search:
                    await self.metal_search.optimize_for_workload(
                        typical_batch_size, typical_query_size
                    )

                # Optimize ANE generator
                if self.ane_generator:
                    self.ane_generator.optimize_for_workload(
                        typical_batch_size, embedding_dim
                    )

                # Update concurrency limits
                if self.adaptive_concurrency:
                    # Let adaptive concurrency learn from the workload
                    pass  # It will adapt automatically

                logger.info("System optimization for workload completed")

            except Exception as e:
                logger.error(f"Workload optimization failed: {e}")

    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down M4 Pro optimized system...")

        try:
            if self.bolt_integration:
                await self.bolt_integration.shutdown()

            if self.adaptive_concurrency:
                self.adaptive_concurrency.shutdown()

            if self.memory_pools:
                self.memory_pools.shutdown()

            logger.info("M4 Pro system shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global system instance
_m4_pro_system: M4ProOptimizedSystem | None = None


async def initialize_m4_pro_system() -> M4ProOptimizedSystem:
    """Initialize and get global M4 Pro optimized system"""
    global _m4_pro_system

    if _m4_pro_system is None:
        _m4_pro_system = M4ProOptimizedSystem()
        await _m4_pro_system.initialize_all_components()

    return _m4_pro_system


def get_m4_pro_system() -> M4ProOptimizedSystem | None:
    """Get global M4 Pro system (if initialized)"""
    return _m4_pro_system


async def deploy_einstein_with_m4_pro_optimizations() -> dict[str, Any]:
    """Deploy Einstein with full M4 Pro optimizations"""
    async with production_error_handling(
        "M4ProOptimizedSystem", "deploy_einstein_with_m4_pro_optimizations"
    ):
        logger.info("Deploying Einstein with M4 Pro optimizations...")

        try:
            # Initialize database connection manager first to avoid lock conflicts
            from .database_connection_manager import get_database_pool

            # Pre-initialize database pools to avoid lock conflicts
            db_paths = [
                ".einstein/analytics.db",
                ".einstein/embeddings.db",
                "data/wheel_trading_master.duckdb",
            ]

            for db_path in db_paths:
                if Path(db_path).exists():
                    pool = get_database_pool(db_path, pool_size=4, db_type="duckdb")
                    await pool.initialize()
                    logger.info(f"Initialized database pool for {db_path}")

            # Initialize the optimized system
            system = await initialize_m4_pro_system()

            # Get deployment report
            report = await system.get_system_performance_report()

            # Log deployment status
            status = report["system_status"]
            active_components = status["active_components"]

            logger.info(
                f"Einstein deployment completed with {active_components}/5 optimizations active"
            )

            if status["benchmark_validation_passed"]:
                logger.info("✅ All performance validations passed")
            else:
                logger.warning("⚠️ Some performance validations failed")

            return report

        except Exception as e:
            logger.error(f"Einstein deployment failed: {e}")
            raise
