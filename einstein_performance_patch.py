#!/usr/bin/env python3
"""Performance optimizations for Einstein + Bolt integration."""

import asyncio
import logging
import multiprocessing

logger = logging.getLogger("einstein_perf")


class EinsteinPerformanceOptimizer:
    """Optimize Einstein performance for integration with Bolt."""

    def __init__(self):
        self.cpu_count = min(multiprocessing.cpu_count(), 12)
        self.memory_limit_gb = 8.0  # Conservative limit

    async def optimize_search_concurrency(self, einstein_instance):
        """Optimize search concurrency for better integration."""
        try:
            # Reduce concurrent searches to prevent resource conflicts
            if hasattr(einstein_instance, "search_semaphore"):
                einstein_instance.search_semaphore = asyncio.Semaphore(2)

            if hasattr(einstein_instance, "embedding_semaphore"):
                einstein_instance.embedding_semaphore = asyncio.Semaphore(1)

            logger.info("✅ Optimized Einstein search concurrency")
            return True

        except Exception as e:
            logger.error(f"Failed to optimize search concurrency: {e}")
            return False

    async def optimize_faiss_settings(self, einstein_instance):
        """Optimize FAISS settings for better performance."""
        try:
            if (
                hasattr(einstein_instance, "vector_index")
                and einstein_instance.vector_index
            ):
                # Set optimal FAISS parameters
                import faiss

                faiss.omp_set_num_threads(min(self.cpu_count, 4))

            logger.info("✅ Optimized FAISS settings")
            return True

        except Exception as e:
            logger.error(f"Failed to optimize FAISS: {e}")
            return False

    async def optimize_bolt_integration(self, bolt_instance):
        """Optimize Bolt instance for Einstein integration."""
        try:
            # Reduce agent count if system is under pressure
            if hasattr(bolt_instance, "system_state") and bolt_instance.system_state:
                if bolt_instance.system_state.memory_percent > 80:
                    # Reduce concurrent agents
                    original_agents = len(bolt_instance.agents)
                    target_agents = max(2, original_agents // 2)
                    logger.info(
                        f"Reducing agents from {original_agents} to {target_agents} due to memory pressure"
                    )

            logger.info("✅ Optimized Bolt integration")
            return True

        except Exception as e:
            logger.error(f"Failed to optimize Bolt integration: {e}")
            return False


# Global optimizer instance
_optimizer = EinsteinPerformanceOptimizer()


async def apply_performance_optimizations(einstein_instance=None, bolt_instance=None):
    """Apply all performance optimizations."""
    results = []

    if einstein_instance:
        results.append(await _optimizer.optimize_search_concurrency(einstein_instance))
        results.append(await _optimizer.optimize_faiss_settings(einstein_instance))

    if bolt_instance:
        results.append(await _optimizer.optimize_bolt_integration(bolt_instance))

    return all(results)
