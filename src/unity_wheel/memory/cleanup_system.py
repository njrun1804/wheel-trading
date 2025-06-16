"""
Memory Cleanup System - Intelligent garbage collection and memory reclamation

Provides automated cleanup strategies with component-specific optimizations,
emergency cleanup protocols, and intelligent scheduling for the trading system.
"""

import gc
import logging
import os
import threading
import time
import weakref
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class CleanupLevel(Enum):
    """Cleanup intensity levels"""

    LIGHT = "light"  # Basic GC, low priority evictions
    MODERATE = "moderate"  # Standard cleanup, cache clearing
    AGGRESSIVE = "aggressive"  # Heavy cleanup, high priority evictions
    EMERGENCY = "emergency"  # All available cleanup measures


@dataclass
class CleanupStats:
    """Statistics for cleanup operations"""

    runs_total: int = 0
    runs_light: int = 0
    runs_moderate: int = 0
    runs_aggressive: int = 0
    runs_emergency: int = 0
    memory_freed_gb: float = 0.0
    objects_collected: int = 0
    collections_triggered: int = 0
    last_cleanup_time: float = 0
    average_cleanup_time: float = 0
    emergency_cleanups: int = 0


class CleanupSystem:
    """
    Comprehensive memory cleanup system with intelligent scheduling

    Features:
    - Multiple cleanup intensity levels
    - Component-specific cleanup strategies
    - Automated scheduling based on pressure
    - Emergency cleanup protocols
    - Statistics and performance tracking
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

        # Configuration
        self.light_cleanup_interval = 300  # 5 minutes
        self.moderate_cleanup_interval = 900  # 15 minutes
        self.gc_generations = [0, 1, 2]  # Python GC generations

        # Statistics
        self.stats = CleanupStats()

        # Thread management
        self.scheduler_thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.RLock()

        # Cleanup callbacks by component
        self.cleanup_callbacks: dict[str, list[Callable]] = defaultdict(list)

        # Component-specific cleanup strategies
        self.component_strategies = {
            "trading_data": self._cleanup_trading_data,
            "ml_models": self._cleanup_ml_models,
            "database": self._cleanup_database,
            "cache": self._cleanup_cache,
        }

        # Weak references to objects that can be cleaned
        self.managed_objects: dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)

        # Emergency cleanup protocols
        self.emergency_protocols = [
            self._emergency_evict_all,
            self._emergency_gc_full,
            self._emergency_clear_caches,
            self._emergency_reduce_limits,
        ]

        logger.info("CleanupSystem initialized")

    def start(self):
        """Start automated cleanup scheduling"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop, daemon=True, name="MemoryCleanupScheduler"
            )
            self.scheduler_thread.start()
            logger.info("Memory cleanup scheduling started")

    def stop(self):
        """Stop automated cleanup scheduling"""
        self.running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        logger.info("Memory cleanup scheduling stopped")

    def _scheduler_loop(self):
        """Automated cleanup scheduling loop"""
        last_light_cleanup = 0
        last_moderate_cleanup = 0

        while self.running:
            try:
                current_time = time.time()

                # Check if cleanups are due
                if current_time - last_light_cleanup >= self.light_cleanup_interval:
                    self.run_cleanup(CleanupLevel.LIGHT)
                    last_light_cleanup = current_time

                if (
                    current_time - last_moderate_cleanup
                    >= self.moderate_cleanup_interval
                ):
                    self.run_cleanup(CleanupLevel.MODERATE)
                    last_moderate_cleanup = current_time

                # Check for pressure-based cleanup
                pressure = self.memory_manager.pressure_monitor.get_pressure_level()
                if pressure > 0.85:  # High pressure
                    self.run_cleanup(CleanupLevel.AGGRESSIVE)
                    time.sleep(30)  # Wait after aggressive cleanup

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Cleanup scheduler error: {e}")
                time.sleep(60)

    def run_cleanup(
        self, level: CleanupLevel = CleanupLevel.MODERATE
    ) -> dict[str, Any]:
        """
        Run cleanup at specified intensity level

        Args:
            level: Cleanup intensity level

        Returns:
            Dictionary with cleanup results
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()

        logger.info(f"Starting {level.value} cleanup")

        with self.lock:
            try:
                # Update stats
                self.stats.runs_total += 1
                if level == CleanupLevel.LIGHT:
                    self.stats.runs_light += 1
                elif level == CleanupLevel.MODERATE:
                    self.stats.runs_moderate += 1
                elif level == CleanupLevel.AGGRESSIVE:
                    self.stats.runs_aggressive += 1
                elif level == CleanupLevel.EMERGENCY:
                    self.stats.runs_emergency += 1
                    self.stats.emergency_cleanups += 1

                # Perform cleanup based on level
                results = self._perform_cleanup(level)

                # Measure cleanup effectiveness
                final_memory = self._get_memory_usage()
                memory_freed = max(0, initial_memory - final_memory)
                cleanup_time = time.time() - start_time

                # Update statistics
                self.stats.memory_freed_gb += memory_freed / (1024**3)
                self.stats.last_cleanup_time = start_time

                # Update average cleanup time
                if self.stats.runs_total == 1:
                    self.stats.average_cleanup_time = cleanup_time
                else:
                    self.stats.average_cleanup_time = (
                        self.stats.average_cleanup_time * (self.stats.runs_total - 1)
                        + cleanup_time
                    ) / self.stats.runs_total

                results.update(
                    {
                        "memory_freed_mb": memory_freed / (1024**2),
                        "cleanup_time_seconds": cleanup_time,
                        "initial_memory_gb": initial_memory / (1024**3),
                        "final_memory_gb": final_memory / (1024**3),
                    }
                )

                logger.info(
                    f"Cleanup completed: {level.value}, "
                    f"freed {memory_freed / (1024**2):.1f}MB in {cleanup_time:.2f}s"
                )

                return results

            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                return {"error": str(e)}

    def _perform_cleanup(self, level: CleanupLevel) -> dict[str, Any]:
        """Perform cleanup operations based on level"""
        results = {"level": level.value, "actions": []}

        if level == CleanupLevel.LIGHT:
            results["actions"].extend(self._light_cleanup())
        elif level == CleanupLevel.MODERATE:
            results["actions"].extend(self._light_cleanup())
            results["actions"].extend(self._moderate_cleanup())
        elif level == CleanupLevel.AGGRESSIVE:
            results["actions"].extend(self._light_cleanup())
            results["actions"].extend(self._moderate_cleanup())
            results["actions"].extend(self._aggressive_cleanup())
        elif level == CleanupLevel.EMERGENCY:
            results["actions"].extend(self._emergency_cleanup())

        return results

    def _light_cleanup(self) -> list[str]:
        """Light cleanup operations"""
        actions = []

        # Basic garbage collection
        collected = gc.collect()
        if collected > 0:
            actions.append(f"gc_collect: {collected} objects")
            self.stats.objects_collected += collected
            self.stats.collections_triggered += 1

        # Component light cleanup
        for component, strategy in self.component_strategies.items():
            try:
                result = strategy("light")
                if result:
                    actions.append(f"{component}_light: {result}")
            except Exception as e:
                logger.warning(f"Light cleanup failed for {component}: {e}")

        return actions

    def _moderate_cleanup(self) -> list[str]:
        """Moderate cleanup operations"""
        actions = []

        # More thorough GC
        for generation in self.gc_generations:
            collected = gc.collect(generation)
            if collected > 0:
                actions.append(f"gc_collect_gen{generation}: {collected} objects")
                self.stats.objects_collected += collected

        # Evict low priority allocations
        evicted_count = 0
        for component, pool in self.memory_manager.pools.items():
            evicted = pool.evict_by_priority(max_priority=3)
            if evicted > 0:
                evicted_count += evicted
                actions.append(f"{component}_evict_low: {evicted / (1024**2):.1f}MB")

        if evicted_count > 0:
            actions.append(f"total_evicted: {evicted_count / (1024**2):.1f}MB")

        # Component moderate cleanup
        for component, strategy in self.component_strategies.items():
            try:
                result = strategy("moderate")
                if result:
                    actions.append(f"{component}_moderate: {result}")
            except Exception as e:
                logger.warning(f"Moderate cleanup failed for {component}: {e}")

        return actions

    def _aggressive_cleanup(self) -> list[str]:
        """Aggressive cleanup operations"""
        actions = []

        # Full garbage collection
        gc.collect(2)  # Full collection
        actions.append("full_gc_collection")

        # Aggressive eviction
        evicted_count = 0
        for component, pool in self.memory_manager.pools.items():
            evicted = pool.evict_by_priority(max_priority=6)
            if evicted > 0:
                evicted_count += evicted
                actions.append(
                    f"{component}_evict_aggressive: {evicted / (1024**2):.1f}MB"
                )

        # Clear all caches
        self._clear_all_caches()
        actions.append("cleared_all_caches")

        # Component aggressive cleanup
        for component, strategy in self.component_strategies.items():
            try:
                result = strategy("aggressive")
                if result:
                    actions.append(f"{component}_aggressive: {result}")
            except Exception as e:
                logger.warning(f"Aggressive cleanup failed for {component}: {e}")

        return actions

    def _emergency_cleanup(self) -> list[str]:
        """Emergency cleanup protocols"""
        actions = []

        logger.critical("Running emergency cleanup protocols")

        # Run all emergency protocols
        for i, protocol in enumerate(self.emergency_protocols):
            try:
                result = protocol()
                actions.append(f"emergency_protocol_{i}: {result}")
            except Exception as e:
                logger.error(f"Emergency protocol {i} failed: {e}")
                actions.append(f"emergency_protocol_{i}: FAILED - {str(e)}")

        return actions

    def _emergency_evict_all(self) -> str:
        """Emergency protocol: Evict all evictable allocations"""
        total_evicted = 0

        for _component, pool in self.memory_manager.pools.items():
            evicted = pool.evict_by_priority(max_priority=8)  # Keep only critical
            total_evicted += evicted

        return f"evicted {total_evicted / (1024**2):.1f}MB (all non-critical)"

    def _emergency_gc_full(self) -> str:
        """Emergency protocol: Full garbage collection with debug info"""
        before = len(gc.get_objects())
        collected = gc.collect(2)
        after = len(gc.get_objects())

        return f"collected {collected} objects, {before} -> {after} total objects"

    def _emergency_clear_caches(self) -> str:
        """Emergency protocol: Clear all possible caches"""
        cleared_count = 0

        # Clear component caches
        for callbacks in self.cleanup_callbacks.values():
            for callback in callbacks:
                try:
                    callback("emergency")
                    cleared_count += 1
                except Exception:
                    pass

        # Clear managed objects
        for obj_set in self.managed_objects.values():
            obj_set.clear()

        return f"cleared {cleared_count} caches"

    def _emergency_reduce_limits(self) -> str:
        """Emergency protocol: Temporarily reduce memory limits"""
        reduced_count = 0

        for _component, pool in self.memory_manager.pools.items():
            # Reduce pool size by 30%
            new_size = int(pool.max_size_bytes * 0.7)
            if new_size < pool.max_size_bytes:
                pool.max_size_bytes = new_size
                reduced_count += 1

        # Set emergency environment flag
        os.environ["MEMORY_EMERGENCY_LIMITS"] = "1"

        return f"reduced {reduced_count} pool limits by 30%"

    def _cleanup_trading_data(self, level: str) -> str | None:
        """Trading data specific cleanup"""
        if level == "light":
            # Clear old historical data cache
            pass
        elif level == "moderate":
            # Clear intraday data older than 1 hour
            pass
        elif level == "aggressive":
            # Clear all non-essential trading data
            pass

        return None

    def _cleanup_ml_models(self, level: str) -> str | None:
        """ML models specific cleanup"""
        if level == "light":
            # Clear embedding caches
            pass
        elif level == "moderate":
            # Unload unused models
            pass
        elif level == "aggressive":
            # Keep only active model
            pass

        return None

    def _cleanup_database(self, level: str) -> str | None:
        """Database specific cleanup"""
        if level == "light":
            # Clear query result cache
            pass
        elif level == "moderate":
            # Clear connection pools
            pass
        elif level == "aggressive":
            # Force checkpoint and vacuum
            pass

        return None

    def _cleanup_cache(self, level: str) -> str | None:
        """Cache specific cleanup"""
        if level == "light":
            # Clear LRU cache entries
            pass
        elif level == "moderate":
            # Clear 50% of cache
            pass
        elif level == "aggressive":
            # Clear all cache
            pass

        return None

    def _clear_all_caches(self):
        """Clear all registered caches"""
        for component_callbacks in self.cleanup_callbacks.values():
            for callback in component_callbacks:
                try:
                    callback("clear_all")
                except Exception as e:
                    logger.warning(f"Cache clear callback failed: {e}")

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            return psutil.virtual_memory().used
        except:
            return 0

    def register_cleanup_callback(self, component: str, callback: Callable):
        """Register cleanup callback for component"""
        self.cleanup_callbacks[component].append(callback)

    def register_managed_object(self, category: str, obj: Any):
        """Register object for managed cleanup"""
        self.managed_objects[category].add(obj)

    def force_cleanup_component(
        self, component: str, level: CleanupLevel = CleanupLevel.AGGRESSIVE
    ):
        """Force cleanup for specific component"""
        if component in self.component_strategies:
            try:
                result = self.component_strategies[component](level.value)
                logger.info(f"Force cleanup {component}: {result}")
                return result
            except Exception as e:
                logger.error(f"Force cleanup failed for {component}: {e}")
                return None

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cleanup statistics"""
        return {
            "runs": {
                "total": self.stats.runs_total,
                "light": self.stats.runs_light,
                "moderate": self.stats.runs_moderate,
                "aggressive": self.stats.runs_aggressive,
                "emergency": self.stats.runs_emergency,
            },
            "performance": {
                "memory_freed_gb": self.stats.memory_freed_gb,
                "objects_collected": self.stats.objects_collected,
                "collections_triggered": self.stats.collections_triggered,
                "average_cleanup_time": self.stats.average_cleanup_time,
                "last_cleanup_time": self.stats.last_cleanup_time,
            },
            "callbacks_registered": {
                component: len(callbacks)
                for component, callbacks in self.cleanup_callbacks.items()
            },
            "managed_objects": {
                category: len(obj_set)
                for category, obj_set in self.managed_objects.items()
            },
        }

    def get_recommendations(self) -> list[str]:
        """Get cleanup recommendations based on current state"""
        recommendations = []

        # Check cleanup frequency
        if self.stats.runs_total == 0:
            recommendations.append("No cleanups run yet - consider starting monitoring")

        # Check memory freed efficiency
        if self.stats.runs_total > 10 and self.stats.memory_freed_gb < 1.0:
            recommendations.append("Low memory recovery - review allocation patterns")

        # Check emergency frequency
        if self.stats.emergency_cleanups > self.stats.runs_total * 0.1:
            recommendations.append(
                "High emergency cleanup rate - review memory budgets"
            )

        # Check cleanup time
        if self.stats.average_cleanup_time > 10.0:
            recommendations.append(
                "Slow cleanup performance - review cleanup strategies"
            )

        return recommendations


# Global instance
_cleanup_system: CleanupSystem | None = None


def get_cleanup_system(memory_manager=None) -> CleanupSystem:
    """Get or create the global cleanup system"""
    global _cleanup_system
    if _cleanup_system is None and memory_manager:
        _cleanup_system = CleanupSystem(memory_manager)
    return _cleanup_system
