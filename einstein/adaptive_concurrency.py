#!/usr/bin/env python3
"""
Adaptive Concurrency Manager for Einstein

Dynamically adjusts concurrency limits based on:
- System resource utilization (CPU, memory)
- Historical performance metrics
- Operation type characteristics
- M4 Pro hardware optimization
"""

import asyncio
import psutil
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric for an operation."""
    operation_type: str
    duration_ms: float
    concurrency_level: int
    cpu_utilization: float
    memory_usage_mb: float
    timestamp: float


@dataclass
class ConcurrencyConfig:
    """Concurrency configuration for an operation type."""
    operation_type: str
    base_limit: int
    min_limit: int
    max_limit: int
    target_duration_ms: float
    current_limit: int


class AdaptiveConcurrencyManager:
    """Manages dynamic concurrency limits for optimal M4 Pro performance."""
    
    def __init__(self, cpu_cores: int = 12):
        self.cpu_cores = cpu_cores
        self.base_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Performance history (sliding window)
        self.performance_history: Dict[str, deque] = {}
        self.history_window_size = 50
        
        # Concurrency configurations for different operation types
        self.configs = {
            'text_search': ConcurrencyConfig(
                operation_type='text_search',
                base_limit=self.cpu_cores,        # Can use all cores
                min_limit=1,
                max_limit=self.cpu_cores * 2,    # Hyperthreading
                target_duration_ms=5.0,          # Very fast target
                current_limit=self.cpu_cores
            ),
            'semantic_search': ConcurrencyConfig(
                operation_type='semantic_search',
                base_limit=4,                     # GPU/MLX bound
                min_limit=1,
                max_limit=8,                      # Don't overwhelm GPU
                target_duration_ms=20.0,         # Reasonable for embeddings
                current_limit=4
            ),
            'structural_search': ConcurrencyConfig(
                operation_type='structural_search',
                base_limit=6,                     # Graph traversal
                min_limit=2,
                max_limit=self.cpu_cores,
                target_duration_ms=10.0,
                current_limit=6
            ),
            'analytical_search': ConcurrencyConfig(
                operation_type='analytical_search',
                base_limit=8,                     # DuckDB can use multiple cores
                min_limit=2,
                max_limit=self.cpu_cores,
                target_duration_ms=15.0,
                current_limit=8
            ),
            'file_analysis': ConcurrencyConfig(
                operation_type='file_analysis',
                base_limit=self.cpu_cores // 2,  # I/O + CPU bound
                min_limit=2,
                max_limit=self.cpu_cores,
                target_duration_ms=100.0,        # File analysis takes longer
                current_limit=self.cpu_cores // 2
            ),
            'embedding_generation': ConcurrencyConfig(
                operation_type='embedding_generation',
                base_limit=2,                     # MLX GPU bound
                min_limit=1,
                max_limit=4,                      # Limited by GPU memory
                target_duration_ms=50.0,
                current_limit=2
            )
        }
        
        # Initialize performance history
        for op_type in self.configs.keys():
            self.performance_history[op_type] = deque(maxlen=self.history_window_size)
        
        # System monitoring
        self.system_load_history = deque(maxlen=20)
        self.adjustment_cooldown = {}  # Last adjustment time per operation
        self.cooldown_period = 5.0  # Seconds between adjustments
        
        logger.info(f"Adaptive concurrency manager initialized for {self.cpu_cores}-core M4 Pro")
    
    async def get_semaphore(self, operation_type: str) -> asyncio.Semaphore:
        """Get an adaptive semaphore for the given operation type."""
        
        if operation_type not in self.configs:
            logger.warning(f"Unknown operation type: {operation_type}, using default")
            return asyncio.Semaphore(4)
        
        config = self.configs[operation_type]
        
        # Update concurrency limit based on recent performance
        await self._update_concurrency_limit(operation_type)
        
        return asyncio.Semaphore(config.current_limit)
    
    async def record_performance(self, operation_type: str, duration_ms: float, 
                               concurrency_level: int) -> None:
        """Record performance metrics for an operation."""
        
        # Get current system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / (1024**2)
        
        # Create performance metric
        metric = PerformanceMetric(
            operation_type=operation_type,
            duration_ms=duration_ms,
            concurrency_level=concurrency_level,
            cpu_utilization=cpu_percent,
            memory_usage_mb=memory_mb,
            timestamp=time.time()
        )
        
        # Store in history
        if operation_type not in self.performance_history:
            self.performance_history[operation_type] = deque(maxlen=self.history_window_size)
        
        self.performance_history[operation_type].append(metric)
        
        # Update system load history
        self.system_load_history.append((cpu_percent, memory_info.percent, time.time()))
    
    async def _update_concurrency_limit(self, operation_type: str) -> None:
        """Update concurrency limit based on performance history."""
        
        config = self.configs[operation_type]
        
        # Check cooldown period
        last_adjustment = self.adjustment_cooldown.get(operation_type, 0)
        if time.time() - last_adjustment < self.cooldown_period:
            return
        
        # Get recent performance data
        history = self.performance_history[operation_type]
        if len(history) < 5:  # Need some data to make decisions
            return
        
        # Calculate recent average performance
        recent_metrics = list(history)[-10:]  # Last 10 operations
        avg_duration = sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
        
        # Get current system load
        current_cpu = psutil.cpu_percent(interval=None)
        current_memory = psutil.virtual_memory().percent
        
        # Decision logic
        old_limit = config.current_limit
        new_limit = old_limit
        
        # Performance-based adjustment
        if avg_duration > config.target_duration_ms * 1.5:
            # Performance is poor, reduce concurrency if CPU is high
            if avg_cpu > 80 or current_cpu > 85:
                new_limit = max(config.min_limit, old_limit - 1)
                logger.debug(f"Reducing {operation_type} concurrency: {old_limit} -> {new_limit} (high CPU)")
        
        elif avg_duration < config.target_duration_ms * 0.7:
            # Performance is good, increase concurrency if resources available
            if avg_cpu < 60 and current_cpu < 70 and current_memory < 80:
                new_limit = min(config.max_limit, old_limit + 1)
                logger.debug(f"Increasing {operation_type} concurrency: {old_limit} -> {new_limit} (low resource usage)")
        
        # System load-based adjustment
        if current_cpu > 90 or current_memory > 90:
            # System under heavy load, be conservative
            new_limit = max(config.min_limit, min(new_limit, config.base_limit // 2))
            logger.debug(f"System overload, limiting {operation_type}: {old_limit} -> {new_limit}")
        
        # Update if changed
        if new_limit != old_limit:
            config.current_limit = new_limit
            self.adjustment_cooldown[operation_type] = time.time()
            
            logger.info(f"Adjusted {operation_type} concurrency: {old_limit} -> {new_limit} "
                       f"(avg_duration: {avg_duration:.1f}ms, target: {config.target_duration_ms:.1f}ms, "
                       f"CPU: {current_cpu:.1f}%)")
    
    def get_system_health(self) -> Dict[str, float]:
        """Get current system health metrics."""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Calculate load trends
        cpu_trend = 0.0
        if len(self.system_load_history) >= 3:
            recent_loads = [load[0] for load in list(self.system_load_history)[-3:]]
            cpu_trend = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_trend': cpu_trend,
            'system_health_score': self._calculate_health_score(cpu_percent, memory.percent)
        }
    
    def _calculate_health_score(self, cpu_percent: float, memory_percent: float) -> float:
        """Calculate overall system health score (0-1, higher is better)."""
        
        # CPU score (exponential decay after 70%)
        if cpu_percent <= 70:
            cpu_score = 1.0
        else:
            cpu_score = max(0.0, 1.0 - ((cpu_percent - 70) / 30) ** 2)
        
        # Memory score (linear decay after 80%)
        if memory_percent <= 80:
            memory_score = 1.0
        else:
            memory_score = max(0.0, 1.0 - (memory_percent - 80) / 20)
        
        # Combined score (weighted average)
        return (cpu_score * 0.6 + memory_score * 0.4)
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all operation types."""
        
        summary = {}
        
        for op_type, history in self.performance_history.items():
            if not history:
                continue
            
            config = self.configs[op_type]
            recent_metrics = list(history)[-20:]  # Last 20 operations
            
            avg_duration = sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
            min_duration = min(m.duration_ms for m in recent_metrics)
            max_duration = max(m.duration_ms for m in recent_metrics)
            avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
            
            summary[op_type] = {
                'current_limit': config.current_limit,
                'target_duration_ms': config.target_duration_ms,
                'avg_duration_ms': avg_duration,
                'min_duration_ms': min_duration,
                'max_duration_ms': max_duration,
                'avg_cpu_percent': avg_cpu,
                'sample_count': len(recent_metrics),
                'performance_ratio': config.target_duration_ms / avg_duration if avg_duration > 0 else 1.0
            }
        
        return summary
    
    async def optimize_for_batch_operation(self, operation_type: str, 
                                         estimated_operations: int) -> None:
        """Temporarily optimize concurrency for a large batch operation."""
        
        if operation_type not in self.configs:
            return
        
        config = self.configs[operation_type]
        system_health = self.get_system_health()
        
        # For large batches, be more aggressive with concurrency if system is healthy
        if system_health['system_health_score'] > 0.8 and estimated_operations > 50:
            # Temporarily increase limit for batch processing
            batch_limit = min(config.max_limit, config.current_limit + 2)
            
            logger.info(f"Optimizing {operation_type} for batch operation: "
                       f"{config.current_limit} -> {batch_limit} "
                       f"({estimated_operations} operations)")
            
            # Store original limit
            original_limit = config.current_limit
            config.current_limit = batch_limit
            
            # Reset after batch (simplified - in practice you'd want a callback)
            async def reset_after_delay():
                await asyncio.sleep(60)  # Reset after 1 minute
                config.current_limit = original_limit
                logger.info(f"Reset {operation_type} concurrency to {original_limit}")
            
            asyncio.create_task(reset_after_delay())


# Global instance
_adaptive_manager: Optional[AdaptiveConcurrencyManager] = None


def get_adaptive_concurrency_manager() -> AdaptiveConcurrencyManager:
    """Get the global adaptive concurrency manager."""
    global _adaptive_manager
    if _adaptive_manager is None:
        _adaptive_manager = AdaptiveConcurrencyManager()
    return _adaptive_manager


# Context manager for performance tracking
class PerformanceTracker:
    """Context manager to automatically track operation performance."""
    
    def __init__(self, operation_type: str, concurrency_level: int = 1):
        self.operation_type = operation_type
        self.concurrency_level = concurrency_level
        self.start_time = None
        self.manager = get_adaptive_concurrency_manager()
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            await self.manager.record_performance(
                self.operation_type, 
                duration_ms, 
                self.concurrency_level
            )


if __name__ == "__main__":
    # Test the adaptive concurrency manager
    async def test_adaptive_concurrency():
        manager = get_adaptive_concurrency_manager()
        
        # Simulate some operations
        for i in range(10):
            # Test text search
            async with PerformanceTracker('text_search', 4):
                await asyncio.sleep(0.005)  # 5ms operation
            
            # Test semantic search  
            async with PerformanceTracker('semantic_search', 2):
                await asyncio.sleep(0.025)  # 25ms operation
        
        # Print performance summary
        summary = manager.get_performance_summary()
        print("Performance Summary:")
        for op_type, metrics in summary.items():
            print(f"  {op_type}:")
            for key, value in metrics.items():
                print(f"    {key}: {value}")
        
        # Print system health
        health = manager.get_system_health()
        print(f"\nSystem Health: {health}")
    
    asyncio.run(test_adaptive_concurrency())