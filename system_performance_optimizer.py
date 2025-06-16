#!/usr/bin/env python3
"""
Comprehensive System Performance Optimizer for M4 Pro Architecture
Optimizes memory, CPU, I/O, and resource utilization across all components
"""

import asyncio
import gc
import logging
import mmap
import os
import resource
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    
    # M4 Pro specific
    p_cores: int = 8
    e_cores: int = 4
    total_cores: int = 12
    total_memory_gb: float = 24.0
    
    # Memory optimization
    memory_target_percent: float = 80.0  # Use up to 80% of available memory
    gc_threshold_mb: int = 1000  # Trigger GC when memory exceeds this
    buffer_pool_size_mb: int = 512  # Buffer pool size
    
    # CPU optimization
    process_pool_size: int = 8  # Use P-cores for compute-intensive tasks
    thread_pool_size: int = 16  # Higher for I/O bound tasks
    cpu_affinity_enabled: bool = True
    
    # I/O optimization
    max_concurrent_reads: int = 24
    read_buffer_size_kb: int = 64
    mmap_threshold_mb: int = 10  # Use mmap for files larger than this
    prefetch_size_mb: int = 32
    
    # Resource limits
    max_open_files: int = 4096
    stack_size_mb: int = 8
    
    # Performance monitoring
    monitoring_interval_seconds: float = 1.0
    performance_log_interval: int = 60  # seconds


@dataclass
class SystemMetrics:
    """Current system performance metrics"""
    
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    open_files: int
    active_threads: int
    gc_collections: int
    
    # Component-specific metrics
    einstein_memory_mb: float = 0.0
    jarvis2_memory_mb: float = 0.0
    unity_wheel_memory_mb: float = 0.0
    
    # Performance indicators
    bottleneck_type: Optional[str] = None
    optimization_suggestions: List[str] = None


class MemoryOptimizer:
    """Advanced memory optimization for M4 Pro unified memory"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.buffer_pool = {}
        self.memory_pressure_callbacks = []
        self._last_gc_time = time.time()
        
    def optimize_memory_allocation(self):
        """Optimize memory allocation patterns"""
        
        # Set garbage collection thresholds
        if hasattr(gc, 'set_threshold'):
            # More aggressive GC for better memory management
            gc.set_threshold(700, 10, 10)  # generation 0, 1, 2 thresholds
        
        # Enable malloc arena optimization (macOS specific)
        try:
            os.environ['MallocNanoZone'] = '0'  # Disable nano zone for large allocations
            os.environ['MallocScribble'] = '0'  # Disable scribbling for performance
        except Exception as e:
            logger.debug(f"Could not set malloc environment: {e}")
    
    def get_buffer(self, size_bytes: int) -> memoryview:
        """Get a reusable buffer to avoid allocations"""
        
        size_key = self._round_to_power_of_2(size_bytes)
        
        if size_key not in self.buffer_pool:
            self.buffer_pool[size_key] = []
        
        pool = self.buffer_pool[size_key]
        
        if pool:
            return pool.pop()
        else:
            # Create new buffer
            return memoryview(bytearray(size_key))
    
    def return_buffer(self, buffer: memoryview):
        """Return buffer to pool for reuse"""
        
        size_key = len(buffer)
        if size_key in self.buffer_pool:
            if len(self.buffer_pool[size_key]) < 10:  # Limit pool size
                self.buffer_pool[size_key].append(buffer)
    
    def _round_to_power_of_2(self, n: int) -> int:
        """Round to next power of 2 for buffer pooling"""
        return 1 << (n - 1).bit_length()
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        
        if not HAS_PSUTIL:
            return False
        
        try:
            memory = psutil.virtual_memory()
            pressure = memory.percent > 85.0
            
            if pressure:
                logger.warning(f"Memory pressure detected: {memory.percent:.1f}% used")
                self._trigger_memory_cleanup()
            
            return pressure
        except Exception as e:
            logger.debug(f"Could not check memory pressure: {e}")
            return False
    
    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup procedures"""
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Emergency GC collected {collected} objects")
        
        # Clear buffer pools
        for pool in self.buffer_pool.values():
            pool.clear()
            
        # Trigger registered callbacks
        for callback in self.memory_pressure_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Memory pressure callback failed: {e}")
    
    def register_pressure_callback(self, callback):
        """Register callback for memory pressure events"""
        self.memory_pressure_callbacks.append(callback)


class CPUOptimizer:
    """CPU optimization for M4 Pro 8P+4E architecture"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.process_pool = None
        self.thread_pool = None
        self.core_assignments = {}
        
    def initialize_pools(self):
        """Initialize optimized process and thread pools"""
        
        # Process pool for CPU-intensive tasks (use P-cores)
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config.process_pool_size,
            mp_context=None  # Use default multiprocessing context
        )
        
        # Thread pool for I/O-bound tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix="wheel_worker"
        )
        
        logger.info(f"Initialized CPU pools: {self.config.process_pool_size} processes, "
                   f"{self.config.thread_pool_size} threads")
    
    def set_cpu_affinity(self, core_list: List[int]):
        """Set CPU affinity for current process"""
        
        if not self.config.cpu_affinity_enabled or not HAS_PSUTIL:
            return
        
        try:
            p = psutil.Process()
            p.cpu_affinity(core_list)
            logger.info(f"Set CPU affinity to cores: {core_list}")
        except Exception as e:
            logger.debug(f"Could not set CPU affinity: {e}")
    
    def optimize_for_task_type(self, task_type: str) -> Dict[str, Any]:
        """Get optimized execution settings for different task types"""
        
        if task_type == "compute_intensive":
            return {
                "executor": self.process_pool,
                "max_workers": self.config.process_pool_size,
                "preferred_cores": list(range(8)),  # P-cores
            }
        elif task_type == "io_bound":
            return {
                "executor": self.thread_pool,
                "max_workers": self.config.thread_pool_size,
                "preferred_cores": list(range(8, 12)),  # E-cores
            }
        elif task_type == "ml_inference":
            # Use GPU when available, otherwise P-cores
            return {
                "executor": self.thread_pool,
                "max_workers": 4,  # Lower for GPU coordination
                "use_gpu": HAS_MLX,
                "preferred_cores": list(range(4)),  # Subset of P-cores
            }
        else:
            return {
                "executor": self.thread_pool,
                "max_workers": 8,
                "preferred_cores": list(range(12)),  # All cores
            }
    
    def shutdown_pools(self):
        """Shutdown executor pools"""
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


class IOOptimizer:
    """I/O performance optimization"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.read_cache = {}
        self.prefetch_cache = {}
        
    def optimize_file_reading(self, file_path: Path, use_mmap: bool = None) -> bytes:
        """Optimized file reading with caching and mmap"""
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = file_path.stat().st_size
        
        # Decide whether to use mmap
        if use_mmap is None:
            use_mmap = file_size > self.config.mmap_threshold_mb * 1024 * 1024
        
        cache_key = str(file_path)
        
        # Check cache first
        if cache_key in self.read_cache:
            return self.read_cache[cache_key]
        
        if use_mmap and file_size > 0:
            return self._read_with_mmap(file_path)
        else:
            return self._read_with_buffer(file_path)
    
    def _read_with_mmap(self, file_path: Path) -> bytes:
        """Read file using memory mapping"""
        
        try:
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    data = bytes(mm)
                    
            # Cache smaller files
            if len(data) < 10 * 1024 * 1024:  # 10MB
                self.read_cache[str(file_path)] = data
                
            return data
        except Exception as e:
            logger.debug(f"mmap failed for {file_path}, falling back to regular read: {e}")
            return self._read_with_buffer(file_path)
    
    def _read_with_buffer(self, file_path: Path) -> bytes:
        """Read file with optimized buffering"""
        
        buffer_size = self.config.read_buffer_size_kb * 1024
        
        with open(file_path, 'rb', buffering=buffer_size) as f:
            data = f.read()
        
        # Cache smaller files
        if len(data) < 5 * 1024 * 1024:  # 5MB
            self.read_cache[str(file_path)] = data
        
        return data
    
    def batch_read_files(self, file_paths: List[Path]) -> Dict[str, bytes]:
        """Read multiple files concurrently"""
        
        results = {}
        
        # Use thread pool for concurrent I/O
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_reads) as executor:
            future_to_path = {
                executor.submit(self.optimize_file_reading, path): path
                for path in file_paths
            }
            
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    results[str(path)] = future.result()
                except Exception as e:
                    logger.error(f"Failed to read {path}: {e}")
                    results[str(path)] = b""
        
        return results
    
    def clear_cache(self):
        """Clear file read cache"""
        self.read_cache.clear()
        self.prefetch_cache.clear()


class ResourceLimitOptimizer:
    """Optimize system resource limits"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        
    def optimize_limits(self):
        """Set optimal resource limits"""
        
        try:
            # Increase file descriptor limit
            resource.setrlimit(resource.RLIMIT_NOFILE, 
                             (self.config.max_open_files, self.config.max_open_files))
            
            # Set stack size
            stack_size_bytes = self.config.stack_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_STACK, 
                             (stack_size_bytes, stack_size_bytes))
            
            logger.info(f"Set resource limits: files={self.config.max_open_files}, "
                       f"stack={self.config.stack_size_mb}MB")
            
        except Exception as e:
            logger.warning(f"Could not set resource limits: {e}")
    
    def get_current_limits(self) -> Dict[str, Tuple[int, int]]:
        """Get current resource limits"""
        
        limits = {}
        
        try:
            limits['files'] = resource.getrlimit(resource.RLIMIT_NOFILE)
            limits['stack'] = resource.getrlimit(resource.RLIMIT_STACK)
            limits['memory'] = resource.getrlimit(resource.RLIMIT_AS)  # Address space
        except Exception as e:
            logger.debug(f"Could not get resource limits: {e}")
        
        return limits


class SystemPerformanceOptimizer:
    """Main system performance optimizer"""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Initialize optimizers
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.cpu_optimizer = CPUOptimizer(self.config)
        self.io_optimizer = IOOptimizer(self.config)
        self.resource_optimizer = ResourceLimitOptimizer(self.config)
        
        # Performance monitoring
        self.metrics_history = []
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Optimization state
        self.optimization_applied = False
        
    def apply_optimizations(self):
        """Apply all performance optimizations"""
        
        if self.optimization_applied:
            logger.info("Performance optimizations already applied")
            return
        
        logger.info("🚀 Applying M4 Pro performance optimizations...")
        
        try:
            # 1. Memory optimizations
            self.memory_optimizer.optimize_memory_allocation()
            
            # 2. CPU optimizations
            self.cpu_optimizer.initialize_pools()
            
            # 3. Resource limit optimizations
            self.resource_optimizer.optimize_limits()
            
            # 4. Set CPU affinity for main process
            if HAS_PSUTIL:
                # Use all cores for main process initially
                self.cpu_optimizer.set_cpu_affinity(list(range(self.config.total_cores)))
            
            # 5. Register memory pressure callbacks
            self._register_cleanup_callbacks()
            
            self.optimization_applied = True
            logger.info("✅ Performance optimizations applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply performance optimizations: {e}")
            raise
    
    def _register_cleanup_callbacks(self):
        """Register cleanup callbacks for memory pressure"""
        
        def clear_io_cache():
            self.io_optimizer.clear_cache()
            
        def force_gc():
            gc.collect()
            
        self.memory_optimizer.register_pressure_callback(clear_io_cache)
        self.memory_optimizer.register_pressure_callback(force_gc)
    
    def start_monitoring(self):
        """Start performance monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("📊 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self.metrics_history) > 300:  # 5 minutes at 1s intervals
                    self.metrics_history.pop(0)
                
                # Check for performance issues
                self._analyze_performance(metrics)
                
                time.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        
        timestamp = time.time()
        
        # Default values
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_used_gb = 0.0
        memory_available_gb = 0.0
        open_files = 0
        active_threads = 0
        
        if HAS_PSUTIL:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                memory_available_gb = memory.available / (1024**3)
                
                # Count open files and threads
                process = psutil.Process()
                open_files = len(process.open_files())
                active_threads = process.num_threads()
                
            except Exception as e:
                logger.debug(f"Metrics collection error: {e}")
        
        # GC statistics
        gc_collections = sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats())))
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            open_files=open_files,
            active_threads=active_threads,
            gc_collections=gc_collections,
        )
    
    def _analyze_performance(self, metrics: SystemMetrics):
        """Analyze performance and suggest optimizations"""
        
        suggestions = []
        bottleneck = None
        
        # CPU analysis
        if metrics.cpu_percent > 90:
            bottleneck = "CPU"
            suggestions.append("Consider reducing CPU-intensive operations")
        
        # Memory analysis
        if metrics.memory_percent > 85:
            bottleneck = "Memory"
            suggestions.append("Consider clearing caches or reducing memory usage")
            # Trigger cleanup
            self.memory_optimizer.check_memory_pressure()
        
        # File handle analysis
        if metrics.open_files > self.config.max_open_files * 0.8:
            suggestions.append("High file handle usage detected")
        
        # Thread analysis
        if metrics.active_threads > 50:
            suggestions.append("High thread count detected")
        
        if suggestions:
            metrics.bottleneck_type = bottleneck
            metrics.optimization_suggestions = suggestions
            
            logger.warning(f"Performance issue detected: {bottleneck}, "
                          f"suggestions: {suggestions}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not self.metrics_history:
            return {"error": "No performance data available"}
        
        recent_metrics = self.metrics_history[-60:]  # Last minute
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        report = {
            "timestamp": time.time(),
            "optimization_status": "applied" if self.optimization_applied else "pending",
            "monitoring_duration_minutes": len(self.metrics_history) / 60,
            
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "peak": max(cpu_values) if cpu_values else 0,
            },
            
            "memory": {
                "current_percent": memory_values[-1] if memory_values else 0,
                "current_used_gb": recent_metrics[-1].memory_used_gb if recent_metrics else 0,
                "current_available_gb": recent_metrics[-1].memory_available_gb if recent_metrics else 0,
                "average_percent": sum(memory_values) / len(memory_values) if memory_values else 0,
                "peak_percent": max(memory_values) if memory_values else 0,
            },
            
            "resources": {
                "open_files": recent_metrics[-1].open_files if recent_metrics else 0,
                "active_threads": recent_metrics[-1].active_threads if recent_metrics else 0,
                "gc_collections": recent_metrics[-1].gc_collections if recent_metrics else 0,
            },
            
            "configuration": asdict(self.config),
            
            "recent_issues": [
                {
                    "timestamp": m.timestamp,
                    "bottleneck": m.bottleneck_type,
                    "suggestions": m.optimization_suggestions,
                }
                for m in recent_metrics[-10:]
                if m.bottleneck_type is not None
            ],
        }
        
        return report
    
    def optimize_for_component(self, component: str):
        """Apply component-specific optimizations"""
        
        if component == "einstein":
            # Einstein search and indexing optimizations
            settings = self.cpu_optimizer.optimize_for_task_type("io_bound")
            logger.info(f"Applied Einstein optimizations: {settings}")
            
        elif component == "jarvis2":
            # Jarvis2 ML and computation optimizations
            settings = self.cpu_optimizer.optimize_for_task_type("ml_inference")
            logger.info(f"Applied Jarvis2 optimizations: {settings}")
            
        elif component == "unity_wheel":
            # Unity Wheel trading optimizations
            settings = self.cpu_optimizer.optimize_for_task_type("compute_intensive")
            logger.info(f"Applied Unity Wheel optimizations: {settings}")
    
    def shutdown(self):
        """Shutdown optimizer and clean up resources"""
        
        logger.info("🛑 Shutting down performance optimizer...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Shutdown CPU pools
        self.cpu_optimizer.shutdown_pools()
        
        # Clear caches
        self.io_optimizer.clear_cache()
        
        logger.info("✅ Performance optimizer shutdown complete")


# Global optimizer instance
_performance_optimizer: Optional[SystemPerformanceOptimizer] = None


def get_performance_optimizer() -> SystemPerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = SystemPerformanceOptimizer()
    return _performance_optimizer


def optimize_system_performance(auto_start_monitoring: bool = True) -> SystemPerformanceOptimizer:
    """Initialize and apply system performance optimizations"""
    
    optimizer = get_performance_optimizer()
    
    if not optimizer.optimization_applied:
        optimizer.apply_optimizations()
    
    if auto_start_monitoring:
        optimizer.start_monitoring()
    
    return optimizer


if __name__ == "__main__":
    
    def main():
        """Test system performance optimizer"""
        
        print("🚀 System Performance Optimizer Test")
        print("=" * 50)
        
        # Initialize optimizer
        optimizer = optimize_system_performance()
        
        # Test different optimization scenarios
        print("\n📊 Testing component optimizations...")
        optimizer.optimize_for_component("einstein")
        optimizer.optimize_for_component("jarvis2")
        optimizer.optimize_for_component("unity_wheel")
        
        # Let it run for a bit to collect metrics
        print("\n⏱️  Collecting performance metrics...")
        time.sleep(10)
        
        # Generate report
        print("\n📈 Performance Report:")
        report = optimizer.get_performance_report()
        
        print(f"  CPU Usage: {report['cpu']['current']:.1f}% (avg: {report['cpu']['average']:.1f}%)")
        print(f"  Memory Usage: {report['memory']['current_percent']:.1f}% "
              f"({report['memory']['current_used_gb']:.1f}GB used)")
        print(f"  Open Files: {report['resources']['open_files']}")
        print(f"  Active Threads: {report['resources']['active_threads']}")
        
        if report['recent_issues']:
            print(f"  Recent Issues: {len(report['recent_issues'])}")
        
        # Shutdown
        optimizer.shutdown()
        print("\n✅ Test completed successfully")
    
    main()