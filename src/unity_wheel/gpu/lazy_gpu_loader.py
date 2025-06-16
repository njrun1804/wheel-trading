"""
Lazy GPU Loading System

Implements lazy loading for GPU acceleration components to minimize startup time.
Components are only loaded when first accessed, with intelligent caching and
background warm-up capabilities.
"""

import asyncio
import logging
import threading
import time
import weakref
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LazyLoader(Generic[T]):
    """Generic lazy loader for expensive resources."""
    
    def __init__(self, loader_func: Callable[[], T], name: str = "unknown"):
        self.loader_func = loader_func
        self.name = name
        self._resource: Optional[T] = None
        self._loading = False
        self._load_error: Optional[Exception] = None
        self._load_time: Optional[float] = None
        self._lock = threading.Lock()
        self._access_count = 0
        self._first_access_time: Optional[float] = None
    
    def get(self) -> T:
        """Get the resource, loading it if necessary."""
        with self._lock:
            self._access_count += 1
            if self._first_access_time is None:
                self._first_access_time = time.perf_counter()
            
            if self._resource is not None:
                return self._resource
            
            if self._load_error is not None:
                raise self._load_error
            
            if self._loading:
                # Another thread is loading, wait for it
                pass
            else:
                self._loading = True
                
                try:
                    start_time = time.perf_counter()
                    logger.debug(f"Loading {self.name}...")
                    
                    self._resource = self.loader_func()
                    
                    self._load_time = time.perf_counter() - start_time
                    logger.info(f"Loaded {self.name} in {self._load_time*1000:.1f}ms")
                    
                    return self._resource
                    
                except Exception as e:
                    self._load_error = e
                    logger.error(f"Failed to load {self.name}: {e}")
                    raise
                finally:
                    self._loading = False
        
        # If we get here, another thread was loading
        while self._loading:
            time.sleep(0.001)  # Small delay to avoid busy waiting
        
        if self._load_error is not None:
            raise self._load_error
        
        return self._resource
    
    def is_loaded(self) -> bool:
        """Check if resource is loaded."""
        return self._resource is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return {
            'name': self.name,
            'loaded': self.is_loaded(),
            'access_count': self._access_count,
            'load_time_ms': self._load_time * 1000 if self._load_time else None,
            'first_access_time': self._first_access_time,
            'has_error': self._load_error is not None
        }


class LazyGPUComponentManager:
    """Manages lazy loading of GPU components."""
    
    def __init__(self):
        self._components: Dict[str, LazyLoader] = {}
        self._warmup_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="GPU-Warmup"
        )
        self._warmup_tasks = set()
        self._warmup_lock = threading.Lock()
    
    def register_component(self, name: str, loader_func: Callable[[], Any]) -> LazyLoader:
        """Register a component for lazy loading."""
        if name in self._components:
            logger.warning(f"Component {name} already registered, replacing")
        
        lazy_loader = LazyLoader(loader_func, name)
        self._components[name] = lazy_loader
        logger.debug(f"Registered lazy component: {name}")
        return lazy_loader
    
    def get_component(self, name: str) -> Any:
        """Get a component, loading it if necessary."""
        if name not in self._components:
            raise KeyError(f"Component {name} not registered")
        
        return self._components[name].get()
    
    def is_component_loaded(self, name: str) -> bool:
        """Check if a component is loaded."""
        if name not in self._components:
            return False
        return self._components[name].is_loaded()
    
    def warmup_component(self, name: str) -> None:
        """Start warming up a component in the background."""
        if name not in self._components:
            logger.warning(f"Cannot warmup unknown component: {name}")
            return
        
        if self._components[name].is_loaded():
            logger.debug(f"Component {name} already loaded, skipping warmup")
            return
        
        with self._warmup_lock:
            # Check if already warming up
            if name in self._warmup_tasks:
                logger.debug(f"Component {name} already warming up")
                return
            
            # Start warmup
            future = self._warmup_executor.submit(self._warmup_component_sync, name)
            self._warmup_tasks.add(name)
            
            # Clean up completed tasks
            def cleanup_task(task_name):
                with self._warmup_lock:
                    self._warmup_tasks.discard(task_name)
            
            future.add_done_callback(lambda f: cleanup_task(name))
            logger.debug(f"Started warmup for component: {name}")
    
    def _warmup_component_sync(self, name: str) -> None:
        """Synchronously warm up a component."""
        try:
            component = self._components[name].get()
            logger.debug(f"Warmed up component: {name}")
        except Exception as e:
            logger.warning(f"Failed to warm up component {name}: {e}")
    
    def warmup_all(self, priority_components: list[str] = None) -> None:
        """Start warming up all components."""
        # Warmup priority components first
        if priority_components:
            for name in priority_components:
                if name in self._components:
                    self.warmup_component(name)
        
        # Then warmup remaining components
        for name in self._components:
            if not priority_components or name not in priority_components:
                self.warmup_component(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all components."""
        return {name: loader.get_stats() for name, loader in self._components.items()}
    
    def shutdown(self) -> None:
        """Shutdown the component manager."""
        self._warmup_executor.shutdown(wait=True)


# Global component manager
_component_manager = LazyGPUComponentManager()


def lazy_gpu_component(name: str):
    """Decorator to register a function as a lazy GPU component."""
    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        _component_manager.register_component(name, func)
        
        @wraps(func)
        def wrapper() -> T:
            return _component_manager.get_component(name)
        
        return wrapper
    return decorator


# Lazy component definitions
@lazy_gpu_component("mlx_core")
def _load_mlx_core():
    """Load MLX core module."""
    import mlx.core as mx
    # Verify it works
    test_array = mx.array([1.0, 2.0, 3.0])
    mx.eval(test_array)
    return mx


@lazy_gpu_component("mlx_nn")
def _load_mlx_nn():
    """Load MLX neural network module."""
    import mlx.nn as nn
    return nn


@lazy_gpu_component("gpu_accelerator")
def _load_gpu_accelerator():
    """Load GPU accelerator from existing module."""
    from ..gpu_acceleration import GPUAccelerator
    return GPUAccelerator()


@lazy_gpu_component("mlx_memory_manager")
def _load_mlx_memory_manager():
    """Load MLX memory manager."""
    from .mlx_memory_manager import get_mlx_memory_manager
    return get_mlx_memory_manager()


@lazy_gpu_component("optimized_initializer")
def _load_optimized_initializer():
    """Load optimized GPU initializer."""
    from .optimized_gpu_init import get_optimized_gpu_initializer
    return get_optimized_gpu_initializer()


# Fast access functions
def get_mlx_core():
    """Get MLX core module (lazy loaded)."""
    return _component_manager.get_component("mlx_core")


def get_mlx_nn():
    """Get MLX neural network module (lazy loaded)."""
    return _component_manager.get_component("mlx_nn")


def get_gpu_accelerator():
    """Get GPU accelerator (lazy loaded)."""
    return _component_manager.get_component("gpu_accelerator")


def get_mlx_memory_manager():
    """Get MLX memory manager (lazy loaded)."""
    return _component_manager.get_component("mlx_memory_manager")


def get_optimized_initializer():
    """Get optimized GPU initializer (lazy loaded)."""
    return _component_manager.get_component("optimized_initializer")


# Convenience functions
def is_gpu_ready() -> bool:
    """Check if GPU components are ready."""
    try:
        return _component_manager.is_component_loaded("mlx_core")
    except:
        return False


def warmup_gpu_components(priority: list[str] = None) -> None:
    """Start warming up GPU components in background."""
    if priority is None:
        priority = ["mlx_core", "gpu_accelerator", "optimized_initializer"]
    
    _component_manager.warmup_all(priority_components=priority)


def get_gpu_component_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all GPU components."""
    return _component_manager.get_all_stats()


# Lazy initialization decorator for GPU functions
def requires_gpu(func):
    """Decorator that ensures GPU is initialized before function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Ensure MLX core is loaded
        if not _component_manager.is_component_loaded("mlx_core"):
            logger.debug(f"Loading GPU for {func.__name__}")
            get_mlx_core()
        
        return func(*args, **kwargs)
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Ensure MLX core is loaded
        if not _component_manager.is_component_loaded("mlx_core"):
            logger.debug(f"Loading GPU for {func.__name__}")
            # Load in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, get_mlx_core)
        
        return await func(*args, **kwargs)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


# Context manager for optimized GPU operations
class LazyGPUContext:
    """Context manager that ensures GPU is ready with minimal overhead."""
    
    def __init__(self, components: list[str] = None):
        self.components = components or ["mlx_core"]
        self._loaded_components = {}
    
    def __enter__(self):
        """Enter context, loading required components."""
        start_time = time.perf_counter()
        
        for component in self.components:
            if component in _component_manager._components:
                self._loaded_components[component] = _component_manager.get_component(component)
        
        load_time = (time.perf_counter() - start_time) * 1000
        if load_time > 10:  # Log if loading takes more than 10ms
            logger.debug(f"GPU context loaded {len(self._loaded_components)} components in {load_time:.1f}ms")
        
        return self._loaded_components
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Components remain loaded for future use
        pass


# Integration with existing GPU acceleration
def integrate_with_existing_gpu_acceleration():
    """Integrate lazy loading with existing GPU acceleration code."""
    try:
        # Patch the existing GPU accelerator to use lazy loading
        from .. import gpu_acceleration
        
        # Store original global accelerator creation
        original_accelerator = getattr(gpu_acceleration, '_accelerator', None)
        
        # Replace with lazy loader
        def get_lazy_accelerator():
            return get_gpu_accelerator()
        
        # Patch the module
        gpu_acceleration._accelerator = LazyLoader(
            lambda: gpu_acceleration.GPUAccelerator(), 
            "gpu_acceleration"
        )
        
        logger.info("Integrated lazy loading with existing GPU acceleration")
        
    except Exception as e:
        logger.warning(f"Failed to integrate with existing GPU acceleration: {e}")


# Auto-warmup on import
def auto_warmup_on_import():
    """Automatically start warming up critical components on import."""
    # Only warmup if this is likely a production environment
    import os
    if os.getenv('WHEEL_TRADING_ENV') == 'production' or os.getenv('WARMUP_GPU', '').lower() == 'true':
        logger.info("Starting automatic GPU component warmup")
        warmup_gpu_components(['mlx_core', 'optimized_initializer'])


# Performance monitoring
class GPULoadingMonitor:
    """Monitor GPU loading performance."""
    
    def __init__(self):
        self.load_events = []
        self._lock = threading.Lock()
    
    def record_load_event(self, component: str, load_time_ms: float, success: bool):
        """Record a component loading event."""
        with self._lock:
            self.load_events.append({
                'component': component,
                'load_time_ms': load_time_ms,
                'success': success,
                'timestamp': time.time()
            })
            
            # Keep only recent events
            if len(self.load_events) > 100:
                self.load_events = self.load_events[-50:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.load_events:
                return {'total_events': 0}
            
            successful_events = [e for e in self.load_events if e['success']]
            failed_events = [e for e in self.load_events if not e['success']]
            
            if successful_events:
                avg_load_time = sum(e['load_time_ms'] for e in successful_events) / len(successful_events)
                max_load_time = max(e['load_time_ms'] for e in successful_events)
                min_load_time = min(e['load_time_ms'] for e in successful_events)
            else:
                avg_load_time = max_load_time = min_load_time = 0
            
            return {
                'total_events': len(self.load_events),
                'successful_loads': len(successful_events),
                'failed_loads': len(failed_events),
                'avg_load_time_ms': avg_load_time,
                'max_load_time_ms': max_load_time,
                'min_load_time_ms': min_load_time,
                'recent_events': self.load_events[-10:]  # Last 10 events
            }


# Global monitor
_loading_monitor = GPULoadingMonitor()


def get_loading_performance_summary() -> Dict[str, Any]:
    """Get GPU loading performance summary."""
    return _loading_monitor.get_performance_summary()


# Initialize on import
if __name__ != "__main__":
    integrate_with_existing_gpu_acceleration()
    auto_warmup_on_import()


if __name__ == "__main__":
    # Test lazy loading system
    async def test_lazy_loading():
        print("Testing Lazy GPU Loading System...")
        
        # Test component registration and loading
        start_time = time.perf_counter()
        
        # Check initial state
        print(f"Initial state:")
        for name, stats in get_gpu_component_stats().items():
            print(f"  {name}: loaded={stats['loaded']}, access_count={stats['access_count']}")
        
        # Test lazy loading
        print(f"\nTesting lazy loading...")
        
        # This should trigger loading
        try:
            mx = get_mlx_core()
            print(f"✅ MLX core loaded successfully")
        except Exception as e:
            print(f"❌ MLX core failed to load: {e}")
        
        # Test GPU accelerator
        try:
            accelerator = get_gpu_accelerator()
            print(f"✅ GPU accelerator loaded successfully")
        except Exception as e:
            print(f"❌ GPU accelerator failed to load: {e}")
        
        # Test context manager
        print(f"\nTesting context manager...")
        with LazyGPUContext(['mlx_core', 'gpu_accelerator']) as ctx:
            print(f"Context loaded {len(ctx)} components")
        
        # Check final state
        print(f"\nFinal state:")
        for name, stats in get_gpu_component_stats().items():
            print(f"  {name}: loaded={stats['loaded']}, access_count={stats['access_count']}, load_time={stats['load_time_ms']}")
        
        # Performance summary
        total_time = (time.perf_counter() - start_time) * 1000
        print(f"\nTotal test time: {total_time:.1f}ms")
        
        # Test warmup
        print(f"\nTesting warmup...")
        warmup_gpu_components()
        
        # Wait a bit for warmup to complete
        await asyncio.sleep(1)
        
        print(f"Warmup completed")
    
    asyncio.run(test_lazy_loading())