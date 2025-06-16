"""
Async Hardware Operation Patterns - Optimized for Sub-5ms Latency

Advanced async patterns for hardware operations including:
1. Async context managers with resource pooling
2. Concurrent hardware detection with intelligent batching
3. Async generators for streaming hardware data
4. Circuit breaker pattern for fault tolerance
5. Async caching with background refresh
6. Cooperative multitasking optimization
"""

import asyncio
import functools
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple, Union

import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class AsyncHardwareConfig:
    """Configuration for async hardware operations."""
    
    # Connection pooling
    max_connections: int = 8
    connection_timeout_ms: float = 100.0
    pool_timeout_ms: float = 50.0
    
    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout_s: float = 10.0
    success_threshold: int = 3
    
    # Caching
    cache_size: int = 1000
    default_ttl_s: float = 300.0
    background_refresh_ratio: float = 0.8
    
    # Batching
    batch_size: int = 10
    batch_timeout_ms: float = 10.0
    max_batch_wait_ms: float = 5.0
    
    # Performance
    max_concurrent_operations: int = 20
    operation_timeout_ms: float = 50.0
    slow_operation_threshold_ms: float = 10.0


class AsyncHardwareError(Exception):
    """Base exception for async hardware operations."""
    pass


class CircuitBreakerOpenError(AsyncHardwareError):
    """Circuit breaker is open, operation rejected."""
    pass


class OperationTimeoutError(AsyncHardwareError):
    """Hardware operation timed out."""
    pass


class AsyncCircuitBreaker:
    """Circuit breaker for hardware operations with async support."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 10.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        async with self._lock:
            # Check if circuit should be closed
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")
        
        # Execute operation
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self) -> None:
        """Record successful operation."""
        async with self._lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
    
    async def _record_failure(self) -> None:
        """Record failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
    
    @property
    def is_available(self) -> bool:
        """Check if circuit breaker allows operations."""
        return self.state != CircuitState.OPEN


class AsyncHardwareConnection:
    """Async hardware connection with resource management."""
    
    def __init__(self, connection_id: str, timeout_ms: float = 100.0):
        self.connection_id = connection_id
        self.timeout_ms = timeout_ms
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.is_healthy = True
        
        # Connection state
        self._semaphore = asyncio.Semaphore(1)  # Single use at a time
        self._closed = False
    
    async def execute(self, operation: str, *args, **kwargs) -> Any:
        """Execute hardware operation through this connection."""
        if self._closed:
            raise AsyncHardwareError("Connection is closed")
        
        async with self._semaphore:
            try:
                # Update usage stats
                self.last_used = time.time()
                self.use_count += 1
                
                # Simulate hardware operation
                await asyncio.sleep(0.001)  # 1ms base operation time
                
                # Operation-specific logic
                if operation == "detect_cpu":
                    return {"cores": 12, "type": "M4 Pro", "detected_at": time.time()}
                elif operation == "detect_memory":
                    return {"total_gb": 36, "available_gb": 24, "detected_at": time.time()}
                elif operation == "detect_gpu":
                    return {"name": "M4 Pro GPU", "cores": 16, "detected_at": time.time()}
                else:
                    return {"operation": operation, "args": args, "kwargs": kwargs}
                
            except Exception as e:
                self.is_healthy = False
                raise AsyncHardwareError(f"Hardware operation failed: {e}")
    
    async def ping(self) -> bool:
        """Check connection health."""
        try:
            await asyncio.wait_for(
                self.execute("ping"), 
                timeout=self.timeout_ms / 1000.0
            )
            self.is_healthy = True
            return True
        except:
            self.is_healthy = False
            return False
    
    async def close(self) -> None:
        """Close connection."""
        self._closed = True
    
    def __str__(self) -> str:
        return f"HardwareConnection({self.connection_id}, uses={self.use_count})"


class AsyncHardwareConnectionPool:
    """High-performance async connection pool for hardware operations."""
    
    def __init__(self, config: AsyncHardwareConfig):
        self.config = config
        self.connections: Dict[str, AsyncHardwareConnection] = {}
        self.available_connections: asyncio.Queue = asyncio.Queue(maxsize=config.max_connections)
        self.active_connections: Set[str] = set()
        
        self._lock = asyncio.Lock()
        self._connection_counter = 0
        self._health_check_task: Optional[asyncio.Task] = None
        self._start_health_checks()
    
    def _start_health_checks(self) -> None:
        """Start background health check task."""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self) -> None:
        """Background health check for connections."""
        while True:
            try:
                await asyncio.sleep(30)  # Health check every 30 seconds
                
                unhealthy_connections = []
                for conn_id, conn in self.connections.items():
                    if not await conn.ping():
                        unhealthy_connections.append(conn_id)
                
                # Remove unhealthy connections
                for conn_id in unhealthy_connections:
                    await self._remove_connection(conn_id)
                    logger.warning(f"Removed unhealthy connection: {conn_id}")
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _create_connection(self) -> AsyncHardwareConnection:
        """Create new hardware connection."""
        self._connection_counter += 1
        conn_id = f"hw_conn_{self._connection_counter}"
        
        connection = AsyncHardwareConnection(
            conn_id, 
            timeout_ms=self.config.connection_timeout_ms
        )
        
        return connection
    
    async def _remove_connection(self, conn_id: str) -> None:
        """Remove connection from pool."""
        async with self._lock:
            if conn_id in self.connections:
                conn = self.connections.pop(conn_id)
                await conn.close()
                self.active_connections.discard(conn_id)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool with automatic cleanup."""
        connection = None
        try:
            # Try to get available connection
            try:
                connection = await asyncio.wait_for(
                    self.available_connections.get(),
                    timeout=self.config.pool_timeout_ms / 1000.0
                )
            except asyncio.TimeoutError:
                # Create new connection if pool is empty and under limit
                async with self._lock:
                    if len(self.connections) < self.config.max_connections:
                        connection = await self._create_connection()
                        self.connections[connection.connection_id] = connection
                    else:
                        raise AsyncHardwareError("Connection pool exhausted")
            
            if connection:
                self.active_connections.add(connection.connection_id)
                yield connection
            else:
                raise AsyncHardwareError("No connection available")
                
        finally:
            # Return connection to pool
            if connection:
                self.active_connections.discard(connection.connection_id)
                try:
                    self.available_connections.put_nowait(connection)
                except asyncio.QueueFull:
                    # Pool full, close connection
                    await connection.close()
    
    async def close_all(self) -> None:
        """Close all connections and stop health checks."""
        if self._health_check_task:
            self._health_check_task.cancel()
        
        for connection in self.connections.values():
            await connection.close()
        
        self.connections.clear()
        self.active_connections.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "total_connections": len(self.connections),
            "active_connections": len(self.active_connections),
            "available_connections": self.available_connections.qsize(),
            "healthy_connections": sum(1 for conn in self.connections.values() if conn.is_healthy),
            "max_connections": self.config.max_connections
        }


class AsyncBatchProcessor:
    """Batch processor for hardware operations to reduce overhead."""
    
    def __init__(self, config: AsyncHardwareConfig, connection_pool: AsyncHardwareConnectionPool):
        self.config = config
        self.connection_pool = connection_pool
        
        # Batching state
        self.pending_requests: Dict[str, List[Tuple[str, asyncio.Future]]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def execute_batch(self, operation_type: str, operation_id: str) -> Any:
        """Execute operation as part of a batch."""
        future = asyncio.Future()
        
        async with self._lock:
            # Add to pending requests
            self.pending_requests[operation_type].append((operation_id, future))
            
            # Start batch timer if not already running
            if operation_type not in self.batch_timers:
                self.batch_timers[operation_type] = asyncio.create_task(
                    self._batch_timer(operation_type)
                )
            
            # Process batch if it's full
            if len(self.pending_requests[operation_type]) >= self.config.batch_size:
                await self._process_batch(operation_type)
        
        return await future
    
    async def _batch_timer(self, operation_type: str) -> None:
        """Timer to process batch after timeout."""
        try:
            await asyncio.sleep(self.config.batch_timeout_ms / 1000.0)
            async with self._lock:
                await self._process_batch(operation_type)
        except asyncio.CancelledError:
            pass
    
    async def _process_batch(self, operation_type: str) -> None:
        """Process a batch of operations."""
        if operation_type not in self.pending_requests:
            return
        
        requests = self.pending_requests.pop(operation_type, [])
        if not requests:
            return
        
        # Cancel timer
        if operation_type in self.batch_timers:
            self.batch_timers[operation_type].cancel()
            del self.batch_timers[operation_type]
        
        # Process batch
        try:
            async with self.connection_pool.get_connection() as conn:
                # Execute all operations in batch
                results = {}
                for operation_id, future in requests:
                    try:
                        result = await conn.execute(operation_type, operation_id)
                        results[operation_id] = result
                    except Exception as e:
                        results[operation_id] = e
                
                # Set results on futures
                for operation_id, future in requests:
                    if not future.cancelled():
                        result = results.get(operation_id)
                        if isinstance(result, Exception):
                            future.set_exception(result)
                        else:
                            future.set_result(result)
        
        except Exception as e:
            # Set exception on all futures
            for _, future in requests:
                if not future.cancelled():
                    future.set_exception(e)


class AsyncHardwareStreamer:
    """Async generator for streaming hardware data with backpressure control."""
    
    def __init__(self, config: AsyncHardwareConfig, connection_pool: AsyncHardwareConnectionPool):
        self.config = config
        self.connection_pool = connection_pool
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout_s
        )
    
    async def stream_hardware_data(self, 
                                 data_types: List[str],
                                 interval_ms: float = 1000.0,
                                 buffer_size: int = 100) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream hardware data with specified interval."""
        buffer = asyncio.Queue(maxsize=buffer_size)
        producer_task = asyncio.create_task(
            self._producer(buffer, data_types, interval_ms)
        )
        
        try:
            while True:
                # Get data from buffer with timeout
                try:
                    data = await asyncio.wait_for(buffer.get(), timeout=5.0)
                    if data is None:  # Sentinel value for end
                        break
                    yield data
                except asyncio.TimeoutError:
                    logger.warning("Hardware data stream timeout")
                    break
        finally:
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass
    
    async def _producer(self, 
                       buffer: asyncio.Queue,
                       data_types: List[str],
                       interval_ms: float) -> None:
        """Producer coroutine for streaming data."""
        try:
            while True:
                # Collect hardware data
                data = {}
                for data_type in data_types:
                    try:
                        if self.circuit_breaker.is_available:
                            result = await self.circuit_breaker.call(
                                self._get_hardware_data, data_type
                            )
                            data[data_type] = result
                        else:
                            data[data_type] = {"error": "circuit_breaker_open"}
                    except Exception as e:
                        data[data_type] = {"error": str(e)}
                
                # Add timestamp
                data["timestamp"] = time.time()
                
                # Put data in buffer (non-blocking)
                try:
                    buffer.put_nowait(data)
                except asyncio.QueueFull:
                    # Buffer full, drop oldest data
                    try:
                        buffer.get_nowait()
                        buffer.put_nowait(data)
                    except asyncio.QueueEmpty:
                        pass
                
                # Wait for next interval
                await asyncio.sleep(interval_ms / 1000.0)
                
        except asyncio.CancelledError:
            # Send sentinel value to signal end
            try:
                buffer.put_nowait(None)
            except asyncio.QueueFull:
                pass
            raise
    
    async def _get_hardware_data(self, data_type: str) -> Dict[str, Any]:
        """Get specific hardware data."""
        async with self.connection_pool.get_connection() as conn:
            return await conn.execute(f"detect_{data_type}")


class AsyncHardwareCache:
    """Async cache with background refresh and intelligent prefetching."""
    
    def __init__(self, config: AsyncHardwareConfig):
        self.config = config
        self.cache: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, timestamp, ttl)
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        self._lock = asyncio.Lock()
        self._refresh_tasks: Set[asyncio.Task] = set()
    
    async def get(self, key: str, fetcher: Callable, ttl: Optional[float] = None) -> Any:
        """Get value from cache with async fetcher."""
        ttl = ttl or self.config.default_ttl_s
        current_time = time.time()
        
        async with self._lock:
            # Record access pattern
            self.access_patterns[key].append(current_time)
            
            # Check cache
            if key in self.cache:
                value, timestamp, cached_ttl = self.cache[key]
                age = current_time - timestamp
                
                if age < cached_ttl:
                    # Cache hit - check if background refresh needed
                    if age > cached_ttl * self.config.background_refresh_ratio:
                        self._schedule_background_refresh(key, fetcher, ttl)
                    return value
            
            # Cache miss - fetch value
            value = await fetcher()
            self.cache[key] = (value, current_time, ttl)
            
            # Evict old entries if cache is full
            if len(self.cache) > self.config.cache_size:
                await self._evict_lru()
            
            return value
    
    def _schedule_background_refresh(self, key: str, fetcher: Callable, ttl: float) -> None:
        """Schedule background refresh for key."""
        task = asyncio.create_task(self._background_refresh(key, fetcher, ttl))
        self._refresh_tasks.add(task)
        task.add_done_callback(self._refresh_tasks.discard)
    
    async def _background_refresh(self, key: str, fetcher: Callable, ttl: float) -> None:
        """Background refresh of cache entry."""
        try:
            new_value = await fetcher()
            async with self._lock:
                self.cache[key] = (new_value, time.time(), ttl)
        except Exception as e:
            logger.debug(f"Background refresh failed for {key}: {e}")
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        # Find entries to evict based on access patterns
        current_time = time.time()
        candidates = []
        
        for key, (value, timestamp, ttl) in self.cache.items():
            last_access = self.access_patterns[key][-1] if self.access_patterns[key] else 0
            score = current_time - last_access  # Higher score = less recently used
            candidates.append((key, score))
        
        # Sort by score and evict bottom 10%
        candidates.sort(key=lambda x: x[1], reverse=True)
        evict_count = max(1, len(candidates) // 10)
        
        for key, _ in candidates[:evict_count]:
            self.cache.pop(key, None)
            self.access_patterns.pop(key, None)
    
    async def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        async with self._lock:
            self.cache.pop(key, None)
            self.access_patterns.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.access_patterns.clear()
        
        # Cancel all refresh tasks
        for task in self._refresh_tasks:
            task.cancel()
        self._refresh_tasks.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.config.cache_size,
            "active_refreshes": len(self._refresh_tasks),
            "tracked_patterns": len(self.access_patterns)
        }


class AsyncHardwareManager:
    """Unified async hardware manager with all optimization patterns."""
    
    def __init__(self, config: Optional[AsyncHardwareConfig] = None):
        self.config = config or AsyncHardwareConfig()
        
        # Core components
        self.connection_pool = AsyncHardwareConnectionPool(self.config)
        self.batch_processor = AsyncBatchProcessor(self.config, self.connection_pool)
        self.cache = AsyncHardwareCache(self.config)
        self.streamer = AsyncHardwareStreamer(self.config, self.connection_pool)
        
        # Performance tracking
        self.operation_times: deque = deque(maxlen=1000)
        self.error_count = 0
        self.total_operations = 0
        
        # Semaphore for concurrent operations
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)
    
    @asynccontextmanager
    async def operation_context(self, operation_name: str):
        """Context manager for hardware operations with performance tracking."""
        start_time = time.perf_counter()
        
        async with self._semaphore:
            try:
                yield
                
                # Record success
                operation_time = (time.perf_counter() - start_time) * 1000
                self.operation_times.append(operation_time)
                self.total_operations += 1
                
                # Log slow operations
                if operation_time > self.config.slow_operation_threshold_ms:
                    logger.warning(f"Slow operation {operation_name}: {operation_time:.2f}ms")
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Operation {operation_name} failed: {e}")
                raise
    
    async def get_hardware_info(self, info_type: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get hardware information with all optimizations."""
        async with self.operation_context(f"get_{info_type}"):
            if use_cache:
                return await self.cache.get(
                    f"hw_{info_type}",
                    lambda: self._fetch_hardware_info(info_type)
                )
            else:
                return await self._fetch_hardware_info(info_type)
    
    async def _fetch_hardware_info(self, info_type: str) -> Dict[str, Any]:
        """Fetch hardware info from connection pool."""
        async with self.connection_pool.get_connection() as conn:
            return await conn.execute(f"detect_{info_type}")
    
    async def get_hardware_info_batch(self, info_types: List[str]) -> Dict[str, Any]:
        """Get multiple hardware info types using batch processing."""
        tasks = []
        for info_type in info_types:
            task = self.batch_processor.execute_batch("detect_" + info_type, info_type)
            tasks.append((info_type, task))
        
        results = {}
        for info_type, task in tasks:
            try:
                result = await task
                results[info_type] = result
            except Exception as e:
                results[info_type] = {"error": str(e)}
        
        return results
    
    async def stream_hardware_monitoring(self, 
                                       data_types: List[str],
                                       interval_ms: float = 1000.0) -> AsyncIterator[Dict[str, Any]]:
        """Stream hardware monitoring data."""
        async for data in self.streamer.stream_hardware_data(data_types, interval_ms):
            yield data
    
    async def close(self) -> None:
        """Close all resources."""
        await self.connection_pool.close_all()
        await self.cache.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        operation_times = list(self.operation_times)
        
        return {
            "total_operations": self.total_operations,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.total_operations),
            "operation_times": {
                "count": len(operation_times),
                "avg_ms": sum(operation_times) / len(operation_times) if operation_times else 0,
                "min_ms": min(operation_times) if operation_times else 0,
                "max_ms": max(operation_times) if operation_times else 0,
                "p95_ms": sorted(operation_times)[int(len(operation_times) * 0.95)] if operation_times else 0,
                "under_5ms": sum(1 for t in operation_times if t < 5.0) / len(operation_times) if operation_times else 0
            },
            "connection_pool": self.connection_pool.get_stats(),
            "cache": self.cache.get_stats()
        }


# Global manager instance
_async_hardware_manager: Optional[AsyncHardwareManager] = None


async def get_async_hardware_manager(config: Optional[AsyncHardwareConfig] = None) -> AsyncHardwareManager:
    """Get or create global async hardware manager."""
    global _async_hardware_manager
    
    if _async_hardware_manager is None:
        _async_hardware_manager = AsyncHardwareManager(config)
    
    return _async_hardware_manager


# Convenience functions
async def get_cpu_info_async() -> Dict[str, Any]:
    """Get CPU info asynchronously."""
    manager = await get_async_hardware_manager()
    return await manager.get_hardware_info("cpu")


async def get_memory_info_async() -> Dict[str, Any]:
    """Get memory info asynchronously."""
    manager = await get_async_hardware_manager()
    return await manager.get_hardware_info("memory")


async def get_all_hardware_info_async() -> Dict[str, Any]:
    """Get all hardware info using batch processing."""
    manager = await get_async_hardware_manager()
    return await manager.get_hardware_info_batch(["cpu", "memory", "gpu"])


if __name__ == "__main__":
    async def test_async_patterns():
        print("🚀 Testing Async Hardware Patterns")
        print("=" * 50)
        
        # Test basic async operations
        print("\n1. Testing basic async operations...")
        cpu_info = await get_cpu_info_async()
        memory_info = await get_memory_info_async()
        print(f"CPU: {cpu_info.get('cores', 'N/A')} cores")
        print(f"Memory: {memory_info.get('total_gb', 'N/A')}GB")
        
        # Test batch processing  
        print("\n2. Testing batch processing...")
        start = time.perf_counter()
        batch_info = await get_all_hardware_info_async()
        batch_time = (time.perf_counter() - start) * 1000
        print(f"Batch operation completed in {batch_time:.2f}ms")
        print(f"Retrieved {len(batch_info)} hardware components")
        
        # Test streaming
        print("\n3. Testing hardware streaming...")
        manager = await get_async_hardware_manager()
        
        stream_count = 0
        async for data in manager.stream_hardware_monitoring(["cpu", "memory"], 100):
            stream_count += 1
            print(f"Stream #{stream_count}: {len(data)} data points")
            if stream_count >= 3:  # Test 3 data points
                break
        
        # Test concurrent access
        print("\n4. Testing concurrent access...")
        tasks = []
        for i in range(20):
            task = manager.get_hardware_info(["cpu", "memory", "gpu"][i % 3])
            tasks.append(task)
        
        concurrent_start = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = (time.perf_counter() - concurrent_start) * 1000
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        print(f"Concurrent operations: {len(successful_results)}/{len(tasks)} succeeded")
        print(f"Total time: {concurrent_time:.2f}ms")
        print(f"Average per operation: {concurrent_time / len(tasks):.2f}ms")
        
        # Get performance stats
        print("\n5. Performance statistics...")
        stats = manager.get_performance_stats()
        print(f"Total operations: {stats['total_operations']}")
        print(f"Average time: {stats['operation_times']['avg_ms']:.2f}ms")
        print(f"P95 time: {stats['operation_times']['p95_ms']:.2f}ms")
        print(f"Under 5ms rate: {stats['operation_times']['under_5ms']:.1%}")
        print(f"Error rate: {stats['error_rate']:.1%}")
        
        # Cleanup
        await manager.close()
        print("\n✅ Async hardware patterns test completed!")
    
    asyncio.run(test_async_patterns())