#!/usr/bin/env python3
"""
GPU Error Handling and Fallback System for Bolt
Comprehensive error recovery and graceful degradation for GPU operations
"""

import asyncio
import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUErrorType(Enum):
    """Types of GPU errors."""
    MEMORY_ERROR = "memory_error"
    COMPUTE_ERROR = "compute_error"
    DEVICE_ERROR = "device_error"
    TIMEOUT_ERROR = "timeout_error"
    COMPILATION_ERROR = "compilation_error"
    DRIVER_ERROR = "driver_error"
    UNKNOWN_ERROR = "unknown_error"


class FallbackStrategy(Enum):
    """Fallback strategies for GPU errors."""
    CPU_FALLBACK = "cpu_fallback"
    REDUCED_PRECISION = "reduced_precision"
    SMALLER_BATCH = "smaller_batch"
    SIMPLIFIED_ALGORITHM = "simplified_algorithm"
    NO_FALLBACK = "no_fallback"


@dataclass
class GPUErrorInfo:
    """Information about a GPU error."""
    error_type: GPUErrorType
    error_message: str
    timestamp: float
    operation_name: str
    fallback_strategy: Optional[FallbackStrategy] = None
    recovery_successful: bool = False
    recovery_time_ms: float = 0.0


@dataclass
class GPUOperationConfig:
    """Configuration for GPU operations with error handling."""
    operation_name: str
    timeout_seconds: float = 30.0
    max_retries: int = 3
    fallback_strategies: List[FallbackStrategy] = None
    memory_limit_mb: float = 500.0
    enable_monitoring: bool = True


class GPUErrorTracker:
    """Track and analyze GPU errors for intelligent fallback decisions."""
    
    def __init__(self, max_error_history: int = 1000):
        self.max_error_history = max_error_history
        self.error_history: List[GPUErrorInfo] = []
        self.error_counts: Dict[GPUErrorType, int] = {}
        self.fallback_success_rates: Dict[FallbackStrategy, float] = {}
        self._lock = asyncio.Lock()
    
    async def record_error(self, error_info: GPUErrorInfo):
        """Record a GPU error."""
        async with self._lock:
            self.error_history.append(error_info)
            
            # Limit history size
            if len(self.error_history) > self.max_error_history:
                self.error_history = self.error_history[-self.max_error_history:]
            
            # Update error counts
            self.error_counts[error_info.error_type] = self.error_counts.get(error_info.error_type, 0) + 1
            
            # Update fallback success rates
            if error_info.fallback_strategy and error_info.recovery_successful:
                current_rate = self.fallback_success_rates.get(error_info.fallback_strategy, 0.0)
                self.fallback_success_rates[error_info.fallback_strategy] = (current_rate + 1.0) / 2.0
    
    async def get_recommended_fallback(self, error_type: GPUErrorType) -> Optional[FallbackStrategy]:
        """Get recommended fallback strategy based on error history."""
        async with self._lock:
            # Analyze recent errors of this type
            recent_errors = [e for e in self.error_history[-100:] if e.error_type == error_type]
            
            if not recent_errors:
                return FallbackStrategy.CPU_FALLBACK  # Default fallback
            
            # Find most successful fallback strategy
            strategy_success = {}
            for error in recent_errors:
                if error.fallback_strategy and error.recovery_successful:
                    strategy = error.fallback_strategy
                    strategy_success[strategy] = strategy_success.get(strategy, 0) + 1
            
            if strategy_success:
                best_strategy = max(strategy_success, key=strategy_success.get)
                return best_strategy
            
            return FallbackStrategy.CPU_FALLBACK
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = len(self.error_history)
        
        return {
            "total_errors": total_errors,
            "error_counts": dict(self.error_counts),
            "fallback_success_rates": dict(self.fallback_success_rates),
            "recent_error_rate": len([e for e in self.error_history if time.time() - e.timestamp < 300]) / 5.0,  # errors per minute
            "most_common_error": max(self.error_counts, key=self.error_counts.get) if self.error_counts else None
        }


class GPUErrorHandler:
    """Comprehensive GPU error handling and recovery system."""
    
    def __init__(self):
        self.error_tracker = GPUErrorTracker()
        self.gpu_available = MLX_AVAILABLE
        self.device_healthy = True
        self.last_health_check = 0.0
        self.health_check_interval = 60.0  # Check every minute
        
        # Error classification patterns
        self.error_patterns = {
            GPUErrorType.MEMORY_ERROR: [
                "out of memory", "memory allocation", "insufficient memory",
                "memory limit", "allocation failed"
            ],
            GPUErrorType.COMPUTE_ERROR: [
                "computation failed", "kernel error", "invalid operation",
                "numerical error", "compute error"
            ],
            GPUErrorType.DEVICE_ERROR: [
                "device error", "gpu error", "metal error", "device unavailable"
            ],
            GPUErrorType.TIMEOUT_ERROR: [
                "timeout", "operation timed out", "execution timeout"
            ],
            GPUErrorType.COMPILATION_ERROR: [
                "compilation failed", "kernel compilation", "shader error"
            ],
            GPUErrorType.DRIVER_ERROR: [
                "driver error", "mlx error", "metal driver"
            ]
        }
    
    def classify_error(self, error: Exception) -> GPUErrorType:
        """Classify GPU error based on error message."""
        error_message = str(error).lower()
        
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type
        
        return GPUErrorType.UNKNOWN_ERROR
    
    async def check_gpu_health(self) -> bool:
        """Check if GPU is healthy and available."""
        current_time = time.time()
        
        if current_time - self.last_health_check < self.health_check_interval:
            return self.device_healthy
        
        try:
            if MLX_AVAILABLE:
                # Simple GPU health test
                test_array = mx.array([1.0, 2.0, 3.0])
                result = mx.sum(test_array)
                mx.eval(result)
                self.device_healthy = True
            else:
                self.device_healthy = False
        except Exception as e:
            logger.warning(f"GPU health check failed: {e}")
            self.device_healthy = False
        
        self.last_health_check = current_time
        return self.device_healthy
    
    async def handle_gpu_error(
        self, 
        error: Exception, 
        operation_name: str,
        original_args: tuple = None,
        original_kwargs: dict = None
    ) -> Optional[Any]:
        """Handle GPU error with intelligent fallback."""
        error_type = self.classify_error(error)
        error_info = GPUErrorInfo(
            error_type=error_type,
            error_message=str(error),
            timestamp=time.time(),
            operation_name=operation_name
        )
        
        logger.warning(f"GPU error in {operation_name}: {error_type.value} - {error}")
        
        # Get recommended fallback strategy
        fallback_strategy = await self.error_tracker.get_recommended_fallback(error_type)
        error_info.fallback_strategy = fallback_strategy
        
        recovery_start = time.time()
        recovery_result = None
        
        try:
            if fallback_strategy == FallbackStrategy.CPU_FALLBACK:
                recovery_result = await self._cpu_fallback(operation_name, original_args, original_kwargs)
                error_info.recovery_successful = recovery_result is not None
            
            elif fallback_strategy == FallbackStrategy.REDUCED_PRECISION:
                recovery_result = await self._reduced_precision_fallback(operation_name, original_args, original_kwargs)
                error_info.recovery_successful = recovery_result is not None
            
            elif fallback_strategy == FallbackStrategy.SMALLER_BATCH:
                recovery_result = await self._smaller_batch_fallback(operation_name, original_args, original_kwargs)
                error_info.recovery_successful = recovery_result is not None
            
            elif fallback_strategy == FallbackStrategy.SIMPLIFIED_ALGORITHM:
                recovery_result = await self._simplified_algorithm_fallback(operation_name, original_args, original_kwargs)
                error_info.recovery_successful = recovery_result is not None
            
        except Exception as fallback_error:
            logger.error(f"Fallback strategy {fallback_strategy.value} failed: {fallback_error}")
            error_info.recovery_successful = False
        
        error_info.recovery_time_ms = (time.time() - recovery_start) * 1000
        await self.error_tracker.record_error(error_info)
        
        return recovery_result
    
    async def _cpu_fallback(self, operation_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Fallback to CPU implementation."""
        try:
            # Import CPU fallback implementations
            from bolt.gpu_acceleration import _cpu_fallback_operation
            
            return _cpu_fallback_operation(operation_name, *args, **kwargs)
        except Exception as e:
            logger.error(f"CPU fallback failed for {operation_name}: {e}")
            return None
    
    async def _reduced_precision_fallback(self, operation_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Fallback using reduced precision."""
        if not MLX_AVAILABLE:
            return await self._cpu_fallback(operation_name, args, kwargs)
        
        try:
            # Convert to lower precision (float16) and retry
            if args:
                reduced_args = []
                for arg in args:
                    if hasattr(arg, 'astype') and hasattr(arg, 'dtype'):
                        reduced_args.append(arg.astype(mx.float16))
                    else:
                        reduced_args.append(arg)
                
                # Retry with reduced precision
                # This would need specific implementation per operation
                logger.info(f"Retrying {operation_name} with reduced precision")
                return None  # Placeholder
            
        except Exception as e:
            logger.error(f"Reduced precision fallback failed: {e}")
            return await self._cpu_fallback(operation_name, args, kwargs)
    
    async def _smaller_batch_fallback(self, operation_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Fallback using smaller batch sizes."""
        try:
            # Reduce batch size and retry
            # This requires operation-specific logic
            logger.info(f"Retrying {operation_name} with smaller batch size")
            return None  # Placeholder
        except Exception as e:
            logger.error(f"Smaller batch fallback failed: {e}")
            return await self._cpu_fallback(operation_name, args, kwargs)
    
    async def _simplified_algorithm_fallback(self, operation_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Fallback using simplified algorithms."""
        try:
            # Use simpler algorithm and retry
            logger.info(f"Retrying {operation_name} with simplified algorithm")
            return None  # Placeholder
        except Exception as e:
            logger.error(f"Simplified algorithm fallback failed: {e}")
            return await self._cpu_fallback(operation_name, args, kwargs)


# Global error handler instance
_gpu_error_handler = GPUErrorHandler()


def gpu_error_recovery(
    operation_name: str = None,
    config: GPUOperationConfig = None
):
    """Decorator for GPU operations with automatic error recovery."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal operation_name
            if operation_name is None:
                operation_name = func.__name__
            
            # Check GPU health first
            if not await _gpu_error_handler.check_gpu_health():
                logger.warning(f"GPU unhealthy, using CPU fallback for {operation_name}")
                return await _gpu_error_handler._cpu_fallback(operation_name, args, kwargs)
            
            max_retries = config.max_retries if config else 3
            timeout = config.timeout_seconds if config else 30.0
            
            for attempt in range(max_retries + 1):
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                    return result
                
                except asyncio.TimeoutError:
                    error = TimeoutError(f"Operation {operation_name} timed out after {timeout}s")
                    if attempt < max_retries:
                        logger.warning(f"Timeout in {operation_name}, retrying ({attempt + 1}/{max_retries})")
                        continue
                    else:
                        return await _gpu_error_handler.handle_gpu_error(error, operation_name, args, kwargs)
                
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Error in {operation_name}, retrying ({attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        return await _gpu_error_handler.handle_gpu_error(e, operation_name, args, kwargs)
            
            return None
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in async context
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def gpu_operation_context(operation_name: str, config: GPUOperationConfig = None):
    """Context manager for GPU operations with automatic error handling."""
    start_time = time.time()
    
    try:
        yield
    except Exception as e:
        logger.error(f"GPU operation {operation_name} failed: {e}")
        # Handle error appropriately
        raise
    finally:
        operation_time = time.time() - start_time
        if config and config.enable_monitoring:
            logger.debug(f"GPU operation {operation_name} completed in {operation_time:.3f}s")


def get_gpu_error_handler() -> GPUErrorHandler:
    """Get the global GPU error handler instance."""
    return _gpu_error_handler


def get_gpu_error_statistics() -> Dict[str, Any]:
    """Get comprehensive GPU error statistics."""
    return _gpu_error_handler.error_tracker.get_error_statistics()


# Production-ready error types for external use
class GPUMemoryError(Exception):
    """GPU memory related error."""
    pass


class GPUComputeError(Exception):
    """GPU computation error."""
    pass


class GPUDeviceError(Exception):
    """GPU device error."""
    pass


class GPUTimeoutError(Exception):
    """GPU operation timeout error."""
    pass


if __name__ == "__main__":
    # Test the error handling system
    async def test_gpu_error_handling():
        print("Testing GPU Error Handling System")
        print("=" * 50)
        
        handler = get_gpu_error_handler()
        
        # Test GPU health check
        is_healthy = await handler.check_gpu_health()
        print(f"GPU Health: {'Healthy' if is_healthy else 'Unhealthy'}")
        
        # Test error classification
        test_errors = [
            Exception("out of memory error"),
            Exception("kernel compilation failed"),
            Exception("device unavailable")
        ]
        
        for error in test_errors:
            error_type = handler.classify_error(error)
            print(f"Error: {error} -> Type: {error_type.value}")
        
        # Get statistics
        stats = await get_gpu_error_statistics()
        print(f"\nError Statistics: {stats}")
    
    asyncio.run(test_gpu_error_handling())