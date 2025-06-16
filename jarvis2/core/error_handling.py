"""Comprehensive error handling and recovery for Jarvis2."""
import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context for error handling."""

    component: str
    operation: str
    severity: ErrorSeverity
    error: Exception
    timestamp: float
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = None


class RecoveryStrategy:
    """Base class for recovery strategies."""

    def recover(self, context: ErrorContext) -> Any:
        """Attempt recovery from error."""
        raise NotImplementedError


class RetryStrategy(RecoveryStrategy):
    """Retry with exponential backoff."""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0):
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def recover(self, context: ErrorContext) -> Any:
        """Retry the operation with backoff."""
        if context.retry_count >= context.max_retries:
            logger.error(
                f"Max retries ({context.max_retries}) exceeded for {context.operation}"
            )
            raise context.error
        delay = min(self.base_delay * 2**context.retry_count, self.max_delay)
        logger.warning(
            f"Retrying {context.operation} after {delay:.1f}s (attempt {context.retry_count + 1})"
        )
        await asyncio.sleep(delay)
        context.retry_count += 1
        return None


class FallbackStrategy(RecoveryStrategy):
    """Use fallback implementation."""

    def __init__(self, fallback_func: Callable):
        self.fallback_func = fallback_func

    async def recover(self, context: ErrorContext) -> Any:
        """Use fallback implementation."""
        logger.warning(f"Using fallback for {context.operation}")
        if asyncio.iscoroutinefunction(self.fallback_func):
            return await self.fallback_func(context.metadata)
        else:
            return self.fallback_func(context.metadata)


class DegradedModeStrategy(RecoveryStrategy):
    """Continue with reduced functionality."""

    def recover(self, context: ErrorContext) -> Any:
        """Return degraded result."""
        logger.warning(f"Operating in degraded mode for {context.operation}")
        if context.component == "neural":
            return {"value": [[0.5]], "policy": [1.0 / 50] * 50, "degraded": True}
        elif context.component == "search":
            return {
                "best_code": """# Error during search - returning template
def solution():
    pass""",
                "confidence": 0.1,
                "alternatives": [],
                "stats": {"error": str(context.error)},
            }
        elif context.component == "vector_index":
            return []
        else:
            return None


class ErrorHandler:
    """Central error handling with recovery strategies."""

    def __init__(self):
        self.strategies = {
            ErrorSeverity.LOW: RetryStrategy(),
            ErrorSeverity.MEDIUM: DegradedModeStrategy(),
            ErrorSeverity.HIGH: FallbackStrategy(self._emergency_fallback),
            ErrorSeverity.CRITICAL: None,
        }
        self.error_counts = {}
        self.circuit_breakers = {}

    async def handle_error(self, context: ErrorContext) -> Any:
        """Handle error with appropriate strategy."""
        logger.error(
            f"Error in {context.component}.{context.operation}: {context.error}",
            exc_info=True,
        )
        key = f"{context.component}.{context.operation}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        if self._is_circuit_open(key):
            logger.error(f"Circuit breaker OPEN for {key}")
            return await self._emergency_fallback(context.metadata)
        strategy = self.strategies.get(context.severity)
        if strategy:
            try:
                return await strategy.recover(context)
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
                context.severity = ErrorSeverity(
                    list(ErrorSeverity)[
                        min(
                            list(ErrorSeverity).index(context.severity) + 1,
                            len(ErrorSeverity) - 1,
                        )
                    ]
                )
                return await self.handle_error(context)
        else:
            logger.critical(f"CRITICAL error in {context.component}: {context.error}")
            raise context.error

    def _is_circuit_open(self, key: str) -> bool:
        """Check if circuit breaker is open."""
        error_count = self.error_counts.get(key, 0)
        last_check = self.circuit_breakers.get(key, 0)
        if error_count > 10:
            if time.time() - last_check < 300:
                return True
            else:
                self.error_counts[key] = 0
                self.circuit_breakers[key] = time.time()
        return False

    def _emergency_fallback(self, metadata: dict | None) -> Any:
        """Emergency fallback when all else fails."""
        logger.critical("Using emergency fallback")
        return {"error": "System unavailable", "fallback": True, "metadata": metadata}


_error_handler = ErrorHandler()


def with_error_handling(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    max_retries: int = 3,
):
    """Decorator for error handling with recovery."""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            context = ErrorContext(
                component=component,
                operation=operation,
                severity=severity,
                error=None,
                timestamp=time.time(),
                retry_count=0,
                max_retries=max_retries,
                metadata={"args": args, "kwargs": kwargs},
            )
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context.error = e
                    result = await _error_handler.handle_error(context)
                    if result is None and context.retry_count < context.max_retries:
                        continue
                    else:
                        return result

        return wrapper

    return decorator


def with_timeout(timeout_seconds: float, default_value: Any = None):
    """Decorator for timeout handling."""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout_seconds
                )
            except TimeoutError:
                logger.error(f"{func.__name__} timed out after {timeout_seconds}s")
                if default_value is not None:
                    return default_value
                raise

        return wrapper

    return decorator


class ResourceGuard:
    """Guard against resource exhaustion."""

    def __init__(self, max_memory_gb: float = 18.0, max_cpu_percent: float = 90.0):
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent
        self._last_check = 0
        self._check_interval = 5.0

    def check_resources(self) -> bool:
        """Check if resources are within limits."""
        now = time.time()
        if now - self._last_check < self._check_interval:
            return True
        self._last_check = now
        import psutil

        memory = psutil.virtual_memory()
        memory_gb = memory.used / 1024**3
        if memory_gb > self.max_memory_gb:
            logger.warning(
                f"Memory usage high: {memory_gb:.1f}GB / {self.max_memory_gb}GB"
            )
            return False
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > self.max_cpu_percent:
            logger.warning(f"CPU usage high: {cpu_percent:.1f}%")
            return False
        return True

    async def wait_for_resources(self, timeout: float = 60.0):
        """Wait for resources to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_resources():
                return True
            logger.info("Waiting for resources to become available...")
            await asyncio.sleep(5)
        return False


async def example_protected_function():
    """Example function with error handling."""

    @with_error_handling("example", "process", ErrorSeverity.LOW)
    @with_timeout(30.0, default_value="timeout_fallback")
    async def process_data(data: str) -> str:
        if not data:
            raise ValueError("No data provided")
        await asyncio.sleep(1)
        return f"Processed: {data}"

    result = await process_data("test")
    return result
