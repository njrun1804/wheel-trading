"""
Enhanced Logging System with Structured Context and Error Tracking

Builds on the existing structured logging to add comprehensive error tracking,
performance monitoring, and debugging visibility.
"""

import asyncio
import functools
import logging
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import uuid4

try:
    from ..logging.structured import LogContext as BaseLogContext
    from ..logging.structured import (
        StructuredLogger,
        execution_context,
        request_id,
    )
    from ..logging.structured import get_logger as get_base_logger
except ImportError:
    # Fallback if structured logging is not available
    import logging
    from contextvars import ContextVar
    from typing import Any

    request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
    execution_context: ContextVar[dict[str, Any]] = ContextVar(
        "execution_context", default={}
    )

    class StructuredLogger(logging.LoggerAdapter):
        def __init__(
            self, logger: logging.Logger, extra: dict[str, Any] | None = None
        ):
            super().__init__(logger, extra or {})

    class BaseLogContext:
        def __init__(self, **context):
            self.context = context

        def __enter__(self):
            execution_context.set(self.context)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            execution_context.set({})

    def get_base_logger(name: str | None = None) -> StructuredLogger:
        if name is None:
            import inspect

            frame = inspect.currentframe()
            if frame and frame.f_back:
                name = frame.f_back.f_globals.get("__name__", "unknown")
            else:
                name = "unknown"
        base_logger = logging.getLogger(name)
        return StructuredLogger(base_logger)


# Enhanced context variables
operation_id: ContextVar[str | None] = ContextVar("operation_id", default=None)
performance_context: ContextVar[dict[str, Any]] = ContextVar(
    "performance_context", default={}
)
error_context: ContextVar[dict[str, Any]] = ContextVar("error_context", default={})

F = TypeVar("F", bound=Callable[..., Any])


class EnhancedLogger(StructuredLogger):
    """
    Enhanced logger with error tracking, performance monitoring, and debugging features.
    """

    def __init__(self, logger: logging.Logger, extra: dict[str, Any] | None = None):
        super().__init__(logger, extra)
        self.error_count = 0
        self.warning_count = 0
        self.performance_samples = []
        self.last_error_time = None

    def error(self, msg: Any, *args, **kwargs) -> None:
        """Enhanced error logging with tracking."""
        self.error_count += 1
        self.last_error_time = time.time()

        # Add error context
        extra = kwargs.get("extra", {})
        if isinstance(extra, dict):
            extra.update(
                {
                    "error_sequence": self.error_count,
                    "logger_name": self.logger.name,
                    "thread_id": threading.current_thread().ident,
                }
            )

            # Add error context from context vars
            err_ctx = error_context.get({})
            if err_ctx:
                extra["error_context"] = err_ctx

            kwargs["extra"] = extra

        super().error(msg, *args, **kwargs)

    def warning(self, msg: Any, *args, **kwargs) -> None:
        """Enhanced warning logging with tracking."""
        self.warning_count += 1

        # Add warning context
        extra = kwargs.get("extra", {})
        if isinstance(extra, dict):
            extra.update(
                {
                    "warning_sequence": self.warning_count,
                    "logger_name": self.logger.name,
                }
            )
            kwargs["extra"] = extra

        super().warning(msg, *args, **kwargs)

    def performance(self, operation: str, duration_ms: float, **context) -> None:
        """Log performance metrics."""
        self.performance_samples.append(
            {
                "operation": operation,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            }
        )

        # Keep only last 100 samples
        if len(self.performance_samples) > 100:
            self.performance_samples = self.performance_samples[-100:]

        self.info(
            f"Performance: {operation} took {duration_ms:.2f}ms",
            extra={
                "metric_type": "performance",
                "operation": operation,
                "duration_ms": duration_ms,
                **context,
            },
        )

    def debug_context(self, msg: Any, **context) -> None:
        """Log debug information with rich context."""
        self.debug(
            msg,
            extra={
                "debug_context": context,
                "execution_context": execution_context.get({}),
                "operation_id": operation_id.get(),
                "request_id": request_id.get(),
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get logger statistics."""
        return {
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "last_error_time": self.last_error_time,
            "performance_samples_count": len(self.performance_samples),
            "avg_performance_ms": (
                sum(s["duration_ms"] for s in self.performance_samples)
                / len(self.performance_samples)
                if self.performance_samples
                else 0.0
            ),
        }


class ErrorLogger:
    """Specialized logger for error reporting and tracking."""

    def __init__(self, logger: EnhancedLogger):
        self.logger = logger
        self.error_patterns = {}
        self.error_frequency = {}

    def log_error(
        self,
        error: Exception,
        *,
        component: str | None = None,
        operation: str | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
        additional_context: dict[str, Any] | None = None,
        stack_trace: bool = True,
    ) -> None:
        """Log error with comprehensive context."""

        error_type = type(error).__name__
        error_message = str(error)

        # Track error patterns
        pattern_key = f"{error_type}:{component}:{operation}"
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1

        # Track error frequency
        current_hour = int(time.time() // 3600)
        freq_key = f"{error_type}:{current_hour}"
        self.error_frequency[freq_key] = self.error_frequency.get(freq_key, 0) + 1

        # Build comprehensive error context
        error_context = {
            "error_type": error_type,
            "error_message": error_message,
            "component": component,
            "operation": operation,
            "user_id": user_id,
            "request_id": request_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "error_pattern_count": self.error_patterns.get(pattern_key, 0),
            "error_frequency_hour": self.error_frequency.get(freq_key, 0),
        }

        # Add additional context
        if additional_context:
            error_context.update(additional_context)

        # Add stack trace if requested and available
        if stack_trace and hasattr(error, "__traceback__"):
            import traceback

            error_context["stack_trace"] = traceback.format_exception(
                type(error), error, error.__traceback__
            )

        # Check if this is a structured UnityWheelError
        if hasattr(error, "to_dict"):
            error_context.update(error.to_dict())

        self.logger.error(
            f"Error in {component or 'unknown'}.{operation or 'unknown'}: {error_message}",
            extra=error_context,
        )

    def get_error_patterns(self) -> dict[str, int]:
        """Get error pattern statistics."""
        return self.error_patterns.copy()

    def get_error_frequency(self) -> dict[str, int]:
        """Get error frequency statistics."""
        return self.error_frequency.copy()


class LogContext(BaseLogContext):
    """Enhanced log context with error tracking and performance monitoring."""

    def __init__(self, **context):
        super().__init__(**context)
        self.operation_id = str(uuid4())[:8]
        self.start_time = time.perf_counter()
        self.performance_metrics = {}

    def __enter__(self):
        result = super().__enter__()
        operation_id.set(self.operation_id)
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate execution time
        execution_time = (time.perf_counter() - self.start_time) * 1000  # ms

        # Log performance metrics
        perf_ctx = performance_context.get({})
        perf_ctx.update(
            {
                "operation_id": self.operation_id,
                "execution_time_ms": execution_time,
                **self.performance_metrics,
            }
        )
        performance_context.set(perf_ctx)

        # If there was an exception, add error context
        if exc_type:
            err_ctx = error_context.get({})
            err_ctx.update(
                {
                    "operation_id": self.operation_id,
                    "execution_time_ms": execution_time,
                    "exception_type": exc_type.__name__ if exc_type else None,
                    "exception_message": str(exc_val) if exc_val else None,
                }
            )
            error_context.set(err_ctx)

        super().__exit__(exc_type, exc_val, exc_tb)
        operation_id.set(None)

    def add_performance_metric(self, name: str, value: Any) -> None:
        """Add a performance metric to the context."""
        self.performance_metrics[name] = value

    def mark_milestone(self, milestone: str) -> None:
        """Mark a timing milestone."""
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.performance_metrics[f"milestone_{milestone}"] = elapsed_ms


class AsyncLogContext:
    """Async context manager for logging with automatic cleanup."""

    def __init__(self, **context):
        self.context = context
        self.operation_id = str(uuid4())[:8]
        self.start_time = time.perf_counter()
        self.task_info = {}

    async def __aenter__(self):
        # Set context variables
        execution_context.set(self.context)
        operation_id.set(self.operation_id)

        # Track async task info
        current_task = asyncio.current_task()
        if current_task:
            self.task_info = {
                "task_name": current_task.get_name(),
                "task_id": str(id(current_task)),
            }

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        execution_time = (time.perf_counter() - self.start_time) * 1000  # ms

        # Update performance context
        perf_ctx = performance_context.get({})
        perf_ctx.update(
            {
                "operation_id": self.operation_id,
                "execution_time_ms": execution_time,
                "async_task": True,
                **self.task_info,
            }
        )
        performance_context.set(perf_ctx)

        # Handle exceptions
        if exc_type:
            err_ctx = error_context.get({})
            err_ctx.update(
                {
                    "operation_id": self.operation_id,
                    "execution_time_ms": execution_time,
                    "async_task": True,
                    "exception_type": exc_type.__name__ if exc_type else None,
                    "exception_message": str(exc_val) if exc_val else None,
                    **self.task_info,
                }
            )
            error_context.set(err_ctx)

        # Clear context
        execution_context.set({})
        operation_id.set(None)


def get_enhanced_logger(name: str | None = None) -> EnhancedLogger:
    """Get an enhanced logger instance."""
    base_logger = get_base_logger(name)
    return EnhancedLogger(base_logger.logger, base_logger.extra)


def log_execution_time(operation: str) -> Callable[[F], F]:
    """Decorator to log function execution time."""

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger = get_enhanced_logger(func.__module__)
                    logger.performance(f"{operation}.{func.__name__}", duration_ms)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger = get_enhanced_logger(func.__module__)
                    logger.performance(f"{operation}.{func.__name__}", duration_ms)

            return sync_wrapper

    return decorator


def log_with_context(**context) -> Callable[[F], F]:
    """Decorator to add context to all log messages in function."""

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with AsyncLogContext(**context):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with LogContext(**context):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def structured_error_log(logger: EnhancedLogger, error: Exception, **context) -> None:
    """Log error with structured information."""
    error_logger = ErrorLogger(logger)
    error_logger.log_error(error, additional_context=context)


def performance_log(
    logger: EnhancedLogger, operation: str, duration_ms: float, **metrics
) -> None:
    """Log performance metrics."""
    logger.performance(operation, duration_ms, **metrics)


# Global error tracking
_global_error_stats = {
    "total_errors": 0,
    "errors_by_type": {},
    "errors_by_component": {},
    "last_error_time": None,
}


def track_global_error(error: Exception, component: str | None = None) -> None:
    """Track global error statistics."""
    _global_error_stats["total_errors"] += 1
    _global_error_stats["last_error_time"] = time.time()

    error_type = type(error).__name__
    _global_error_stats["errors_by_type"][error_type] = (
        _global_error_stats["errors_by_type"].get(error_type, 0) + 1
    )

    if component:
        _global_error_stats["errors_by_component"][component] = (
            _global_error_stats["errors_by_component"].get(component, 0) + 1
        )


def get_global_error_stats() -> dict[str, Any]:
    """Get global error statistics."""
    return _global_error_stats.copy()


def reset_global_error_stats() -> None:
    """Reset global error statistics."""
    global _global_error_stats
    _global_error_stats = {
        "total_errors": 0,
        "errors_by_type": {},
        "errors_by_component": {},
        "last_error_time": None,
    }
