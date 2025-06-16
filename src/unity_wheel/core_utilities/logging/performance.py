"""
Performance logging utilities.
"""
import functools
import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class PerformanceLogger:
    """Logger for performance metrics."""

    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(name)
        self.metrics: dict[str, float] = {}

    def log_timing(self, operation: str, duration: float, **kwargs):
        """Log timing information."""
        self.metrics[operation] = duration
        self.logger.info(f"{operation} completed in {duration:.3f}s", extra=kwargs)

    def log_memory_usage(self, operation: str, memory_mb: float, **kwargs):
        """Log memory usage."""
        self.logger.info(f"{operation} used {memory_mb:.1f}MB", extra=kwargs)

    def get_metrics(self) -> dict[str, float]:
        """Get collected metrics."""
        return self.metrics.copy()


def timed_operation(
    operation_name: str | None = None, logger: PerformanceLogger | None = None
):
    """Decorator to time function execution."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                if logger:
                    logger.log_timing(name, duration)
                else:
                    logging.debug(f"{name} completed in {duration:.3f}s")

        return wrapper

    return decorator


def measure_time(func: Callable) -> Callable:
    """Simple timing decorator."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"{func.__name__} took {end - start:.3f}s")
        return result

    return wrapper


# Global performance logger instance
performance_logger = PerformanceLogger()
