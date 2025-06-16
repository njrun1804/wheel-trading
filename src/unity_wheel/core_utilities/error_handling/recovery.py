"""
Advanced Recovery and Retry System

Provides comprehensive retry mechanisms, backoff strategies, and recovery patterns
for both sync and async operations with circuit breaker integration.
"""

import asyncio
import builtins
import functools
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from .exceptions import TimeoutError, UnityWheelError
from .logging_enhanced import get_enhanced_logger, track_global_error

F = TypeVar("F", bound=Callable[..., Any])


class RecoveryStrategy(Enum):
    """Recovery strategy types."""

    RETRY = "retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class BackoffStrategy(Enum):
    """Backoff calculation strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    FIBONACCI = "fibonacci"
    POLYNOMIAL = "polynomial"


class RetryStrategy(Enum):
    """Retry strategies for error handling."""

    IMMEDIATE = "immediate"
    DELAY = "delay"
    BACKOFF = "backoff"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_max_pct: float = 0.1
    timeout_seconds: float | None = None
    retry_on_exceptions: list[type] | None = None
    stop_on_exceptions: list[type] | None = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    success_threshold: int = 2  # Successful calls needed to close circuit
    half_open_max_calls: int = 3  # Max calls allowed in half-open state


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""

    attempt_number: int
    delay_seconds: float
    exception: Exception | None
    timestamp: float
    success: bool
    execution_time_ms: float


class BackoffCalculator:
    """Calculates backoff delays using various strategies."""

    @staticmethod
    def exponential(
        attempt: int,
        base_delay: float,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
    ) -> float:
        """Calculate exponential backoff delay."""
        delay = base_delay * (multiplier ** (attempt - 1))
        return min(delay, max_delay)

    @staticmethod
    def linear(
        attempt: int,
        base_delay: float,
        multiplier: float = 1.0,
        max_delay: float = 60.0,
    ) -> float:
        """Calculate linear backoff delay."""
        delay = base_delay * attempt * multiplier
        return min(delay, max_delay)

    @staticmethod
    def fibonacci(attempt: int, base_delay: float, max_delay: float = 60.0) -> float:
        """Calculate Fibonacci backoff delay."""

        def fib(n):
            if n <= 2:
                return 1
            return fib(n - 1) + fib(n - 2)

        delay = base_delay * fib(attempt)
        return min(delay, max_delay)

    @staticmethod
    def polynomial(
        attempt: int, base_delay: float, power: float = 2.0, max_delay: float = 60.0
    ) -> float:
        """Calculate polynomial backoff delay."""
        delay = base_delay * (attempt**power)
        return min(delay, max_delay)

    @staticmethod
    def add_jitter(delay: float, max_jitter_pct: float = 0.1) -> float:
        """Add jitter to delay to prevent thundering herd."""
        jitter_amount = delay * max_jitter_pct
        jitter = random.uniform(-jitter_amount, jitter_amount)
        return max(0, delay + jitter)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(self, config: CircuitBreakerConfig, name: str):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self.logger = get_enhanced_logger(f"circuit_breaker.{name}")

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (
                time.time() - self.last_failure_time
                > self.config.recovery_timeout_seconds
            ):
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                self.logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
                return True
            return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
        else:
            self.failure_count = 0  # Reset failure count on success

    def record_failure(self) -> None:
        """Record a failed execution."""
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failure during half-open, go back to open
            self.state = CircuitBreakerState.OPEN
            self.half_open_calls = 0
            self.logger.warning(
                f"Circuit breaker {self.name} back to OPEN after half-open failure"
            )
        else:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(
                    f"Circuit breaker {self.name} OPENED after {self.failure_count} failures"
                )

    def execute_call(self) -> None:
        """Mark that a call is being executed (for half-open state)."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self.half_open_calls,
        }


class RecoveryManager:
    """Manages recovery strategies and circuit breakers."""

    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.retry_stats: dict[str, list[RetryAttempt]] = {}
        self.logger = get_enhanced_logger("recovery_manager")

    def get_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(config, name)
        return self.circuit_breakers[name]

    def record_retry_attempt(self, operation: str, attempt: RetryAttempt) -> None:
        """Record a retry attempt for statistics."""
        if operation not in self.retry_stats:
            self.retry_stats[operation] = []

        self.retry_stats[operation].append(attempt)

        # Keep only last 100 attempts per operation
        if len(self.retry_stats[operation]) > 100:
            self.retry_stats[operation] = self.retry_stats[operation][-100:]

    def get_retry_stats(self, operation: str) -> dict[str, Any]:
        """Get retry statistics for an operation."""
        attempts = self.retry_stats.get(operation, [])
        if not attempts:
            return {"operation": operation, "total_attempts": 0}

        successful_attempts = [a for a in attempts if a.success]
        failed_attempts = [a for a in attempts if not a.success]

        return {
            "operation": operation,
            "total_attempts": len(attempts),
            "successful_attempts": len(successful_attempts),
            "failed_attempts": len(failed_attempts),
            "success_rate": len(successful_attempts) / len(attempts),
            "average_execution_time_ms": sum(a.execution_time_ms for a in attempts)
            / len(attempts),
            "average_attempts_to_success": (
                sum(a.attempt_number for a in successful_attempts)
                / len(successful_attempts)
                if successful_attempts
                else 0
            ),
        }

    def get_all_stats(self) -> dict[str, Any]:
        """Get all recovery statistics."""
        return {
            "circuit_breakers": {
                name: cb.get_status() for name, cb in self.circuit_breakers.items()
            },
            "retry_operations": {
                op: self.get_retry_stats(op) for op in self.retry_stats.keys()
            },
        }


# Global recovery manager
_recovery_manager = RecoveryManager()


def exponential_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    jitter: bool = True,
    retry_on: list[type] | None = None,
    stop_on: list[type] | None = None,
) -> Callable[[F], F]:
    """Decorator for exponential backoff retry."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        backoff_multiplier=multiplier,
        jitter=jitter,
        retry_on_exceptions=retry_on,
        stop_on_exceptions=stop_on,
    )
    return with_retry(config)


def linear_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 1.0,
    jitter: bool = True,
    retry_on: list[type] | None = None,
    stop_on: list[type] | None = None,
) -> Callable[[F], F]:
    """Decorator for linear backoff retry."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
        backoff_strategy=BackoffStrategy.LINEAR,
        backoff_multiplier=multiplier,
        jitter=jitter,
        retry_on_exceptions=retry_on,
        stop_on_exceptions=stop_on,
    )
    return with_retry(config)


def with_retry(config: RetryConfig) -> Callable[[F], F]:
    """Decorator that adds retry logic to functions."""

    def decorator(func: F) -> F:
        operation_name = f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _async_retry_execute(
                    func, config, operation_name, *args, **kwargs
                )

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return _sync_retry_execute(
                    func, config, operation_name, *args, **kwargs
                )

            return sync_wrapper

    return decorator


def async_with_retry(config: RetryConfig) -> Callable[[F], F]:
    """Decorator specifically for async functions with retry logic."""

    def decorator(func: F) -> F:
        operation_name = f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await _async_retry_execute(
                func, config, operation_name, *args, **kwargs
            )

        return wrapper

    return decorator


def _should_retry(exception: Exception, config: RetryConfig) -> bool:
    """Determine if an exception should trigger a retry."""
    # Check stop conditions first
    if config.stop_on_exceptions:
        if isinstance(exception, tuple(config.stop_on_exceptions)):
            return False

    # Check retry conditions
    if config.retry_on_exceptions:
        return isinstance(exception, tuple(config.retry_on_exceptions))

    # Default: retry on UnityWheelError if it's retryable
    if isinstance(exception, UnityWheelError):
        return exception.is_retryable()

    # Don't retry on system errors by default
    return not isinstance(exception, (KeyboardInterrupt, SystemExit, MemoryError))


def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt."""
    if config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
        delay = BackoffCalculator.exponential(
            attempt,
            config.base_delay_seconds,
            config.backoff_multiplier,
            config.max_delay_seconds,
        )
    elif config.backoff_strategy == BackoffStrategy.LINEAR:
        delay = BackoffCalculator.linear(
            attempt,
            config.base_delay_seconds,
            config.backoff_multiplier,
            config.max_delay_seconds,
        )
    elif config.backoff_strategy == BackoffStrategy.FIBONACCI:
        delay = BackoffCalculator.fibonacci(
            attempt, config.base_delay_seconds, config.max_delay_seconds
        )
    elif config.backoff_strategy == BackoffStrategy.POLYNOMIAL:
        delay = BackoffCalculator.polynomial(
            attempt, config.base_delay_seconds, 2.0, config.max_delay_seconds
        )
    else:  # FIXED
        delay = config.base_delay_seconds

    # Add jitter if enabled
    if config.jitter:
        delay = BackoffCalculator.add_jitter(delay, config.jitter_max_pct)

    return delay


def _sync_retry_execute(
    func: Callable, config: RetryConfig, operation_name: str, *args, **kwargs
) -> Any:
    """Execute function with sync retry logic."""
    logger = get_enhanced_logger(func.__module__)
    last_exception = None

    for attempt in range(1, config.max_attempts + 1):
        start_time = time.perf_counter()
        attempt_start = time.time()

        try:
            # Apply timeout if configured
            if config.timeout_seconds:
                # For sync functions, we can't easily add timeout without threading
                # Just execute normally
                result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - record attempt and return
            execution_time = (time.perf_counter() - start_time) * 1000

            attempt_record = RetryAttempt(
                attempt_number=attempt,
                delay_seconds=0,
                exception=None,
                timestamp=attempt_start,
                success=True,
                execution_time_ms=execution_time,
            )
            _recovery_manager.record_retry_attempt(operation_name, attempt_record)

            if attempt > 1:
                logger.info(
                    f"Retry successful for {operation_name} on attempt {attempt}"
                )

            return result

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            last_exception = e

            # Check if we should retry
            if attempt >= config.max_attempts or not _should_retry(e, config):
                # Record final failed attempt
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    delay_seconds=0,
                    exception=e,
                    timestamp=attempt_start,
                    success=False,
                    execution_time_ms=execution_time,
                )
                _recovery_manager.record_retry_attempt(operation_name, attempt_record)

                # Track global error
                track_global_error(e, func.__module__)

                # Re-raise the exception
                raise

            # Calculate delay for next attempt
            delay = _calculate_delay(attempt, config)

            # Record retry attempt
            attempt_record = RetryAttempt(
                attempt_number=attempt,
                delay_seconds=delay,
                exception=e,
                timestamp=attempt_start,
                success=False,
                execution_time_ms=execution_time,
            )
            _recovery_manager.record_retry_attempt(operation_name, attempt_record)

            logger.warning(
                f"Attempt {attempt} failed for {operation_name}, retrying in {delay:.2f}s: {str(e)}"
            )

            # Wait before next attempt
            time.sleep(delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


async def _async_retry_execute(
    func: Callable, config: RetryConfig, operation_name: str, *args, **kwargs
) -> Any:
    """Execute async function with retry logic."""
    logger = get_enhanced_logger(func.__module__)
    last_exception = None

    for attempt in range(1, config.max_attempts + 1):
        start_time = time.perf_counter()
        attempt_start = time.time()

        try:
            # Apply timeout if configured
            if config.timeout_seconds:
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=config.timeout_seconds
                )
            else:
                result = await func(*args, **kwargs)

            # Success - record attempt and return
            execution_time = (time.perf_counter() - start_time) * 1000

            attempt_record = RetryAttempt(
                attempt_number=attempt,
                delay_seconds=0,
                exception=None,
                timestamp=attempt_start,
                success=True,
                execution_time_ms=execution_time,
            )
            _recovery_manager.record_retry_attempt(operation_name, attempt_record)

            if attempt > 1:
                logger.info(
                    f"Async retry successful for {operation_name} on attempt {attempt}"
                )

            return result

        except builtins.TimeoutError as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            timeout_error = TimeoutError(
                f"Operation {operation_name} timed out after {config.timeout_seconds}s on attempt {attempt}",
                timeout_seconds=config.timeout_seconds,
                elapsed_seconds=execution_time / 1000,
                operation=operation_name,
                cause=e,
            )
            last_exception = timeout_error

            # Record timeout attempt
            attempt_record = RetryAttempt(
                attempt_number=attempt,
                delay_seconds=0,
                exception=timeout_error,
                timestamp=attempt_start,
                success=False,
                execution_time_ms=execution_time,
            )
            _recovery_manager.record_retry_attempt(operation_name, attempt_record)

            # Check if we should retry timeouts
            if attempt >= config.max_attempts or not _should_retry(
                timeout_error, config
            ):
                track_global_error(timeout_error, func.__module__)
                raise timeout_error

            # Calculate delay and continue
            delay = _calculate_delay(attempt, config)
            logger.warning(
                f"Timeout on attempt {attempt} for {operation_name}, retrying in {delay:.2f}s"
            )
            await asyncio.sleep(delay)

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            last_exception = e

            # Check if we should retry
            if attempt >= config.max_attempts or not _should_retry(e, config):
                # Record final failed attempt
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    delay_seconds=0,
                    exception=e,
                    timestamp=attempt_start,
                    success=False,
                    execution_time_ms=execution_time,
                )
                _recovery_manager.record_retry_attempt(operation_name, attempt_record)

                # Track global error
                track_global_error(e, func.__module__)

                # Re-raise the exception
                raise

            # Calculate delay for next attempt
            delay = _calculate_delay(attempt, config)

            # Record retry attempt
            attempt_record = RetryAttempt(
                attempt_number=attempt,
                delay_seconds=delay,
                exception=e,
                timestamp=attempt_start,
                success=False,
                execution_time_ms=execution_time,
            )
            _recovery_manager.record_retry_attempt(operation_name, attempt_record)

            logger.warning(
                f"Async attempt {attempt} failed for {operation_name}, retrying in {delay:.2f}s: {str(e)}"
            )

            # Wait before next attempt
            await asyncio.sleep(delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


def get_recovery_manager() -> RecoveryManager:
    """Get the global recovery manager."""
    return _recovery_manager
