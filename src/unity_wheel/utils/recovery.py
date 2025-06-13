"""Error recovery and resilience mechanisms for autonomous operation."""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class RecoveryStrategy(Enum):
    """Available recovery strategies."""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    DEGRADE = "degrade"
    SKIP = "skip"


@dataclass
class RecoveryContext:
    """Context for recovery operations."""

    operation: str
    attempt: int = 0
    max_attempts: int = 3
    errors: List[Exception] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        """Time elapsed since first attempt."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    @property
    def should_retry(self) -> bool:
        """Check if we should retry."""
        return self.attempt < self.max_attempts

    def record_error(self, error: Exception) -> None:
        """Record an error."""
        self.errors.append(error)
        self.attempt += 1


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(minutes=5),
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._state = "closed"  # closed, open, half_open

    @property
    def state(self) -> str:
        """Get current circuit state."""
        if self._state == "open":
            # Check if we should transition to half-open
            if self._last_failure_time:
                elapsed = datetime.now(timezone.utc) - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = "half_open"
                    self._half_open_calls = 0
                    logger.info(f"Circuit breaker transitioning to half-open")
        return self._state

    def call_succeeded(self) -> None:
        """Record successful call."""
        if self.state == "half_open":
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                # Enough successful calls, close the circuit
                self._state = "closed"
                self._failure_count = 0
                self._last_failure_time = None
                logger.info("Circuit breaker closed after successful recovery")
        elif self.state == "closed":
            # Reset failure count on success
            self._failure_count = 0

    def call_failed(self) -> None:
        """Record failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        if self.state == "half_open":
            # Failure in half-open state reopens circuit
            self._state = "open"
            logger.warning("Circuit breaker reopened due to failure in half-open state")
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(f"Circuit breaker opened after {self._failure_count} failures")

    def is_open(self) -> bool:
        """Check if circuit is open (calls should be blocked)."""
        return self.state == "open"

    def is_available(self) -> bool:
        """Check if calls are allowed."""
        return self.state in ("closed", "half_open")


class RecoveryManager:
    """Manages recovery strategies for different operations."""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_values: Dict[str, Any] = {}
        self.degraded_operations: set[str] = set()

    def get_circuit_breaker(self, operation: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker()
        return self.circuit_breakers[operation]

    def set_fallback(self, operation: str, value: Any) -> None:
        """Set fallback value for operation."""
        self.fallback_values[operation] = value

    def mark_degraded(self, operation: str) -> None:
        """Mark operation as running in degraded mode."""
        self.degraded_operations.add(operation)
        logger.warning(f"Operation {operation} marked as degraded")

    def is_degraded(self, operation: str) -> bool:
        """Check if operation is degraded."""
        return operation in self.degraded_operations

    def recover_operation(self, operation: str) -> None:
        """Mark operation as recovered."""
        self.degraded_operations.discard(operation)
        logger.info(f"Operation {operation} recovered from degraded state")


# Global recovery manager
recovery_manager = RecoveryManager()


def with_recovery(
    *,
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    fallback_value: Any = None,
    operation_name: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator for adding recovery logic to functions."""

    def decorator(func: F) -> F:
        operation = operation_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = RecoveryContext(operation=operation, max_attempts=max_attempts)

            while True:
                try:
                    # Check circuit breaker
                    breaker = recovery_manager.get_circuit_breaker(operation)
                    if breaker.is_open():
                        raise Exception(f"Circuit breaker open for {operation}")

                    # Attempt operation
                    result = func(*args, **kwargs)

                    # Success - notify circuit breaker
                    breaker.call_succeeded()

                    # Clear degraded state if previously degraded
                    if recovery_manager.is_degraded(operation):
                        recovery_manager.recover_operation(operation)

                    return result

                except (ValueError, KeyError, AttributeError) as e:
                    context.record_error(e)
                    breaker = recovery_manager.get_circuit_breaker(operation)
                    breaker.call_failed()

                    logger.error(
                        f"Operation {operation} failed (attempt {context.attempt}/{max_attempts})",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "attempt": context.attempt,
                            "operation": operation,
                        },
                    )

                    # Apply recovery strategy
                    if strategy == RecoveryStrategy.RETRY and context.should_retry:
                        # Exponential backoff
                        wait_time = backoff_factor ** (context.attempt - 1)
                        logger.info(f"Retrying {operation} after {wait_time:.1f}s")
                        time.sleep(wait_time)
                        continue

                    elif strategy == RecoveryStrategy.FALLBACK and fallback_value is not None:
                        logger.warning(f"Using fallback value for {operation}")
                        return fallback_value

                    elif strategy == RecoveryStrategy.DEGRADE:
                        recovery_manager.mark_degraded(operation)
                        # Try to return a degraded result
                        if hasattr(func, "__degraded_mode__"):
                            return func.__degraded_mode__(*args, **kwargs)

                    elif strategy == RecoveryStrategy.SKIP:
                        logger.warning(f"Skipping failed operation {operation}")
                        return None

                    # No recovery possible
                    raise

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = RecoveryContext(operation=operation, max_attempts=max_attempts)

            while True:
                try:
                    # Check circuit breaker
                    breaker = recovery_manager.get_circuit_breaker(operation)
                    if breaker.is_open():
                        raise Exception(f"Circuit breaker open for {operation}")

                    # Attempt operation
                    result = await func(*args, **kwargs)

                    # Success - notify circuit breaker
                    breaker.call_succeeded()

                    # Clear degraded state if previously degraded
                    if recovery_manager.is_degraded(operation):
                        recovery_manager.recover_operation(operation)

                    return result

                except (ValueError, KeyError, AttributeError) as e:
                    context.record_error(e)
                    breaker = recovery_manager.get_circuit_breaker(operation)
                    breaker.call_failed()

                    logger.error(
                        f"Operation {operation} failed (attempt {context.attempt}/{max_attempts})",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "attempt": context.attempt,
                            "operation": operation,
                        },
                    )

                    # Apply recovery strategy
                    if strategy == RecoveryStrategy.RETRY and context.should_retry:
                        # Exponential backoff
                        wait_time = backoff_factor ** (context.attempt - 1)
                        logger.info(f"Retrying {operation} after {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        continue

                    elif strategy == RecoveryStrategy.FALLBACK and fallback_value is not None:
                        logger.warning(f"Using fallback value for {operation}")
                        return fallback_value

                    elif strategy == RecoveryStrategy.DEGRADE:
                        recovery_manager.mark_degraded(operation)
                        # Try to return a degraded result
                        if hasattr(func, "__degraded_mode__"):
                            return await func.__degraded_mode__(*args, **kwargs)

                    elif strategy == RecoveryStrategy.SKIP:
                        logger.warning(f"Skipping failed operation {operation}")
                        return None

                    # No recovery possible
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


@contextmanager
def recovery_context(operation: str, fallback: Any = None) -> None:
    """Context manager for manual recovery handling."""
    context = RecoveryContext(operation=operation)

    try:
        yield context
    except (ValueError, KeyError, AttributeError) as e:
        context.record_error(e)
        logger.error(
            f"Error in recovery context for {operation}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "operation": operation,
            },
        )

        if fallback is not None:
            logger.warning(f"Using fallback value for {operation}")
            return fallback
        raise


def validate_and_recover(
    value: Any,
    validator: Callable[[Any], bool],
    recovery_func: Callable[[], Any],
    operation: str,
) -> Any:
    """Validate a value and recover if invalid."""
    try:
        if validator(value):
            return value
        else:
            logger.warning(
                f"Validation failed for {operation}, attempting recovery",
                extra={"operation": operation, "invalid_value": str(value)},
            )
            return recovery_func()
    except (ValueError, KeyError, AttributeError) as e:
        logger.error(
            f"Recovery failed for {operation}",
            extra={
                "operation": operation,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


class GracefulDegradation:
    """Manager for graceful degradation of features."""

    def __init__(self):
        self.disabled_features: set[str] = set()
        self.feature_scores: Dict[str, float] = {}

    def disable_feature(self, feature: str, reason: str) -> None:
        """Disable a feature."""
        self.disabled_features.add(feature)
        logger.warning(f"Feature {feature} disabled", extra={"feature": feature, "reason": reason})

    def is_enabled(self, feature: str) -> bool:
        """Check if feature is enabled."""
        return feature not in self.disabled_features

    def update_feature_score(self, feature: str, score: float) -> None:
        """Update feature reliability score."""
        self.feature_scores[feature] = score

        # Auto-disable if score too low
        if score < 0.3:
            self.disable_feature(feature, f"Low reliability score: {score:.2f}")

    def get_enabled_features(self, features: List[str]) -> List[str]:
        """Filter list to only enabled features."""
        return [f for f in features if self.is_enabled(f)]


# Global degradation manager
degradation_manager = GracefulDegradation()
