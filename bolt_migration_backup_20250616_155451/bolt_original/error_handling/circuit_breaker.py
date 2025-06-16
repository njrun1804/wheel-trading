"""
Circuit Breaker Pattern Implementation

Implements circuit breaker pattern to prevent cascading failures and provide
fail-fast behavior when services are unavailable.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)
from typing import Any


class CircuitState(Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures to open circuit
    success_threshold: int = 3  # Number of successes to close circuit in half-open
    timeout: float = 60.0  # Timeout before trying half-open (seconds)
    reset_timeout: float = 300.0  # How long to keep records (seconds)
    max_requests_half_open: int = 5  # Max requests in half-open state

    @classmethod
    def from_bolt_config(cls):
        """Create CircuitBreakerConfig from Bolt configuration."""
        try:
            from bolt.core.config import get_default_config

            bolt_config = get_default_config()
            cb_config = bolt_config.circuit_breaker
            return cls(
                failure_threshold=cb_config.failure_threshold,
                success_threshold=cb_config.success_threshold,
                timeout=cb_config.timeout_s,
                reset_timeout=cb_config.reset_timeout_s,
                max_requests_half_open=cb_config.max_requests_half_open,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load circuit breaker config, using defaults: {e}"
            )
            return cls()  # Use default values


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    state_changed_time: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, circuit_name: str, failure_count: int):
        self.circuit_name = circuit_name
        self.failure_count = failure_count
        super().__init__(
            f"Circuit breaker '{circuit_name}' is open "
            f"({failure_count} consecutive failures)"
        )


class CircuitBreaker:
    """Circuit breaker implementation with async support."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker.{name}")
        self._lock = threading.RLock()

        # Half-open state tracking
        self._half_open_requests = 0

        # Callbacks
        self._on_state_change: list[
            Callable[[str, CircuitState, CircuitState], None]
        ] = []
        self._on_failure: list[Callable[[str, Exception], None]] = []
        self._on_success: list[Callable[[str, Any], None]] = []

    def add_state_change_callback(
        self, callback: Callable[[str, CircuitState, CircuitState], None]
    ):
        """Add callback for state changes."""
        self._on_state_change.append(callback)

    def add_failure_callback(self, callback: Callable[[str, Exception], None]):
        """Add callback for failures."""
        self._on_failure.append(callback)

    def add_success_callback(self, callback: Callable[[str, Any], None]):
        """Add callback for successes."""
        self._on_success.append(callback)

    @contextmanager
    def call(self):
        """Context manager for circuit breaker protection."""
        if self._should_reject():
            raise CircuitBreakerOpenException(
                self.name, self.stats.consecutive_failures
            )

        start_time = time.time()
        try:
            yield
            self._on_success_internal(time.time() - start_time)
        except Exception as e:
            self._on_failure_internal(e, time.time() - start_time)
            raise

    async def call_async(self, coro_func: Callable[..., Any], *args, **kwargs):
        """Async wrapper for circuit breaker protection."""
        if self._should_reject():
            raise CircuitBreakerOpenException(
                self.name, self.stats.consecutive_failures
            )

        start_time = time.time()
        try:
            result = await coro_func(*args, **kwargs)
            self._on_success_internal(time.time() - start_time)
            return result
        except Exception as e:
            self._on_failure_internal(e, time.time() - start_time)
            raise

    def call_sync(self, func: Callable[..., Any], *args, **kwargs):
        """Synchronous wrapper for circuit breaker protection."""
        if self._should_reject():
            raise CircuitBreakerOpenException(
                self.name, self.stats.consecutive_failures
            )

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            self._on_success_internal(time.time() - start_time)
            return result
        except Exception as e:
            self._on_failure_internal(e, time.time() - start_time)
            raise

    def _should_reject(self) -> bool:
        """Check if request should be rejected."""
        with self._lock:
            current_time = time.time()

            # Clean up old failure records
            self._cleanup_old_records(current_time)

            if self.stats.state == CircuitState.CLOSED:
                return False

            elif self.stats.state == CircuitState.OPEN:
                # Check if timeout has passed to try half-open
                if (
                    self.stats.last_failure_time
                    and current_time - self.stats.last_failure_time
                    >= self.config.timeout
                ):
                    self._change_state(CircuitState.HALF_OPEN)
                    self._half_open_requests = 0
                    self.logger.info(
                        f"Circuit breaker '{self.name}' trying half-open state"
                    )
                    return False
                return True

            elif self.stats.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self._half_open_requests >= self.config.max_requests_half_open:
                    return True
                self._half_open_requests += 1
                return False

            return False

    def _on_success_internal(self, duration: float):
        """Handle successful operation."""
        with self._lock:
            current_time = time.time()

            self.stats.total_requests += 1
            self.stats.success_count += 1
            self.stats.last_success_time = current_time
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0

            if self.stats.state == CircuitState.HALF_OPEN:
                # Check if we should close the circuit
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    self._change_state(CircuitState.CLOSED)

            elif self.stats.state == CircuitState.OPEN:
                # This shouldn't happen, but reset if it does
                self._change_state(CircuitState.CLOSED)

            # Notify callbacks
            for callback in self._on_success:
                try:
                    callback(self.name, duration)
                except Exception as e:
                    self.logger.warning(f"Success callback failed: {e}")

    def _on_failure_internal(self, exception: Exception, duration: float):
        """Handle failed operation."""
        with self._lock:
            current_time = time.time()

            self.stats.total_requests += 1
            self.stats.failure_count += 1
            self.stats.last_failure_time = current_time
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0

            # Check if we should open the circuit
            if (
                self.stats.state == CircuitState.CLOSED
                and self.stats.consecutive_failures >= self.config.failure_threshold
            ):
                self._change_state(CircuitState.OPEN)

            elif (
                self.stats.state == CircuitState.HALF_OPEN
                and self.stats.consecutive_failures >= 1
            ):
                # Go back to open on any failure in half-open
                self._change_state(CircuitState.OPEN)

            # Notify callbacks
            for callback in self._on_failure:
                try:
                    callback(self.name, exception)
                except Exception as e:
                    self.logger.warning(f"Failure callback failed: {e}")

    def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state."""
        old_state = self.stats.state
        if old_state != new_state:
            self.stats.state = new_state
            self.stats.state_changed_time = time.time()

            self.logger.info(
                f"Circuit breaker '{self.name}' state changed: "
                f"{old_state.value} -> {new_state.value}"
            )

            # Reset counters on state change
            if new_state == CircuitState.CLOSED:
                self.stats.consecutive_failures = 0
                self.stats.consecutive_successes = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._half_open_requests = 0

            # Notify callbacks
            for callback in self._on_state_change:
                try:
                    callback(self.name, old_state, new_state)
                except Exception as e:
                    self.logger.warning(f"State change callback failed: {e}")

    def force_open(self):
        """Manually force circuit breaker open."""
        with self._lock:
            self._change_state(CircuitState.OPEN)
            self.logger.warning(f"Circuit breaker '{self.name}' manually forced open")

    def force_close(self):
        """Manually force circuit breaker closed."""
        with self._lock:
            self._change_state(CircuitState.CLOSED)
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes = 0
            self.logger.info(f"Circuit breaker '{self.name}' manually forced closed")

    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            old_stats = self.stats
            self.stats = CircuitBreakerStats()
            self._half_open_requests = 0

            self.logger.info(f"Circuit breaker '{self.name}' reset")

            # Notify state change if needed
            if old_stats.state != CircuitState.CLOSED:
                for callback in self._on_state_change:
                    try:
                        callback(self.name, old_stats.state, CircuitState.CLOSED)
                    except Exception as e:
                        self.logger.warning(f"State change callback failed: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            current_time = time.time()
            return {
                "name": self.name,
                "state": self.stats.state.value,
                "failure_count": self.stats.failure_count,
                "success_count": self.stats.success_count,
                "total_requests": self.stats.total_requests,
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "last_failure_time": self.stats.last_failure_time,
                "last_success_time": self.stats.last_success_time,
                "state_changed_time": self.stats.state_changed_time,
                "failure_rate": (
                    self.stats.failure_count / self.stats.total_requests
                    if self.stats.total_requests > 0
                    else 0.0
                ),
                "success_rate": (
                    self.stats.success_count / self.stats.total_requests
                    if self.stats.total_requests > 0
                    else 0.0
                ),
                "time_in_current_state": current_time - self.stats.state_changed_time,
                "time_to_next_attempt": max(
                    0,
                    (self.stats.last_failure_time + self.config.timeout - current_time)
                    if self.stats.state == CircuitState.OPEN
                    and self.stats.last_failure_time
                    else 0,
                ),
                "half_open_requests": self._half_open_requests
                if self.stats.state == CircuitState.HALF_OPEN
                else 0,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout,
                    "reset_timeout": self.config.reset_timeout,
                    "max_requests_half_open": self.config.max_requests_half_open,
                },
            }

    def is_available(self) -> bool:
        """Check if circuit breaker is available for requests."""
        return not self._should_reject()

    def _cleanup_old_records(self, current_time: float):
        """Clean up old failure/success records beyond reset timeout."""
        if (
            self.stats.last_failure_time
            and (current_time - self.stats.last_failure_time)
            > self.config.reset_timeout
        ):
            # Reset failure counts if enough time has passed
            if self.stats.state == CircuitState.OPEN:
                self.logger.info(
                    f"Circuit breaker '{self.name}' auto-reset due to timeout"
                )
                self._change_state(CircuitState.CLOSED)

    def get_health_status(self) -> dict[str, Any]:
        """Get detailed health status for monitoring."""
        with self._lock:
            current_time = time.time()

            # Calculate recent failure rate (last hour)
            recent_failures = 0
            recent_total = 0
            cutoff_time = current_time - 3600  # 1 hour

            if (
                self.stats.last_failure_time
                and self.stats.last_failure_time > cutoff_time
            ):
                recent_failures = min(
                    self.stats.consecutive_failures, 10
                )  # Cap for calculation
            if (
                self.stats.last_success_time
                and self.stats.last_success_time > cutoff_time
            ):
                recent_total = recent_failures + min(
                    self.stats.consecutive_successes, 10
                )

            health_score = 100.0
            if self.stats.state == CircuitState.OPEN:
                health_score = 0.0
            elif self.stats.state == CircuitState.HALF_OPEN:
                health_score = 50.0
            elif self.stats.failure_count > 0:
                failure_rate = self.stats.failure_count / max(
                    self.stats.total_requests, 1
                )
                health_score = max(0, 100 - (failure_rate * 100))

            return {
                "name": self.name,
                "healthy": self.stats.state == CircuitState.CLOSED,
                "health_score": health_score,
                "state": self.stats.state.value,
                "available": self.is_available(),
                "recent_failure_rate": recent_failures / max(recent_total, 1)
                if recent_total > 0
                else 0,
                "consecutive_failures": self.stats.consecutive_failures,
                "time_since_last_failure": current_time - self.stats.last_failure_time
                if self.stats.last_failure_time
                else None,
                "recommended_action": self._get_recommended_action(),
            }

    def _get_recommended_action(self) -> str:
        """Get recommended action based on current state."""
        if self.stats.state == CircuitState.OPEN:
            return f"Circuit is open. Wait {self.config.timeout}s or manually reset."
        elif self.stats.state == CircuitState.HALF_OPEN:
            return "Circuit is testing recovery. Monitor success rate."
        elif self.stats.consecutive_failures > self.config.failure_threshold // 2:
            return "Warning: Approaching failure threshold. Monitor closely."
        else:
            return "Circuit is healthy and operating normally."


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""

    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()
        self.logger = logging.getLogger(f"{__name__}.CircuitBreakerManager")

    def get_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            cb_config = config or self.default_config
            circuit_breaker = CircuitBreaker(name, cb_config)

            # Add default callbacks
            circuit_breaker.add_state_change_callback(self._on_state_change)
            circuit_breaker.add_failure_callback(self._on_failure)

            self.circuit_breakers[name] = circuit_breaker
            self.logger.info(f"Created circuit breaker: {name}")

        return self.circuit_breakers[name]

    def _on_state_change(
        self, name: str, old_state: CircuitState, new_state: CircuitState
    ):
        """Handle circuit breaker state changes."""
        self.logger.info(
            f"Circuit breaker {name}: {old_state.value} -> {new_state.value}"
        )

    def _on_failure(self, name: str, exception: Exception):
        """Handle circuit breaker failures."""
        self.logger.warning(f"Circuit breaker {name} failure: {exception}")

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}

    def reset_all(self):
        """Reset all circuit breakers."""
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.reset()
        self.logger.info("Reset all circuit breakers")

    def force_close_all(self):
        """Force close all circuit breakers."""
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.force_close()
        self.logger.info("Forced close all circuit breakers")

    def get_open_circuits(self) -> list[str]:
        """Get list of open circuit breakers."""
        return [
            name
            for name, cb in self.circuit_breakers.items()
            if cb.stats.state == CircuitState.OPEN
        ]

    def get_available_circuits(self) -> list[str]:
        """Get list of available circuit breakers."""
        return [name for name, cb in self.circuit_breakers.items() if cb.is_available()]


# Global circuit breaker manager
_circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker(
    name: str, config: CircuitBreakerConfig | None = None
) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    return _circuit_breaker_manager.get_circuit_breaker(name, config)


def circuit_breaker(
    name: str | None = None, config: CircuitBreakerConfig | None = None
):
    """Decorator for circuit breaker protection."""

    def decorator(func):
        cb_name = name or f"{func.__module__}.{func.__name__}"
        circuit_breaker_instance = get_circuit_breaker(cb_name, config)

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                return await circuit_breaker_instance.call_async(func, *args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                return circuit_breaker_instance.call_sync(func, *args, **kwargs)

            return sync_wrapper

    return decorator


# Utility functions for common circuit breaker patterns


def create_resource_circuit_breaker(
    resource_name: str, failure_threshold: int = 3, timeout: float = 30.0
) -> CircuitBreaker:
    """Create a circuit breaker for resource access."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
        success_threshold=2,
        max_requests_half_open=3,
    )
    return get_circuit_breaker(f"resource_{resource_name}", config)


def create_service_circuit_breaker(
    service_name: str, failure_threshold: int = 5, timeout: float = 60.0
) -> CircuitBreaker:
    """Create a circuit breaker for external service calls."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
        success_threshold=3,
        max_requests_half_open=5,
    )
    return get_circuit_breaker(f"service_{service_name}", config)


def create_agent_circuit_breaker(
    agent_id: str, failure_threshold: int = 3, timeout: float = 45.0
) -> CircuitBreaker:
    """Create a circuit breaker for agent operations."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
        success_threshold=2,
        max_requests_half_open=3,
    )
    return get_circuit_breaker(f"agent_{agent_id}", config)
