"""
from __future__ import annotations

Advanced rate limiting with circuit breaker pattern.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional

from unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket.

        Returns:
            True if tokens were available, False otherwise
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate seconds until tokens will be available."""
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        needed = tokens - self.tokens
        return needed / self.refill_rate

    def _refill(self) -> None:
        """Refill bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3

    failures: int = 0
    last_failure_time: Optional[float] = None
    half_open_calls: int = 0
    state: str = "closed"  # closed, open, half_open


class RateLimiter:
    """Advanced rate limiter with multiple strategies."""

    def __init__(
        self,
        requests_per_second: float = 10,
        burst_capacity: int = 20,
        enable_circuit_breaker: bool = True,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_second: Sustained request rate
            burst_capacity: Maximum burst size
            enable_circuit_breaker: Enable circuit breaker pattern
        """
        self.bucket = RateLimitBucket(capacity=burst_capacity, refill_rate=requests_per_second)

        self.circuit_breaker = CircuitBreakerState() if enable_circuit_breaker else None

        # Sliding window for request tracking
        self.request_times: Deque[float] = deque(maxlen=1000)

        # Per-endpoint rate limits
        self.endpoint_buckets: Dict[str, RateLimitBucket] = {}

    async def acquire(self, endpoint: Optional[str] = None, priority: int = 1) -> None:
        """Acquire permission to make a request.

        Args:
            endpoint: Optional endpoint-specific rate limiting
            priority: Request priority (higher = more important)

        Raises:
            RateLimitError: If rate limit exceeded and circuit open
        """
        # Check circuit breaker
        if self.circuit_breaker:
            self._check_circuit_breaker()

        # Check endpoint-specific limit
        if endpoint and endpoint in self.endpoint_buckets:
            bucket = self.endpoint_buckets[endpoint]
        else:
            bucket = self.bucket

        # Try to consume tokens
        while not bucket.consume(1):
            wait_time = bucket.time_until_available(1)

            # Apply priority-based wait reduction
            if priority > 1:
                wait_time = wait_time / priority

            logger.debug(
                "acquire",
                action="rate_limited",
                wait_time=wait_time,
                endpoint=endpoint,
                tokens_available=bucket.tokens,
            )

            await asyncio.sleep(wait_time)

        # Track request
        self.request_times.append(time.time())

        # Update circuit breaker for half-open state
        if self.circuit_breaker and self.circuit_breaker.state == "half_open":
            self.circuit_breaker.half_open_calls += 1

    def report_success(self) -> None:
        """Report successful request to update circuit breaker."""
        if not self.circuit_breaker:
            return

        if self.circuit_breaker.state == "half_open":
            # Successful call in half-open state
            self.circuit_breaker.failures = 0
            self.circuit_breaker.state = "closed"
            logger.info("report_success", action="circuit_closed")

    def report_failure(self, is_rate_limit: bool = False) -> None:
        """Report failed request to update circuit breaker.

        Args:
            is_rate_limit: Whether failure was due to rate limiting
        """
        if not self.circuit_breaker:
            return

        self.circuit_breaker.failures += 1
        self.circuit_breaker.last_failure_time = time.time()

        if self.circuit_breaker.failures >= self.circuit_breaker.failure_threshold:
            self.circuit_breaker.state = "open"
            logger.warning(
                "report_failure", action="circuit_opened", failures=self.circuit_breaker.failures
            )

    def _check_circuit_breaker(self) -> None:
        """Check and update circuit breaker state."""
        if not self.circuit_breaker:
            return

        cb = self.circuit_breaker

        if cb.state == "open":
            # Check if recovery timeout has passed
            if cb.last_failure_time:
                elapsed = time.time() - cb.last_failure_time
                if elapsed >= cb.recovery_timeout:
                    cb.state = "half_open"
                    cb.half_open_calls = 0
                    logger.info("_check_circuit_breaker", action="circuit_half_open")
                else:
                    from unity_wheel.auth.exceptions import RateLimitError

                    raise RateLimitError(
                        retry_after=int(cb.recovery_timeout - elapsed),
                        message="Circuit breaker is open",
                    )

        elif cb.state == "half_open":
            # Check if we've made enough successful calls
            if cb.half_open_calls >= cb.half_open_max_calls:
                cb.state = "closed"
                cb.failures = 0
                logger.info("_check_circuit_breaker", action="circuit_closed")

    def add_endpoint_limit(
        self, endpoint: str, requests_per_second: float, burst_capacity: int
    ) -> None:
        """Add endpoint-specific rate limit.

        Args:
            endpoint: API endpoint pattern
            requests_per_second: Sustained rate for endpoint
            burst_capacity: Burst capacity for endpoint
        """
        self.endpoint_buckets[endpoint] = RateLimitBucket(
            capacity=burst_capacity, refill_rate=requests_per_second
        )

        logger.info(
            "add_endpoint_limit", endpoint=endpoint, rps=requests_per_second, burst=burst_capacity
        )

    def get_current_rate(self, window_seconds: int = 60) -> float:
        """Get current request rate.

        Args:
            window_seconds: Time window to calculate rate over

        Returns:
            Requests per second in window
        """
        if not self.request_times:
            return 0.0

        now = time.time()
        cutoff = now - window_seconds

        # Count requests in window
        recent_requests = sum(1 for t in self.request_times if t > cutoff)

        return recent_requests / window_seconds

    def get_status(self) -> Dict[str, any]:
        """Get rate limiter status."""
        status = {
            "tokens_available": self.bucket.tokens,
            "capacity": self.bucket.capacity,
            "refill_rate": self.bucket.refill_rate,
            "current_rate": self.get_current_rate(),
            "circuit_breaker": None,
        }

        if self.circuit_breaker:
            status["circuit_breaker"] = {
                "state": self.circuit_breaker.state,
                "failures": self.circuit_breaker.failures,
                "threshold": self.circuit_breaker.failure_threshold,
            }

        return status