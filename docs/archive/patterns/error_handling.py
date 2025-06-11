"""Standard error handling patterns for Codex to follow.

This module provides canonical examples of error handling patterns
that should be used throughout the codebase. Codex should reference
these patterns when generating new code.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Tuple, Union

import numpy as np

from ..models.greeks import CalculationResult
from ..utils.decorators import RecoveryStrategy, cached, timed_operation, with_recovery
from ..utils.logging import get_logger

logger = get_logger(__name__)


# PATTERN 1: Validated calculations with confidence scores
@with_recovery(strategy=RecoveryStrategy.FALLBACK)
@timed_operation(threshold_ms=10.0)
def calculate_with_validation(
    input_value: float,
    parameters: dict,
) -> CalculationResult:
    """
    Standard pattern for calculations that return confidence scores.

    CODEX PATTERN:
    1. Validate inputs first
    2. Perform calculation with error handling
    3. Score confidence based on conditions
    4. Return structured result with warnings
    """
    warnings = []

    # Step 1: Input validation
    if input_value <= 0:
        logger.warning("Invalid input value", extra={"value": input_value})
        return CalculationResult(np.nan, 0.0, ["Input must be positive"])

    if not parameters:
        return CalculationResult(np.nan, 0.0, ["Parameters required"])

    try:
        # Step 2: Perform calculation
        result = input_value * parameters.get("multiplier", 1.0)

        # Step 3: Confidence scoring
        confidence = 0.99  # Start with high confidence

        if input_value > 1000:
            confidence *= 0.9
            warnings.append("Large input value may reduce accuracy")

        if abs(result) < 1e-10:
            confidence *= 0.8
            warnings.append("Result near zero")

        # Step 4: Return structured result
        logger.debug(
            "Calculation completed",
            extra={
                "function": "calculate_with_validation",
                "input": input_value,
                "result": result,
                "confidence": confidence,
            },
        )

        return CalculationResult(result, confidence, warnings)

    except (ValueError, TypeError, KeyError) as e:
        # Specific exception handling with context
        logger.error(
            f"Calculation failed: {e}",
            extra={
                "function": "calculate_with_validation",
                "input": input_value,
                "parameters": parameters,
                "error_type": type(e).__name__,
            },
        )
        return CalculationResult(np.nan, 0.0, [f"Error: {str(e)}"])


# PATTERN 2: External API calls with retry and fallback
@with_recovery(strategy=RecoveryStrategy.RETRY, max_attempts=3)
async def fetch_external_data(
    endpoint: str,
    params: dict,
    timeout: float = 5.0,
) -> Tuple[Optional[dict], float]:
    """
    Pattern for external API calls with proper error handling.

    CODEX PATTERN:
    1. Use specific timeout values
    2. Catch network-specific exceptions
    3. Log with full context
    4. Return data with confidence
    """
    import asyncio

    import aiohttp

    confidence = 1.0

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                endpoint, params=params, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(
                        "API call successful",
                        extra={
                            "endpoint": endpoint,
                            "status": response.status,
                            "response_time": response.headers.get("X-Response-Time"),
                        },
                    )
                    return data, confidence

                elif response.status == 429:  # Rate limited
                    logger.warning(
                        "Rate limited",
                        extra={
                            "endpoint": endpoint,
                            "retry_after": response.headers.get("Retry-After"),
                        },
                    )
                    return None, 0.0

                else:
                    logger.error(
                        f"API error: {response.status}",
                        extra={"endpoint": endpoint, "response": await response.text()},
                    )
                    return None, 0.3  # Low confidence fallback

    except asyncio.TimeoutError:
        logger.error("API timeout", extra={"endpoint": endpoint, "timeout": timeout})
        return None, 0.0

    except aiohttp.ClientError as e:
        logger.error(
            f"Network error: {e}",
            extra={
                "endpoint": endpoint,
                "error_type": type(e).__name__,
                "details": str(e),
            },
        )
        return None, 0.0

    except Exception as e:
        # Only catch Exception for truly unexpected errors
        logger.critical(
            f"Unexpected error in API call: {e}",
            extra={"endpoint": endpoint, "error_type": type(e).__name__},
            exc_info=True,  # Include full traceback
        )
        return None, 0.0


# PATTERN 3: Data validation with die() for requirements
def validate_required_data(market_data: dict) -> dict:
    """
    Pattern for validating required data using die().

    CODEX PATTERN:
    1. Use die() for absolutely required fields
    2. Use validate_x() for optional fields with defaults
    3. Log validation failures with context
    """
    from ..utils.validate import die, validate_positive

    # Required fields - will raise if missing
    ticker = die(market_data.get("ticker"), "Ticker symbol required")
    price = die(market_data.get("price"), "Current price required")

    # Validate numeric constraints
    price = die(validate_positive(price), f"Price must be positive, got {price}")

    # Optional fields with defaults
    volume = market_data.get("volume", 0)
    if volume < 0:
        logger.warning("Negative volume, using 0", extra={"ticker": ticker, "volume": volume})
        volume = 0

    volatility = market_data.get("volatility")
    if volatility is None:
        logger.info(
            "No volatility provided, using default", extra={"ticker": ticker, "default": 0.20}
        )
        volatility = 0.20

    return {
        "ticker": ticker,
        "price": price,
        "volume": volume,
        "volatility": volatility,
    }


# PATTERN 4: Batch operations with partial failure handling
def process_batch_with_recovery(
    items: list,
    processor_func: callable,
) -> Tuple[list, list, float]:
    """
    Pattern for processing batches where some items may fail.

    CODEX PATTERN:
    1. Continue processing even if some items fail
    2. Collect both successes and failures
    3. Calculate overall confidence
    4. Log summary statistics
    """
    successes = []
    failures = []

    for i, item in enumerate(items):
        try:
            result = processor_func(item)
            successes.append(result)

        except (ValueError, TypeError, KeyError) as e:
            # Expected errors - log and continue
            logger.warning(
                f"Failed to process item {i}: {e}",
                extra={
                    "item_index": i,
                    "item_preview": str(item)[:100],
                    "error_type": type(e).__name__,
                },
            )
            failures.append({"item": item, "error": str(e)})

        except Exception as e:
            # Unexpected errors - log with full context but continue
            logger.error(
                f"Unexpected error processing item {i}: {e}",
                extra={
                    "item_index": i,
                    "item": item,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            failures.append({"item": item, "error": f"Unexpected: {str(e)}"})

    # Calculate confidence based on success rate
    total = len(items)
    success_rate = len(successes) / total if total > 0 else 0.0
    confidence = success_rate

    # Adjust confidence based on failure types
    if any("Unexpected" in f["error"] for f in failures):
        confidence *= 0.8  # Lower confidence for unexpected errors

    logger.info(
        "Batch processing complete",
        extra={
            "total": total,
            "successes": len(successes),
            "failures": len(failures),
            "confidence": confidence,
        },
    )

    return successes, failures, confidence


# PATTERN 5: Circuit breaker for repeated failures
class CircuitBreakerPattern:
    """
    Pattern for circuit breaker to prevent cascading failures.

    CODEX PATTERN:
    1. Track consecutive failures
    2. Open circuit after threshold
    3. Retry after cooldown period
    4. Log state transitions
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.consecutive_failures = 0
        self.last_failure_time = None
        self.is_open = False

    def call_with_breaker(self, func: callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        import time

        # Check if circuit should be reset
        if self.is_open and self.last_failure_time:
            if time.time() - self.last_failure_time > self.reset_timeout:
                logger.info(
                    "Circuit breaker reset",
                    extra={"failures": self.consecutive_failures, "timeout": self.reset_timeout},
                )
                self.is_open = False
                self.consecutive_failures = 0

        # If circuit is open, fail fast
        if self.is_open:
            logger.warning("Circuit breaker is open", extra={"failures": self.consecutive_failures})
            raise Exception("Circuit breaker is open")

        try:
            # Try the operation
            result = func(*args, **kwargs)

            # Success - reset failure count
            if self.consecutive_failures > 0:
                logger.info(
                    "Operation succeeded, resetting failure count",
                    extra={"previous_failures": self.consecutive_failures},
                )
            self.consecutive_failures = 0

            return result

        except Exception as e:
            # Increment failure count
            self.consecutive_failures += 1
            self.last_failure_time = time.time()

            # Check if we should open the circuit
            if self.consecutive_failures >= self.failure_threshold:
                self.is_open = True
                logger.error(
                    "Circuit breaker opened",
                    extra={
                        "failures": self.consecutive_failures,
                        "threshold": self.failure_threshold,
                        "error": str(e),
                    },
                )

            raise


# PATTERN 6: Graceful degradation with fallbacks
def calculate_with_fallback(
    primary_func: callable, fallback_func: callable, *args, **kwargs
) -> Tuple[any, float, str]:
    """
    Pattern for graceful degradation when primary method fails.

    CODEX PATTERN:
    1. Try primary method first
    2. Fall back to simpler method on failure
    3. Track which method was used
    4. Adjust confidence accordingly
    """
    method_used = "primary"

    try:
        result = primary_func(*args, **kwargs)
        confidence = 0.99

        logger.debug("Primary method succeeded", extra={"function": primary_func.__name__})

        return result, confidence, method_used

    except (ValueError, TypeError, ZeroDivisionError) as e:
        logger.warning(
            f"Primary method failed, using fallback: {e}",
            extra={
                "primary": primary_func.__name__,
                "fallback": fallback_func.__name__,
                "error": str(e),
            },
        )

        try:
            result = fallback_func(*args, **kwargs)
            confidence = 0.70  # Lower confidence for fallback
            method_used = "fallback"

            return result, confidence, method_used

        except Exception as fallback_error:
            logger.error(
                f"Both primary and fallback failed",
                extra={
                    "primary_error": str(e),
                    "fallback_error": str(fallback_error),
                },
            )
            # Return safe default
            return 0.0, 0.0, "failed"
