"""
Enhanced Exception Classes with Structured Information

Provides comprehensive exception hierarchy with debugging context,
recovery strategies, and structured error information.
"""

import asyncio
import builtins
import functools
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar
from uuid import uuid4


class ErrorSeverity(Enum):
    """Error severity levels for prioritization and handling."""

    CRITICAL = "critical"  # System-breaking, requires immediate attention
    HIGH = "high"  # Major functionality impacted
    MEDIUM = "medium"  # Minor functionality impacted
    LOW = "low"  # Informational or recoverable
    DEBUG = "debug"  # Development/debugging information


class ErrorCategory(Enum):
    """Error categories for classification and routing."""

    VALIDATION = "validation"  # Data validation errors
    AUTHENTICATION = "authentication"  # Auth/authorization errors
    AUTHORIZATION = "authorization"  # Permission errors
    DATABASE = "database"  # Database-related errors
    EXTERNAL_SERVICE = "external_service"  # Third-party service errors
    NETWORK = "network"  # Network connectivity errors
    TIMEOUT = "timeout"  # Operation timeout errors
    RESOURCE = "resource"  # Resource exhaustion errors
    CONFIGURATION = "configuration"  # Configuration errors
    BUSINESS_LOGIC = "business_logic"  # Business rule violations
    SYSTEM = "system"  # System-level errors
    ASYNC_OPERATION = "async_operation"  # Async execution errors


class UnityWheelError(Exception):
    """
    Base exception class for Unity Wheel system with enhanced debugging information.

    Provides structured error information, context tracking, and recovery hints.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
        recovery_hint: str | None = None,
        component: str | None = None,
        operation: str | None = None,
        user_message: str | None = None,
        sensitive_data: bool = False,
    ):
        super().__init__(message)

        # Core error information
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.category = category
        self.severity = severity
        self.user_message = user_message or message

        # Context and debugging
        self.context = context or {}
        self.component = component
        self.operation = operation
        self.recovery_hint = recovery_hint
        self.sensitive_data = sensitive_data

        # Timestamp and tracking
        self.timestamp = datetime.now(UTC)
        self.trace_id = str(uuid4())[:8]

        # Exception chaining
        self.cause = cause
        if cause:
            self.__cause__ = cause

        # Capture stack trace information
        self.stack_trace = (
            traceback.format_exc()
            if traceback.format_exc() != "NoneType: None\n"
            else None
        )

        # Performance metrics
        self.created_at = time.perf_counter()

    def _generate_error_code(self) -> str:
        """Generate a unique error code."""
        class_name = self.__class__.__name__
        timestamp = int(time.time() * 1000) % 1000000  # Last 6 digits of timestamp
        return f"{class_name}_{timestamp}"

    def to_dict(self) -> dict[str, Any]:
        """Convert error to structured dictionary for logging."""
        result = {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "component": self.component,
            "operation": self.operation,
        }

        # Add context if not sensitive
        if self.context and not self.sensitive_data:
            result["context"] = self.context
        elif self.sensitive_data:
            result["context_keys"] = list(self.context.keys()) if self.context else []

        # Add recovery information
        if self.recovery_hint:
            result["recovery_hint"] = self.recovery_hint

        # Add cause information
        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }

        # Add stack trace in debug mode
        if self.stack_trace and self.severity in [
            ErrorSeverity.CRITICAL,
            ErrorSeverity.HIGH,
        ]:
            result["stack_trace"] = self.stack_trace

        return result

    def is_retryable(self) -> bool:
        """Determine if this error is retryable."""
        retryable_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.RESOURCE,
        }
        return self.category in retryable_categories

    def get_user_message(self) -> str:
        """Get user-friendly error message."""
        return self.user_message

    def __str__(self) -> str:
        """String representation with debugging information."""
        parts = [f"[{self.error_code}] {self.message}"]

        if self.component:
            parts.append(f"Component: {self.component}")

        if self.operation:
            parts.append(f"Operation: {self.operation}")

        if self.context and not self.sensitive_data:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        return " | ".join(parts)


class ValidationError(UnityWheelError):
    """Data validation errors with field-specific information."""

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: Any = None,
        constraint: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if field:
            context["field"] = field
        if constraint:
            context["constraint"] = constraint
        if value is not None and not kwargs.get("sensitive_data", False):
            context["value"] = str(value)

        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs,
        )

        self.field = field
        self.value = value
        self.constraint = constraint


class DatabaseError(UnityWheelError):
    """Database operation errors with query information."""

    def __init__(
        self,
        message: str,
        *,
        query: str | None = None,
        table: str | None = None,
        operation: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if table:
            context["table"] = table
        if query and not kwargs.get("sensitive_data", False):
            context["query"] = query[:200] + "..." if len(query) > 200 else query

        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            operation=operation or "database_operation",
            context=context,
            **kwargs,
        )

        self.query = query
        self.table = table


class ExternalServiceError(UnityWheelError):
    """External service errors with service information."""

    def __init__(
        self,
        message: str,
        *,
        service_name: str | None = None,
        endpoint: str | None = None,
        status_code: int | None = None,
        response_time_ms: float | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if service_name:
            context["service_name"] = service_name
        if endpoint:
            context["endpoint"] = endpoint
        if status_code:
            context["status_code"] = status_code
        if response_time_ms:
            context["response_time_ms"] = response_time_ms

        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )

        self.service_name = service_name
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_time_ms = response_time_ms


class TimeoutError(UnityWheelError):
    """Timeout errors with duration information."""

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float | None = None,
        elapsed_seconds: float | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
        if elapsed_seconds:
            context["elapsed_seconds"] = elapsed_seconds

        recovery_hint = "Consider increasing timeout or optimizing the operation"
        if (
            timeout_seconds
            and elapsed_seconds
            and elapsed_seconds > timeout_seconds * 2
        ):
            recovery_hint = "Operation is significantly slower than expected - investigate performance"

        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_hint=recovery_hint,
            **kwargs,
        )

        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class AsyncOperationError(UnityWheelError):
    """Async operation errors with task information."""

    def __init__(
        self,
        message: str,
        *,
        task_name: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if task_name:
            context["task_name"] = task_name
        if task_id:
            context["task_id"] = task_id

        super().__init__(
            message,
            category=ErrorCategory.ASYNC_OPERATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs,
        )

        self.task_name = task_name
        self.task_id = task_id


class ResourceError(UnityWheelError):
    """Resource exhaustion errors."""

    def __init__(
        self,
        message: str,
        *,
        resource_type: str | None = None,
        current_usage: float | None = None,
        limit: float | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if resource_type:
            context["resource_type"] = resource_type
        if current_usage is not None:
            context["current_usage"] = current_usage
        if limit is not None:
            context["limit"] = limit

        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_hint="Free up resources or increase limits",
            **kwargs,
        )

        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class ConfigurationError(UnityWheelError):
    """Configuration errors."""

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        config_file: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        if config_file:
            context["config_file"] = config_file

        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_hint="Check configuration file and restart service",
            **kwargs,
        )

        self.config_key = config_key
        self.config_file = config_file


class CircuitBreakerError(UnityWheelError):
    """Circuit breaker errors."""

    def __init__(
        self,
        message: str,
        *,
        service_name: str,
        failure_count: int,
        threshold: int,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update(
            {
                "service_name": service_name,
                "failure_count": failure_count,
                "threshold": threshold,
            }
        )

        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_hint=f"Circuit breaker open for {service_name}. Wait for recovery or check service health",
            **kwargs,
        )

        self.service_name = service_name
        self.failure_count = failure_count
        self.threshold = threshold


# Type variable for decorator return types
F = TypeVar("F", bound=Callable[..., Any])


def error_handler(
    *,
    default_return: Any = None,
    log_errors: bool = True,
    component: str | None = None,
    operation: str | None = None,
    reraise: bool = True,
    catch_types: list[type] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for consistent error handling and logging.

    Args:
        default_return: Value to return on error (if not reraising)
        log_errors: Whether to log caught errors
        component: Component name for context
        operation: Operation name for context
        reraise: Whether to reraise exceptions after handling
        catch_types: Specific exception types to catch (None = all)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we should catch this exception type
                if catch_types and not isinstance(e, tuple(catch_types)):
                    raise

                # Convert to UnityWheelError if not already
                if not isinstance(e, UnityWheelError):
                    error = UnityWheelError(
                        f"Error in {func.__name__}: {str(e)}",
                        cause=e,
                        component=component or func.__module__,
                        operation=operation or func.__name__,
                        context={
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                        },
                    )
                else:
                    error = e
                    if not error.component:
                        error.component = component or func.__module__
                    if not error.operation:
                        error.operation = operation or func.__name__

                # Log error if requested
                if log_errors:
                    from .logging_enhanced import get_enhanced_logger

                    logger = get_enhanced_logger(func.__module__)
                    logger.error(f"Error in {func.__name__}", extra=error.to_dict())

                if reraise:
                    raise error
                return default_return

        return wrapper

    return decorator


def async_error_handler(
    *,
    default_return: Any = None,
    log_errors: bool = True,
    component: str | None = None,
    operation: str | None = None,
    reraise: bool = True,
    catch_types: list[type] | None = None,
    timeout_seconds: float | None = None,
) -> Callable[[F], F]:
    """
    Decorator for async function error handling with timeout support.

    Args:
        default_return: Value to return on error (if not reraising)
        log_errors: Whether to log caught errors
        component: Component name for context
        operation: Operation name for context
        reraise: Whether to reraise exceptions after handling
        catch_types: Specific exception types to catch (None = all)
        timeout_seconds: Optional timeout for the operation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            try:
                # Apply timeout if specified
                if timeout_seconds:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout_seconds
                    )
                else:
                    result = await func(*args, **kwargs)
                return result

            except builtins.TimeoutError as e:
                elapsed = time.perf_counter() - start_time
                error = TimeoutError(
                    f"Operation {func.__name__} timed out after {elapsed:.2f}s",
                    timeout_seconds=timeout_seconds,
                    elapsed_seconds=elapsed,
                    component=component or func.__module__,
                    operation=operation or func.__name__,
                    cause=e,
                )

                if log_errors:
                    from .logging_enhanced import get_enhanced_logger

                    logger = get_enhanced_logger(func.__module__)
                    logger.error(f"Timeout in {func.__name__}", extra=error.to_dict())

                if reraise:
                    raise error
                return default_return

            except Exception as e:
                # Check if we should catch this exception type
                if catch_types and not isinstance(e, tuple(catch_types)):
                    raise

                elapsed = time.perf_counter() - start_time

                # Convert to UnityWheelError if not already
                if not isinstance(e, UnityWheelError):
                    error = AsyncOperationError(
                        f"Error in async {func.__name__}: {str(e)}",
                        cause=e,
                        component=component or func.__module__,
                        operation=operation or func.__name__,
                        context={
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                            "execution_time_seconds": elapsed,
                        },
                    )
                else:
                    error = e
                    if not error.component:
                        error.component = component or func.__module__
                    if not error.operation:
                        error.operation = operation or func.__name__
                    # Add execution time to context
                    if error.context:
                        error.context["execution_time_seconds"] = elapsed

                # Log error if requested
                if log_errors:
                    from .logging_enhanced import get_enhanced_logger

                    logger = get_enhanced_logger(func.__module__)
                    logger.error(
                        f"Error in async {func.__name__}", extra=error.to_dict()
                    )

                if reraise:
                    raise error
                return default_return

        return wrapper

    return decorator


def timeout_handler(timeout_seconds: float) -> Callable[[F], F]:
    """
    Decorator to add timeout handling to functions.

    Args:
        timeout_seconds: Timeout in seconds
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout_seconds
                    )
                except builtins.TimeoutError as e:
                    raise TimeoutError(
                        f"Function {func.__name__} timed out after {timeout_seconds}s",
                        timeout_seconds=timeout_seconds,
                        operation=func.__name__,
                        component=func.__module__,
                        cause=e,
                    )

            return async_wrapper
        else:
            # For sync functions, we can't easily add timeout without threading
            # Return the original function with a warning
            import warnings

            warnings.warn(
                f"Timeout decorator applied to sync function {func.__name__} - no effect"
            )
            return func

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    timeout_seconds: float = 60.0,
    expected_exception: type = Exception,
) -> Callable[[F], F]:
    """
    Circuit breaker pattern decorator.

    Args:
        failure_threshold: Number of failures before opening circuit
        timeout_seconds: Time to wait before trying again
        expected_exception: Exception type that counts as failure
    """

    def decorator(func: F) -> F:
        # Circuit state
        failure_count = 0
        last_failure_time = 0
        circuit_open = False

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, circuit_open

            # Check if circuit should be reset
            if circuit_open and time.time() - last_failure_time > timeout_seconds:
                circuit_open = False
                failure_count = 0

            # If circuit is open, raise immediately
            if circuit_open:
                raise CircuitBreakerError(
                    f"Circuit breaker open for {func.__name__}",
                    service_name=func.__name__,
                    failure_count=failure_count,
                    threshold=failure_threshold,
                )

            try:
                result = func(*args, **kwargs)
                # Success - reset failure count
                failure_count = 0
                return result
            except expected_exception:
                # Failure - increment counter
                failure_count += 1
                last_failure_time = time.time()

                if failure_count >= failure_threshold:
                    circuit_open = True

                # Re-raise the original exception
                raise

        return wrapper

    return decorator
