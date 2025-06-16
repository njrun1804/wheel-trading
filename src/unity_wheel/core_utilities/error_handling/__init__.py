"""
Comprehensive Error Handling and Logging Framework

Provides structured error handling, propagation, and logging patterns
across both Einstein and Bolt systems with debugging visibility.
"""

from .exceptions import (
    AsyncOperationError,
    CircuitBreakerError,
    ConfigurationError,
    DatabaseError,
    ErrorCategory,
    ErrorSeverity,
    ExternalServiceError,
    ResourceError,
    TimeoutError,
    UnityWheelError,
    ValidationError,
    async_error_handler,
    circuit_breaker,
    error_handler,
    timeout_handler,
)
from .logging_enhanced import (
    AsyncLogContext,
    ErrorLogger,
    LogContext,
    get_enhanced_logger,
    log_execution_time,
    log_with_context,
    performance_log,
    structured_error_log,
)
from .monitoring import (
    ErrorMetrics,
    ErrorMonitor,
    HealthChecker,
    SystemStatus,
    alert_on_error,
    async_health_check_decorator,
    track_error_patterns,
)
from .recovery import (
    BackoffStrategy,
    RecoveryManager,
    async_with_retry,
    exponential_backoff,
    get_recovery_manager,
    linear_backoff,
    with_retry,
)

__all__ = [
    # Core exceptions
    "UnityWheelError",
    "AsyncOperationError",
    "DatabaseError",
    "ExternalServiceError",
    "ValidationError",
    "ConfigurationError",
    "ResourceError",
    "TimeoutError",
    "CircuitBreakerError",
    "ErrorSeverity",
    "ErrorCategory",
    # Decorators
    "error_handler",
    "async_error_handler",
    "timeout_handler",
    "circuit_breaker",
    # Enhanced logging
    "get_enhanced_logger",
    "LogContext",
    "AsyncLogContext",
    "log_execution_time",
    "log_with_context",
    "structured_error_log",
    "performance_log",
    "ErrorLogger",
    # Recovery
    "RecoveryManager",
    "BackoffStrategy",
    "exponential_backoff",
    "linear_backoff",
    "with_retry",
    "async_with_retry",
    # Monitoring
    "ErrorMonitor",
    "ErrorMetrics",
    "HealthChecker",
    "SystemStatus",
    "alert_on_error",
    "track_error_patterns",
    "async_health_check_decorator",
]
