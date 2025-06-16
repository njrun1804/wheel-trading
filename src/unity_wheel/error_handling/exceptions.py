"""
Unified Exception Hierarchy

Consolidates exception types from bolt, einstein, jarvis2, and unity_wheel
into a single, coherent hierarchy with structured error context.
"""

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Unified error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Unified error categories."""

    SYSTEM = "system"
    COMPONENT = "component"  # bolt, einstein, jarvis2, unity_wheel
    RESOURCE = "resource"  # memory, GPU, disk, network
    CONFIGURATION = "configuration"
    EXTERNAL = "external"  # APIs, databases, services
    VALIDATION = "validation"
    PERFORMANCE = "performance"


class RecoveryStrategy(Enum):
    """Unified recovery strategies (consolidated from all components)."""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    DEGRADE = "degrade"
    FAILOVER = "failover"
    RESTART = "restart"
    SKIP = "skip"
    MANUAL_INTERVENTION = "manual_intervention"
    NONE = "none"


@dataclass
class ErrorContext:
    """Unified error context with component-specific metadata."""

    timestamp: float = field(default_factory=time.time)
    component: str | None = None  # bolt, einstein, jarvis2, unity_wheel
    operation: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    system_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Performance context
    performance_impact: str | None = None  # "low", "medium", "high"
    resource_usage: dict[str, float] | None = None

    # Component-specific context
    bolt_context: dict[str, Any] | None = None
    einstein_context: dict[str, Any] | None = None
    jarvis2_context: dict[str, Any] | None = None
    unity_wheel_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "timestamp": self.timestamp,
            "component": self.component,
            "operation": self.operation,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "system_state": self.system_state,
            "metadata": self.metadata,
            "performance_impact": self.performance_impact,
            "resource_usage": self.resource_usage,
            "component_contexts": {
                "bolt": self.bolt_context,
                "einstein": self.einstein_context,
                "jarvis2": self.jarvis2_context,
                "unity_wheel": self.unity_wheel_context,
            },
        }


class UnifiedError(Exception):
    """
    Base unified exception class for all components.

    Consolidates features from:
    - bolt.error_handling.exceptions.BoltException
    - jarvis2.core.error_handling.ErrorContext
    - src.patterns.error_handling patterns
    - unity_wheel.utils.recovery patterns
    """

    def __init__(
        self,
        message: str,
        *,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        error_code: str | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
        recovery_hints: list[str] | None = None,
        diagnostic_data: dict[str, Any] | None = None,
        confidence_impact: float = 0.0,  # Impact on confidence scores (0.0-1.0)
    ):
        super().__init__(message)

        self.message = message
        self.severity = severity
        self.category = category
        self.recovery_strategy = recovery_strategy
        self.error_code = error_code or self._generate_error_code()
        self.context = context or ErrorContext()
        self.cause = cause
        self.recovery_hints = recovery_hints or []
        self.diagnostic_data = diagnostic_data or {}
        self.confidence_impact = confidence_impact

        # Capture stack trace
        self.stack_trace = traceback.format_exc()

        # Set cause chain for Python exception handling
        if cause:
            self.__cause__ = cause

    def _generate_error_code(self) -> str:
        """Generate unique error code."""
        class_name = self.__class__.__name__
        return f"UNIFIED_{class_name.upper()}"

    def add_recovery_hint(self, hint: str) -> None:
        """Add a recovery hint."""
        self.recovery_hints.append(hint)

    def add_diagnostic_data(self, key: str, value: Any) -> None:
        """Add diagnostic data."""
        self.diagnostic_data[key] = value

    def set_context(self, **kwargs) -> None:
        """Update error context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.metadata[key] = value

    def set_component_context(self, component: str, **kwargs) -> None:
        """Set component-specific context."""
        context_attr = f"{component}_context"
        if hasattr(self.context, context_attr):
            current_context = getattr(self.context, context_attr) or {}
            current_context.update(kwargs)
            setattr(self.context, context_attr, current_context)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recovery_strategy": self.recovery_strategy.value,
            "context": self.context.to_dict(),
            "recovery_hints": self.recovery_hints,
            "diagnostic_data": self.diagnostic_data,
            "confidence_impact": self.confidence_impact,
            "cause": str(self.cause) if self.cause else None,
            "stack_trace": self.stack_trace,
        }

    def is_recoverable(self) -> bool:
        """Check if error is recoverable."""
        return self.recovery_strategy in [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.FALLBACK,
            RecoveryStrategy.DEGRADE,
            RecoveryStrategy.FAILOVER,
            RecoveryStrategy.RESTART,
        ]

    def is_critical(self) -> bool:
        """Check if error is critical."""
        return self.severity == ErrorSeverity.CRITICAL

    def should_retry(self) -> bool:
        """Check if error suggests retry."""
        return (
            self.recovery_strategy == RecoveryStrategy.RETRY
            and self.context.retry_count < self.context.max_retries
        )

    def affects_confidence(self) -> bool:
        """Check if error affects confidence scores."""
        return self.confidence_impact > 0.0

    def __str__(self) -> str:
        """String representation with error code and severity."""
        return f"[{self.error_code}:{self.severity.value}] {self.message}"


class ComponentError(UnifiedError):
    """Component-specific errors (bolt, einstein, jarvis2, unity_wheel)."""

    def __init__(self, message: str, component: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.COMPONENT)
        super().__init__(message, **kwargs)
        self.component = component
        self.set_context(component=component)


class BoltError(ComponentError):
    """Bolt component errors (compatibility with existing bolt code)."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="bolt", **kwargs)


class EinsteinError(ComponentError):
    """Einstein component errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="einstein", **kwargs)


class Jarvis2Error(ComponentError):
    """Jarvis2 component errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="jarvis2", **kwargs)


class UnityWheelError(ComponentError):
    """Unity Wheel component errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, component="unity_wheel", **kwargs)


class ResourceError(UnifiedError):
    """Resource exhaustion and management errors."""

    def __init__(
        self,
        message: str,
        resource_type: str,
        current_usage: float | None = None,
        limit: float | None = None,
        **kwargs,
    ):
        kwargs.setdefault("category", ErrorCategory.RESOURCE)
        kwargs.setdefault("recovery_strategy", RecoveryStrategy.DEGRADE)
        super().__init__(message, **kwargs)

        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit

        # Add resource-specific diagnostic data
        self.add_diagnostic_data("resource_type", resource_type)
        if current_usage is not None:
            self.add_diagnostic_data("current_usage", current_usage)
        if limit is not None:
            self.add_diagnostic_data("limit", limit)
            if current_usage is not None:
                usage_percent = (current_usage / limit) * 100
                self.add_diagnostic_data("usage_percent", usage_percent)


class MemoryError(ResourceError):
    """Memory-related errors (consolidates bolt and unity_wheel memory errors)."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, resource_type="memory", **kwargs)
        self.add_recovery_hint("Clear caches and temporary data")
        self.add_recovery_hint("Reduce batch sizes")
        self.add_recovery_hint("Enable garbage collection")


class GPUError(ResourceError):
    """GPU-related errors."""

    def __init__(self, message: str, gpu_backend: str = "unknown", **kwargs):
        super().__init__(message, resource_type="gpu", **kwargs)
        self.gpu_backend = gpu_backend
        self.add_diagnostic_data("gpu_backend", gpu_backend)
        self.add_recovery_hint("Fall back to CPU processing")
        self.add_recovery_hint("Clear GPU cache")


class NetworkError(UnifiedError):
    """Network and external service errors."""

    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        status_code: int | None = None,
        **kwargs,
    ):
        kwargs.setdefault("category", ErrorCategory.EXTERNAL)
        kwargs.setdefault("recovery_strategy", RecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)

        self.endpoint = endpoint
        self.status_code = status_code

        if endpoint:
            self.add_diagnostic_data("endpoint", endpoint)
        if status_code:
            self.add_diagnostic_data("status_code", status_code)

        self.add_recovery_hint("Check network connectivity")
        self.add_recovery_hint("Retry with exponential backoff")


class ConfigurationError(UnifiedError):
    """Configuration and validation errors."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any | None = None,
        **kwargs,
    ):
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("recovery_strategy", RecoveryStrategy.MANUAL_INTERVENTION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)

        self.config_key = config_key
        self.config_value = config_value

        if config_key:
            self.add_diagnostic_data("config_key", config_key)
        if config_value is not None:
            self.add_diagnostic_data("config_value", config_value)

        self.add_recovery_hint("Check configuration file syntax")
        self.add_recovery_hint("Validate configuration values")


class ValidationError(UnifiedError):
    """Data validation errors."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        field_value: Any | None = None,
        expected_type: str | None = None,
        **kwargs,
    ):
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("recovery_strategy", RecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)

        self.field_name = field_name
        self.field_value = field_value
        self.expected_type = expected_type

        if field_name:
            self.add_diagnostic_data("field_name", field_name)
        if field_value is not None:
            self.add_diagnostic_data("field_value", str(field_value))
        if expected_type:
            self.add_diagnostic_data("expected_type", expected_type)


class PerformanceError(UnifiedError):
    """Performance threshold violations."""

    def __init__(
        self,
        message: str,
        operation: str,
        actual_duration: float,
        threshold_duration: float,
        **kwargs,
    ):
        kwargs.setdefault("category", ErrorCategory.PERFORMANCE)
        kwargs.setdefault("recovery_strategy", RecoveryStrategy.DEGRADE)
        super().__init__(message, **kwargs)

        self.operation = operation
        self.actual_duration = actual_duration
        self.threshold_duration = threshold_duration

        self.add_diagnostic_data("operation", operation)
        self.add_diagnostic_data("actual_duration", actual_duration)
        self.add_diagnostic_data("threshold_duration", threshold_duration)
        self.add_diagnostic_data(
            "performance_ratio", actual_duration / threshold_duration
        )


# Compatibility aliases for existing code
BoltException = BoltError  # Compatibility with bolt error handling
EinsteinException = EinsteinError
Jarvis2Exception = Jarvis2Error
UnityWheelException = UnityWheelError

# Legacy exception mappings for migration
LEGACY_EXCEPTION_MAPPING = {
    # Bolt exceptions
    "BoltException": UnifiedError,
    "BoltSystemException": ComponentError,
    "BoltResourceException": ResourceError,
    "BoltMemoryException": MemoryError,
    "BoltGPUException": GPUError,
    "BoltNetworkException": NetworkError,
    "BoltConfigurationException": ConfigurationError,
    # Unity Wheel exceptions
    "WheelException": UnityWheelError,
    "ValidationException": ValidationError,
    # Jarvis2 exceptions
    "JarvisException": Jarvis2Error,
}


def wrap_legacy_exception(
    exc: Exception, component: str = "unknown", **kwargs
) -> UnifiedError:
    """
    Wrap legacy exceptions in unified error format.

    Used for gradual migration from existing error handling.
    """
    exc_name = type(exc).__name__

    # Check if it's already a unified error
    if isinstance(exc, UnifiedError):
        return exc

    # Map legacy exceptions
    unified_class = LEGACY_EXCEPTION_MAPPING.get(exc_name, ComponentError)

    # Determine appropriate category and recovery strategy
    error_message = str(exc)

    if "memory" in error_message.lower():
        unified_class = MemoryError
    elif "gpu" in error_message.lower():
        unified_class = GPUError
    elif "network" in error_message.lower() or "connection" in error_message.lower():
        unified_class = NetworkError
    elif "config" in error_message.lower():
        unified_class = ConfigurationError
    elif "validation" in error_message.lower():
        unified_class = ValidationError

    # Create unified error with context
    return unified_class(error_message, component=component, cause=exc, **kwargs)
