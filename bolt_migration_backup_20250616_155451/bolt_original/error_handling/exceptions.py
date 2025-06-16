"""
Bolt Exception Hierarchy

Defines a comprehensive exception hierarchy for the Bolt system with
structured error information, recovery hints, and diagnostic data.
"""

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """Severity levels for Bolt errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in the Bolt system."""

    SYSTEM = "system"
    RESOURCE = "resource"
    AGENT = "agent"
    TASK = "task"
    NETWORK = "network"
    HARDWARE = "hardware"
    CONFIGURATION = "configuration"
    EXTERNAL = "external"


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of errors."""

    RETRY = "retry"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAILOVER = "failover"
    RESTART = "restart"
    MANUAL_INTERVENTION = "manual_intervention"
    NONE = "none"


@dataclass
class ErrorContext:
    """Contextual information about an error."""

    timestamp: float = field(default_factory=time.time)
    operation: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    component: str | None = None
    system_state: dict[str, Any] | None = None
    recovery_attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "component": self.component,
            "system_state": self.system_state,
            "recovery_attempts": self.recovery_attempts,
            "metadata": self.metadata,
        }


class BoltException(Exception):
    """Base exception class for all Bolt errors.

    Provides structured error information, recovery hints, and diagnostic data.
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

        # Capture stack trace
        self.stack_trace = traceback.format_exc()

        # Set the cause chain
        if cause:
            self.__cause__ = cause

    def _generate_error_code(self) -> str:
        """Generate a unique error code based on exception class."""
        class_name = self.__class__.__name__
        return f"BOLT_{class_name.upper()}"

    def add_recovery_hint(self, hint: str) -> None:
        """Add a recovery hint to the exception."""
        self.recovery_hints.append(hint)

    def add_diagnostic_data(self, key: str, value: Any) -> None:
        """Add diagnostic data to the exception."""
        self.diagnostic_data[key] = value

    def set_context(self, **kwargs) -> None:
        """Update error context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.metadata[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recovery_strategy": self.recovery_strategy.value,
            "context": self.context.to_dict(),
            "recovery_hints": self.recovery_hints,
            "diagnostic_data": self.diagnostic_data,
            "cause": str(self.cause) if self.cause else None,
            "stack_trace": self.stack_trace,
        }

    def is_recoverable(self) -> bool:
        """Check if the error is potentially recoverable."""
        return self.recovery_strategy in [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            RecoveryStrategy.FAILOVER,
            RecoveryStrategy.RESTART,
        ]

    def is_critical(self) -> bool:
        """Check if the error is critical."""
        return self.severity == ErrorSeverity.CRITICAL

    def should_retry(self) -> bool:
        """Check if the error suggests a retry."""
        return self.recovery_strategy == RecoveryStrategy.RETRY

    def __str__(self) -> str:
        """String representation including error code and severity."""
        return f"[{self.error_code}:{self.severity.value}] {self.message}"


class BoltSystemException(BoltException):
    """System-level errors in the Bolt framework."""

    def __init__(self, message: str, **kwargs):
        # Remove conflicting parameters from kwargs to avoid duplication
        severity = kwargs.pop("severity", ErrorSeverity.HIGH)
        kwargs.pop("category", None)  # Always remove category - we enforce SYSTEM
        super().__init__(
            message, severity=severity, category=ErrorCategory.SYSTEM, **kwargs
        )


class BoltResourceException(BoltException):
    """Resource exhaustion and management errors."""

    def __init__(
        self,
        message: str,
        resource_type: str,
        current_usage: float | None = None,
        limit: float | None = None,
        **kwargs,
    ):
        # Remove conflicting kwargs to avoid duplication
        severity = kwargs.pop("severity", ErrorSeverity.HIGH)
        kwargs.pop("category", None)  # Always remove category - we enforce RESOURCE
        recovery_strategy = kwargs.pop(
            "recovery_strategy", RecoveryStrategy.GRACEFUL_DEGRADATION
        )
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.RESOURCE,
            recovery_strategy=recovery_strategy,
            **kwargs,
        )

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
                self.add_diagnostic_data("usage_percent", (current_usage / limit) * 100)


class BoltMemoryException(BoltResourceException):
    """Memory-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            resource_type="memory",
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            **kwargs,
        )

        # Add memory-specific recovery hints
        self.add_recovery_hint("Reduce batch sizes")
        self.add_recovery_hint("Clear caches")
        self.add_recovery_hint("Trigger garbage collection")
        self.add_recovery_hint("Restart agents with lower memory footprint")


class BoltGPUException(BoltResourceException):
    """GPU-related errors."""

    def __init__(self, message: str, gpu_backend: str = "unknown", **kwargs):
        super().__init__(
            message,
            resource_type="gpu",
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            **kwargs,
        )

        self.gpu_backend = gpu_backend
        self.add_diagnostic_data("gpu_backend", gpu_backend)

        # Add GPU-specific recovery hints
        self.add_recovery_hint("Fall back to CPU processing")
        self.add_recovery_hint("Reduce GPU memory allocation")
        self.add_recovery_hint("Clear GPU cache")
        self.add_recovery_hint("Restart GPU-dependent components")


class BoltAgentException(BoltException):
    """Agent-related errors."""

    def __init__(
        self, message: str, agent_id: str, agent_status: str | None = None, **kwargs
    ):
        # Remove conflicting kwargs to avoid duplication
        severity = kwargs.pop("severity", ErrorSeverity.MEDIUM)
        kwargs.pop("category", None)  # Always remove category - we enforce AGENT
        recovery_strategy = kwargs.pop("recovery_strategy", RecoveryStrategy.RETRY)
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.AGENT,
            recovery_strategy=recovery_strategy,
            **kwargs,
        )

        self.agent_id = agent_id
        self.agent_status = agent_status

        # Set context
        self.set_context(agent_id=agent_id)
        self.add_diagnostic_data("agent_status", agent_status)

        # Add agent-specific recovery hints
        self.add_recovery_hint("Restart the agent")
        self.add_recovery_hint("Reassign tasks to other agents")
        self.add_recovery_hint("Check agent resource allocation")


class BoltTaskException(BoltException):
    """Task execution errors."""

    def __init__(
        self,
        message: str,
        task_id: str,
        task_type: str | None = None,
        agent_id: str | None = None,
        **kwargs,
    ):
        # Remove conflicting kwargs to avoid duplication
        severity = kwargs.pop("severity", ErrorSeverity.MEDIUM)
        kwargs.pop("category", None)  # Always remove category - we enforce TASK
        recovery_strategy = kwargs.pop("recovery_strategy", RecoveryStrategy.RETRY)
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.TASK,
            recovery_strategy=recovery_strategy,
            **kwargs,
        )

        self.task_id = task_id
        self.task_type = task_type

        # Set context
        self.set_context(task_id=task_id, agent_id=agent_id)
        self.add_diagnostic_data("task_type", task_type)

        # Add task-specific recovery hints
        self.add_recovery_hint("Retry task with different parameters")
        self.add_recovery_hint("Split task into smaller subtasks")
        self.add_recovery_hint("Assign task to different agent")
        self.add_recovery_hint("Check task dependencies")


class BoltTimeoutException(BoltException):
    """Timeout-related errors."""

    def __init__(self, message: str, timeout_duration: float, operation: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs,
        )

        self.timeout_duration = timeout_duration
        self.operation = operation

        # Set context
        self.set_context(operation=operation)
        self.add_diagnostic_data("timeout_duration", timeout_duration)

        # Add timeout-specific recovery hints
        self.add_recovery_hint("Increase timeout duration")
        self.add_recovery_hint("Optimize operation performance")
        self.add_recovery_hint("Break operation into smaller parts")
        self.add_recovery_hint("Check system resource availability")


class BoltConfigurationException(BoltException):
    """Configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any | None = None,
        **kwargs,
    ):
        # Remove conflicting kwargs to avoid duplication
        severity = kwargs.pop("severity", ErrorSeverity.HIGH)
        kwargs.pop(
            "category", None
        )  # Always remove category - we enforce CONFIGURATION
        recovery_strategy = kwargs.pop(
            "recovery_strategy", RecoveryStrategy.MANUAL_INTERVENTION
        )
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.CONFIGURATION,
            recovery_strategy=recovery_strategy,
            **kwargs,
        )

        self.config_key = config_key
        self.config_value = config_value

        if config_key:
            self.add_diagnostic_data("config_key", config_key)
        if config_value is not None:
            self.add_diagnostic_data("config_value", config_value)

        # Add configuration-specific recovery hints
        self.add_recovery_hint("Check configuration file syntax")
        self.add_recovery_hint("Validate configuration values")
        self.add_recovery_hint("Reset to default configuration")
        self.add_recovery_hint("Check environment variables")


class BoltHardwareException(BoltException):
    """Hardware-related errors."""

    def __init__(self, message: str, hardware_component: str, **kwargs):
        # Remove conflicting kwargs to avoid duplication
        severity = kwargs.pop("severity", ErrorSeverity.HIGH)
        kwargs.pop("category", None)  # Always remove category - we enforce HARDWARE
        recovery_strategy = kwargs.pop(
            "recovery_strategy", RecoveryStrategy.GRACEFUL_DEGRADATION
        )
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.HARDWARE,
            recovery_strategy=recovery_strategy,
            **kwargs,
        )

        self.hardware_component = hardware_component
        self.add_diagnostic_data("hardware_component", hardware_component)

        # Add hardware-specific recovery hints
        self.add_recovery_hint("Check hardware availability")
        self.add_recovery_hint("Fall back to alternative hardware")
        self.add_recovery_hint("Reduce hardware requirements")
        self.add_recovery_hint("Restart hardware components")


class BoltNetworkException(BoltException):
    """Network-related errors."""

    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        status_code: int | None = None,
        **kwargs,
    ):
        # Remove conflicting kwargs to avoid duplication
        severity = kwargs.pop("severity", ErrorSeverity.MEDIUM)
        kwargs.pop("category", None)  # Always remove category - we enforce NETWORK
        recovery_strategy = kwargs.pop("recovery_strategy", RecoveryStrategy.RETRY)
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.NETWORK,
            recovery_strategy=recovery_strategy,
            **kwargs,
        )

        self.endpoint = endpoint
        self.status_code = status_code

        if endpoint:
            self.add_diagnostic_data("endpoint", endpoint)
        if status_code:
            self.add_diagnostic_data("status_code", status_code)

        # Add network-specific recovery hints
        self.add_recovery_hint("Check network connectivity")
        self.add_recovery_hint("Retry with exponential backoff")
        self.add_recovery_hint("Use alternative endpoint")
        self.add_recovery_hint("Check firewall and proxy settings")


class BoltExternalException(BoltException):
    """External service/dependency errors."""

    def __init__(self, message: str, service_name: str, **kwargs):
        # Remove conflicting kwargs to avoid duplication
        severity = kwargs.pop("severity", ErrorSeverity.MEDIUM)
        kwargs.pop("category", None)  # Always remove category - we enforce EXTERNAL
        recovery_strategy = kwargs.pop(
            "recovery_strategy", RecoveryStrategy.GRACEFUL_DEGRADATION
        )
        super().__init__(
            message,
            severity=severity,
            category=ErrorCategory.EXTERNAL,
            recovery_strategy=recovery_strategy,
            **kwargs,
        )

        self.service_name = service_name
        self.add_diagnostic_data("service_name", service_name)

        # Add external service recovery hints
        self.add_recovery_hint("Check external service status")
        self.add_recovery_hint("Use cached/fallback data")
        self.add_recovery_hint("Retry with different service endpoint")
        self.add_recovery_hint("Enable degraded mode operation")


# Exception utilities


def wrap_exception(
    exc: Exception, message: str | None = None, **kwargs
) -> BoltException:
    """Wrap a generic exception in a BoltException with context."""

    # Determine appropriate BoltException type based on original exception
    bolt_exc_class: type[BoltException]
    if isinstance(exc, MemoryError | OSError) and "memory" in str(exc).lower():
        bolt_exc_class = BoltMemoryException
    elif isinstance(exc, TimeoutError):
        bolt_exc_class = BoltTimeoutException
        kwargs.setdefault("timeout_duration", 30.0)
        kwargs.setdefault("operation", "unknown")
    elif isinstance(exc, ConnectionError | OSError) and any(
        term in str(exc).lower() for term in ["connection", "network", "socket"]
    ):
        bolt_exc_class = BoltNetworkException
    else:
        bolt_exc_class = BoltException

    # Use provided message or extract from original exception
    error_message = message or str(exc)

    # Create wrapped exception
    wrapped = bolt_exc_class(error_message, cause=exc, **kwargs)

    return wrapped


def create_recovery_hint(error_type: str, context: dict[str, Any]) -> list[str]:
    """Generate contextual recovery hints based on error type and context."""

    hints = []

    if error_type == "memory":
        usage = context.get("memory_percent", 0)
        if usage > 90:
            hints.extend(
                [
                    "System memory critically low - restart system",
                    "Enable emergency memory reclamation",
                    "Reduce agent count and batch sizes",
                ]
            )
        elif usage > 80:
            hints.extend(
                [
                    "Clear caches and temporary data",
                    "Reduce concurrent operations",
                    "Monitor memory usage patterns",
                ]
            )

    elif error_type == "gpu":
        backend = context.get("gpu_backend", "unknown")
        hints.extend(
            [
                f"GPU backend ({backend}) may be unstable",
                "Consider falling back to CPU processing",
                "Check GPU driver and framework versions",
            ]
        )

    elif error_type == "agent":
        agent_count = context.get("active_agents", 0)
        if agent_count == 0:
            hints.append("No active agents - system may be shutting down")
        else:
            hints.append("Check individual agent health and logs")

    elif error_type == "task":
        retry_count = context.get("retry_count", 0)
        if retry_count > 3:
            hints.extend(
                [
                    "Task repeatedly failing - check task validity",
                    "Consider breaking task into smaller parts",
                    "Check task dependencies and prerequisites",
                ]
            )

    return hints


def categorize_exception(exc: Exception) -> ErrorCategory:
    """Automatically categorize an exception."""

    exc_str = str(exc).lower()
    type(exc).__name__.lower()

    if any(term in exc_str for term in ["memory", "allocation", "heap"]):
        return ErrorCategory.RESOURCE
    elif any(term in exc_str for term in ["network", "connection", "socket", "http"]):
        return ErrorCategory.NETWORK
    elif any(term in exc_str for term in ["gpu", "metal", "cuda", "opencl"]):
        return ErrorCategory.HARDWARE
    elif any(term in exc_str for term in ["config", "setting", "parameter"]):
        return ErrorCategory.CONFIGURATION
    elif "agent" in exc_str:
        return ErrorCategory.AGENT
    elif "task" in exc_str:
        return ErrorCategory.TASK
    else:
        return ErrorCategory.SYSTEM
