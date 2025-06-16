"""
Bolt Error Handling Framework

Comprehensive error handling system for production-ready Bolt deployment.
Provides robust error management, recovery mechanisms, and diagnostic tools.
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerManager
from .diagnostics import DiagnosticCollector, SystemHealthChecker
from .exceptions import (
    BoltAgentException,
    BoltConfigurationException,
    BoltException,
    BoltExternalException,
    BoltGPUException,
    BoltHardwareException,
    BoltMemoryException,
    BoltNetworkException,
    BoltResourceException,
    BoltSystemException,
    BoltTaskException,
    BoltTimeoutException,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    RecoveryStrategy,
    categorize_exception,
    create_recovery_hint,
    wrap_exception,
)
from .graceful_degradation import GracefulDegradationManager
from .monitoring import ErrorMonitor
from .recovery import ErrorRecoveryManager
from .resource_guards import ResourceGuard, ResourceGuardManager
from .system import BoltErrorHandlingSystem

__all__ = [
    # Exceptions
    "BoltException",
    "BoltSystemException",
    "BoltResourceException",
    "BoltAgentException",
    "BoltTaskException",
    "BoltTimeoutException",
    "BoltConfigurationException",
    "BoltHardwareException",
    "BoltMemoryException",
    "BoltGPUException",
    "BoltNetworkException",
    "BoltExternalException",
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryStrategy",
    "ErrorContext",
    "wrap_exception",
    "create_recovery_hint",
    "categorize_exception",
    # Core Components
    "ErrorRecoveryManager",
    "BoltErrorHandlingSystem",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "ResourceGuard",
    "ResourceGuardManager",
    "GracefulDegradationManager",
    "ErrorMonitor",
    "DiagnosticCollector",
    "SystemHealthChecker",
]
