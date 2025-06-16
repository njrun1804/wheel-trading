"""
Unified Error Handling System

Consolidates error handling across bolt, einstein, jarvis2, and unity_wheel components.
Provides hardware-optimized error management with M4 Pro parallel processing.
"""

from .circuit_breaker import (
    CircuitBreakerManager,
    UnifiedCircuitBreaker,
    circuit_breaker,
)
from .decorators import (
    handle_errors_gracefully,
    monitor_performance,
    with_circuit_breaker,
    with_error_handling,
)
from .diagnostics import (
    DiagnosticResult,
    SystemHealthChecker,
    UnifiedDiagnostics,
    with_diagnostics,
)
from .exceptions import (
    ComponentError,
    ConfigurationError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    NetworkError,
    ResourceError,
    UnifiedError,
)
from .monitoring import (
    ErrorAlert,
    ErrorPattern,
    UnifiedErrorMonitor,
    get_error_monitor,
)
from .recovery import (
    RecoveryConfiguration,
    RecoveryStrategy,
    UnifiedRecoveryManager,
    with_recovery,
)

__all__ = [
    # Core exceptions
    "UnifiedError",
    "ComponentError",
    "ResourceError",
    "NetworkError",
    "ConfigurationError",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    # Recovery system
    "UnifiedRecoveryManager",
    "RecoveryStrategy",
    "RecoveryConfiguration",
    "with_recovery",
    # Circuit breaker
    "UnifiedCircuitBreaker",
    "CircuitBreakerManager",
    "circuit_breaker",
    # Monitoring
    "UnifiedErrorMonitor",
    "ErrorPattern",
    "ErrorAlert",
    "get_error_monitor",
    # Diagnostics
    "UnifiedDiagnostics",
    "DiagnosticResult",
    "SystemHealthChecker",
    "with_diagnostics",
    # Decorators
    "with_error_handling",
    "monitor_performance",
    "with_circuit_breaker",
    "handle_errors_gracefully",
]

# Version info
__version__ = "1.0.0"
__author__ = "Agent 5 (P-Core 4)"
__description__ = "Unified error handling system for wheel-trading components"
