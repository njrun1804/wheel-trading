"""
Einstein Error Handling System

Comprehensive error handling for the Einstein indexing and search system,
providing structured exceptions, recovery mechanisms, and user-friendly
error messages with diagnostic information.
"""

from .diagnostics import (
    EinsteinDiagnostics,
    SystemHealthChecker,
)
from .exceptions import (
    EinsteinConfigurationException,
    EinsteinDatabaseException,
    EinsteinEmbeddingException,
    EinsteinException,
    EinsteinFileWatcherException,
    EinsteinIndexException,
    EinsteinResourceException,
    EinsteinSearchException,
    wrap_einstein_exception,
)
from .fallbacks import (
    EinsteinFallbackManager,
    EmbeddingFallbackChain,
    SearchFallbackChain,
)
from .recovery import (
    EinsteinRecoveryManager,
    RecoveryConfiguration,
    RecoveryState,
)

__all__ = [
    # Exceptions
    "EinsteinException",
    "EinsteinIndexException",
    "EinsteinSearchException",
    "EinsteinEmbeddingException",
    "EinsteinFileWatcherException",
    "EinsteinDatabaseException",
    "EinsteinConfigurationException",
    "EinsteinResourceException",
    "wrap_einstein_exception",
    # Recovery
    "EinsteinRecoveryManager",
    "RecoveryConfiguration",
    "RecoveryState",
    # Fallbacks
    "EinsteinFallbackManager",
    "SearchFallbackChain",
    "EmbeddingFallbackChain",
    # Diagnostics
    "EinsteinDiagnostics",
    "SystemHealthChecker",
]
