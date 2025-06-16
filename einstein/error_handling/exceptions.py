"""
Einstein Exception Hierarchy

Comprehensive exception system for Einstein indexing and search components,
providing structured error information, recovery hints, and diagnostic data.
"""

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Type


class EinsteinErrorSeverity(Enum):
    """Severity levels for Einstein errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EinsteinErrorCategory(Enum):
    """Categories of errors in the Einstein system."""
    INDEX = "index"
    SEARCH = "search"
    EMBEDDING = "embedding"
    FILEWATCHER = "file_watcher"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    FAISS = "faiss"
    NEURAL = "neural"


class EinsteinRecoveryStrategy(Enum):
    """Recovery strategies for different types of Einstein errors."""
    RETRY = "retry"
    FALLBACK = "fallback"
    REBUILD_INDEX = "rebuild_index"
    REDUCE_SCOPE = "reduce_scope"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"
    IGNORE = "ignore"


@dataclass
class EinsteinErrorContext:
    """Contextual information about an Einstein error."""
    timestamp: float = field(default_factory=time.time)
    operation: str | None = None
    component: str | None = None
    file_path: str | None = None
    query: str | None = None
    index_size: int | None = None
    search_params: dict[str, Any] | None = None
    system_state: dict[str, Any] | None = None
    recovery_attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'timestamp': self.timestamp,
            'operation': self.operation,
            'component': self.component,
            'file_path': self.file_path,
            'query': self.query,
            'index_size': self.index_size,
            'search_params': self.search_params,
            'system_state': self.system_state,
            'recovery_attempts': self.recovery_attempts,
            'metadata': self.metadata
        }


class EinsteinException(Exception):
    """Base exception class for all Einstein errors.
    
    Provides structured error information, recovery hints, and diagnostic data
    specifically tailored for Einstein indexing and search operations.
    """
    
    def __init__(
        self,
        message: str,
        *,
        severity: EinsteinErrorSeverity = EinsteinErrorSeverity.MEDIUM,
        category: EinsteinErrorCategory = EinsteinErrorCategory.INDEX,
        recovery_strategy: EinsteinRecoveryStrategy = EinsteinRecoveryStrategy.RETRY,
        error_code: str | None = None,
        context: EinsteinErrorContext | None = None,
        cause: Exception | None = None,
        recovery_hints: list[str] | None = None,
        diagnostic_data: dict[str, Any] | None = None,
        user_message: str | None = None
    ):
        super().__init__(message)
        
        self.message = message
        self.severity = severity
        self.category = category
        self.recovery_strategy = recovery_strategy
        self.error_code = error_code or self._generate_error_code()
        self.context = context or EinsteinErrorContext()
        self.cause = cause
        self.recovery_hints = recovery_hints or []
        self.diagnostic_data = diagnostic_data or {}
        self.user_message = user_message or self._generate_user_message()
        
        # Capture stack trace
        self.stack_trace = traceback.format_exc()
        
        # Set the cause chain
        if cause:
            self.__cause__ = cause
    
    def _generate_error_code(self) -> str:
        """Generate a unique error code based on exception class."""
        class_name = self.__class__.__name__
        return f"EINSTEIN_{class_name.upper()}"
    
    def _generate_user_message(self) -> str:
        """Generate a user-friendly error message."""
        if self.category == EinsteinErrorCategory.INDEX:
            return "There was an issue with the search index. The system will attempt to rebuild it automatically."
        elif self.category == EinsteinErrorCategory.SEARCH:
            return "Your search could not be completed. Please try a different query or check back in a moment."
        elif self.category == EinsteinErrorCategory.EMBEDDING:
            return "There was an issue processing your query. The system will try alternative methods."
        elif self.category == EinsteinErrorCategory.FILEWATCHER:
            return "File monitoring has encountered an issue. Manual refresh may be needed."
        elif self.category == EinsteinErrorCategory.DATABASE:
            return "There was an issue accessing the search database. Please try again."
        elif self.category == EinsteinErrorCategory.CONFIGURATION:
            return "There's a configuration issue that needs attention. Please check your settings."
        elif self.category == EinsteinErrorCategory.RESOURCE:
            return "System resources are constrained. The system will operate in reduced capacity mode."
        else:
            return "An unexpected issue occurred. The system will attempt automatic recovery."
    
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
            'error_code': self.error_code,
            'message': self.message,
            'user_message': self.user_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'recovery_strategy': self.recovery_strategy.value,
            'context': self.context.to_dict(),
            'recovery_hints': self.recovery_hints,
            'diagnostic_data': self.diagnostic_data,
            'cause': str(self.cause) if self.cause else None,
            'stack_trace': self.stack_trace
        }
    
    def is_recoverable(self) -> bool:
        """Check if the error is potentially recoverable."""
        return self.recovery_strategy in [
            EinsteinRecoveryStrategy.RETRY,
            EinsteinRecoveryStrategy.FALLBACK,
            EinsteinRecoveryStrategy.REBUILD_INDEX,
            EinsteinRecoveryStrategy.REDUCE_SCOPE,
            EinsteinRecoveryStrategy.GRACEFUL_DEGRADATION
        ]
    
    def is_critical(self) -> bool:
        """Check if the error is critical."""
        return self.severity == EinsteinErrorSeverity.CRITICAL
    
    def should_retry(self) -> bool:
        """Check if the error suggests a retry."""
        return self.recovery_strategy == EinsteinRecoveryStrategy.RETRY
    
    def __str__(self) -> str:
        """String representation including error code and severity."""
        return f"[{self.error_code}:{self.severity.value}] {self.message}"


class EinsteinIndexException(EinsteinException):
    """Index-related errors in Einstein."""
    
    def __init__(
        self,
        message: str,
        index_type: str | None = None,
        index_size: int | None = None,
        **kwargs
    ):
        kwargs.setdefault('category', EinsteinErrorCategory.INDEX)
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.REBUILD_INDEX)
        super().__init__(message, **kwargs)
        
        self.index_type = index_type
        self.index_size = index_size
        
        if index_type:
            self.add_diagnostic_data('index_type', index_type)
        if index_size is not None:
            self.add_diagnostic_data('index_size', index_size)
        
        # Add index-specific recovery hints
        self.add_recovery_hint("Try rebuilding the search index")
        self.add_recovery_hint("Check available disk space")
        self.add_recovery_hint("Verify file permissions")


class EinsteinSearchException(EinsteinException):
    """Search operation errors in Einstein."""
    
    def __init__(
        self,
        message: str,
        query: str | None = None,
        search_type: str | None = None,
        result_count: int | None = None,
        **kwargs
    ):
        kwargs.setdefault('category', EinsteinErrorCategory.SEARCH)
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)
        
        self.query = query
        self.search_type = search_type
        self.result_count = result_count
        
        if query:
            self.set_context(query=query)
            self.add_diagnostic_data('query_length', len(query))
        if search_type:
            self.add_diagnostic_data('search_type', search_type)
        if result_count is not None:
            self.add_diagnostic_data('result_count', result_count)
        
        # Add search-specific recovery hints
        self.add_recovery_hint("Try simplifying your search query")
        self.add_recovery_hint("Use different search terms")
        self.add_recovery_hint("Try text-based search instead of semantic search")


class EinsteinEmbeddingException(EinsteinException):
    """Embedding generation and processing errors."""
    
    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        embedding_dim: int | None = None,
        text_length: int | None = None,
        **kwargs
    ):
        kwargs.setdefault('category', EinsteinErrorCategory.EMBEDDING)
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.text_length = text_length
        
        if model_name:
            self.add_diagnostic_data('model_name', model_name)
        if embedding_dim is not None:
            self.add_diagnostic_data('embedding_dim', embedding_dim)
        if text_length is not None:
            self.add_diagnostic_data('text_length', text_length)
        
        # Add embedding-specific recovery hints
        self.add_recovery_hint("Try using a different embedding model")
        self.add_recovery_hint("Break down large text into smaller chunks")
        self.add_recovery_hint("Use CPU-based embedding as fallback")


class EinsteinFileWatcherException(EinsteinException):
    """File watching and monitoring errors."""
    
    def __init__(
        self,
        message: str,
        watched_path: str | None = None,
        event_type: str | None = None,
        **kwargs
    ):
        kwargs.setdefault('category', EinsteinErrorCategory.FILEWATCHER)
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.GRACEFUL_DEGRADATION)
        super().__init__(message, **kwargs)
        
        self.watched_path = watched_path
        self.event_type = event_type
        
        if watched_path:
            self.set_context(file_path=watched_path)
        if event_type:
            self.add_diagnostic_data('event_type', event_type)
        
        # Add file watcher recovery hints
        self.add_recovery_hint("Restart file monitoring service")
        self.add_recovery_hint("Check file system permissions")
        self.add_recovery_hint("Manually refresh the index")


class EinsteinDatabaseException(EinsteinException):
    """Database access and operation errors."""
    
    def __init__(
        self,
        message: str,
        database_type: str | None = None,
        query_type: str | None = None,
        **kwargs
    ):
        kwargs.setdefault('category', EinsteinErrorCategory.DATABASE)
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.RETRY)
        super().__init__(message, **kwargs)
        
        self.database_type = database_type
        self.query_type = query_type
        
        if database_type:
            self.add_diagnostic_data('database_type', database_type)
        if query_type:
            self.add_diagnostic_data('query_type', query_type)
        
        # Add database-specific recovery hints
        self.add_recovery_hint("Check database connection")
        self.add_recovery_hint("Verify database file integrity")
        self.add_recovery_hint("Try recreating database indexes")


class EinsteinConfigurationException(EinsteinException):
    """Configuration and setup errors."""
    
    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any | None = None,
        **kwargs
    ):
        kwargs.setdefault('category', EinsteinErrorCategory.CONFIGURATION)
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.MANUAL_INTERVENTION)
        kwargs.setdefault('severity', EinsteinErrorSeverity.HIGH)
        super().__init__(message, **kwargs)
        
        self.config_key = config_key
        self.config_value = config_value
        
        if config_key:
            self.add_diagnostic_data('config_key', config_key)
        if config_value is not None:
            self.add_diagnostic_data('config_value', str(config_value))
        
        # Add configuration-specific recovery hints
        self.add_recovery_hint("Check configuration file syntax")
        self.add_recovery_hint("Verify all required settings are present")
        self.add_recovery_hint("Reset to default configuration")


class EinsteinResourceException(EinsteinException):
    """Resource exhaustion and management errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        current_usage: float | None = None,
        limit: float | None = None,
        **kwargs
    ):
        kwargs.setdefault('category', EinsteinErrorCategory.RESOURCE)
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.GRACEFUL_DEGRADATION)
        kwargs.setdefault('severity', EinsteinErrorSeverity.HIGH)
        super().__init__(message, **kwargs)
        
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        
        self.add_diagnostic_data('resource_type', resource_type)
        if current_usage is not None:
            self.add_diagnostic_data('current_usage', current_usage)
        if limit is not None:
            self.add_diagnostic_data('limit', limit)
            if current_usage is not None:
                self.add_diagnostic_data('usage_percent', (current_usage / limit) * 100)
        
        # Add resource-specific recovery hints
        if resource_type == "memory":
            self.add_recovery_hint("Reduce index size or batch processing")
            self.add_recovery_hint("Clear caches and temporary data")
            self.add_recovery_hint("Enable memory-efficient mode")
        elif resource_type == "disk":
            self.add_recovery_hint("Clean up temporary files")
            self.add_recovery_hint("Archive old index data")
            self.add_recovery_hint("Check available disk space")
        elif resource_type == "cpu":
            self.add_recovery_hint("Reduce concurrent operations")
            self.add_recovery_hint("Lower processing priority")
            self.add_recovery_hint("Enable throttling mode")


class EinsteinFAISSException(EinsteinException):
    """FAISS-specific errors."""
    
    def __init__(
        self,
        message: str,
        faiss_operation: str | None = None,
        vector_dim: int | None = None,
        **kwargs
    ):
        kwargs.setdefault('category', EinsteinErrorCategory.FAISS)
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.FALLBACK)
        super().__init__(message, **kwargs)
        
        self.faiss_operation = faiss_operation
        self.vector_dim = vector_dim
        
        if faiss_operation:
            self.add_diagnostic_data('faiss_operation', faiss_operation)
        if vector_dim is not None:
            self.add_diagnostic_data('vector_dim', vector_dim)
        
        # Add FAISS-specific recovery hints
        self.add_recovery_hint("Try rebuilding FAISS index")
        self.add_recovery_hint("Use alternative search method")
        self.add_recovery_hint("Check FAISS library installation")


class EinsteinDependencyException(EinsteinException):
    """Dependency graph and analysis errors."""
    
    def __init__(
        self,
        message: str,
        dependency_type: str | None = None,
        symbol_name: str | None = None,
        **kwargs
    ):
        kwargs.setdefault('category', EinsteinErrorCategory.DEPENDENCY)
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.REDUCE_SCOPE)
        super().__init__(message, **kwargs)
        
        self.dependency_type = dependency_type
        self.symbol_name = symbol_name
        
        if dependency_type:
            self.add_diagnostic_data('dependency_type', dependency_type)
        if symbol_name:
            self.add_diagnostic_data('symbol_name', symbol_name)
        
        # Add dependency-specific recovery hints
        self.add_recovery_hint("Try rebuilding dependency graph")
        self.add_recovery_hint("Skip dependency analysis for now")
        self.add_recovery_hint("Use basic text search instead")


# Exception utilities

def wrap_einstein_exception(
    exc: Exception,
    message: str | None = None,
    category: EinsteinErrorCategory | None = None,
    **kwargs
) -> EinsteinException:
    """Wrap a generic exception in an EinsteinException with context."""
    
    # Determine appropriate EinsteinException type based on original exception
    einstein_exc_class: Type[EinsteinException]
    
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        einstein_exc_class = EinsteinConfigurationException
        kwargs.setdefault('recovery_strategy', EinsteinRecoveryStrategy.MANUAL_INTERVENTION)
    elif isinstance(exc, MemoryError):
        einstein_exc_class = EinsteinResourceException
        kwargs.setdefault('resource_type', 'memory')
    elif isinstance(exc, (IOError, OSError)) and "disk" in str(exc).lower():
        einstein_exc_class = EinsteinResourceException
        kwargs.setdefault('resource_type', 'disk')
    elif isinstance(exc, FileNotFoundError):
        einstein_exc_class = EinsteinFileWatcherException
    elif isinstance(exc, (ConnectionError, TimeoutError)):
        einstein_exc_class = EinsteinDatabaseException
    elif "faiss" in str(exc).lower():
        einstein_exc_class = EinsteinFAISSException
    elif "embedding" in str(exc).lower():
        einstein_exc_class = EinsteinEmbeddingException
    elif "search" in str(exc).lower():
        einstein_exc_class = EinsteinSearchException
    elif "index" in str(exc).lower():
        einstein_exc_class = EinsteinIndexException
    else:
        einstein_exc_class = EinsteinException
    
    # Use provided message or extract from original exception
    error_message = message or str(exc)
    
    # Set category if not provided
    if category:
        kwargs['category'] = category
    
    # Create wrapped exception
    wrapped = einstein_exc_class(
        error_message,
        cause=exc,
        **kwargs
    )
    
    return wrapped


def create_einstein_recovery_hint(
    error_type: str,
    context: dict[str, Any]
) -> list[str]:
    """Generate contextual recovery hints based on error type and context."""
    
    hints = []
    
    if error_type == "index":
        index_size = context.get('index_size', 0)
        if index_size == 0:
            hints.extend([
                "Index appears to be empty - try rebuilding from scratch",
                "Check if source files are accessible",
                "Verify indexing permissions"
            ])
        else:
            hints.extend([
                "Try rebuilding the corrupted index",
                "Check for sufficient disk space",
                "Verify index file integrity"
            ])
    
    elif error_type == "search":
        query_length = context.get('query_length', 0)
        if query_length > 1000:
            hints.extend([
                "Query is very long - try breaking it into smaller parts",
                "Use more specific search terms",
                "Consider using structured search instead"
            ])
        elif query_length == 0:
            hints.extend([
                "Empty query provided - add search terms",
                "Check query formatting"
            ])
    
    elif error_type == "embedding":
        model_name = context.get('model_name')
        if model_name:
            hints.extend([
                f"Embedding model ({model_name}) may be unavailable",
                "Try switching to a different embedding model",
                "Use CPU-based embedding as fallback"
            ])
    
    elif error_type == "resource":
        resource_type = context.get('resource_type')
        usage_percent = context.get('usage_percent', 0)
        
        if resource_type == "memory" and usage_percent > 90:
            hints.extend([
                "System memory critically low - restart application",
                "Clear all caches and temporary data",
                "Reduce index size or processing batch size"
            ])
        elif resource_type == "disk" and usage_percent > 95:
            hints.extend([
                "Disk space critically low - clean up files",
                "Archive old index data",
                "Move indexes to different storage location"
            ])
    
    elif error_type == "faiss":
        vector_dim = context.get('vector_dim')
        if vector_dim:
            hints.extend([
                f"FAISS index dimension mismatch (expected: {vector_dim})",
                "Rebuild FAISS index with correct dimensions",
                "Check embedding model compatibility"
            ])
    
    return hints


def categorize_einstein_exception(exc: Exception) -> EinsteinErrorCategory:
    """Automatically categorize an exception for Einstein context."""
    
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    
    if any(term in exc_str for term in ['index', 'faiss']):
        return EinsteinErrorCategory.INDEX
    elif any(term in exc_str for term in ['search', 'query']):
        return EinsteinErrorCategory.SEARCH
    elif any(term in exc_str for term in ['embedding', 'model', 'vector']):
        return EinsteinErrorCategory.EMBEDDING
    elif any(term in exc_str for term in ['file', 'watch', 'monitor']):
        return EinsteinErrorCategory.FILEWATCHER
    elif any(term in exc_str for term in ['database', 'db', 'sql']):
        return EinsteinErrorCategory.DATABASE
    elif any(term in exc_str for term in ['config', 'setting', 'parameter']):
        return EinsteinErrorCategory.CONFIGURATION
    elif any(term in exc_str for term in ['memory', 'disk', 'cpu', 'resource']):
        return EinsteinErrorCategory.RESOURCE
    elif any(term in exc_str for term in ['dependency', 'import', 'module']):
        return EinsteinErrorCategory.DEPENDENCY
    elif 'faiss' in exc_str:
        return EinsteinErrorCategory.FAISS
    else:
        return EinsteinErrorCategory.INDEX  # Default to index for unknown errors