"""
Structured Logging - Unified implementation with context management.

Consolidates structured logging across all components.
"""

import json
import logging
import threading
import time
from collections.abc import MutableMapping
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

# Context variables for request tracking
request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
execution_context: ContextVar[dict[str, Any]] = ContextVar(
    "execution_context", default={}
)

# Thread-local storage for performance
_local = threading.local()


class LogContext:
    """Context manager for scoped logging context."""

    def __init__(self, **context):
        self.context = context
        self.previous_context = None
        self.request_id = str(uuid4())[:8]

    def __enter__(self):
        # Store previous context
        self.previous_context = execution_context.get({})

        # Merge contexts
        new_context = {**self.previous_context, **self.context}
        execution_context.set(new_context)
        request_id.set(self.request_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        execution_context.set(self.previous_context or {})
        request_id.set(None)


class StructuredLogger(logging.LoggerAdapter):
    """
    Logger adapter that adds structured context to all log messages.

    Optimized for high-performance logging with minimal overhead.
    """

    def __init__(self, logger: logging.Logger, extra: dict[str, Any] | None = None):
        super().__init__(logger, extra or {})
        self._cached_context = {}
        self._last_context_update = 0

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        """Process log message to add structured context."""
        try:
            # Get current timestamp
            timestamp = datetime.now(UTC).isoformat()

            # Get context efficiently
            ctx = self._get_cached_context()
            req_id = request_id.get()

            # Build structured log entry
            log_entry = {
                "timestamp": timestamp,
                "level": kwargs.get("levelname", getattr(logging, "INFO", 20)),
                "message": str(msg),
                "module": self.logger.name,
                "thread_id": threading.current_thread().ident,
            }

            # Add request ID if available
            if req_id:
                log_entry["request_id"] = req_id

            # Add execution context
            if ctx:
                log_entry["context"] = ctx

            # Add extra fields from kwargs, filtering out LogRecord built-ins
            if "extra" in kwargs:
                extra_data = kwargs["extra"]
                if isinstance(extra_data, dict):
                    # Filter out built-in LogRecord attributes
                    safe_extra = {
                        k: v
                        for k, v in extra_data.items()
                        if k not in self._builtin_fields
                    }
                    log_entry.update(safe_extra)

                # Clean kwargs
                kwargs = {k: v for k, v in kwargs.items() if k != "extra"}

            # Format as JSON
            formatted_msg = json.dumps(log_entry, default=self._json_serializer)

            return formatted_msg, kwargs

        except Exception as e:
            # Fallback to simple logging if structured logging fails
            return f"[LOGGING_ERROR: {e}] {msg}", kwargs

    def _get_cached_context(self) -> dict[str, Any]:
        """Get context with efficient caching."""
        current_time = time.time()

        # Cache context for 1 second to avoid excessive ContextVar lookups
        if current_time - self._last_context_update > 1.0:
            self._cached_context = execution_context.get({})
            self._last_context_update = current_time

        return self._cached_context

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for non-serializable objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return str(obj)
        else:
            return repr(obj)

    # Built-in LogRecord fields to filter from extra data
    _builtin_fields = {
        "name",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "msg",
        "args",
        "exc_info",
        "exc_text",
        "stack_info",
    }


def get_logger(name: str | None = None) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (uses calling module if None)

    Returns:
        StructuredLogger instance
    """
    if name is None:
        # Get caller's module name
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "unknown")
        else:
            name = "unknown"

    base_logger = logging.getLogger(name)
    return StructuredLogger(base_logger)


def set_context(**kwargs):
    """Set execution context for current async/thread context."""
    ctx = execution_context.get({})
    ctx.update(kwargs)
    execution_context.set(ctx)


def clear_context():
    """Clear execution context."""
    execution_context.set({})


def get_context() -> dict[str, Any]:
    """Get current execution context."""
    return execution_context.get({})


# Performance-optimized logger for high-frequency calls
class FastLogger:
    """Minimal overhead logger for performance-critical paths."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._enabled = logger.isEnabledFor(logging.INFO)

    def log(self, level: int, msg: str, **kwargs):
        """Log with minimal overhead."""
        if self.logger.isEnabledFor(level):
            # Fast path - minimal formatting
            timestamp = time.time()
            log_data = {"ts": timestamp, "msg": msg}
            if kwargs:
                log_data.update(kwargs)

            record = logging.LogRecord(
                name=self.logger.name,
                level=level,
                pathname="",
                lineno=0,
                msg=json.dumps(log_data, separators=(",", ":")),
                args=(),
                exc_info=None,
            )
            self.logger.handle(record)
