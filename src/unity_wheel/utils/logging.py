"""Structured logging for autonomous operation with machine-parseable output."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Context variables for request tracking
request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
execution_context: ContextVar[Dict[str, Any]] = ContextVar("execution_context", default={})

F = TypeVar("F", bound=Callable[..., Any])


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter that adds structured context to all log messages."""

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Process log message to add structured context."""
        # Get execution context
        ctx = execution_context.get()
        req_id = request_id.get()

        # Build structured log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": kwargs.get("levelname", "INFO"),
            "message": msg,
            "module": self.logger.name,
        }

        # Add request ID if available
        if req_id:
            log_entry["request_id"] = req_id

        # Add execution context
        if ctx:
            log_entry["context"] = ctx

        # Add any extra fields from kwargs
        if "extra" in kwargs:
            log_entry.update(kwargs["extra"])

        # Format as JSON for machine parsing
        formatted_msg = json.dumps(log_entry, default=str)

        return formatted_msg, kwargs


class PerformanceLogger:
    """Context manager for logging performance metrics."""

    def __init__(self, operation: str, logger: StructuredLogger, threshold_ms: float = 200):
        self.operation = operation
        self.logger = logger
        self.threshold_ms = threshold_ms
        self.start_time: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

    def __enter__(self) -> PerformanceLogger:
        self.start_time = time.time()
        self.logger.debug(
            f"Starting {self.operation}",
            extra={
                "operation": self.operation,
                "status": "started",
            },
        )
        return self

    def add_metadata(self, **kwargs: Any) -> None:
        """Add metadata to be logged with performance metrics."""
        self.metadata.update(kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = (time.time() - self.start_time) * 1000

        extra_data = {
            "operation": self.operation,
            "duration_ms": round(duration_ms, 2),
            "status": "failed" if exc_type else "completed",
            **self.metadata,
        }

        if exc_type:
            extra_data["error"] = str(exc_val)
            extra_data["error_type"] = exc_type.__name__

        # Log with appropriate level based on duration
        if duration_ms > self.threshold_ms:
            self.logger.warning(
                f"{self.operation} took {duration_ms:.0f}ms (threshold: {self.threshold_ms}ms)",
                extra=extra_data,
            )
        else:
            self.logger.info(f"Completed {self.operation} in {duration_ms:.0f}ms", extra=extra_data)


def timed_operation(threshold_ms: float = 200, include_args: bool = False) -> Callable[[F], F]:
    """Decorator to automatically log operation timing."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            operation = f"{func.__module__}.{func.__name__}"

            with PerformanceLogger(operation, logger, threshold_ms) as perf:
                if include_args:
                    # Log first few args (avoid logging sensitive data)
                    safe_args = str(args[:3]) if len(args) <= 3 else f"{str(args[:3])}..."
                    perf.add_metadata(args=safe_args)

                result = func(*args, **kwargs)

                # Add result metadata if it's simple
                if isinstance(result, (int, float, str, bool)):
                    perf.add_metadata(result=result)
                elif isinstance(result, dict) and "confidence" in result:
                    perf.add_metadata(confidence=result["confidence"])

                return result

        return wrapper  # type: ignore

    return decorator


class DecisionLogger:
    """Specialized logger for trading decisions with confidence tracking."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.decision_history: List[Dict[str, Any]] = []

    def log_decision(
        self,
        action: str,
        rationale: str,
        confidence: float,
        risk_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a trading decision with full context."""
        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "rationale": rationale,
            "confidence": confidence,
            "risk_metrics": risk_metrics or {},
            "metadata": metadata or {},
        }

        self.decision_history.append(decision)

        self.logger.info(
            f"Trading decision: {action} (confidence: {confidence:.1%})",
            extra={
                "decision": decision,
                "type": "trading_decision",
            },
        )

    def log_validation_failure(
        self,
        check: str,
        reason: str,
        severity: str = "WARNING",
        recovery_action: Optional[str] = None,
    ) -> None:
        """Log validation failures with recovery suggestions."""
        self.logger.warning(
            f"Validation failed: {check} - {reason}",
            extra={
                "validation": {
                    "check": check,
                    "reason": reason,
                    "severity": severity,
                    "recovery_action": recovery_action,
                },
                "type": "validation_failure",
            },
        )


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    include_stdout: bool = True,
) -> None:
    """Set up structured logging for the application."""

    # Create formatters
    class StructuredFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            # Already formatted by StructuredLogger
            if (
                hasattr(record, "msg")
                and isinstance(record.msg, str)
                and record.msg.startswith("{")
            ):
                return record.msg
            # Fallback for regular loggers
            return json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "module": record.name,
                    "message": record.getMessage(),
                }
            )

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add structured formatter to all handlers
    formatter = StructuredFormatter()

    if include_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(logging.getLogger(name))


def set_execution_context(**kwargs: Any) -> None:
    """Set execution context for current async context."""
    ctx = execution_context.get()
    ctx.update(kwargs)
    execution_context.set(ctx)


def clear_execution_context() -> None:
    """Clear execution context."""
    execution_context.set({})


# Export convenience loggers
logger = get_logger(__name__)
decision_logger = DecisionLogger(get_logger("decisions"))
