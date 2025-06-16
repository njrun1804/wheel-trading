#!/usr/bin/env python3
"""
Comprehensive logging configuration for the Bolt 8-Agent System.

Provides structured logging with multiple levels, performance metrics, error tracking,
hardware monitoring integration, and debugging modes optimized for M4 Pro hardware.

Features:
- Structured JSON logging with context
- Performance-aware logging with minimal overhead
- Hardware-specific log categories
- Error correlation and tracking
- GPU/MLX operation logging
- Memory pressure alerts
- Agent execution tracing
- Einstein search operation logging
- Real-time log streaming for debugging
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import psutil


class LogLevel(Enum):
    """Log levels for different system components."""

    TRACE = 5  # Extremely detailed tracing
    DEBUG = 10  # Debug information
    INFO = 20  # General information
    WARNING = 30  # Warning conditions
    ERROR = 40  # Error conditions
    CRITICAL = 50  # Critical errors


class LogCategory(Enum):
    """Categories for different types of operations."""

    SYSTEM = "system"
    HARDWARE = "hardware"
    AGENT = "agent"
    GPU = "gpu"
    MEMORY = "memory"
    EINSTEIN = "einstein"
    TASK = "task"
    PERFORMANCE = "performance"
    ERROR = "error"
    NETWORK = "network"
    DATABASE = "database"
    TOOLS = "tools"


@dataclass
class LogContext:
    """Context information for structured logging."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str | None = None
    task_id: str | None = None
    operation: str | None = None
    category: LogCategory = LogCategory.SYSTEM
    thread_id: int | None = None
    process_id: int = field(default_factory=os.getpid)
    timestamp: float = field(default_factory=time.time)

    # Hardware context
    cpu_percent: float | None = None
    memory_percent: float | None = None
    gpu_memory_gb: float | None = None

    # Performance context
    duration_ms: float | None = None
    error_count: int = 0
    retry_count: int = 0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """Tracks performance metrics for logging optimization."""

    def __init__(self):
        self.operation_times: dict[str, list[float]] = {}
        self.error_counts: dict[str, int] = {}
        self.memory_samples: list[float] = []
        self.cpu_samples: list[float] = []

    def record_operation(self, operation: str, duration_ms: float):
        """Record operation timing."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(duration_ms)

        # Keep only last 100 samples per operation
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation] = self.operation_times[operation][-100:]

    def record_error(self, category: str):
        """Record error occurrence."""
        self.error_counts[category] = self.error_counts.get(category, 0) + 1

    def record_system_metrics(self, cpu_percent: float, memory_percent: float):
        """Record system metrics for trend analysis."""
        self.cpu_samples.append(cpu_percent)
        self.memory_samples.append(memory_percent)

        # Keep only last 1000 samples
        if len(self.cpu_samples) > 1000:
            self.cpu_samples = self.cpu_samples[-1000:]
            self.memory_samples = self.memory_samples[-1000:]

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "operations": {},
            "errors": dict(self.error_counts),
            "system_trends": {
                "avg_cpu": sum(self.cpu_samples) / len(self.cpu_samples)
                if self.cpu_samples
                else 0,
                "avg_memory": sum(self.memory_samples) / len(self.memory_samples)
                if self.memory_samples
                else 0,
            },
        }

        for op, times in self.operation_times.items():
            if times:
                stats["operations"][op] = {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "p95_ms": sorted(times)[int(len(times) * 0.95)]
                    if len(times) > 20
                    else max(times),
                }

        return stats


# Global context variable for log correlation
_log_context: ContextVar[LogContext | None] = ContextVar("log_context", default=None)

# Global performance tracker
_performance_tracker = PerformanceTracker()


class BoltFormatter(logging.Formatter):
    """Custom formatter for structured Bolt logging."""

    def __init__(self, include_hardware_info: bool = True):
        super().__init__()
        self.include_hardware_info = include_hardware_info

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data."""
        # Get current context
        context = _log_context.get()

        # Build log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context information
        if context:
            log_entry.update(
                {
                    "session_id": context.session_id,
                    "agent_id": context.agent_id,
                    "task_id": context.task_id,
                    "operation": context.operation,
                    "category": context.category.value if context.category else None,
                    "process_id": context.process_id,
                    "duration_ms": context.duration_ms,
                    "error_count": context.error_count,
                    "retry_count": context.retry_count,
                }
            )

            # Add hardware info if available and enabled
            if self.include_hardware_info:
                if context.cpu_percent is not None:
                    log_entry["cpu_percent"] = context.cpu_percent
                if context.memory_percent is not None:
                    log_entry["memory_percent"] = context.memory_percent
                if context.gpu_memory_gb is not None:
                    log_entry["gpu_memory_gb"] = context.gpu_memory_gb

            # Add metadata
            if context.metadata:
                log_entry["metadata"] = context.metadata

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
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
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, default=str, separators=(",", ":"))


class HardwareAwareHandler(logging.Handler):
    """Logging handler that adapts based on system performance."""

    def __init__(self, base_handler: logging.Handler, memory_threshold: float = 85.0):
        super().__init__()
        self.base_handler = base_handler
        self.memory_threshold = memory_threshold
        self.last_memory_check = 0
        self.memory_check_interval = 5.0  # seconds
        self.high_memory_mode = False

    def emit(self, record: logging.LogRecord):
        """Emit log record with memory awareness."""
        current_time = time.time()

        # Check memory usage periodically
        if current_time - self.last_memory_check > self.memory_check_interval:
            try:
                memory_percent = psutil.virtual_memory().percent
                self.high_memory_mode = memory_percent > self.memory_threshold
                self.last_memory_check = current_time

                # Record system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                _performance_tracker.record_system_metrics(cpu_percent, memory_percent)

            except Exception:
                pass  # Don't fail logging due to memory check errors

        # In high memory mode, reduce log verbosity
        if self.high_memory_mode and record.levelno < logging.WARNING:
            return

        # Emit through base handler
        try:
            self.base_handler.emit(record)
        except Exception:
            self.handleError(record)

    def setFormatter(self, formatter):
        """Set formatter on base handler."""
        self.base_handler.setFormatter(formatter)

    def close(self):
        """Close base handler."""
        self.base_handler.close()
        super().close()


class BoltLogger:
    """Main logger class for the Bolt system."""

    def __init__(
        self,
        name: str = "bolt",
        log_dir: Path | None = None,
        log_level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_performance_logging: bool = True,
        max_file_size_mb: int = 50,
        backup_count: int = 5,
        enable_hardware_monitoring: bool = True,
    ):
        self.name = name
        self.log_dir = log_dir or Path.cwd() / "logs"
        self.log_level = log_level
        self.enable_performance_logging = enable_performance_logging
        self.enable_hardware_monitoring = enable_hardware_monitoring

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.value)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_handlers(
            enable_console, enable_file, max_file_size_mb, backup_count
        )

        # Setup performance monitoring
        if enable_performance_logging:
            self._setup_performance_monitoring()

    def _setup_handlers(
        self,
        enable_console: bool,
        enable_file: bool,
        max_file_size_mb: int,
        backup_count: int,
    ):
        """Setup logging handlers."""
        formatter = BoltFormatter(include_hardware_info=self.enable_hardware_monitoring)

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level.value)

            # Use hardware-aware handler
            hw_console_handler = HardwareAwareHandler(console_handler)
            hw_console_handler.setFormatter(formatter)
            self.logger.addHandler(hw_console_handler)

        # File handlers
        if enable_file:
            # Main log file
            main_log_file = self.log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
            )
            file_handler.setLevel(self.log_level.value)

            hw_file_handler = HardwareAwareHandler(file_handler)
            hw_file_handler.setFormatter(formatter)
            self.logger.addHandler(hw_file_handler)

            # Error log file (WARNING and above)
            error_log_file = self.log_dir / f"{self.name}_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
            )
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(formatter)
            self.logger.addHandler(error_handler)

            # Performance log file (if enabled)
            if self.enable_performance_logging:
                perf_log_file = self.log_dir / f"{self.name}_performance.log"
                perf_handler = logging.handlers.RotatingFileHandler(
                    perf_log_file,
                    maxBytes=max_file_size_mb * 1024 * 1024,
                    backupCount=backup_count,
                )
                perf_handler.setLevel(LogLevel.TRACE.value)

                # Create performance-specific formatter
                perf_formatter = BoltFormatter(include_hardware_info=True)
                perf_handler.setFormatter(perf_formatter)

                # Only log performance-related messages
                perf_handler.addFilter(
                    lambda record: hasattr(record, "category")
                    and record.category == LogCategory.PERFORMANCE.value
                )

                self.logger.addHandler(perf_handler)

    def _setup_performance_monitoring(self):
        """Setup performance monitoring task."""
        asyncio.create_task(self._performance_monitoring_loop())

    async def _performance_monitoring_loop(self):
        """Background task for performance monitoring."""
        while True:
            try:
                # Log performance stats every 60 seconds
                await asyncio.sleep(60)

                stats = _performance_tracker.get_stats()
                self.performance(
                    "System performance stats",
                    extra={"category": LogCategory.PERFORMANCE.value, "stats": stats},
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error(f"Performance monitoring error: {e}")

    def trace(self, message: str, **kwargs):
        """Log trace level message."""
        self.logger.log(LogLevel.TRACE.value, message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info level message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error level message."""
        self.logger.error(message, **kwargs)

        # Track error in performance metrics
        context = _log_context.get()
        category = context.category.value if context and context.category else "unknown"
        _performance_tracker.record_error(category)

    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self.logger.critical(message, **kwargs)

        # Track critical error
        context = _log_context.get()
        category = context.category.value if context and context.category else "unknown"
        _performance_tracker.record_error(f"critical_{category}")

    def performance(
        self, message: str, operation: str = None, duration_ms: float = None, **kwargs
    ):
        """Log performance information."""
        if operation and duration_ms:
            _performance_tracker.record_operation(operation, duration_ms)

        self.logger.log(
            LogLevel.TRACE.value,
            message,
            extra={
                "category": LogCategory.PERFORMANCE.value,
                "operation": operation,
                "duration_ms": duration_ms,
                **kwargs,
            },
        )

    def hardware_event(self, message: str, **kwargs):
        """Log hardware-related event."""
        self.logger.info(
            message, extra={"category": LogCategory.HARDWARE.value, **kwargs}
        )

    def agent_event(self, message: str, agent_id: str, **kwargs):
        """Log agent-related event."""
        self.logger.info(
            message,
            extra={"category": LogCategory.AGENT.value, "agent_id": agent_id, **kwargs},
        )

    def task_event(self, message: str, task_id: str, **kwargs):
        """Log task-related event."""
        self.logger.info(
            message,
            extra={"category": LogCategory.TASK.value, "task_id": task_id, **kwargs},
        )

    def gpu_event(self, message: str, **kwargs):
        """Log GPU-related event."""
        self.logger.info(message, extra={"category": LogCategory.GPU.value, **kwargs})

    def memory_event(self, message: str, **kwargs):
        """Log memory-related event."""
        self.logger.info(
            message, extra={"category": LogCategory.MEMORY.value, **kwargs}
        )

    def einstein_event(self, message: str, **kwargs):
        """Log Einstein search-related event."""
        self.logger.info(
            message, extra={"category": LogCategory.EINSTEIN.value, **kwargs}
        )


class LogContextManager:
    """Context manager for setting log context."""

    def __init__(self, context: LogContext):
        self.context = context
        self.token = None

    def __enter__(self):
        self.token = _log_context.set(self.context)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            _log_context.reset(self.token)


class OperationTimer:
    """Context manager for timing operations with automatic logging."""

    def __init__(
        self, logger: BoltLogger, operation: str, log_level: LogLevel = LogLevel.DEBUG
    ):
        self.logger = logger
        self.operation = operation
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000

            if exc_type:
                self.logger.error(
                    f"Operation {self.operation} failed after {duration_ms:.2f}ms: {exc_val}"
                )
            else:
                message = f"Operation {self.operation} completed in {duration_ms:.2f}ms"

                if self.log_level == LogLevel.DEBUG:
                    self.logger.debug(message)
                elif self.log_level == LogLevel.INFO:
                    self.logger.info(message)
                elif self.log_level == LogLevel.TRACE:
                    self.logger.trace(message)

                # Always log to performance tracker
                self.logger.performance(
                    message, operation=self.operation, duration_ms=duration_ms
                )


# Global logger instance
_default_logger: BoltLogger | None = None


def get_logger(name: str = "bolt", **kwargs) -> BoltLogger:
    """Get or create a logger instance."""
    global _default_logger

    if _default_logger is None or _default_logger.name != name:
        _default_logger = BoltLogger(name=name, **kwargs)

    return _default_logger


def set_log_context(**kwargs) -> LogContextManager:
    """Set logging context for structured logging."""
    current_context = _log_context.get()

    # Create new context with updated values
    if current_context:
        context_dict = asdict(current_context)
        context_dict.update(kwargs)
        new_context = LogContext(**context_dict)
    else:
        new_context = LogContext(**kwargs)

    return LogContextManager(new_context)


def time_operation(
    operation: str, logger: BoltLogger = None, log_level: LogLevel = LogLevel.DEBUG
):
    """Decorator/context manager for timing operations."""
    if logger is None:
        logger = get_logger()

    return OperationTimer(logger, operation, log_level)


def configure_bolt_logging(
    log_level: LogLevel = LogLevel.INFO,
    log_dir: Path | None = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_performance_logging: bool = True,
    enable_hardware_monitoring: bool = True,
    debug_mode: bool = False,
) -> BoltLogger:
    """Configure logging for the entire Bolt system."""

    if debug_mode:
        log_level = LogLevel.DEBUG
        enable_performance_logging = True
        enable_hardware_monitoring = True

    logger = BoltLogger(
        name="bolt",
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_performance_logging=enable_performance_logging,
        enable_hardware_monitoring=enable_hardware_monitoring,
    )

    # Configure standard library logging to use our system
    logging.getLogger().addHandler(logging.NullHandler())

    return logger


# Performance helper functions
def log_memory_usage(logger: BoltLogger, operation: str = "memory_check"):
    """Log current memory usage."""
    try:
        vm = psutil.virtual_memory()
        logger.memory_event(
            f"Memory usage: {vm.percent:.1f}% ({vm.used / (1024**3):.1f}GB / {vm.total / (1024**3):.1f}GB)",
            operation=operation,
            memory_percent=vm.percent,
            memory_used_gb=vm.used / (1024**3),
            memory_total_gb=vm.total / (1024**3),
            memory_available_gb=vm.available / (1024**3),
        )
    except Exception as e:
        logger.error(f"Failed to log memory usage: {e}")


def log_cpu_usage(logger: BoltLogger, operation: str = "cpu_check"):
    """Log current CPU usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()

        logger.hardware_event(
            f"CPU usage: {cpu_percent:.1f}% on {cpu_count} cores",
            operation=operation,
            cpu_percent=cpu_percent,
            cpu_cores=cpu_count,
            cpu_freq_mhz=cpu_freq.current if cpu_freq else None,
        )
    except Exception as e:
        logger.error(f"Failed to log CPU usage: {e}")


def log_system_health(logger: BoltLogger):
    """Log comprehensive system health check."""
    with set_log_context(operation="system_health_check", category=LogCategory.SYSTEM):
        log_memory_usage(logger, "health_check_memory")
        log_cpu_usage(logger, "health_check_cpu")

        # Log disk usage
        try:
            disk = psutil.disk_usage("/")
            logger.hardware_event(
                f"Disk usage: {disk.percent:.1f}% ({disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB)",
                operation="health_check_disk",
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024**3),
                disk_total_gb=disk.total / (1024**3),
            )
        except Exception as e:
            logger.error(f"Failed to log disk usage: {e}")


# Example usage and test functions
async def test_logging_system():
    """Test the logging system with various scenarios."""
    logger = configure_bolt_logging(
        log_level=LogLevel.DEBUG, debug_mode=True, log_dir=Path("test_logs")
    )

    # Test basic logging
    logger.info("Starting logging system test")

    # Test context management
    with set_log_context(agent_id="test_agent", operation="test_operation"):
        logger.debug("Testing context logging")

        # Test operation timing
        with time_operation("test_timed_operation", logger):
            await asyncio.sleep(0.1)  # Simulate work

        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            logger.error(f"Caught test error: {e}", exc_info=True)

    # Test hardware logging
    log_system_health(logger)

    # Test performance logging
    logger.performance(
        "Test performance message", operation="test_perf", duration_ms=15.5
    )

    # Test different event types
    logger.agent_event("Agent started", agent_id="agent_1")
    logger.task_event("Task queued", task_id="task_123")
    logger.gpu_event("GPU operation completed")
    logger.memory_event("Memory allocation successful")
    logger.einstein_event("Search completed", query="test query")

    logger.info("Logging system test completed")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_logging_system())
