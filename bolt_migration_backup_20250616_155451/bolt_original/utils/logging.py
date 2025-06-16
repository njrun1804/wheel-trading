"""
Logging configuration for Bolt system.

Provides centralized logging setup with appropriate handlers and formatters.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Setup logging configuration for Bolt system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """

    # Create root logger for Bolt
    logger = logging.getLogger("bolt")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "bolt") -> logging.Logger:
    """Get a logger instance for Bolt subsystem."""
    return logging.getLogger(name)


def setup_component_logger(component_name: str) -> logging.Logger:
    """Setup a logger for a specific Bolt component."""
    logger_name = f"bolt.{component_name}"
    return logging.getLogger(logger_name)


class BoltLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds Bolt-specific context."""

    def process(self, msg, kwargs):
        """Add context information to log messages."""
        context = self.extra

        # Add component and operation context if available
        if "component" in context:
            msg = f"[{context['component']}] {msg}"

        if "operation" in context:
            msg = f"({context['operation']}) {msg}"

        return msg, kwargs


def get_component_logger(component_name: str, **extra_context) -> BoltLoggerAdapter:
    """Get a logger adapter for a specific component with context."""
    base_logger = get_logger()

    context = {"component": component_name, **extra_context}

    return BoltLoggerAdapter(base_logger, context)


# Performance logging utilities
class PerformanceLogger:
    """Logger for performance metrics and timing."""

    def __init__(self, component_name: str):
        self.logger = get_component_logger(component_name)
        self.timings = {}

    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""
        import time

        self.timings[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")

    def end_timing(self, operation: str) -> float:
        """End timing an operation and log the duration."""
        import time

        if operation not in self.timings:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0

        duration = time.time() - self.timings[operation]
        del self.timings[operation]

        self.logger.info(f"Operation '{operation}' completed in {duration:.3f}s")
        return duration

    def log_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log a performance metric."""
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"Metric {metric_name}: {value:.3f}{unit_str}")


def configure_third_party_logging():
    """Configure logging for third-party libraries used by Bolt."""

    # Reduce verbosity of common noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # MLX and torch logging
    logging.getLogger("mlx").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
