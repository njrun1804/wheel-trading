"""
Fallback tracing implementation using logging.

Provides real tracing functionality when accelerated tools are not available.
"""

import asyncio
import contextvars
import json
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Context variable for tracking spans
current_span = contextvars.ContextVar("current_span", default=None)


@dataclass
class SpanData:
    """Data structure for a tracing span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    status: str = "ok"  # ok, error, timeout
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class TraceFallback:
    """Fallback tracing implementation using structured logging."""

    def __init__(self, service_name: str = "bolt", enable_console: bool = True):
        self.service_name = service_name
        self.enable_console = enable_console
        self.spans = {}
        self.completed_spans = []
        self.max_completed_spans = 1000

        # Setup tracing logger
        self.trace_logger = logging.getLogger(f"{__name__}.trace")
        self.trace_logger.setLevel(logging.INFO)

        if enable_console and not self.trace_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [TRACE] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.trace_logger.addHandler(handler)

    @contextmanager
    def trace_span(self, operation_name: str, **tags):
        """Create a tracing span context manager."""
        span_id = str(uuid4())
        parent_span = current_span.get()
        parent_span_id = parent_span.span_id if parent_span else None
        trace_id = parent_span.trace_id if parent_span else str(uuid4())

        span = SpanData(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags,
        )

        self.spans[span_id] = span

        # Set as current span
        token = current_span.set(span)

        self.trace_logger.info(f"SPAN_START: {operation_name} [{span_id}]")

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.error = str(e)
            self.trace_logger.error(f"SPAN_ERROR: {operation_name} [{span_id}] - {e}")
            raise
        finally:
            # Finish span
            span.end_time = time.time()
            span.duration_ms = (span.end_time - span.start_time) * 1000

            self.trace_logger.info(
                f"SPAN_END: {operation_name} [{span_id}] "
                f"duration={span.duration_ms:.2f}ms status={span.status}"
            )

            # Reset context
            current_span.reset(token)

            # Move to completed spans
            self.spans.pop(span_id, None)
            self.completed_spans.append(span)

            # Limit completed spans
            if len(self.completed_spans) > self.max_completed_spans:
                self.completed_spans.pop(0)

    @asynccontextmanager
    async def trace_span_async(self, operation_name: str, **tags):
        """Async version of trace_span."""
        span_id = str(uuid4())
        parent_span = current_span.get()
        parent_span_id = parent_span.span_id if parent_span else None
        trace_id = parent_span.trace_id if parent_span else str(uuid4())

        span = SpanData(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags,
        )

        self.spans[span_id] = span

        # Set as current span
        token = current_span.set(span)

        self.trace_logger.info(f"SPAN_START: {operation_name} [{span_id}]")

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.error = str(e)
            self.trace_logger.error(f"SPAN_ERROR: {operation_name} [{span_id}] - {e}")
            raise
        finally:
            # Finish span
            span.end_time = time.time()
            span.duration_ms = (span.end_time - span.start_time) * 1000

            self.trace_logger.info(
                f"SPAN_END: {operation_name} [{span_id}] "
                f"duration={span.duration_ms:.2f}ms status={span.status}"
            )

            # Reset context
            current_span.reset(token)

            # Move to completed spans
            self.spans.pop(span_id, None)
            self.completed_spans.append(span)

            # Limit completed spans
            if len(self.completed_spans) > self.max_completed_spans:
                self.completed_spans.pop(0)

    def log_event(self, message: str, level: str = "info", **attributes):
        """Log an event in the current span."""
        span = current_span.get()

        event_data = {
            "timestamp": time.time(),
            "message": message,
            "level": level,
            **attributes,
        }

        if span:
            span.logs.append(event_data)
            self.trace_logger.info(
                f"SPAN_LOG: [{span.span_id}] {level.upper()}: {message}"
            )
        else:
            self.trace_logger.info(f"EVENT: {level.upper()}: {message}")

    def add_span_tag(self, key: str, value: Any):
        """Add a tag to the current span."""
        span = current_span.get()
        if span:
            span.tags[key] = value

    def get_current_trace_id(self) -> str | None:
        """Get the current trace ID."""
        span = current_span.get()
        return span.trace_id if span else None

    def get_current_span_id(self) -> str | None:
        """Get the current span ID."""
        span = current_span.get()
        return span.span_id if span else None

    def get_trace_data(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all spans for a trace."""
        trace_spans = []

        # Check active spans
        for span in self.spans.values():
            if span.trace_id == trace_id:
                trace_spans.append(asdict(span))

        # Check completed spans
        for span in self.completed_spans:
            if span.trace_id == trace_id:
                trace_spans.append(asdict(span))

        # Sort by start time
        trace_spans.sort(key=lambda x: x["start_time"])
        return trace_spans

    def get_active_spans(self) -> list[dict[str, Any]]:
        """Get all currently active spans."""
        return [asdict(span) for span in self.spans.values()]

    def get_recent_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent completed traces."""
        # Group spans by trace_id
        traces = {}
        for span in self.completed_spans[
            -limit * 10 :
        ]:  # Get more spans to ensure we have enough traces
            trace_id = span.trace_id
            if trace_id not in traces:
                traces[trace_id] = []
            traces[trace_id].append(asdict(span))

        # Convert to list and limit
        trace_list = []
        for trace_id, spans in traces.items():
            spans.sort(key=lambda x: x["start_time"])
            root_span = min(spans, key=lambda x: x["start_time"])
            trace_list.append(
                {
                    "trace_id": trace_id,
                    "operation_name": root_span["operation_name"],
                    "start_time": root_span["start_time"],
                    "duration_ms": max(s["duration_ms"] or 0 for s in spans),
                    "span_count": len(spans),
                    "status": "error"
                    if any(s["status"] == "error" for s in spans)
                    else "ok",
                }
            )

        # Sort by start time and limit
        trace_list.sort(key=lambda x: x["start_time"], reverse=True)
        return trace_list[:limit]

    def export_traces_json(self, output_file: str, trace_ids: list[str] = None):
        """Export traces to JSON file."""
        if trace_ids:
            traces_data = {}
            for trace_id in trace_ids:
                traces_data[trace_id] = self.get_trace_data(trace_id)
        else:
            # Export all completed traces
            traces_data = {}
            for span in self.completed_spans:
                trace_id = span.trace_id
                if trace_id not in traces_data:
                    traces_data[trace_id] = []
                traces_data[trace_id].append(asdict(span))

        with open(output_file, "w") as f:
            json.dump(traces_data, f, indent=2, default=str)

        logger.info(f"Exported {len(traces_data)} traces to {output_file}")

    def clear_completed_spans(self):
        """Clear all completed spans."""
        self.completed_spans.clear()
        logger.info("Cleared all completed spans")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get tracing performance statistics."""
        if self.completed_spans:
            durations = [s.duration_ms for s in self.completed_spans if s.duration_ms]
            avg_duration = sum(durations) / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            error_count = sum(1 for s in self.completed_spans if s.status == "error")
        else:
            avg_duration = 0
            max_duration = 0
            error_count = 0

        return {
            "backend": "logging_trace",
            "service_name": self.service_name,
            "active_spans": len(self.spans),
            "completed_spans": len(self.completed_spans),
            "average_duration_ms": avg_duration,
            "max_duration_ms": max_duration,
            "error_rate": error_count / len(self.completed_spans)
            if self.completed_spans
            else 0,
            "console_output": self.enable_console,
        }


# Global instance
_trace_fallback = None


def get_trace_turbo(service_name: str = "bolt") -> TraceFallback:
    """Get global trace fallback instance."""
    global _trace_fallback
    if _trace_fallback is None:
        _trace_fallback = TraceFallback(service_name)
    return _trace_fallback


# Convenience functions
def trace_function(operation_name: str = None, **tags):
    """Decorator for tracing functions."""

    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                tracer = get_trace_turbo()
                async with tracer.trace_span_async(operation_name, **tags):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                tracer = get_trace_turbo()
                with tracer.trace_span(operation_name, **tags):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
