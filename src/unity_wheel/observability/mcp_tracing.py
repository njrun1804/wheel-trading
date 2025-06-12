"""
OpenTelemetry tracing for MCP operations.
Tracks: ripgrep scan, graph build, DuckDB query, embed call, LLM call.
"""

import time
import functools
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.semconv.trace import SpanAttributes

# Initialize tracer
resource = Resource.create({
    "service.name": "wheel-trading-mcp",
    "service.version": "1.0.0",
    "deployment.environment": "local"
})

provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Configure OTLP exporter (to Phoenix or Jaeger)
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318",
    insecure=True
)

# Add batch processor for performance
span_processor = BatchSpanProcessor(
    otlp_exporter,
    max_queue_size=2048,
    max_export_batch_size=256,
    export_timeout_millis=5000
)
provider.add_span_processor(span_processor)

# Get tracer
tracer = trace.get_tracer("mcp.operations", "1.0.0")

@contextmanager
def trace_operation(operation: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for tracing MCP operations."""
    with tracer.start_as_current_span(operation) as span:
        # Add common attributes
        span.set_attribute("mcp.operation", operation)
        
        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(f"mcp.{key}", value)
        
        start_time = time.time()
        try:
            yield span
            # Success - record duration
            duration_ms = (time.time() - start_time) * 1000
            span.set_attribute("mcp.duration_ms", duration_ms)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            # Error - record exception
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

def trace_mcp_call(operation: str):
    """Decorator for tracing MCP function calls."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract meaningful attributes from args/kwargs
            attributes = {
                "function": func.__name__,
                "module": func.__module__,
            }
            
            # Add specific attributes based on operation
            if operation == "ripgrep.scan":
                if args:
                    attributes["pattern"] = str(args[0])[:100]  # First 100 chars
                if "path" in kwargs:
                    attributes["path"] = kwargs["path"]
            elif operation == "duckdb.query":
                if args:
                    attributes["query_length"] = len(str(args[0]))
            elif operation == "dependency_graph.build":
                if "file_count" in kwargs:
                    attributes["file_count"] = kwargs["file_count"]
            
            with trace_operation(operation, attributes):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Specific trace decorators for each MCP operation
trace_ripgrep = trace_mcp_call("ripgrep.scan")
trace_dependency_graph = trace_mcp_call("dependency_graph.build")
trace_duckdb_query = trace_mcp_call("duckdb.query")
trace_embedding_call = trace_mcp_call("embedding.generate")
trace_llm_call = trace_mcp_call("llm.inference")

# Example spans for common patterns
@contextmanager
def trace_mcp_workflow(workflow_name: str, total_files: int = 0):
    """Trace an entire MCP workflow with nested spans."""
    with tracer.start_as_current_span(f"mcp.workflow.{workflow_name}") as span:
        span.set_attribute("workflow.name", workflow_name)
        span.set_attribute("workflow.total_files", total_files)
        
        workflow_start = time.time()
        
        # Track individual step timings
        step_timings = {}
        
        def record_step(step_name: str, duration_ms: float):
            step_timings[step_name] = duration_ms
            span.add_event(f"Step completed: {step_name}", {
                "step.duration_ms": duration_ms
            })
        
        span.record_step = record_step
        
        try:
            yield span
            
            # Record total workflow time
            total_duration_ms = (time.time() - workflow_start) * 1000
            span.set_attribute("workflow.duration_ms", total_duration_ms)
            
            # Add step timing summary
            for step, duration in step_timings.items():
                span.set_attribute(f"workflow.step.{step}_ms", duration)
            
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

# Utility functions for performance monitoring
def record_token_usage(span: trace.Span, prompt_tokens: int, completion_tokens: int):
    """Record token usage metrics on a span."""
    span.set_attribute("llm.prompt_tokens", prompt_tokens)
    span.set_attribute("llm.completion_tokens", completion_tokens)
    span.set_attribute("llm.total_tokens", prompt_tokens + completion_tokens)

def record_db_metrics(span: trace.Span, rows_scanned: int, rows_returned: int, cache_hit: bool = False):
    """Record database query metrics."""
    span.set_attribute("db.rows_scanned", rows_scanned)
    span.set_attribute("db.rows_returned", rows_returned)
    span.set_attribute("db.cache_hit", cache_hit)
    
    if rows_scanned > 0:
        selectivity = rows_returned / rows_scanned
        span.set_attribute("db.selectivity", selectivity)

# Performance SLO monitoring
class LatencySLOMonitor:
    """Monitor latency SLOs for MCP operations."""
    
    def __init__(self, slo_ms: Dict[str, float]):
        self.slo_ms = slo_ms
        self.violations = []
    
    def check_slo(self, operation: str, duration_ms: float) -> bool:
        """Check if operation met its SLO."""
        if operation in self.slo_ms:
            slo = self.slo_ms[operation]
            if duration_ms > slo:
                self.violations.append({
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "slo_ms": slo,
                    "timestamp": time.time()
                })
                return False
        return True
    
    def get_p95_violations(self) -> Dict[str, float]:
        """Get operations that violate SLO > 5% of the time."""
        # Implementation would track rolling window
        # For now, return empty dict
        return {}

# Default SLOs for MCP operations (milliseconds)
DEFAULT_SLOS = {
    "ripgrep.scan": 500,
    "dependency_graph.build": 1000,
    "duckdb.query": 2000,
    "embedding.generate": 1000,
    "llm.inference": 25000,  # 25 seconds for end-to-end
}

slo_monitor = LatencySLOMonitor(DEFAULT_SLOS)