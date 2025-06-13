"""Unified trace implementation replacing trace, trace-opik, and trace-phoenix MCPs.
Optimized for M4 Pro with direct SDK integration."""

import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor

# Direct integrations (no MCP overhead)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Phoenix direct integration
try:
    from phoenix.trace import TraceDataset, register_dataset
    from phoenix.trace.span import Span as PhoenixSpan
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

# Opik direct integration
try:
    import opik
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False


@dataclass
class TraceConfig:
    """Configuration for trace backends."""
    phoenix_url: str = "http://localhost:6006"
    opik_url: str = "http://localhost:5173"
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "unity-wheel-trading"
    batch_size: int = 100
    export_interval_ms: int = 5000
    max_workers: int = 8  # M4 Pro performance cores


class TraceTurbo:
    """Hardware-accelerated unified tracing for M4 Pro."""
    
    def __init__(self, config: Optional[TraceConfig] = None):
        self.config = config or TraceConfig()
        
        # Load hardware optimization
        with open("optimization_config.json") as f:
            self.hw_config = json.load(f)
        
        # Thread pool for async exports (use performance cores)
        self._executor = ThreadPoolExecutor(
            max_workers=self.hw_config["cpu"]["max_workers"]
        )
        
        # Initialize providers
        self._init_providers()
        
        # Batch buffers for high throughput
        self._span_buffer = []
        self._buffer_lock = asyncio.Lock()
        
        # Start background export task
        self._export_task = asyncio.create_task(self._export_loop())
        
    def _init_providers(self):
        """Initialize all trace providers."""
        # OpenTelemetry setup
        resource = Resource.create({
            "service.name": self.config.service_name,
            "hardware.type": "M4 Pro",
            "hardware.cores": self.hw_config["cpu"]["max_workers"],
            "hardware.memory_gb": self.hw_config["memory"]["max_allocation_gb"]
        })
        
        provider = TracerProvider(resource=resource)
        
        # OTLP exporter with batching
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True
        )
        
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=10000,
            max_export_batch_size=self.config.batch_size,
            schedule_delay_millis=self.config.export_interval_ms
        )
        
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)
        
        # Initialize backend clients
        if PHOENIX_AVAILABLE:
            self.phoenix_client = PhoenixClient(self.config.phoenix_url)
        
        if OPIK_AVAILABLE:
            opik.configure(api_url=self.config.opik_url)
            self.opik_client = opik
    
    @asynccontextmanager
    async def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None,
                        span_type: str = "general") -> Dict[str, Any]:
        """Create a traced span with automatic backend routing."""
        start_time = time.perf_counter()
        
        # OpenTelemetry span
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                span.set_attributes(attributes)
            
            span_data = {
                "name": name,
                "trace_id": format(span.get_span_context().trace_id, '032x'),
                "span_id": format(span.get_span_context().span_id, '016x'),
                "start_time": datetime.utcnow().isoformat(),
                "attributes": attributes or {},
                "type": span_type
            }
            
            try:
                yield span_data
                
                # Record success
                span.set_status(trace.StatusCode.OK)
                span_data["status"] = "success"
                
            except Exception as e:
                # Record error
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)
                span_data["status"] = "error"
                span_data["error"] = str(e)
                raise
                
            finally:
                # Calculate duration
                duration_ms = (time.perf_counter() - start_time) * 1000
                span_data["duration_ms"] = duration_ms
                span.set_attribute("duration_ms", duration_ms)
                
                # Add to buffer for batch export
                async with self._buffer_lock:
                    self._span_buffer.append(span_data)
    
    async def trace_function(self, func_name: str, args: tuple = (), 
                           kwargs: dict = None) -> Any:
        """Trace a function execution with all backends."""
        kwargs = kwargs or {}
        
        attributes = {
            "function.name": func_name,
            "function.args_count": len(args),
            "function.kwargs_count": len(kwargs)
        }
        
        async with self.trace_span(f"function.{func_name}", attributes, "function") as span:
            # Execute function
            if asyncio.iscoroutinefunction(func_name):
                result = await func_name(*args, **kwargs)
            else:
                # Run in thread pool for CPU-bound functions
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor, func_name, *args
                )
            
            span["result_type"] = type(result).__name__
            return result
    
    async def trace_batch(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Trace multiple operations in parallel using all cores."""
        tasks = []
        
        for op in operations:
            name = op.get("name", "operation")
            attributes = op.get("attributes", {})
            
            async def traced_op(operation):
                async with self.trace_span(name, attributes) as span:
                    if "function" in operation:
                        return await operation["function"]()
                    else:
                        return operation
            
            tasks.append(traced_op(op))
        
        return await asyncio.gather(*tasks)
    
    async def _export_loop(self):
        """Background task to export spans in batches."""
        while True:
            await asyncio.sleep(self.config.export_interval_ms / 1000)
            
            async with self._buffer_lock:
                if not self._span_buffer:
                    continue
                    
                spans_to_export = self._span_buffer[:self.config.batch_size]
                self._span_buffer = self._span_buffer[self.config.batch_size:]
            
            # Export to different backends in parallel
            export_tasks = []
            
            if PHOENIX_AVAILABLE:
                export_tasks.append(
                    self._export_to_phoenix(spans_to_export)
                )
            
            if OPIK_AVAILABLE:
                export_tasks.append(
                    self._export_to_opik(spans_to_export)
                )
            
            if export_tasks:
                await asyncio.gather(*export_tasks, return_exceptions=True)
    
    async def _export_to_phoenix(self, spans: List[Dict[str, Any]]):
        """Export spans to Phoenix."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self.phoenix_client.export_spans,
            spans
        )
    
    async def _export_to_opik(self, spans: List[Dict[str, Any]]):
        """Export spans to Opik."""
        for span in spans:
            self.opik_client.trace(
                name=span["name"],
                tags={"type": span["type"]},
                metadata=span["attributes"],
                start_time=span["start_time"],
                end_time=datetime.utcnow().isoformat(),
                input=span.get("input"),
                output=span.get("output")
            )
    
    async def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        async with self._buffer_lock:
            buffer_size = len(self._span_buffer)
        
        return {
            "buffer_size": buffer_size,
            "backends": {
                "otlp": True,
                "phoenix": PHOENIX_AVAILABLE,
                "opik": OPIK_AVAILABLE
            },
            "performance": {
                "cpu_cores": self.hw_config["cpu"]["max_workers"],
                "batch_size": self.config.batch_size,
                "export_interval_ms": self.config.export_interval_ms
            }
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self._export_task.cancel()
        
        # Export remaining spans
        async with self._buffer_lock:
            if self._span_buffer:
                await self._export_to_phoenix(self._span_buffer)
                await self._export_to_opik(self._span_buffer)
        
        self._executor.shutdown(wait=False)


class PhoenixClient:
    """Direct Phoenix client without MCP overhead."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    def export_spans(self, spans: List[Dict[str, Any]]):
        """Export spans to Phoenix."""
        # Convert to Phoenix format
        phoenix_spans = []
        
        for span in spans:
            ps = PhoenixSpan(
                name=span["name"],
                span_id=span["span_id"],
                trace_id=span["trace_id"],
                start_time=span["start_time"],
                latency_ms=span.get("duration_ms", 0),
                status=span.get("status", "success"),
                attributes=span["attributes"]
            )
            phoenix_spans.append(ps)
        
        # Register dataset
        dataset = TraceDataset(phoenix_spans)
        register_dataset(dataset, name="unity-wheel-traces")


# Singleton instance
_trace_instance: Optional[TraceTurbo] = None


def get_trace_turbo(config: Optional[TraceConfig] = None) -> TraceTurbo:
    """Get or create the turbo trace instance."""
    global _trace_instance
    if _trace_instance is None:
        _trace_instance = TraceTurbo(config)
    return _trace_instance


# Drop-in replacements for MCP functions
async def start_trace(name: str, attributes: Optional[Dict[str, Any]] = None) -> str:
    """Drop-in replacement for MCP trace.start_trace."""
    tracer = get_trace_turbo()
    
    async with tracer.trace_span(name, attributes) as span:
        return f"Trace started: {span['trace_id']}"


async def end_trace(trace_id: str, status: str = "success") -> str:
    """Drop-in replacement for MCP trace.end_trace."""
    # In the new model, traces are auto-ended with context manager
    return f"Trace {trace_id} ended with status: {status}"


async def add_span_attribute(key: str, value: Any) -> str:
    """Drop-in replacement for MCP trace.add_span_attribute."""
    # Get current span from OpenTelemetry context
    current_span = trace.get_current_span()
    if current_span:
        current_span.set_attribute(key, value)
        return f"Added attribute {key}={value}"
    return "No active span"