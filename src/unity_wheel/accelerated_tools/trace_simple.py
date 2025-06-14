"""Simplified trace implementation for testing - no external dependencies."""

import asyncio
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor


@dataclass
class TraceConfig:
    """Configuration for trace backends."""
    service_name: str = "unity-wheel-trading"
    batch_size: int = 100
    export_interval_ms: int = 5000
    max_workers: int = 8  # M4 Pro performance cores


class TraceTurbo:
    """Simplified hardware-accelerated tracing."""
    
    def __init__(self, config: Optional[TraceConfig] = None):
        self.config = config or TraceConfig()
        
        # Load hardware optimization
        with open("optimization_config.json") as f:
            self.hw_config = json.load(f)
        
        # Thread pool for async exports
        self._executor = ThreadPoolExecutor(
            max_workers=self.hw_config["cpu"]["max_workers"]
        )
        
        # Batch buffers
        self._span_buffer = []
        self._buffer_lock = asyncio.Lock()
        
        # Trace storage (in-memory for testing)
        self._traces = []
        
        # Start background export task when there's an event loop
        self._export_task = None
        self._started = False
        
    async def _ensure_started(self):
        """Ensure background task is started."""
        if not self._started:
            self._started = True
            self._export_task = asyncio.create_task(self._export_loop())
    
    @asynccontextmanager
    async def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None,
                        span_type: str = "general") -> Dict[str, Any]:
        """Create a traced span."""
        await self._ensure_started()
        start_time = time.perf_counter()
        
        # Create span data
        span_data = {
            "name": name,
            "trace_id": f"trace_{int(time.time() * 1000000)}",
            "span_id": f"span_{int(time.time() * 1000000)}_{id(self)}",
            "start_time": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
            "type": span_type
        }
        
        try:
            yield span_data
            span_data["status"] = "success"
            
        except Exception as e:
            span_data["status"] = "error"
            span_data["error"] = str(e)
            raise
            
        finally:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            span_data["duration_ms"] = duration_ms
            
            # Add to buffer for batch export
            async with self._buffer_lock:
                self._span_buffer.append(span_data)
    
    async def trace_function(self, func_name: str, args: tuple = (), 
                           kwargs: dict = None) -> Any:
        """Trace a function execution."""
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
        """Trace multiple operations in parallel."""
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
            try:
                await asyncio.sleep(self.config.export_interval_ms / 1000)
                
                async with self._buffer_lock:
                    if not self._span_buffer:
                        continue
                        
                    spans_to_export = self._span_buffer[:self.config.batch_size]
                    self._span_buffer = self._span_buffer[self.config.batch_size:]
                
                # "Export" to in-memory storage
                self._traces.extend(spans_to_export)
                
            except asyncio.CancelledError:
                break
            except Exception:
                pass  # Ignore export errors
    
    async def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        async with self._buffer_lock:
            buffer_size = len(self._span_buffer)
        
        return {
            "buffer_size": buffer_size,
            "exported_traces": len(self._traces),
            "backends": {
                "in_memory": True,
                "otlp": False,
                "phoenix": False,
                "opik": False
            },
            "performance": {
                "cpu_cores": self.hw_config["cpu"]["max_workers"],
                "batch_size": self.config.batch_size,
                "export_interval_ms": self.config.export_interval_ms
            }
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
        
        # Export remaining spans
        async with self._buffer_lock:
            self._traces.extend(self._span_buffer)
            self._span_buffer.clear()
        
        self._executor.shutdown(wait=False)


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
    
    # Create a dummy span
    span_data = {
        "name": name,
        "trace_id": f"trace_{int(time.time() * 1000000)}",
        "attributes": attributes or {}
    }
    
    async with tracer._buffer_lock:
        tracer._span_buffer.append(span_data)
    
    return f"Trace started: {span_data['trace_id']}"


async def end_trace(trace_id: str, status: str = "success") -> str:
    """Drop-in replacement for MCP trace.end_trace."""
    return f"Trace {trace_id} ended with status: {status}"


async def add_span_attribute(key: str, value: Any) -> str:
    """Drop-in replacement for MCP trace.add_span_attribute."""
    return f"Added attribute {key}={value}"