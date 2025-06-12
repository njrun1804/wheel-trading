#!/usr/bin/env python3
"""Enhanced trace MCP server for debugging and logging."""

from mcp.server import FastMCP
import json
import time
from datetime import datetime
from pathlib import Path
import traceback
import sys
import os

mcp = FastMCP("trace")

# Create trace directory
TRACE_DIR = Path.home() / ".local" / "share" / "wheel-trading" / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

@mcp.tool()
def trace_log(
    message: str,
    level: str = "INFO",
    context: dict = None
) -> str:
    """Log a trace message with context.
    
    Args:
        message: The message to log
        level: Log level (DEBUG, INFO, WARN, ERROR)
        context: Additional context data
    """
    timestamp = datetime.now().isoformat()
    trace_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "context": context or {}
    }
    
    # Write to trace file
    trace_file = TRACE_DIR / f"trace_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(trace_file, 'a') as f:
        f.write(json.dumps(trace_entry) + '\n')
    
    return f"Traced at {timestamp}: [{level}] {message}"

@mcp.tool()
def trace_error(
    error_type: str,
    error_message: str,
    stack_trace: str = None
) -> str:
    """Log an error with full stack trace.
    
    Args:
        error_type: Type of error
        error_message: Error message
        stack_trace: Optional stack trace
    """
    if not stack_trace:
        stack_trace = traceback.format_exc()
    
    context = {
        "error_type": error_type,
        "stack_trace": stack_trace,
        "python_version": sys.version,
        "cwd": os.getcwd()
    }
    
    return trace_log(error_message, "ERROR", context)

@mcp.tool()
def trace_performance(
    operation: str,
    duration_ms: float,
    metadata: dict = None
) -> str:
    """Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        metadata: Additional performance metadata
    """
    context = {
        "operation": operation,
        "duration_ms": duration_ms,
        "metadata": metadata or {}
    }
    
    return trace_log(f"Performance: {operation} took {duration_ms}ms", "INFO", context)

@mcp.tool()
def get_traces(
    level: str = None,
    limit: int = 100,
    date: str = None
) -> str:
    """Retrieve recent traces.
    
    Args:
        level: Filter by log level
        limit: Maximum number of traces to return
        date: Date to retrieve traces for (YYYYMMDD format)
    """
    if date:
        trace_file = TRACE_DIR / f"trace_{date}.jsonl"
    else:
        trace_file = TRACE_DIR / f"trace_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    if not trace_file.exists():
        return "No traces found for the specified date"
    
    traces = []
    with open(trace_file, 'r') as f:
        for line in f:
            try:
                trace = json.loads(line.strip())
                if not level or trace.get('level') == level:
                    traces.append(trace)
            except json.JSONDecodeError:
                continue
    
    # Return most recent traces up to limit
    traces = traces[-limit:]
    return json.dumps(traces, indent=2)

@mcp.tool()
def clear_traces(date: str = None) -> str:
    """Clear trace logs.
    
    Args:
        date: Specific date to clear (YYYYMMDD format), or all if not specified
    """
    if date:
        trace_file = TRACE_DIR / f"trace_{date}.jsonl"
        if trace_file.exists():
            trace_file.unlink()
            return f"Cleared traces for {date}"
        return f"No traces found for {date}"
    else:
        # Clear all trace files
        count = 0
        for trace_file in TRACE_DIR.glob("trace_*.jsonl"):
            trace_file.unlink()
            count += 1
        return f"Cleared {count} trace files"

if __name__ == "__main__":
    print("Starting enhanced trace MCP server...")
    mcp.run()
