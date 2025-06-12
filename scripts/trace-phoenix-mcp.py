#!/usr/bin/env python3
"""Phoenix trace MCP server for rich observability."""

from mcp.server import FastMCP
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

mcp = FastMCP("trace-phoenix")

# Phoenix configuration
PHOENIX_BASE_URL = os.environ.get("PHOENIX_BASE_URL", "http://localhost:6006")
PHOENIX_API_URL = f"{PHOENIX_BASE_URL}/v1"

@mcp.tool()
def get_recent_traces(minutes: int = 5, service: str = "wheel-trading") -> str:
    """Get recent traces from Phoenix.
    
    Args:
        minutes: Look back this many minutes (default: 5)
        service: Service name to filter (default: wheel-trading)
    """
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        
        params = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "service.name": service,
            "limit": 100
        }
        
        response = requests.get(f"{PHOENIX_API_URL}/traces", params=params)
        response.raise_for_status()
        
        traces = response.json()
        
        if not traces:
            return f"No traces found in the last {minutes} minutes"
        
        result = f"Found {len(traces)} traces in the last {minutes} minutes:\n\n"
        
        for trace in traces[:10]:  # Show first 10
            span_name = trace.get("name", "Unknown")
            duration_ms = trace.get("duration_ms", 0)
            status = trace.get("status", {}).get("code", "OK")
            start = trace.get("start_time", "")
            
            result += f"• {span_name}\n"
            result += f"  Duration: {duration_ms}ms\n"
            result += f"  Status: {status}\n"
            result += f"  Start: {start}\n\n"
        
        if len(traces) > 10:
            result += f"... and {len(traces) - 10} more traces"
        
        return result
        
    except Exception as e:
        return f"Error fetching traces: {str(e)}"

@mcp.tool()
def analyze_slow_operations(threshold_ms: int = 1000) -> str:
    """Find operations slower than threshold.
    
    Args:
        threshold_ms: Minimum duration in milliseconds (default: 1000)
    """
    try:
        params = {
            "min_duration_ms": threshold_ms,
            "limit": 50,
            "sort": "duration_desc"
        }
        
        response = requests.get(f"{PHOENIX_API_URL}/traces", params=params)
        response.raise_for_status()
        
        traces = response.json()
        
        if not traces:
            return f"No operations found slower than {threshold_ms}ms"
        
        result = f"Found {len(traces)} slow operations (>{threshold_ms}ms):\n\n"
        
        # Group by operation name
        slow_ops = {}
        for trace in traces:
            name = trace.get("name", "Unknown")
            duration = trace.get("duration_ms", 0)
            
            if name not in slow_ops:
                slow_ops[name] = []
            slow_ops[name].append(duration)
        
        # Sort by average duration
        sorted_ops = sorted(
            slow_ops.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True
        )
        
        for op_name, durations in sorted_ops[:10]:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            count = len(durations)
            
            result += f"• {op_name}\n"
            result += f"  Count: {count}\n"
            result += f"  Avg: {avg_duration:.0f}ms\n"
            result += f"  Max: {max_duration:.0f}ms\n\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing slow operations: {str(e)}"

@mcp.tool()
def get_error_traces(minutes: int = 30) -> str:
    """Get traces with errors in the last N minutes.
    
    Args:
        minutes: Look back this many minutes (default: 30)
    """
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        
        params = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "status.code": "ERROR",
            "limit": 100
        }
        
        response = requests.get(f"{PHOENIX_API_URL}/traces", params=params)
        response.raise_for_status()
        
        traces = response.json()
        
        if not traces:
            return f"No error traces found in the last {minutes} minutes"
        
        result = f"Found {len(traces)} error traces:\n\n"
        
        # Group errors by type
        errors_by_type = {}
        for trace in traces:
            error_msg = trace.get("status", {}).get("message", "Unknown error")
            name = trace.get("name", "Unknown")
            
            key = f"{name}: {error_msg}"
            if key not in errors_by_type:
                errors_by_type[key] = 0
            errors_by_type[key] += 1
        
        # Sort by frequency
        sorted_errors = sorted(
            errors_by_type.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for error_desc, count in sorted_errors[:10]:
            result += f"• {error_desc}\n"
            result += f"  Count: {count}\n\n"
        
        return result
        
    except Exception as e:
        return f"Error fetching error traces: {str(e)}"

@mcp.tool()
def get_span_details(trace_id: str, span_id: str) -> str:
    """Get detailed information about a specific span.
    
    Args:
        trace_id: The trace ID
        span_id: The span ID
    """
    try:
        response = requests.get(f"{PHOENIX_API_URL}/traces/{trace_id}/spans/{span_id}")
        response.raise_for_status()
        
        span = response.json()
        
        result = f"Span Details:\n\n"
        result += f"Name: {span.get('name', 'Unknown')}\n"
        result += f"Duration: {span.get('duration_ms', 0)}ms\n"
        result += f"Status: {span.get('status', {}).get('code', 'OK')}\n"
        
        # Attributes
        attrs = span.get('attributes', {})
        if attrs:
            result += "\nAttributes:\n"
            for key, value in list(attrs.items())[:10]:
                result += f"  {key}: {value}\n"
        
        # Events
        events = span.get('events', [])
        if events:
            result += f"\nEvents ({len(events)}):\n"
            for event in events[:5]:
                result += f"  • {event.get('name', 'Unknown')}\n"
        
        return result
        
    except Exception as e:
        return f"Error fetching span details: {str(e)}"

@mcp.tool()
def analyze_db_queries(minutes: int = 5) -> str:
    """Analyze database query performance.
    
    Args:
        minutes: Look back this many minutes (default: 5)
    """
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        
        params = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "span.kind": "CLIENT",
            "db.system": "duckdb",
            "limit": 200
        }
        
        response = requests.get(f"{PHOENIX_API_URL}/traces", params=params)
        response.raise_for_status()
        
        traces = response.json()
        
        if not traces:
            return f"No database queries found in the last {minutes} minutes"
        
        # Analyze queries
        queries = {}
        for trace in traces:
            attrs = trace.get('attributes', {})
            query = attrs.get('db.statement', 'Unknown query')
            duration = trace.get('duration_ms', 0)
            
            # Normalize query (remove specific values)
            normalized = query
            for word in ['WHERE', 'AND', 'OR']:
                if word in normalized:
                    normalized = normalized.split(word)[0] + word + " ..."
                    break
            
            if normalized not in queries:
                queries[normalized] = []
            queries[normalized].append(duration)
        
        result = f"Database Query Analysis ({len(traces)} queries):\n\n"
        
        # Sort by total time
        sorted_queries = sorted(
            queries.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )
        
        for query, durations in sorted_queries[:10]:
            total_time = sum(durations)
            avg_time = total_time / len(durations)
            count = len(durations)
            
            result += f"Query: {query[:80]}...\n"
            result += f"  Count: {count}\n"
            result += f"  Total: {total_time}ms\n"
            result += f"  Avg: {avg_time:.1f}ms\n\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing database queries: {str(e)}"

if __name__ == "__main__":
    import asyncio
    print(f"Phoenix trace server starting (base URL: {PHOENIX_BASE_URL})")
    mcp.run()