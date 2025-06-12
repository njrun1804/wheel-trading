#!/usr/bin/env python3
"""Logfire trace MCP server for comprehensive observability."""

from mcp.server import FastMCP
import logfire
from datetime import datetime
from typing import Dict, Optional, Any
import json
import os

mcp = FastMCP("trace-logfire")

# Initialize Logfire with token from environment
LOGFIRE_TOKEN = os.environ.get("LOGFIRE_TOKEN", os.environ.get("LOGFIRE_READ_TOKEN", ""))
if LOGFIRE_TOKEN:
    logfire.configure(token=LOGFIRE_TOKEN)
else:
    logfire.configure()  # Will use local mode if no token

@mcp.tool()
def log_trace(
    name: str,
    level: str = "info",
    attributes: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[int] = None
) -> str:
    """Log a trace event to Logfire.
    
    Args:
        name: Name of the operation/event
        level: Log level (debug, info, warning, error)
        attributes: Additional attributes as JSON
        duration_ms: Duration in milliseconds (optional)
    """
    try:
        # Parse attributes if provided as string
        if attributes and isinstance(attributes, str):
            attributes = json.loads(attributes)
        
        log_attrs = attributes or {}
        log_attrs["timestamp"] = datetime.now().isoformat()
        
        if duration_ms:
            log_attrs["duration_ms"] = duration_ms
        
        # Log based on level
        if level == "debug":
            logfire.debug(name, **log_attrs)
        elif level == "warning":
            logfire.warning(name, **log_attrs)
        elif level == "error":
            logfire.error(name, **log_attrs)
        else:
            logfire.info(name, **log_attrs)
        
        return f"Logged {level}: {name} (duration: {duration_ms}ms)" if duration_ms else f"Logged {level}: {name}"
        
    except Exception as e:
        return f"Error logging trace: {str(e)}"

@mcp.tool()
def log_metric(
    name: str,
    value: float,
    unit: str = "",
    tags: Optional[Dict[str, str]] = None
) -> str:
    """Log a metric value to Logfire.
    
    Args:
        name: Metric name (e.g., "trade_profit", "position_delta")
        value: Numeric value
        unit: Unit of measurement (e.g., "USD", "ms", "count")
        tags: Additional tags for categorization
    """
    try:
        metric_data = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        if unit:
            metric_data["unit"] = unit
            
        if tags:
            metric_data.update(tags)
        
        logfire.info(f"metric.{name}", **metric_data)
        
        return f"Logged metric: {name} = {value}{unit}"
        
    except Exception as e:
        return f"Error logging metric: {str(e)}"

@mcp.tool()
def log_span(
    name: str,
    span_type: str = "operation",
    parent_span_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> str:
    """Start a traced span in Logfire.
    
    Args:
        name: Span name
        span_type: Type of span (operation, http, db, etc.)
        parent_span_id: Parent span ID for nested traces
        attributes: Additional span attributes
    """
    try:
        span_attrs = {
            "span.type": span_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if attributes:
            span_attrs.update(attributes)
            
        if parent_span_id:
            span_attrs["parent.span_id"] = parent_span_id
        
        with logfire.span(name, **span_attrs) as span:
            span_id = str(id(span))
            logfire.info(f"Started span: {name} (ID: {span_id})")
            
        return f"Started span: {name} (type: {span_type})"
        
    except Exception as e:
        return f"Error creating span: {str(e)}"

@mcp.tool()
def log_error(
    error_type: str,
    message: str,
    stack_trace: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Log an error with full context to Logfire.
    
    Args:
        error_type: Type of error (e.g., "ValidationError", "APIError")
        message: Error message
        stack_trace: Stack trace if available
        context: Additional context about the error
    """
    try:
        error_data = {
            "error.type": error_type,
            "error.message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if stack_trace:
            error_data["error.stack_trace"] = stack_trace
            
        if context:
            error_data["error.context"] = context
        
        logfire.error(f"error.{error_type}", **error_data)
        
        return f"Logged error: {error_type} - {message}"
        
    except Exception as e:
        return f"Error logging error: {str(e)}"

@mcp.tool()
def log_trading_event(
    event_type: str,
    symbol: str,
    action: str,
    quantity: Optional[int] = None,
    price: Optional[float] = None,
    profit_loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Log a trading-specific event to Logfire.
    
    Args:
        event_type: Type of event (order, fill, position_update, etc.)
        symbol: Trading symbol
        action: Action taken (buy, sell, hold, etc.)
        quantity: Number of contracts/shares
        price: Price per unit
        profit_loss: P&L if applicable
        metadata: Additional metadata
    """
    try:
        event_data = {
            "trading.event_type": event_type,
            "trading.symbol": symbol,
            "trading.action": action,
            "timestamp": datetime.now().isoformat()
        }
        
        if quantity is not None:
            event_data["trading.quantity"] = quantity
            
        if price is not None:
            event_data["trading.price"] = price
            
        if profit_loss is not None:
            event_data["trading.profit_loss"] = profit_loss
            
        if metadata:
            event_data.update(metadata)
        
        logfire.info(f"trading.{event_type}", **event_data)
        
        result = f"Logged trading event: {event_type} - {action} {symbol}"
        if quantity and price:
            result += f" ({quantity} @ ${price})"
        if profit_loss is not None:
            result += f" P&L: ${profit_loss}"
            
        return result
        
    except Exception as e:
        return f"Error logging trading event: {str(e)}"

@mcp.tool()
def query_logs(
    query: str,
    time_range_minutes: int = 60,
    limit: int = 100
) -> str:
    """Query recent logs from Logfire.
    
    Args:
        query: Search query
        time_range_minutes: How far back to search
        limit: Maximum results
    """
    try:
        # Note: This is a placeholder as Logfire's query API may vary
        # In practice, you'd use Logfire's actual query interface
        return f"Query functionality depends on Logfire's API. Query: '{query}' for last {time_range_minutes} minutes"
        
    except Exception as e:
        return f"Error querying logs: {str(e)}"

if __name__ == "__main__":
    import asyncio
    print(f"Logfire trace server starting (token configured: {bool(LOGFIRE_TOKEN)})")
    mcp.run()