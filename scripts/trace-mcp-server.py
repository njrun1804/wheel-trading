#!/usr/bin/env python3
from mcp.server import FastMCP

mcp = FastMCP("trace")

@mcp.tool()
def trace_log(message: str) -> str:
    """Log a trace message"""
    return f"Traced: {message}"

if __name__ == "__main__":
    import asyncio
    mcp.run()
