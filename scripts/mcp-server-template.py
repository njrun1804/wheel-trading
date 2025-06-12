#!/usr/bin/env python3
"""
MCP Server Template - Use this as base for new MCP servers.

Usage:
    1. Copy this file to create a new MCP server
    2. Rename the server name in FastMCP()
    3. Add your tools using @mcp.tool() decorator
    4. Run directly - no asyncio.run() needed!
"""

from mcp.server import FastMCP
from typing import Dict, Any, Optional
import os
import sys
import json
import logging

# Initialize server
mcp = FastMCP("template-server")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@mcp.tool()
def example_tool(input: str) -> str:
    """Example tool that echoes input.
    
    Args:
        input: String to echo back
        
    Returns:
        The input string with a prefix
    """
    return f"Echo: {input}"

@mcp.tool()
def healthz() -> Dict[str, Any]:
    """Health check endpoint for monitoring.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "server": "template-server",
        "timestamp": str(datetime.now()),
        "pid": os.getpid()
    }

# Important: Just call mcp.run() directly - FastMCP handles the async loop!
if __name__ == "__main__":
    mcp.run()
