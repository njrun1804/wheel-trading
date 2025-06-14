#!/usr/bin/env python3
"""Test the FastMCP API"""

from mcp.server import FastMCP, Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Check what's in stdio
print("stdio module contents:", dir(mcp.server.stdio))

# Try creating a simple server
try:
    # Test with FastMCP
    mcp = FastMCP("test-server")
    
    @mcp.tool()
    def test_tool(message: str) -> str:
        return f"Echo: {message}"
    
    print("✓ FastMCP server created successfully")
except Exception as e:
    print(f"✗ FastMCP failed: {e}")

# Try the Server class
try:
    app = Server("test-server")
    print("✓ Server class instantiated")
    print("Server methods:", [m for m in dir(app) if not m.startswith('_') and callable(getattr(app, m))])
except Exception as e:
    print(f"✗ Server class failed: {e}")

# Find stdio server creation
import inspect
stdio_funcs = [name for name, obj in inspect.getmembers(mcp.server.stdio) if inspect.isfunction(obj)]
print("\nFunctions in stdio:", stdio_funcs)