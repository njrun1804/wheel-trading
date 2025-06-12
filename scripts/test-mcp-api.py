#!/usr/bin/env python3
"""Test which MCP API actually works"""

print("Testing MCP imports...")

try:
    # Test 1: Old style imports
    from mcp.server import create_stdio_server
    from mcp.types import Tool, TextContent
    from mcp import server
    print("✓ Old style imports work")
    print(f"  - create_stdio_server: {create_stdio_server}")
    print(f"  - server.Server: {server.Server}")
except Exception as e:
    print(f"✗ Old style imports failed: {e}")

try:
    # Test 2: New style imports
    from mcp.server import stdio
    from mcp.server.stdio import StdioServer
    print("✓ New style imports work")
    print(f"  - stdio: {stdio}")
    print(f"  - StdioServer: {StdioServer}")
except Exception as e:
    print(f"✗ New style imports failed: {e}")

try:
    # Test 3: FastMCP
    from mcp.server import FastMCP
    print("✓ FastMCP import works")
    print(f"  - FastMCP: {FastMCP}")
except Exception as e:
    print(f"✗ FastMCP import failed: {e}")

# Show what's actually in mcp.server
import mcp.server
print("\nAvailable in mcp.server:", [x for x in dir(mcp.server) if not x.startswith('_')])