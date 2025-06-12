#!/usr/bin/env python3
"""Test script for the python_analysis MCP server."""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
import sys

async def test_server():
    """Test the python_analysis MCP server."""
    # Create server parameters
    server_params = StdioServerParameters(
        command="/Users/mikeedwards/.pyenv/shims/python3",
        args=["/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-analysis-mcp.py"],
        env={
            "PYTHONPATH": "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading",
            "WORKSPACE_ROOT": "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
        }
    )
    
    async with ClientSession(server_params) as session:
        # Initialize the session
        await session.initialize()
        
        print("✓ Connected to python_analysis MCP server")
        
        # List available tools
        tools_response = await session.list_tools()
        print(f"\nAvailable tools: {len(tools_response.tools)}")
        for tool in tools_response.tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Test health check
        print("\nTesting health check...")
        result = await session.call_tool("healthz", {})
        health_data = result.content[0].text if result.content else "{}"
        health = json.loads(health_data)
        print(f"  Status: {health.get('status', 'unknown')}")
        print(f"  PID: {health.get('pid', 'unknown')}")
        
        # Test system monitor
        print("\nTesting system monitor...")
        result = await session.call_tool("monitor_system", {})
        print(result.content[0].text[:200] + "..." if result.content else "No result")
        
        print("\n✓ All tests passed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_server())
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)