#!/usr/bin/env python3
"""Test MCP server communication and API consistency."""

import json
import subprocess
import sys
import asyncio
from pathlib import Path

async def test_mcp_protocol(server_script: str):
    """Test basic MCP protocol communication with a server."""
    # MCP protocol test messages
    init_message = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-12",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.1.0"}
        },
        "id": 1
    }
    
    list_tools_message = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2
    }
    
    # Start the server
    proc = await asyncio.create_subprocess_exec(
        sys.executable, server_script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        # Send initialization
        proc.stdin.write((json.dumps(init_message) + '\n').encode())
        await proc.stdin.drain()
        
        # Read response
        response = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
        init_response = json.loads(response.decode())
        
        # Send list tools
        proc.stdin.write((json.dumps(list_tools_message) + '\n').encode())
        await proc.stdin.drain()
        
        # Read tools response
        response = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
        tools_response = json.loads(response.decode())
        
        return {
            "success": True,
            "server": Path(server_script).name,
            "init_response": init_response,
            "tools": tools_response.get("result", {}).get("tools", [])
        }
        
    except asyncio.TimeoutError:
        return {
            "success": False,
            "server": Path(server_script).name,
            "error": "Timeout waiting for response"
        }
    except Exception as e:
        return {
            "success": False,
            "server": Path(server_script).name,
            "error": str(e)
        }
    finally:
        proc.terminate()
        await proc.wait()

async def main():
    """Test all MCP servers."""
    scripts_dir = Path(__file__).parent
    
    # List of confirmed MCP servers from our verification
    mcp_servers = [
        "dependency-graph-mcp-enhanced.py",
        "dependency-graph-mcp.py",
        "python-mcp-server.py",
        "ripgrep-mcp.py",
        "trace-mcp-server.py",
        "trace-phoenix-mcp.py"
    ]
    
    print("Testing MCP Server Communication\n")
    
    results = []
    for server_name in mcp_servers:
        server_path = scripts_dir / server_name
        if server_path.exists():
            print(f"Testing {server_name}...", end=" ", flush=True)
            result = await test_mcp_protocol(str(server_path))
            results.append(result)
            
            if result["success"]:
                print(f"✓ OK - {len(result['tools'])} tools")
            else:
                print(f"✗ Failed - {result['error']}")
    
    # Summary
    print("\n" + "="*60)
    print("MCP COMMUNICATION TEST SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results if r["success"])
    print(f"Servers tested: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    
    if success_count < len(results):
        print("\nFailed servers:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['server']}: {r['error']}")
    
    # Write results
    results_file = scripts_dir / "mcp-communication-test.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results: {results_file}")

if __name__ == "__main__":
    asyncio.run(main())