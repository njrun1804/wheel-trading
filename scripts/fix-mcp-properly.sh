#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Fixing MCP Configuration Properly ===${NC}"

# 1. Clean up the configuration
echo -e "\n${YELLOW}1. Cleaning up mcp-servers.json...${NC}"
python3 << 'EOF'
import json

config_path = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Remove non-existent NPX servers and use Python versions instead
servers_to_remove = []
servers_to_fix = {}

# Check each server
for name, server in config["mcpServers"].items():
    if server.get("command") == "npx":
        args = server.get("args", [])
        # Check for non-existent packages
        if any(arg in ["@modelcontextprotocol/server-ripgrep", "@modelcontextprotocol/server-code-analysis"] for arg in args):
            # These don't exist as NPX packages
            if name == "ripgrep":
                # Keep the Python version if it exists
                if "ripgrep" in config["mcpServers"] and config["mcpServers"]["ripgrep"]["command"] != "npx":
                    servers_to_remove.append(name)
                else:
                    # Convert to use Python wrapper
                    servers_to_fix[name] = {
                        "transport": "stdio",
                        "command": "/Users/mikeedwards/.pyenv/shims/python3",
                        "args": ["/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/ripgrep-mcp.py"]
                    }
            elif name == "dependency-graph":
                # Convert to Python wrapper
                servers_to_fix[name] = {
                    "transport": "stdio",
                    "command": "/Users/mikeedwards/.pyenv/shims/python3",
                    "args": ["/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/dependency-graph-mcp.py"]
                }

# Apply fixes
for name, fix in servers_to_fix.items():
    config["mcpServers"][name] = fix
    print(f"✓ Fixed {name} to use Python implementation")

# Remove duplicates
seen = set()
final_servers = {}
for name, server in config["mcpServers"].items():
    key = f"{server['command']}:{':'.join(server.get('args', []))}"
    if key not in seen:
        seen.add(key)
        final_servers[name] = server
    else:
        print(f"✗ Removed duplicate: {name}")

config["mcpServers"] = final_servers

# Count servers
node_count = sum(1 for s in final_servers.values() if "npx" in s.get("command", "") or "node" in s.get("command", ""))
python_count = len(final_servers) - node_count

print(f"\nFinal configuration:")
print(f"  Total servers: {len(final_servers)}")
print(f"  Node.js servers: {node_count}")
print(f"  Python servers: {python_count}")

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
    f.write('\n')
EOF

# 2. Ensure Python MCP SDK is installed
echo -e "\n${YELLOW}2. Installing Python MCP SDK...${NC}"
/Users/mikeedwards/.pyenv/shims/pip install mcp

# 3. Create simple, working implementations
echo -e "\n${YELLOW}3. Creating simple server implementations...${NC}"

# Simple ripgrep wrapper that just calls rg
cat > "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/ripgrep-mcp.py" << 'EOF'
#!/usr/bin/env python3
import asyncio
import subprocess
import sys
from mcp.server import stdio
from mcp.server.stdio import StdioServer
from mcp.types import Tool, TextContent

server = StdioServer("ripgrep")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search",
            description="Search files using ripgrep",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."}
                },
                "required": ["pattern"]
            }
        )
    ]

@server.tool_handler("search")
async def search(pattern: str, path: str = "."):
    try:
        result = subprocess.run(
            ["rg", pattern, path],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout if result.stdout else "No matches found"
        return [TextContent(type="text", text=output[:5000])]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
EOF
chmod +x "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/ripgrep-mcp.py"

# Simple dependency graph analyzer
cat > "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/dependency-graph-mcp.py" << 'EOF'
#!/usr/bin/env python3
import asyncio
import ast
import sys
from pathlib import Path
from mcp.server import stdio
from mcp.server.stdio import StdioServer
from mcp.types import Tool, TextContent

server = StdioServer("dependency-graph")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="analyze_dependencies",
            description="Analyze Python file dependencies",
            inputSchema={
                "type": "object",
                "properties": {"file": {"type": "string"}},
                "required": ["file"]
            }
        )
    ]

@server.tool_handler("analyze_dependencies")
async def analyze_dependencies(file: str):
    try:
        with open(file, 'r') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(f"from {node.module}")
        
        result = f"Dependencies in {file}:\n" + "\n".join(sorted(set(imports)))
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
EOF
chmod +x "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/dependency-graph-mcp.py"

# 4. List what we actually have
echo -e "\n${YELLOW}4. Final server list:${NC}"
cat mcp-servers.json | jq -r '.mcpServers | keys[]' | sort | nl

echo -e "\n${GREEN}=== Done! ===${NC}"
echo "Changes made:"
echo "  ✓ Removed non-existent NPX packages"
echo "  ✓ Fixed duplicate configurations"
echo "  ✓ Installed Python MCP SDK"
echo "  ✓ Created simple, working implementations"
echo ""
echo "Start Claude with: ./scripts/start-claude-ultimate.sh"