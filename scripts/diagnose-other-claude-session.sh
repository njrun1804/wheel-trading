#!/bin/bash

echo "=== Diagnosing Other Claude Session MCP Failures ==="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Check running MCP processes
echo -e "\n${YELLOW}Currently Running MCP Processes:${NC}"
ps aux | grep -E "mcp|python.*mcp.*server|ripgrep-mcp|dependency-graph-mcp" | grep -v grep | awk '{print $11,$12,$13}' | sort | uniq -c

# 2. Check for error patterns in running processes
echo -e "\n${YELLOW}Checking for common issues:${NC}"

# Check if ripgrep binary exists
if ! command -v rg &> /dev/null; then
    echo -e "${RED}✗ ripgrep binary (rg) not installed${NC}"
    echo "  Fix: brew install ripgrep"
else
    echo -e "${GREEN}✓ ripgrep binary installed${NC}"
fi

# Check Python MCP modules
echo -e "\n${YELLOW}Python MCP Modules:${NC}"
for module in mcp mcp_server_stats mcp_server_duckdb mcp_py_repl; do
    if /Users/mikeedwards/.pyenv/shims/python3 -c "import $module" 2>/dev/null; then
        echo -e "${GREEN}✓ $module${NC}"
    else
        echo -e "${RED}✗ $module - not installed${NC}"
    fi
done

# 3. Test each server individually with EXACT commands from mcp-servers.json
echo -e "\n${YELLOW}Testing Servers with Exact Config Commands:${NC}"

# Read the actual commands from mcp-servers.json
python3 << 'EOF'
import json
import subprocess
import sys

config_path = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"
with open(config_path, 'r') as f:
    config = json.load(f)

failed_servers = []

for server_name, server_config in config["mcpServers"].items():
    cmd = server_config["command"]
    args = server_config.get("args", [])
    
    # Build full command
    if cmd == "npx":
        # Skip NPX tests for now - they often fail in test but work in Claude
        continue
    
    full_cmd = [cmd] + args
    
    # Test with MCP protocol
    test_input = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
    
    try:
        # For Python scripts with spaces in path, we need special handling
        if " " in " ".join(full_cmd):
            # This is likely python_analysis or trace with spaces in path
            result = subprocess.run(
                full_cmd,
                input=test_input,
                capture_output=True,
                text=True,
                timeout=3,
                shell=False
            )
        else:
            result = subprocess.run(
                full_cmd,
                input=test_input,
                capture_output=True,
                text=True,
                timeout=3
            )
        
        if '"result"' in result.stdout and '"serverInfo"' in result.stdout:
            print(f"✓ {server_name}")
        else:
            print(f"✗ {server_name}")
            if result.stderr:
                print(f"  Error: {result.stderr.strip()[:100]}")
            failed_servers.append(server_name)
    except subprocess.TimeoutExpired:
        print(f"✗ {server_name} - timeout")
        failed_servers.append(server_name)
    except Exception as e:
        print(f"✗ {server_name} - {str(e)[:100]}")
        failed_servers.append(server_name)

print(f"\nFailed servers: {', '.join(failed_servers)}")
EOF

# 4. Check specific file paths
echo -e "\n${YELLOW}Checking Server File Paths:${NC}"
files=(
    "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/ripgrep-mcp.py"
    "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/dependency-graph-mcp.py"
    "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py"
    "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-mcp-server.py"
    "/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py"
    "/Users/mikeedwards/mcp-servers/community/mcp-optionsflow/optionsflow.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file - NOT FOUND${NC}"
    fi
done

# 5. Check DuckDB file
echo -e "\n${YELLOW}Checking DuckDB:${NC}"
if [ -f "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/data/cache/wheel_cache.duckdb" ]; then
    echo -e "${GREEN}✓ DuckDB file exists${NC}"
else
    echo -e "${RED}✗ DuckDB file missing${NC}"
fi

echo -e "\n${YELLOW}=== Diagnosis Complete ===${NC}"
echo "Common fixes:"
echo "1. Install ripgrep: brew install ripgrep"
echo "2. Install Python modules: pip install mcp mcp-server-stats mcp-server-duckdb mcp-py-repl"
echo "3. Restart Claude to reload configuration"