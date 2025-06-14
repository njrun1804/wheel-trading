#!/bin/bash

echo "=== Verifying MCP Servers with Protocol ==="

# Test with proper MCP initialize message
test_mcp_server() {
    local name=$1
    local cmd=$2
    printf "%-20s ... " "$name"
    
    # Send initialize request
    response=$(echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | timeout 5s $cmd 2>&1)
    
    if echo "$response" | grep -q '"result"'; then
        echo "✓ Working"
    else
        echo "✗ Failed"
        echo "  Error: $(echo "$response" | head -1)"
    fi
}

# Test Python servers
echo "Python Servers:"
test_mcp_server "github" "/Users/mikeedwards/.pyenv/shims/python3 -m mcp_server_github"
test_mcp_server "statsource" "/Users/mikeedwards/.pyenv/shims/python3 -m mcp_server_stats"
test_mcp_server "duckdb" "/Users/mikeedwards/.pyenv/shims/python3 -m mcp_server_duckdb :memory:"
test_mcp_server "pyrepl" "/Users/mikeedwards/.pyenv/shims/python3 -m mcp_py_repl"
test_mcp_server "python_analysis" "/Users/mikeedwards/.pyenv/shims/python3 /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py"
test_mcp_server "trace" "/Users/mikeedwards/.pyenv/shims/python3 /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-mcp-server.py"

echo ""
echo "Node.js Servers:"
test_mcp_server "filesystem" "npx -y @modelcontextprotocol/server-filesystem@latest /tmp"
test_mcp_server "memory" "npx -y @modelcontextprotocol/server-memory@latest"

echo ""
echo "Community Servers:"
# Check if files exist
for server in mlflow sklearn optionsflow; do
    case $server in
        mlflow) file="/Users/mikeedwards/mcp-servers/community/mlflowMCPServer/mlflow_server.py" ;;
        sklearn) file="/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py" ;;
        optionsflow) file="/Users/mikeedwards/mcp-servers/community/mcp-optionsflow/optionsflow.py" ;;
    esac
    
    printf "%-20s ... " "$server"
    if [ -f "$file" ]; then
        echo "✓ File exists"
    else
        echo "✗ File missing: $file"
    fi
done

echo ""
echo "Note: Some servers may show as 'failed' in Claude UI but still work properly."
echo "This happens when they take time to initialize or need specific environment setup."