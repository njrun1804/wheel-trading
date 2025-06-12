#!/bin/bash

echo "=== Diagnosing Current Claude Session MCP Servers ==="

# Get the Claude process PID
CLAUDE_PID=$(ps aux | grep -E "claude.*mcp-config" | grep -v grep | head -1 | awk '{print $2}')
echo "Claude PID: $CLAUDE_PID"

# Check which servers are actually loaded in config
echo -e "\n--- Configured Servers ---"
cat mcp-servers.json | jq -r '.mcpServers | keys[]' | sort

echo -e "\n--- Running MCP Processes ---"
ps aux | grep -E "mcp|python.*server" | grep -v grep | grep -v "lsp_server" | awk '{
    if ($0 ~ /mcp-server-filesystem/) print "✓ filesystem"
    else if ($0 ~ /mcp-server-brave-search/) print "✓ brave"
    else if ($0 ~ /mcp-server-memory/) print "✓ memory"
    else if ($0 ~ /mcp-server-sequential-thinking/) print "✓ sequential-thinking"
    else if ($0 ~ /mcp-server-puppeteer/) print "✓ puppeteer"
    else if ($0 ~ /mcp-server-github/) print "✓ github"
    else if ($0 ~ /mcp_server_stats/) print "✓ statsource"
    else if ($0 ~ /mcp_server_duckdb/) print "✓ duckdb"
    else if ($0 ~ /mcp_py_repl/) print "✓ pyrepl"
    else if ($0 ~ /mlflow_server.py/) print "✓ mlflow"
    else if ($0 ~ /optionsflow.py/) print "✓ optionsflow"
    else if ($0 ~ /logfire-mcp/) print "✓ logfire"
    else if ($0 ~ /sklearn.*server.py/) print "? sklearn"
    else if ($0 ~ /python-mcp-server.py/) print "? python_analysis"
    else if ($0 ~ /trace-mcp-server.py/) print "? trace"
    else if ($0 ~ /ripgrep/) print "? ripgrep"
    else if ($0 ~ /code-analysis/) print "? dependency-graph"
}' | sort | uniq

echo -e "\n--- Missing Servers ---"
# Compare configured vs running
CONFIGURED=$(cat mcp-servers.json | jq -r '.mcpServers | keys[]' | sort)
RUNNING=$(ps aux | grep -E "mcp|python.*server" | grep -v grep | grep -v "lsp_server" | awk '{
    if ($0 ~ /mcp-server-filesystem/) print "filesystem"
    else if ($0 ~ /mcp-server-brave-search/) print "brave"
    else if ($0 ~ /mcp-server-memory/) print "memory"
    else if ($0 ~ /mcp-server-sequential-thinking/) print "sequential-thinking"
    else if ($0 ~ /mcp-server-puppeteer/) print "puppeteer"
    else if ($0 ~ /mcp-server-github/) print "github"
    else if ($0 ~ /mcp_server_stats/) print "statsource"
    else if ($0 ~ /mcp_server_duckdb/) print "duckdb"
    else if ($0 ~ /mcp_py_repl/) print "pyrepl"
    else if ($0 ~ /mlflow_server.py/) print "mlflow"
    else if ($0 ~ /optionsflow.py/) print "optionsflow"
    else if ($0 ~ /logfire-mcp/) print "logfire"
}' | sort | uniq)

for server in $CONFIGURED; do
    if ! echo "$RUNNING" | grep -q "^$server$"; then
        echo "✗ $server"
    fi
done

# Check for specific issues
echo -e "\n--- Checking Specific Issues ---"

# Check sklearn path
if [ ! -f "/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py" ]; then
    echo "✗ sklearn server file missing"
fi

# Check python analysis
if [ ! -f "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py" ]; then
    echo "✗ python_analysis server file missing"
fi

# Check trace
if [ ! -f "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-mcp-server.py" ]; then
    echo "✗ trace server file missing"
fi

# Check for NPX issues
echo -e "\n--- NPX Server Status ---"
if ! npx @modelcontextprotocol/server-ripgrep --version 2>/dev/null; then
    echo "✗ ripgrep not available via npx"
else
    echo "✓ ripgrep available"
fi

if ! npx @modelcontextprotocol/server-code-analysis --version 2>/dev/null; then
    echo "✗ dependency-graph not available via npx"
else
    echo "✓ dependency-graph available"
fi