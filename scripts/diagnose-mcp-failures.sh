#!/bin/bash

echo "=== Testing Each MCP Server Individually ==="

# Test each server command
servers=(
  "filesystem:npx -y @modelcontextprotocol/server-filesystem@latest /tmp"
  "brave:npx -y @modelcontextprotocol/server-brave-search@latest"
  "memory:npx -y @modelcontextprotocol/server-memory@latest"
  "sequential-thinking:npx -y @modelcontextprotocol/server-sequential-thinking@latest"
  "puppeteer:npx -y @modelcontextprotocol/server-puppeteer@latest"
  "ripgrep:npx -y @modelcontextprotocol/server-ripgrep@latest"
  "dependency-graph:npx -y @modelcontextprotocol/server-code-analysis@latest"
  ":npx -y @modelcontextprotocol/server-opik"
  "github:mcp-server-github"
  "statsource:python3 -m mcp_server_stats"
  "duckdb:python3 -m mcp_server_duckdb :memory:"
  "mlflow:python3 /Users/mikeedwards/mcp-servers/community/mlflowMCPServer/mlflow_server.py"
  "pyrepl:python3 -m mcp_py_repl"
  "sklearn:python3 /Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py"
  "optionsflow:python3 /Users/mikeedwards/mcp-servers/community/mcp-optionsflow/optionsflow.py"
  "python_analysis:python3 /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py"
  "trace:python3 /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-mcp-server.py"
  ":npx -y phoenix-trace-mcp"
  "logfire:uvx logfire-mcp"
)

failed=0
passed=0

for server_cmd in "${servers[@]}"; do
  name="${server_cmd%%:*}"
  cmd="${server_cmd#*:}"
  
  printf "Testing %-20s ... " "$name"
  
  # Test if command exists and can start
  if timeout 3s bash -c "echo '' | $cmd" >/dev/null 2>&1; then
    echo "✓ PASS"
    ((passed++))
  else
    echo "✗ FAIL"
    # Show error
    echo "  Error: $(timeout 2s bash -c "echo '' | $cmd" 2>&1 | head -1)"
    ((failed++))
  fi
done

echo ""
echo "Summary: $passed passed, $failed failed"