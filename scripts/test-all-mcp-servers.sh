#!/bin/bash

echo "=== Testing All 17 MCP Servers ===="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test with proper MCP protocol
test_server() {
    local name=$1
    local cmd=$2
    printf "%-20s ... " "$name"
    
    # Send initialize request with timeout
    response=$(echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | timeout 5s $cmd 2>&1)
    
    if echo "$response" | grep -q '"result".*"serverInfo"'; then
        echo -e "${GREEN}✓ Working${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed${NC}"
        if [ -n "$response" ]; then
            echo "  Error: $(echo "$response" | head -1)"
        else
            echo "  Error: No response (timeout)"
        fi
        return 1
    fi
}

failed=0
passed=0

echo -e "\n${YELLOW}Node.js Servers:${NC}"
test_server "filesystem" "npx -y @modelcontextprotocol/server-filesystem@latest /tmp" && ((passed++)) || ((failed++))
test_server "brave" "npx -y @modelcontextprotocol/server-brave-search@latest" && ((passed++)) || ((failed++))
test_server "memory" "npx -y @modelcontextprotocol/server-memory@latest" && ((passed++)) || ((failed++))
test_server "sequential-thinking" "npx -y @modelcontextprotocol/server-sequential-thinking@latest" && ((passed++)) || ((failed++))
test_server "puppeteer" "npx -y @modelcontextprotocol/server-puppeteer@latest" && ((passed++)) || ((failed++))
test_server "ripgrep" "npx -y @modelcontextprotocol/server-ripgrep" && ((passed++)) || ((failed++))
test_server "dependency-graph" "npx -y @modelcontextprotocol/server-code-analysis" && ((passed++)) || ((failed++))

echo -e "\n${YELLOW}Python Servers:${NC}"
test_server "github" "mcp-server-github" && ((passed++)) || ((failed++))
test_server "statsource" "/Users/mikeedwards/.pyenv/shims/python3 -m mcp_server_stats" && ((passed++)) || ((failed++))
test_server "duckdb" "/Users/mikeedwards/.pyenv/shims/python3 -m mcp_server_duckdb :memory:" && ((passed++)) || ((failed++))
test_server "mlflow" "/Users/mikeedwards/.pyenv/shims/python3 /Users/mikeedwards/mcp-servers/community/mlflowMCPServer/mlflow_server.py" && ((passed++)) || ((failed++))
test_server "pyrepl" "/Users/mikeedwards/.pyenv/shims/python3 -m mcp_py_repl" && ((passed++)) || ((failed++))
test_server "sklearn" "/Users/mikeedwards/.pyenv/shims/python3 /Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py" && ((passed++)) || ((failed++))
test_server "optionsflow" "/Users/mikeedwards/.pyenv/shims/python3 /Users/mikeedwards/mcp-servers/community/mcp-optionsflow/optionsflow.py" && ((passed++)) || ((failed++))
test_server "python_analysis" "/Users/mikeedwards/.pyenv/shims/python3 '/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py'" && ((passed++)) || ((failed++))
test_server "trace" "/Users/mikeedwards/.pyenv/shims/python3 '/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-mcp-server.py'" && ((passed++)) || ((failed++))
test_server "logfire" "uvx logfire-mcp --read-token pylf_v1_us_00l06NMSXxWp1V9cTNJWJLvjRPs5HPRVsFtmdTSS1YC2" && ((passed++)) || ((failed++))

echo -e "\n${YELLOW}=== Summary ===${NC}"
echo "Passed: $passed"
echo "Failed: $failed"

if [ $failed -gt 0 ]; then
    echo -e "\n${RED}$failed servers failed!${NC}"
    echo "Common issues:"
    echo "  • NPX packages not installed - they download on first use"
    echo "  • Python modules missing - run: pip install mcp-server-stats mcp-server-duckdb mcp-py-repl"
    echo "  • Community server files missing - check /Users/mikeedwards/mcp-servers/community/"
    echo "  • Wrong Python path - using system python instead of pyenv"
else
    echo -e "\n${GREEN}All servers working!${NC}"
fi