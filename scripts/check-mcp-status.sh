#!/bin/bash

# Quick MCP Server Status Check

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== MCP Server Quick Status ===${NC}"
echo ""

# Count servers
TOTAL=18
WORKING=0

# Check Node.js
if command -v node >/dev/null 2>&1; then
    ((WORKING+=8))  # All NPX servers work
    echo -e "Node.js: ${GREEN}✓${NC} (8 servers ready)"
else
    echo -e "Node.js: ${RED}✗${NC}"
fi

# Check Python servers
if command -v python3 >/dev/null 2>&1; then
    # Check each Python module/script
    PYTHON_OK=0
    
    # Modules
    for module in mcp_server_stats mcp_server_duckdb mcp_py_repl; do
        if python3 -c "import $module" 2>/dev/null; then
            ((PYTHON_OK++))
        fi
    done
    
    # Scripts
    SCRIPTS=(
        "/Users/mikeedwards/mcp-servers/community/mlflowMCPServer/mlflow_server.py"
        "/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py"
        "/Users/mikeedwards/mcp-servers/community/mcp-optionsflow/optionsflow.py"
        "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py"
    )
    
    for script in "${SCRIPTS[@]}"; do
        if [ -f "$script" ]; then
            ((PYTHON_OK++))
        fi
    done
    
    # Check github binary
    if command -v mcp-server-github >/dev/null 2>&1; then
        ((PYTHON_OK++))
    fi
    
    # Check logfire
    if [ -f ~/.logfire/default.toml ]; then
        ((PYTHON_OK++))
    fi
    
    ((WORKING+=PYTHON_OK))
    echo -e "Python: ${GREEN}✓${NC} ($PYTHON_OK/10 servers ready)"
else
    echo -e "Python: ${RED}✗${NC}"
fi

# Check environment variables
echo -e "\nEnvironment:"
vars=("GITHUB_TOKEN" "BRAVE_API_KEY" "DATABENTO_API_KEY" "FRED_API_KEY")
ENV_OK=0
for var in "${vars[@]}"; do
    if [ -n "${!var}" ]; then
        ((ENV_OK++))
    fi
done
echo -e "API Keys: ${GREEN}$ENV_OK/4 set${NC}"

# Check Logfire token
if security find-generic-password -a "$USER" -s "logfire-mcp" -w >/dev/null 2>&1; then
    echo -e "Logfire: ${GREEN}✓${NC} (token in keychain)"
else
    echo -e "Logfire: ${RED}✗${NC}"
fi

# Performance tools
echo -e "\nPerformance Tools:"
for tool in pnpm bun watchman eza uv fd bat rg; do
    if command -v $tool >/dev/null 2>&1; then
        echo -ne "${GREEN}✓${NC} "
    else
        echo -ne "${RED}✗${NC} "
    fi
    echo -n "$tool "
done
echo ""

# Summary
echo -e "\n${YELLOW}Summary:${NC} $WORKING/$TOTAL servers ready"

if [ $WORKING -eq $TOTAL ]; then
    echo -e "${GREEN}All systems go! Run: ./scripts/start-claude-full.sh${NC}"
else
    echo -e "${YELLOW}Some servers need setup. Run: ./scripts/install-additional-mcps.sh${NC}"
fi