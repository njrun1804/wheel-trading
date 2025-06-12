#!/bin/bash
echo "=== MCP Readiness Check ==="

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

issues=0

# Check ripgrep
if command -v rg &> /dev/null; then
    echo -e "${GREEN}✓ ripgrep installed${NC}"
else
    echo -e "${RED}✗ ripgrep missing - brew install ripgrep${NC}"
    ((issues++))
fi

# Check Python modules
for mod in mcp mcp_server_stats mcp_server_duckdb mcp_py_repl; do
    if /Users/mikeedwards/.pyenv/shims/python3 -c "import $mod" 2>/dev/null; then
        echo -e "${GREEN}✓ $mod${NC}"
    else
        echo -e "${RED}✗ $mod missing${NC}"
        ((issues++))
    fi
done

# Check server files
files=(
    "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/ripgrep-mcp.py"
    "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/dependency-graph-mcp.py"
    "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py"
    "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-mcp-server.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $(basename $file)${NC}"
    else
        echo -e "${RED}✗ $(basename $file) missing${NC}"
        ((issues++))
    fi
done

if [ $issues -eq 0 ]; then
    echo -e "\n${GREEN}All MCP servers ready!${NC}"
else
    echo -e "\n${RED}$issues issues found - run ./scripts/ensure-mcp-always-works.sh${NC}"
fi
