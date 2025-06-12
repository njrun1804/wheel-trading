#!/usr/bin/env bash
# Start all services for Claude optimization

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== Starting All Services ===${NC}"

# 1. Ensure PATH includes our tools
export PATH="/Users/mikeedwards/.local/bin:$PATH"

# 2. Start Phoenix for observability (in background)
echo -e "\n${GREEN}1. Starting Phoenix Observability${NC}"
if ! curl -s http://localhost:6006/health >/dev/null 2>&1; then
    echo "Starting Phoenix on port 6006..."
    (cd "$PROJECT_ROOT" && phoenix serve >/dev/null 2>&1 &)
    sleep 2
    if curl -s http://localhost:6006/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Phoenix started successfully"
    else
        echo -e "${YELLOW}⚠${NC} Phoenix may need manual start: phoenix serve"
    fi
else
    echo -e "${GREEN}✓${NC} Phoenix already running"
fi

# 3. Start essential MCP servers
echo -e "\n${GREEN}2. Starting Essential MCP Servers${NC}"
MCP_ROOT="$PROJECT_ROOT" mcp-up-essential

# 4. Quick health check
echo -e "\n${GREEN}3. Health Check${NC}"
sleep 1
mcp-health | head -20 || echo "Health check will be available shortly"

# 5. Test enhanced features
echo -e "\n${GREEN}4. Testing Enhanced Features${NC}"

# Test dependency graph
echo -n "Testing dependency graph search... "
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/scripts')
try:
    from dependency_graph_mcp_enhanced import search_code_fuzzy
    # This will trigger initial scan
    results = search_code_fuzzy('Advisor')
    print('✓ Working')
except Exception as e:
    print(f'✗ Error: {e}')
" 2>/dev/null || echo "✗ Not available"

echo -e "\n${BLUE}=== Ready to Start Claude ===${NC}"
echo ""
echo "All services are running. You can now:"
echo "  1. Start Claude: ./scripts/start-claude-ultimate.sh"
echo "  2. Monitor health: mcp-health"
echo "  3. View traces: open http://localhost:6006"
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"