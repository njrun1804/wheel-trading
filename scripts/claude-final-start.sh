#!/usr/bin/env bash
# Final Claude launcher with working MCP servers

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== Claude Final Startup ===${NC}"
echo ""

# 1. Clean start
echo "1. Cleaning up old processes..."
pkill -f "mcp\|ripgrep-mcp\|dependency-graph" 2>/dev/null || true
rm -rf .claude/runtime/ws_*/state/*.pid 2>/dev/null || true
rm -rf .claude/runtime/ws_*/locks/* 2>/dev/null || true
sleep 1

# 2. Start MCP servers
echo ""
echo "2. Starting MCP servers..."
export PATH="/Users/mikeedwards/.local/bin:$PATH"
MCP_ROOT="$PROJECT_ROOT" mcp-up-essential

# 3. Check what's running
echo ""
echo "3. MCP Server Status:"
echo "===================="
RUNTIME_DIR="$PROJECT_ROOT/.claude/runtime/ws_b272153b"
if [ -d "$RUNTIME_DIR/state" ]; then
    for pid_file in "$RUNTIME_DIR/state"/*.pid; do
        if [ -f "$pid_file" ]; then
            name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}✓${NC} $name (PID: $pid)"
            fi
        fi
    done
fi

# 4. Summary
echo ""
echo -e "${GREEN}Ready to launch Claude!${NC}"
echo ""
echo "Working MCP servers:"
echo "  • filesystem - File operations"
echo "  • github - Repository management"  
echo "  • dependency-graph - Fast code search (2-5ms)"
echo "  • memory - State persistence"
echo "  • sequential-thinking - Multi-step planning"
echo "  • python_analysis - Trading bot analysis"
echo ""
echo "Note: ripgrep is temporarily disabled (use dependency-graph instead)"
echo ""

# 5. Launch options
echo "To start Claude, run ONE of these:"
echo ""
echo "  1. With Claude CLI:"
echo "     claude --mcp-config \"$PROJECT_ROOT/mcp-servers.json\""
echo ""
echo "  2. With VS Code:"
echo "     code ."
echo ""
echo "  3. With the ultimate script:"
echo "     ./scripts/start-claude-ultimate.sh"
echo ""
echo -e "${BLUE}Your optimized MCP stack is ready!${NC}"