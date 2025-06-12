#!/usr/bin/env bash
# Quick start Claude without optional services

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== Quick Start Claude ===${NC}"
echo ""

# 1. Kill any existing MCP servers
echo "1. Cleaning up any existing servers..."
export PATH="/Users/mikeedwards/.local/bin:$PATH"
mcp-down 2>/dev/null || true

# 2. Start fresh MCP servers
echo ""
echo "2. Starting essential MCP servers..."
MCP_ROOT="$PROJECT_ROOT" mcp-up-essential

# 3. Quick status
echo ""
echo "3. Server status:"
sleep 2
mcp-status 2>/dev/null | head -10 || echo "Servers are starting..."

# 4. Launch Claude
echo ""
echo -e "${GREEN}4. Launching Claude...${NC}"

# Find Claude command
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
else
    echo -e "${YELLOW}Claude CLI not found!${NC}"
    echo "Please install from: https://claude.ai/code"
    echo ""
    echo "Your MCP servers are running. Once Claude is installed, run:"
    echo "  claude --mcp-config \"$PROJECT_ROOT/mcp-servers.json\""
    exit 0
fi

# Launch with MCP config
echo "Starting Claude with MCP configuration..."
exec "$CLAUDE_CMD" --mcp-config "$PROJECT_ROOT/mcp-servers.json"