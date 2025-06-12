#!/bin/bash

# Start Claude with local trace servers (no API keys needed!)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}Unity Wheel Trading - Claude Code LOCAL Trace Servers${NC}"
echo "====================================================="

# Check if local servers are running
echo -e "\n${YELLOW}Checking local trace servers...${NC}"

# Check Opik
if curl -s http://localhost:5173/api/health &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Opik platform running on http://localhost:5173"
else
    echo -e "  ${RED}✗${NC} Opik platform not running"
    echo -e "  ${YELLOW}Run: cd ~/mcp-servers/opik-platform && ./opik.sh start${NC}"
fi

# Check Phoenix
if curl -s http://localhost:6006/healthz &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Phoenix platform running on http://localhost:6006"
else
    echo -e "  ${RED}✗${NC} Phoenix platform not running"
    echo -e "  ${YELLOW}Run: phoenix serve${NC}"
fi

# Use local MCP config
MCP_CONFIG="$PROJECT_ROOT/mcp-servers-local.json"

# Find Claude
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
else
    CLAUDE_CMD="$(alias claude 2>/dev/null | sed "s/alias claude='//" | sed "s/'$//")"
fi

# Load tokens from keychain
LOGFIRE_READ_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null)
if [ -n "$LOGFIRE_READ_TOKEN" ]; then
    export LOGFIRE_READ_TOKEN
fi

echo -e "\n${GREEN}Starting Claude with 18 servers (including local trace servers)...${NC}"
eval "$CLAUDE_CMD --mcp-config \"$MCP_CONFIG\""
