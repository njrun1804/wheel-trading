#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Claude with M4 Max optimizations...${NC}"

# Set high priority
sudo renice -20 $$ 2>/dev/null || true

# Export performance settings
export MCP_PERFORMANCE_MODE=true
export MCP_CONCURRENCY=8
export MCP_CACHE_SIZE=10GB
export OTEL_EXPORTER_OTLP_ENDPOINT='http://127.0.0.1:4318'
export NODE_OPTIONS="--max-old-space-size=8192"
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Set QoS to high performance
taskpolicy -B -t 5 $$ 2>/dev/null || true

# Find Claude command
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
else
    echo -e "${YELLOW}Claude command not found. Please install Claude Code.${NC}"
    exit 1
fi

# MCP config path
MCP_CONFIG="${HOME}/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"

# Start Claude with optimizations
echo -e "${GREEN}✅ M4 Max optimizations applied${NC}"
echo -e "${BLUE}📊 Allocated:${NC}"
echo -e "  • Maximum process priority (-20)"
echo -e "  • High performance QoS tier"
echo -e "  • 8GB Node.js heap"
echo -e "  • Python optimizations enabled"
echo ""

# Launch Claude
echo -e "${GREEN}Launching Claude Code...${NC}"
eval "$CLAUDE_CMD --mcp-config \"$MCP_CONFIG\""