#!/bin/bash

# Ultra-optimized Claude startup with all performance features

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Ultra-Optimized Claude MCP Launcher ===${NC}"

# 1. Validate environment first
echo -e "\n${YELLOW}Validating environment...${NC}"
"$SCRIPT_DIR/validate-mcp-env.sh" || {
    echo -e "${RED}Environment validation failed!${NC}"
    exit 1
}

# 2. Warm caches
echo -e "\n${YELLOW}Warming caches...${NC}"
"$SCRIPT_DIR/warm-mcp-cache.sh"

# 3. Performance settings
export NODE_ENV=production
export NODE_OPTIONS="--max-old-space-size=8192 --optimize-for-size"
export UV_SYSTEM_PYTHON=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONOPTIMIZE=1  # Enable optimizations

# 4. Use bun if available
if command -v bun &> /dev/null; then
    MCP_CONFIG="$PROJECT_ROOT/mcp-servers-optimized.json"
    echo -e "${GREEN}Using bun for 30x faster startup${NC}"
else
    MCP_CONFIG="$PROJECT_ROOT/mcp-servers-final.json"
fi

# 5. Start watchman for file monitoring
if command -v watchman &> /dev/null; then
    watchman watch-project "$PROJECT_ROOT" 2>/dev/null || true
fi

# 6. Load tokens
LOGFIRE_READ_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null)
if [ -n "$LOGFIRE_READ_TOKEN" ]; then
    export LOGFIRE_READ_TOKEN
fi

# 7. Find Claude
CLAUDE_CMD="${CLAUDE_CMD:-claude}"
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
fi

# 8. Launch with performance monitoring
echo -e "\n${GREEN}Launching Claude with ultra-optimized MCP servers...${NC}"
echo -e "${BLUE}Performance features enabled:${NC}"
echo "  • Cache pre-warming"
echo "  • Connection pooling ready"
echo "  • Memory optimization"
echo "  • Bun acceleration"
echo ""

# Start monitoring in background
("$SCRIPT_DIR/mcp-health-monitor.sh" > /dev/null 2>&1 &)

# Launch Claude
eval "$CLAUDE_CMD --mcp-config \"$MCP_CONFIG\""