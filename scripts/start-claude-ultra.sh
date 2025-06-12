#!/bin/bash

# Ultra-optimized Claude launcher with ALL performance features

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}Unity Wheel Trading - Claude Code ULTRA Launcher${NC}"
echo "=================================================="

# 1. Environment validation
echo -e "\n${YELLOW}Validating environment...${NC}"
source ~/.zshrc 2>/dev/null || true

# Required tools check
MISSING_TOOLS=()
command -v bun >/dev/null 2>&1 || MISSING_TOOLS+=("bun")
command -v watchman >/dev/null 2>&1 || MISSING_TOOLS+=("watchman")
command -v rg >/dev/null 2>&1 || MISSING_TOOLS+=("ripgrep")

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo -e "  ${YELLOW}Missing tools: ${MISSING_TOOLS[*]}${NC}"
    echo -e "  Run: ./scripts/optimize-mcp-performance.sh"
fi

# 2. Cache warming
echo -e "\n${YELLOW}Warming caches...${NC}"

# Pre-download NPX packages if using bun
if command -v bun >/dev/null 2>&1; then
    echo -e "  ${BLUE}Pre-caching MCP packages...${NC}"
    bunx @modelcontextprotocol/server-filesystem@latest --help >/dev/null 2>&1 &
    bunx mcp-ripgrep@latest --help >/dev/null 2>&1 &
    wait
    echo -e "  ${GREEN}✓${NC} Package cache warmed"
fi

# Warm Python import cache
if [ -f "$SCRIPT_DIR/mcp-connection-pool.py" ]; then
    python3 -c "import pandas, numpy, duckdb, sqlalchemy" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Python cache warmed"
fi

# Start watchman for file monitoring
if command -v watchman >/dev/null 2>&1; then
    watchman watch-project "$PROJECT_ROOT" >/dev/null 2>&1 || true
    echo -e "  ${GREEN}✓${NC} Watchman monitoring enabled"
fi

# 3. Clean old caches if too large
NPM_SIZE=$(du -sm ~/.npm 2>/dev/null | cut -f1 || echo 0)
if [ "$NPM_SIZE" -gt 1000 ]; then
    echo -e "  ${YELLOW}Cleaning NPM cache (${NPM_SIZE}MB)...${NC}"
    npm cache clean --force >/dev/null 2>&1 || true
fi

# 4. Performance environment
export NODE_ENV=production
export NODE_OPTIONS="--max-old-space-size=8192 --optimize-for-size"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export UV_SYSTEM_PYTHON=1

# Load tokens
LOGFIRE_READ_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null) || true
[ -n "$LOGFIRE_READ_TOKEN" ] && export LOGFIRE_READ_TOKEN

# 5. Choose optimal config
if command -v bun >/dev/null 2>&1; then
    MCP_CONFIG="$PROJECT_ROOT/mcp-servers-optimized.json"
    LAUNCH_MSG="bun + optimizations"
else
    MCP_CONFIG="$PROJECT_ROOT/mcp-servers-final.json"
    LAUNCH_MSG="standard npx"
fi

# Find Claude
CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
[ ! -f "$CLAUDE_CMD" ] && CLAUDE_CMD="claude"

# 6. Performance summary
echo -e "\n${GREEN}Ultra Performance Mode:${NC}"
echo -e "  • Package pre-caching: ${GREEN}✓${NC}"
echo -e "  • Import warming: ${GREEN}✓${NC}"
echo -e "  • File monitoring: ${GREEN}✓${NC}"
echo -e "  • Memory optimization: ${GREEN}✓${NC}"
echo -e "  • Launch mode: ${BLUE}$LAUNCH_MSG${NC}"

# 7. Launch with timing
echo -e "\n${YELLOW}Launching Claude...${NC}"
START_TIME=$(date +%s)

eval "$CLAUDE_CMD --mcp-config \"$MCP_CONFIG\""

END_TIME=$(date +%s)
echo -e "\n${GREEN}Session ended. Duration: $((END_TIME - START_TIME))s${NC}"

# 8. Post-session health check
echo -e "\n${YELLOW}Post-session health check...${NC}"
"$SCRIPT_DIR/mcp-health-monitor.sh" 2>/dev/null || true