#!/bin/bash

# Start Claude Code with OPTIMIZED MCP servers using performance tools

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}Unity Wheel Trading - Claude Code OPTIMIZED Launcher${NC}"
echo "====================================================="

# Check for performance tools
echo -e "\n${YELLOW}Checking performance optimizations...${NC}"

# Check bun
if command -v bun &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} bun installed ($(bun --version)) - 30x faster than npm!"
    BUNX_AVAILABLE=true
else
    echo -e "  ${YELLOW}⚠${NC} bun not installed - falling back to npx"
    echo -e "    Install with: curl -fsSL https://bun.sh/install | bash"
    BUNX_AVAILABLE=false
fi

# Check pnpm
if command -v pnpm &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} pnpm installed ($(pnpm --version)) - efficient package management"
else
    echo -e "  ${YELLOW}⚠${NC} pnpm not installed"
fi

# Check watchman
if command -v watchman &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} watchman installed - efficient file watching"
    # Start watchman if not running
    watchman watch-project "$PROJECT_ROOT" 2>/dev/null || true
else
    echo -e "  ${YELLOW}⚠${NC} watchman not installed"
fi

# Use optimized config if bun is available
if [ "$BUNX_AVAILABLE" = true ]; then
    MCP_CONFIG="$PROJECT_ROOT/mcp-servers-optimized.json"
    echo -e "\n${GREEN}Using optimized configuration with bun${NC}"
else
    MCP_CONFIG="$PROJECT_ROOT/mcp-servers-final.json"
    echo -e "\n${YELLOW}Using standard configuration with npx${NC}"
fi

# Performance environment variables
export NODE_ENV=production
export UV_SYSTEM_PYTHON=1  # Use system Python with uv
export PYTHONDONTWRITEBYTECODE=1  # Don't create .pyc files
export PYTHONUNBUFFERED=1  # Unbuffered output

# Enable Node.js performance features
export NODE_OPTIONS="--max-old-space-size=8192 --optimize-for-size"

# Load tokens
echo -e "\n${YELLOW}Loading tokens...${NC}"
LOGFIRE_READ_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null)
if [ -n "$LOGFIRE_READ_TOKEN" ]; then
    export LOGFIRE_READ_TOKEN
    echo -e "  ${GREEN}✓${NC} Loaded LOGFIRE_READ_TOKEN from keychain"
fi

# Check environment
echo -e "\n${YELLOW}Checking environment...${NC}"
MISSING_VARS=()

for var in GITHUB_TOKEN BRAVE_API_KEY DATABENTO_API_KEY FRED_API_KEY; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}Warning: Missing environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo -e "Continue anyway? (y/n): \c"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "  ${GREEN}✓${NC} All required environment variables set"
fi

# Find Claude
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
else
    CLAUDE_CMD="$(alias claude 2>/dev/null | sed "s/alias claude='//" | sed "s/'$//")"
fi

# Performance tips
echo -e "\n${BLUE}Performance Optimizations Active:${NC}"
echo -e "  • Bun for Node.js servers (30x faster startup)"
echo -e "  • Python -O flag (optimized bytecode)"
echo -e "  • Unbuffered I/O (faster responses)"
echo -e "  • Increased memory limits"
echo -e "  • Watchman file monitoring"
echo ""

# Start Claude
echo -e "${GREEN}Starting Claude Code with optimized MCP servers...${NC}"
echo -e "Config: ${YELLOW}$MCP_CONFIG${NC}"
echo -e "\n${YELLOW}Launching...${NC}\n"

# Clear npm/pnpm cache if it's getting too large
CACHE_SIZE=$(du -sh ~/.npm 2>/dev/null | cut -f1 || echo "0")
echo -e "NPM cache size: $CACHE_SIZE"

eval "$CLAUDE_CMD --mcp-config \"$MCP_CONFIG\""