#!/usr/bin/env bash
# Ultimate one-click Claude launcher with maximum performance

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
cd "$PROJECT_ROOT"

echo -e "${PURPLE}=== CLAUDE ULTIMATE ONE-CLICK LAUNCHER ===${NC}"
echo -e "${BLUE}Maximum performance configuration for 24GB M4 Mac${NC}"
echo ""

# 1. Set maximum performance environment
echo -e "${YELLOW}[1/5] Setting maximum performance...${NC}"
export CLAUDE_CODE_THINKING_BUDGET_TOKENS=500000  # Half million thinking tokens
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=100000      # 100K output tokens
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=200000     # 200K context window
export CLAUDE_CODE_PARALLELISM=12                # Use all cores
export NODE_OPTIONS="--max-old-space-size=16384" # 16GB for Node.js
export PYTHONOPTIMIZE=2                          # Maximum Python optimization
export PYTHON_MEMORY_LIMIT="16G"                 # 16GB for Python
export MCP_PERFORMANCE_MODE=true
export MCP_CACHE_SIZE="8GB"
ulimit -n 10240                                  # Max file descriptors

# 2. Clean and prepare
echo -e "${YELLOW}[2/5] Preparing environment...${NC}"
# Kill any existing MCP servers
pkill -f "mcp\|ripgrep-mcp\|dependency-graph" 2>/dev/null || true
rm -rf "$PROJECT_ROOT/.claude/runtime/ws_*/state/*.pid" 2>/dev/null || true
rm -rf "$PROJECT_ROOT/.claude/runtime/ws_*/locks/*" 2>/dev/null || true
sleep 1

# 3. Fix python-mcp-server if needed
if grep -q "asyncio.run(mcp.run())" "$PROJECT_ROOT/scripts/python-mcp-server.py" 2>/dev/null; then
    echo "  Fixing python-mcp-server..."
    sed -i.bak 's/asyncio.run(mcp.run())/mcp.run()/' "$PROJECT_ROOT/scripts/python-mcp-server.py"
fi

# 4. Start all MCP servers
echo -e "${YELLOW}[3/5] Starting MCP servers...${NC}"
export PATH="/Users/mikeedwards/.local/bin:$PATH"

# Start essential servers using our working script
MCP_ROOT="$PROJECT_ROOT" mcp-up-essential

# Show what's running
echo ""
echo -e "${YELLOW}[4/5] MCP Server Status:${NC}"
RUNTIME_DIR="$PROJECT_ROOT/.claude/runtime/ws_b272153b"
server_count=0
if [ -d "$RUNTIME_DIR/state" ]; then
    for pid_file in "$RUNTIME_DIR/state"/*.pid; do
        if [ -f "$pid_file" ]; then
            name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} $name (PID: $pid)"
                server_count=$((server_count + 1))
            fi
        fi
    done
fi
echo -e "  Total: ${GREEN}$server_count servers running${NC}"

# 5. Launch Claude
echo ""
echo -e "${YELLOW}[5/5] Launching Claude...${NC}"

# Find Claude command
CLAUDE_CMD=""
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
fi

if [ -z "$CLAUDE_CMD" ]; then
    echo -e "${YELLOW}Claude CLI not found. Please install from: https://claude.ai/code${NC}"
    echo ""
    echo "Your MCP servers are running. Once Claude is installed, run:"
    echo "  claude --mcp-config \"$PROJECT_ROOT/mcp-servers.json\""
    echo ""
    echo -e "${BLUE}Configuration Summary:${NC}"
    echo "  • Thinking tokens: 500,000"
    echo "  • Output tokens: 100,000"
    echo "  • Context window: 200,000"
    echo "  • Memory: 16GB Node.js, 16GB Python"
    echo "  • CPU: All 12 cores available"
    echo "  • MCP servers: $server_count running"
else
    echo -e "${GREEN}Launching Claude with maximum performance...${NC}"
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo "  • Thinking tokens: 500,000"
    echo "  • Output tokens: 100,000"
    echo "  • Context window: 200,000"
    echo "  • Memory: 16GB Node.js, 16GB Python"
    echo "  • CPU: All 12 cores available"
    echo "  • MCP servers: $server_count running"
    echo ""
    
    # Launch Claude with MCP configuration
    exec "$CLAUDE_CMD" --mcp-config "$PROJECT_ROOT/mcp-servers.json"
fi