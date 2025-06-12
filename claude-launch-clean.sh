#!/usr/bin/env bash
# Claude launcher - clean version with just the essentials

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

export PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== Claude Optimal Launcher ===${NC}"
echo ""

# Set environment variables per Anthropic docs
export MAX_THINKING_TOKENS=50000                   # More thinking tokens
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192         # Max output
export ANTHROPIC_MODEL="claude-opus-4-20250514"   # Claude 4 Opus
export NODE_OPTIONS="--max-old-space-size=4096"   # 4GB for Node (conservative)
export MCP_TIMEOUT=30000                          # 30s timeout
export MCP_TOOL_TIMEOUT=60000                     # 60s for long operations

echo "Configuration:"
echo "  • Model: Claude 4 Opus"
echo "  • Thinking: 50,000 tokens"
echo "  • Output: 8,192 tokens"
echo "  • MCP Timeout: 30s"
echo ""

# Find Claude
CLAUDE_CMD=""
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
else
    echo "Claude not found! Install from https://claude.ai/code"
    exit 1
fi

# Configure MCP servers using Claude's built-in commands
echo "Configuring MCP servers..."

"$CLAUDE_CMD" mcp add filesystem -s project -f -- \
    npx -y @modelcontextprotocol/server-filesystem@latest "$PROJECT_ROOT" 2>/dev/null || true

if [ -n "${GITHUB_TOKEN:-}" ]; then
    "$CLAUDE_CMD" mcp add github -s user -f --env GITHUB_TOKEN="$GITHUB_TOKEN" -- \
        mcp-server-github 2>/dev/null || true
fi

"$CLAUDE_CMD" mcp add memory -s project -f -- \
    npx -y @modelcontextprotocol/server-memory@latest 2>/dev/null || true

"$CLAUDE_CMD" mcp add sequential-thinking -s project -f -- \
    npx -y @modelcontextprotocol/server-sequential-thinking@latest 2>/dev/null || true

# Add dependency-graph if the enhanced version exists
if [ -f "$PROJECT_ROOT/scripts/dependency-graph-mcp-enhanced.py" ]; then
    "$CLAUDE_CMD" mcp add dependency-graph -s project -f -- \
        /Users/mikeedwards/.pyenv/shims/python3 \
        "$PROJECT_ROOT/scripts/dependency-graph-mcp-enhanced.py" 2>/dev/null || true
fi

echo -e "${GREEN}✓ MCP servers configured${NC}"
echo ""
echo "Launching Claude..."
exec "$CLAUDE_CMD"