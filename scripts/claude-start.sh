#!/usr/bin/env bash
# Simple one-command startup for Claude with all optimizations

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=== Starting Claude with Full Optimization ===${NC}"
echo ""

# 1. Ensure environment is loaded
if [ -f ".envrc" ]; then
    source .envrc
fi

# 2. Start Phoenix (optional, for observability)
echo "1. Checking Phoenix observability..."
./scripts/start-phoenix.sh || echo "   Phoenix is optional - continuing..."

# 3. Start MCP servers
echo ""
echo "2. Starting MCP servers..."
export PATH="/Users/mikeedwards/.local/bin:$PATH"
MCP_ROOT="$PROJECT_ROOT" mcp-up-essential

# 4. Quick health check
echo ""
echo "3. Quick health check..."
sleep 2
if command -v mcp-health >/dev/null; then
    mcp-health | head -10 || true
fi

# 5. Launch Claude
echo ""
echo -e "${GREEN}4. Launching Claude...${NC}"
./scripts/start-claude-ultimate.sh

echo ""
echo -e "${BLUE}Claude is starting with all optimizations!${NC}"