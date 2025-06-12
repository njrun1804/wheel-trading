#!/bin/bash

# Quick test to verify Claude can be started

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Claude MCP Test ===${NC}"
echo ""

# Find claude
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
    echo -e "Claude found at: ${GREEN}$CLAUDE_CMD${NC}"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
    echo -e "Claude found in PATH: ${GREEN}$(which claude)${NC}"
else
    # Claude might be aliased
    CLAUDE_CMD="$(alias claude 2>/dev/null | sed "s/alias claude='//" | sed "s/'$//")"
    if [ -n "$CLAUDE_CMD" ] && [ -f "$CLAUDE_CMD" ]; then
        echo -e "Claude found via alias: ${GREEN}$CLAUDE_CMD${NC}"
    else
        echo -e "${RED}Claude not found!${NC}"
        exit 1
    fi
fi

echo ""
echo "To start Claude with MCP servers, use one of these:"
echo -e "  ${YELLOW}./scripts/start-claude-fixed.sh${NC}  - 13 original servers"
echo -e "  ${YELLOW}./scripts/start-claude-full.sh${NC}   - All 18 servers"
echo ""
echo -e "${GREEN}Ready to go!${NC}"