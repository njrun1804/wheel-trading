#!/bin/bash

# Simple Claude launcher with minimal MCP servers

set -e

# Find Claude command
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
else
    echo "Error: Claude command not found"
    exit 1
fi

# Configuration options
CONFIG_FILE="${1:-mcp-servers-minimal.json}"
DEBUG_MODE="${MCP_DEBUG:-0}"

# Show what we're doing
echo "Starting Claude with MCP servers"
echo "Configuration: $CONFIG_FILE"
[ "$DEBUG_MODE" = "1" ] && echo "Debug mode: ON"

# Launch Claude
if [ "$DEBUG_MODE" = "1" ]; then
    MCP_DEBUG=1 $CLAUDE_CMD --mcp-config "$CONFIG_FILE"
else
    $CLAUDE_CMD --mcp-config "$CONFIG_FILE"
fi
