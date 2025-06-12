#!/bin/bash

# Quick fix for python_analysis server hanging issue
# Updates mcp-servers.json to use the enhanced version

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Fixing python_analysis server..."

# Backup current config
cp "$PROJECT_ROOT/mcp-servers.json" "$PROJECT_ROOT/mcp-servers.json.backup.$(date +%Y%m%d_%H%M%S)"

# Update to use enhanced version
if command -v jq &> /dev/null; then
    # Use jq for clean JSON manipulation
    jq '.mcpServers.python_analysis = {
        "transport": "stdio",
        "command": "/Users/mikeedwards/.pyenv/shims/python3",
        "args": [
            "'"$SCRIPT_DIR/python-mcp-server-enhanced.py"'"
        ],
        "env": {
            "PYTHONPATH": "'"$PROJECT_ROOT"'",
            "WORKSPACE_ROOT": "'"$PROJECT_ROOT"'",
            "DATABENTO_API_KEY": "${DATABENTO_API_KEY}",
            "FRED_API_KEY": "${FRED_API_KEY}",
            "PYTHONOPTIMIZE": "1"
        }
    }' "$PROJECT_ROOT/mcp-servers.json" > "$PROJECT_ROOT/mcp-servers.json.tmp" && \
    mv "$PROJECT_ROOT/mcp-servers.json.tmp" "$PROJECT_ROOT/mcp-servers.json"
else
    # Fallback to sed
    sed -i.bak 's|python-mcp-server\.py|python-mcp-server-enhanced.py|g' "$PROJECT_ROOT/mcp-servers.json"
fi

echo "✓ Fixed python_analysis server to use enhanced version"
echo "  The enhanced version includes:"
echo "  - Health monitoring"
echo "  - Actual trading analysis integration"
echo "  - System monitoring"
echo "  - Test runner"
echo ""
echo "Run './scripts/start-claude-ultimate.sh' to launch Claude with all fixes"