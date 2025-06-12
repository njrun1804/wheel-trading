#!/bin/bash
# Script to add Logfire MCP server configuration with token from keychain

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Get the Logfire token from keychain directly
LOGFIRE_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null)

if [ -z "$LOGFIRE_TOKEN" ]; then
    echo "✗ Failed to retrieve Logfire token from keychain" >&2
    echo "  Please run: security add-generic-password -a \"$USER\" -s \"logfire-mcp\" -w \"YOUR_TOKEN\" -U"
    exit 1
fi

# Create a temporary file with the updated MCP configuration
MCP_CONFIG="$PROJECT_ROOT/mcp-servers.json"
TEMP_FILE=$(mktemp)

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "✗ jq is required but not installed. Please install it with: brew install jq" >&2
    exit 1
fi

# Add the Logfire MCP server to the configuration
if [ -f "$MCP_CONFIG" ]; then
    # Check if logfire server already exists
    if jq -e '.mcpServers.logfire' "$MCP_CONFIG" > /dev/null 2>&1; then
        echo "ℹ️  Logfire MCP server already configured in mcp-servers.json"
        # Update the token
        jq --arg token "$LOGFIRE_TOKEN" \
           '.mcpServers.logfire.env.LOGFIRE_READ_TOKEN = $token' \
           "$MCP_CONFIG" > "$TEMP_FILE"
    else
        # Add the logfire server
        jq --arg token "$LOGFIRE_TOKEN" \
           '.mcpServers.logfire = {
              "transport": "stdio",
              "command": "uvx",
              "args": ["logfire-mcp"],
              "env": {
                "LOGFIRE_READ_TOKEN": $token
              }
            }' \
           "$MCP_CONFIG" > "$TEMP_FILE"
    fi
    
    # Replace the original file
    mv "$TEMP_FILE" "$MCP_CONFIG"
    echo "✓ Logfire MCP server configuration updated in mcp-servers.json"
else
    echo "✗ MCP configuration file not found at: $MCP_CONFIG" >&2
    exit 1
fi

# Test the configuration by running the Logfire MCP server
echo "Testing Logfire MCP server connection..."
if LOGFIRE_READ_TOKEN="$LOGFIRE_TOKEN" timeout 5 uvx logfire-mcp --help > /dev/null 2>&1; then
    echo "✓ Logfire MCP server is accessible"
else
    echo "⚠️  Could not verify Logfire MCP server (this might be normal if it requires Python 3.12+)"
fi

echo ""
echo "To use the Logfire token in your shell, run:"
echo "  source \"$SCRIPT_DIR/setup-logfire-env.sh\""