#!/bin/bash
# Setup Logfire environment variables from keychain

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the Logfire token from keychain directly
LOGFIRE_READ_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null)

if [ -n "$LOGFIRE_READ_TOKEN" ]; then
    export LOGFIRE_READ_TOKEN
    echo "✓ Exported LOGFIRE_READ_TOKEN from keychain"
else
    echo "✗ Failed to retrieve logfire-mcp from keychain" >&2
    return 1 2>/dev/null || exit 1
fi

# Optional: Export other Logfire-related environment variables if needed
# export_from_keychain "LOGFIRE_WRITE_TOKEN" "logfire-mcp-write"
# export_from_keychain "LOGFIRE_PROJECT_NAME" "logfire-project"

# Return success if token was exported
if [ -n "$LOGFIRE_READ_TOKEN" ]; then
    echo "✓ Logfire environment configured successfully"
    return 0 2>/dev/null || exit 0
else
    echo "✗ Failed to configure Logfire environment" >&2
    return 1 2>/dev/null || exit 1
fi