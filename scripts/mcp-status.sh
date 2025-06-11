#!/bin/bash

echo "MCP Server Status"
echo "================="
echo ""

# Check for MCP processes (they run on stdio, not network ports)
if pgrep -f "server-filesystem" > /dev/null; then
    echo "✓ Filesystem server running (stdio mode)"
else
    echo "✗ Filesystem server not running"
fi

if pgrep -f "server-github" > /dev/null; then
    echo "✓ GitHub server running (stdio mode)"
else
    echo "✗ GitHub server not running"
fi

if pgrep -f "mcp-web-search" > /dev/null; then
    echo "✓ Web search server running (stdio mode)"
else
    echo "✗ Web search server not running"
fi

if pgrep -f "wikipedia-mcp" > /dev/null; then
    echo "✓ Wikipedia server running (stdio mode)"
else
    echo "✗ Wikipedia server not running"
fi

echo ""
echo "Note: MCP servers run on stdio for Claude integration, not network ports"
echo ""
echo "To start servers: ./scripts/start-mcp-servers.sh"
echo "To stop servers:  pkill -f 'modelcontextprotocol|mcp-web-search|wikipedia-mcp'"
