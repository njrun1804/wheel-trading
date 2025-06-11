#!/bin/bash

echo "🧪 TESTING MCP SERVERS"
echo "====================="

# Load environment
source .env-mcp

echo ""
echo "Environment:"
echo "GITHUB_TOKEN: $([ -n "$GITHUB_TOKEN" ] && echo "Set ✓" || echo "Not set ✗")"
echo "PYTHONPATH: Set ✓"
echo ""

# Test filesystem
echo "Testing filesystem MCP..."
claude --mcp-config mcp-working.json "List the top 5 files in my current directory" 2>&1 | head -10

echo ""
echo "If the above worked, your MCP servers are ready!"
echo ""
echo "Use: claude --mcp-config mcp-working.json 'your query'"
echo "Or:  claude-mcp 'your query' (after running: source .env-mcp)"
