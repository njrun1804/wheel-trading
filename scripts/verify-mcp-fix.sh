#!/bin/bash

echo "🔧 MCP CONFIGURATION FIX VERIFICATION"
echo "====================================="

echo "✅ FIXES APPLIED:"
echo "1. Updated .mcp.json with full paths (/usr/local/bin/npx)"
echo "2. Fixed Python MCP server to output proper JSON-RPC messages"
echo "3. Added proper MCP protocol initialization"
echo "4. Synchronized both .mcp.json and mcp-working.json"

echo ""
echo "📋 CURRENT MCP SERVER STATUS:"

# Test filesystem server
echo "🗂️  Filesystem: $(timeout 2s /usr/local/bin/npx -y @modelcontextprotocol/server-filesystem /Users/mikeedwards <<< '' 2>&1 | head -1 | grep -q "Secure MCP" && echo "✅ Working" || echo "❌ Failed")"

# Test GitHub server
echo "🐙 GitHub: $(timeout 2s /usr/local/bin/npx -y @modelcontextprotocol/server-github <<< '' 2>&1 | head -1 | grep -q "GitHub MCP" && echo "✅ Working" || echo "❌ Failed")"

# Test Python server
echo "🐍 Python Analysis: $(timeout 2s /Users/mikeedwards/.pyenv/shims/python3 "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py" <<< '' 2>&1 | head -1 | grep -q "jsonrpc" && echo "✅ Working" || echo "❌ Failed")"

echo ""
echo "🚀 NEXT STEPS:"
echo "1. Restart Claude Code to load new MCP configuration"
echo "2. Test with: claude 'List files in current directory'"
echo "3. Your MCP superpowers should now be fully functional!"

echo ""
echo "💡 USAGE EXAMPLES:"
echo "- 'List my trading data files'"
echo "- 'Analyze my current wheel position'"
echo "- 'Check data quality in master database'"
echo "- 'Search GitHub for options pricing code'"
