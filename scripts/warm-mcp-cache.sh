#!/bin/bash
# Pre-download NPX packages for faster startup
echo "Pre-warming MCP cache..."
npx -y @modelcontextprotocol/server-filesystem@latest --help >/dev/null 2>&1 &
npx -y @modelcontextprotocol/server-brave-search@latest --help >/dev/null 2>&1 &
npx -y @modelcontextprotocol/server-memory@latest --help >/dev/null 2>&1 &
npx -y @modelcontextprotocol/server-sequential-thinking@latest --help >/dev/null 2>&1 &
npx -y @modelcontextprotocol/server-puppeteer@latest --help >/dev/null 2>&1 &
npx -y @modelcontextprotocol/server-ripgrep --help >/dev/null 2>&1 &
npx -y @modelcontextprotocol/server-code-analysis --help >/dev/null 2>&1 &
wait
echo "✓ Cache warmed"
