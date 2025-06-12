#!/bin/bash

echo "=== Ripgrep MCP Server Diagnostic ==="
echo "Date: $(date)"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check and report
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        echo "  Error: $2"
    fi
}

echo "1. Checking System Environment"
echo "=============================="

# Check NPM/NPX
echo -n "NPM version: "
npm --version 2>/dev/null || echo "NOT FOUND"

echo -n "NPX version: "
npx --version 2>/dev/null || echo "NOT FOUND"

echo -n "Node.js version: "
node --version 2>/dev/null || echo "NOT FOUND"

# Check which npx
echo -n "NPX location: "
which npx 2>/dev/null || echo "NOT FOUND"

# Check npm registry
echo -n "NPM registry: "
npm config get registry

# Check if behind proxy
echo -n "HTTP proxy: "
echo ${HTTP_PROXY:-"not set"}
echo -n "HTTPS proxy: "
echo ${HTTPS_PROXY:-"not set"}

echo ""
echo "2. Testing Package Name Variations"
echo "=================================="

# Test different package names
packages=(
    "@modelcontextprotocol/server-ripgrep"
    "@modelcontextprotocol/ripgrep-server"
    "mcp-server-ripgrep"
    "@mcp/server-ripgrep"
)

for pkg in "${packages[@]}"; do
    echo -n "Checking $pkg ... "
    if npm view "$pkg" version 2>/dev/null; then
        echo -e "${GREEN}EXISTS${NC}"
        echo "  Latest version: $(npm view "$pkg" version 2>/dev/null)"
    else
        echo -e "${RED}NOT FOUND${NC}"
    fi
done

echo ""
echo "3. Testing NPX Execution Methods"
echo "================================"

# Method 1: Standard npx with -y flag
echo "Method 1: npx -y"
timeout 10s npx -y @modelcontextprotocol/server-ripgrep@latest --help 2>&1 | head -5
check "npx -y execution" "$(timeout 5s npx -y @modelcontextprotocol/server-ripgrep@latest --help 2>&1 | head -1)"

# Method 2: Without -y flag
echo -e "\nMethod 2: npx without -y"
timeout 10s npx @modelcontextprotocol/server-ripgrep@latest --help 2>&1 | head -5
check "npx without -y" "$(timeout 5s npx @modelcontextprotocol/server-ripgrep@latest --help 2>&1 | head -1)"

# Method 3: Pre-install globally
echo -e "\nMethod 3: Global install"
npm list -g @modelcontextprotocol/server-ripgrep 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Package not installed globally. To install:"
    echo "  npm install -g @modelcontextprotocol/server-ripgrep"
fi

echo ""
echo "4. Checking Ripgrep Binary"
echo "=========================="

# Check if ripgrep is installed
echo -n "Ripgrep (rg) version: "
rg --version 2>/dev/null | head -1 || echo "NOT FOUND"

echo -n "Ripgrep location: "
which rg 2>/dev/null || echo "NOT FOUND"

# If not found, suggest installation
if ! command -v rg &> /dev/null; then
    echo -e "${YELLOW}Ripgrep not found. Install with:${NC}"
    echo "  brew install ripgrep    # macOS"
    echo "  apt install ripgrep     # Ubuntu/Debian"
    echo "  yum install ripgrep     # RHEL/CentOS"
fi

echo ""
echo "5. Testing Alternative Approaches"
echo "================================="

# Check if we can use local installation
echo "Checking for local node_modules:"
if [ -d "node_modules/@modelcontextprotocol/server-ripgrep" ]; then
    echo -e "${GREEN}Found local installation${NC}"
    echo "  Path: node_modules/@modelcontextprotocol/server-ripgrep"
else
    echo "No local installation found"
fi

# Test with full path
echo -e "\nTesting with npm exec:"
timeout 10s npm exec -y -- @modelcontextprotocol/server-ripgrep --help 2>&1 | head -5

echo ""
echo "6. Network Connectivity Test"
echo "============================"

# Test npm registry connectivity
echo -n "NPM registry reachable: "
curl -s -o /dev/null -w "%{http_code}" https://registry.npmjs.org/ | grep -q "200" && echo -e "${GREEN}YES${NC}" || echo -e "${RED}NO${NC}"

# Test specific package info
echo -n "Package metadata accessible: "
curl -s https://registry.npmjs.org/@modelcontextprotocol/server-ripgrep | jq -r .name 2>/dev/null && echo -e "${GREEN}YES${NC}" || echo -e "${RED}NO${NC}"

echo ""
echo "7. MCP Configuration Check"
echo "=========================="

# Check current config
echo "Current ripgrep config in mcp-servers.json:"
jq '.mcpServers.ripgrep' /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json 2>/dev/null || echo "Config not found"

echo ""
echo "8. Suggested Solutions"
echo "====================="

echo -e "${YELLOW}Based on the diagnostics, try these solutions:${NC}"
echo ""
echo "Solution 1: Install ripgrep binary first"
echo "  brew install ripgrep"
echo ""
echo "Solution 2: Use a Python-based alternative"
echo '  Create a Python MCP server that wraps ripgrep functionality'
echo ""
echo "Solution 3: Use pre-installed package"
echo "  npm install -g @modelcontextprotocol/server-ripgrep"
echo '  Then update mcp-servers.json to use direct command instead of npx'
echo ""
echo "Solution 4: Use local installation"
echo "  cd /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
echo "  npm install @modelcontextprotocol/server-ripgrep"
echo '  Update mcp-servers.json to use node_modules/.bin/ripgrep-server'

echo ""
echo "=== Diagnostic Complete ==="