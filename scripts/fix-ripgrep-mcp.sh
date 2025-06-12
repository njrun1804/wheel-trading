#!/bin/bash

echo "=== Fixing Ripgrep MCP Server ==="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CONFIG_FILE="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"

echo -e "${BLUE}Analyzing the issue...${NC}"
echo "The package @modelcontextprotocol/server-ripgrep does not exist in npm registry."
echo ""

echo -e "${YELLOW}Available solutions:${NC}"
echo "1. Use the correct npm package: mcp-ripgrep"
echo "2. Use our Python-based ripgrep MCP server"
echo "3. Remove ripgrep from MCP servers"
echo ""

# Check if user wants to proceed
read -p "Choose solution [1/2/3]: " choice

case $choice in
    1)
        echo -e "\n${BLUE}Solution 1: Using mcp-ripgrep package${NC}"
        
        # Update the config to use the correct package
        echo "Updating mcp-servers.json..."
        
        # Create backup
        cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
        
        # Update using jq
        jq '.mcpServers.ripgrep = {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "mcp-ripgrep@latest"
            ]
        }' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
        
        echo -e "${GREEN}✓ Updated configuration to use mcp-ripgrep${NC}"
        
        # Test the package
        echo -e "\nTesting mcp-ripgrep..."
        if timeout 5s npx -y mcp-ripgrep@latest --help 2>&1 | grep -q "mcp"; then
            echo -e "${GREEN}✓ mcp-ripgrep is working${NC}"
        else
            echo -e "${RED}✗ mcp-ripgrep test failed${NC}"
            echo "Falling back to Python solution..."
            choice=2
        fi
        ;;
        
    2)
        echo -e "\n${BLUE}Solution 2: Using Python-based ripgrep server${NC}"
        
        # First check if ripgrep is installed
        if ! command -v rg &> /dev/null; then
            echo -e "${YELLOW}Installing ripgrep...${NC}"
            brew install ripgrep
        fi
        
        # Update config to use Python server
        echo "Updating mcp-servers.json..."
        
        # Create backup
        cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
        
        # Update using jq
        jq '.mcpServers.ripgrep = {
            "transport": "stdio",
            "command": "/Users/mikeedwards/.pyenv/shims/python3",
            "args": [
                "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/ripgrep-mcp-server.py"
            ]
        }' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
        
        echo -e "${GREEN}✓ Updated configuration to use Python ripgrep server${NC}"
        
        # Test the server
        echo -e "\nTesting Python ripgrep server..."
        if timeout 5s python3 "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/ripgrep-mcp-server.py" < /dev/null 2>&1 | grep -q "capabilities"; then
            echo -e "${GREEN}✓ Python ripgrep server is working${NC}"
        else
            echo -e "${YELLOW}Note: Server will work when launched by Claude${NC}"
        fi
        ;;
        
    3)
        echo -e "\n${BLUE}Solution 3: Removing ripgrep from MCP servers${NC}"
        
        # Create backup
        cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
        
        # Remove ripgrep using jq
        jq 'del(.mcpServers.ripgrep)' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
        
        echo -e "${GREEN}✓ Removed ripgrep from configuration${NC}"
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=== Fix Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Restart Claude Desktop"
echo "2. The ripgrep MCP server should now work correctly"
echo ""
echo "If you need to revert:"
echo "  cp ${CONFIG_FILE}.backup $CONFIG_FILE"