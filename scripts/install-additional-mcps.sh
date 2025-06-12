#!/bin/bash

# Install script for additional MCP servers (ripgrep, dependency-graph, trace)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Installing Additional MCP Servers ===${NC}"
echo ""

# 1. Install ripgrep binary if not present
echo -e "${YELLOW}1. Checking ripgrep installation...${NC}"
if command -v rg &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} ripgrep already installed ($(rg --version | head -1))"
else
    echo -e "  ${BLUE}Installing ripgrep...${NC}"
    brew install ripgrep
    echo -e "  ${GREEN}✓${NC} ripgrep installed"
fi

# 2. Install Node.js if not present
echo -e "\n${YELLOW}2. Checking Node.js installation...${NC}"
if command -v node &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Node.js already installed ($(node --version))"
else
    echo -e "  ${BLUE}Installing Node.js...${NC}"
    brew install node@20
    echo -e "  ${GREEN}✓${NC} Node.js installed"
fi

# 3. Install dependency-graph MCP server
echo -e "\n${YELLOW}3. Installing dependency-graph MCP server...${NC}"
MCP_DIR="$HOME/mcp-servers"
mkdir -p "$MCP_DIR"

if [ -d "$MCP_DIR/dependency-mcp" ]; then
    echo -e "  ${YELLOW}dependency-mcp already exists, updating...${NC}"
    cd "$MCP_DIR/dependency-mcp"
    git pull
else
    echo -e "  ${BLUE}Cloning dependency-mcp...${NC}"
    cd "$MCP_DIR"
    git clone https://github.com/mkearl/dependency-mcp.git
    cd dependency-mcp
fi

echo -e "  ${BLUE}Installing dependencies...${NC}"
npm install
echo -e "  ${BLUE}Building...${NC}"
npm run build
echo -e "  ${GREEN}✓${NC} dependency-graph MCP server installed"

# 4. Install Python-based trace MCP server
echo -e "\n${YELLOW}4. Installing trace (logfire) MCP server...${NC}"

# Check if uvx is available (it's part of uv)
if ! command -v uvx &> /dev/null; then
    echo -e "  ${BLUE}Installing uv (includes uvx)...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo -e "  ${GREEN}✓${NC} uv/uvx installed"
fi

# Install logfire-mcp package
if pip show logfire-mcp &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} logfire-mcp already installed"
else
    echo -e "  ${BLUE}Installing logfire-mcp...${NC}"
    pip install logfire-mcp logfire
    echo -e "  ${GREEN}✓${NC} logfire-mcp installed"
fi

# Verify uvx can find logfire-mcp
echo -e "  ${BLUE}Verifying logfire-mcp accessibility...${NC}"
if uvx --help &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} uvx command available"
else
    echo -e "  ${RED}✗${NC} uvx command not found - trace server may not work"
fi

# 5. Setup Logfire token in keychain
echo -e "\n${YELLOW}5. Setting up Logfire token...${NC}"

# Check if token is already in keychain
EXISTING_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null)
if [ -n "$EXISTING_TOKEN" ]; then
    echo -e "  ${GREEN}✓${NC} LOGFIRE_READ_TOKEN already configured in keychain"
else
    # Use the provided token
    LOGFIRE_TOKEN="pylf_v1_us_00l06NMSXxWp1V9cTNJWJLvjRPs5HPRVsFtmdTSS1YC2"
    echo -e "  ${BLUE}Adding Logfire token to keychain...${NC}"
    security add-generic-password -a "$USER" -s "logfire-mcp" -w "$LOGFIRE_TOKEN" -U
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} LOGFIRE_READ_TOKEN added to keychain"
    else
        echo -e "  ${RED}✗${NC} Failed to add token to keychain"
    fi
fi

# Test token retrieval
TEST_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null)
if [ -n "$TEST_TOKEN" ]; then
    export LOGFIRE_READ_TOKEN="$TEST_TOKEN"
    echo -e "  ${GREEN}✓${NC} Token successfully retrieved from keychain"
else
    echo -e "  ${RED}✗${NC} Failed to retrieve token from keychain"
fi

# 6. Test ripgrep MCP (it uses npx so no installation needed)
echo -e "\n${YELLOW}6. Testing ripgrep MCP server...${NC}"
echo -e "  ${BLUE}Testing npx command...${NC}"
if npx -y mcp-ripgrep@latest --help &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} ripgrep MCP server is available via npx"
else
    echo -e "  ${RED}✗${NC} Failed to run ripgrep MCP server"
fi

# 7. Install Opik MCP server
echo -e "\n${YELLOW}7. Installing Opik MCP server...${NC}"
if [ -d "$MCP_DIR/opik-mcp" ]; then
    echo -e "  ${YELLOW}opik-mcp already exists, updating...${NC}"
    cd "$MCP_DIR/opik-mcp"
    git pull
else
    echo -e "  ${BLUE}Cloning opik-mcp...${NC}"
    cd "$MCP_DIR"
    git clone https://github.com/comet-ml/opik-mcp.git
    cd opik-mcp
fi

echo -e "  ${BLUE}Installing dependencies...${NC}"
npm install
echo -e "  ${BLUE}Building...${NC}"
npm run build
echo -e "  ${GREEN}✓${NC} Opik MCP server installed"

# Setup .env file if not exists
if [ ! -f "$MCP_DIR/opik-mcp/.env" ] && [ -f "$MCP_DIR/opik-mcp/.env.example" ]; then
    cp "$MCP_DIR/opik-mcp/.env.example" "$MCP_DIR/opik-mcp/.env"
    echo -e "  ${YELLOW}Note: Configure OPIK_API_KEY in $MCP_DIR/opik-mcp/.env${NC}"
fi

# 8. Install Phoenix Trace MCP server
echo -e "\n${YELLOW}8. Installing Phoenix Trace MCP server...${NC}"
if pip show arize-phoenix &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} arize-phoenix already installed"
else
    echo -e "  ${BLUE}Installing arize-phoenix...${NC}"
    pip install arize-phoenix arize-phoenix-otel
    echo -e "  ${GREEN}✓${NC} arize-phoenix installed"
fi

if pip show phoenix-trace-mcp &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} phoenix-trace-mcp already installed"
else
    echo -e "  ${BLUE}Installing phoenix-trace-mcp...${NC}"
    pip install phoenix-trace-mcp
    echo -e "  ${GREEN}✓${NC} phoenix-trace-mcp installed"
fi

echo -e "\n${GREEN}=== Installation Summary ===${NC}"
echo -e "  ${GREEN}✓${NC} ripgrep binary installed"
echo -e "  ${GREEN}✓${NC} Node.js installed"
echo -e "  ${GREEN}✓${NC} dependency-graph MCP server built at: $MCP_DIR/dependency-mcp"
echo -e "  ${GREEN}✓${NC} trace (logfire) MCP server installed"
echo -e "  ${GREEN}✓${NC}  MCP server built at: $MCP_DIR/opik-mcp"
echo -e "  ${GREEN}✓${NC}  MCP server installed"
echo ""
echo -e "${YELLOW}Total MCP servers available: 18${NC}"
echo -e "  Original: 13 servers"
echo -e "  New: 5 servers (ripgrep, dependency-graph, trace, )"
echo ""
echo -e "${BLUE}To start Claude with all 18 MCP servers:${NC}"
echo -e "  ./scripts/start-claude-full.sh"
echo ""
echo -e "${YELLOW}Environment variables needed:${NC}"
echo -e "  - OPIK_API_KEY (get from comet.com/opik)"
echo -e "  - PHOENIX_COLLECTOR_ENDPOINT (optional, for Phoenix Cloud)"
echo -e "  - PHOENIX_CLIENT_HEADERS (optional, for Phoenix Cloud)"
echo ""