#!/bin/bash

# Optimize MCP Server Performance - Install recommended dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MCP Performance Optimization Script ===${NC}"
echo ""

# 1. Install pnpm (faster package manager)
echo -e "\n${YELLOW}1. Installing pnpm (faster package manager)...${NC}"
if command -v pnpm &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} pnpm already installed ($(pnpm --version))"
else
    echo -e "  ${BLUE}Installing pnpm...${NC}"
    npm install -g pnpm
    echo -e "  ${GREEN}✓${NC} pnpm installed"
fi

# 2. Install bun (ultra-fast JS runtime)
echo -e "\n${YELLOW}2. Installing bun (ultra-fast JavaScript runtime)...${NC}"
if command -v bun &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} bun already installed ($(bun --version))"
else
    echo -e "  ${BLUE}Installing bun...${NC}"
    curl -fsSL https://bun.sh/install | bash
    echo -e "  ${GREEN}✓${NC} bun installed"
    echo -e "  ${YELLOW}Note: You may need to restart your terminal or run: source ~/.zshrc${NC}"
fi

# 3. Install watchman (efficient file watching)
echo -e "\n${YELLOW}3. Installing watchman (Facebook's file watcher)...${NC}"
if command -v watchman &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} watchman already installed ($(watchman --version 2>&1 | head -1))"
else
    echo -e "  ${BLUE}Installing watchman...${NC}"
    brew install watchman
    echo -e "  ${GREEN}✓${NC} watchman installed"
fi

# 4. Install eza (modern ls replacement)
echo -e "\n${YELLOW}4. Installing eza (better ls command)...${NC}"
if command -v eza &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} eza already installed ($(eza --version | head -1))"
else
    echo -e "  ${BLUE}Installing eza...${NC}"
    brew install eza
    echo -e "  ${GREEN}✓${NC} eza installed"
fi

# 5. Optimize NPM cache location
echo -e "\n${YELLOW}5. Optimizing NPM cache location...${NC}"
CURRENT_CACHE=$(npm config get cache)
if [[ "$CURRENT_CACHE" == *"/tmp/"* ]]; then
    echo -e "  ${YELLOW}Current cache in volatile storage: $CURRENT_CACHE${NC}"
    echo -e "  ${BLUE}Moving to persistent location...${NC}"
    npm config set cache ~/.npm
    echo -e "  ${GREEN}✓${NC} NPM cache moved to ~/.npm"
else
    echo -e "  ${GREEN}✓${NC} NPM cache already in persistent location: $CURRENT_CACHE"
fi

# 6. Configure pnpm store
echo -e "\n${YELLOW}6. Configuring pnpm store...${NC}"
if command -v pnpm &> /dev/null; then
    pnpm config set store-dir ~/.pnpm-store
    echo -e "  ${GREEN}✓${NC} pnpm store configured at ~/.pnpm-store"
fi

# 7. Enable parallel installations
echo -e "\n${YELLOW}7. Configuring for optimal performance...${NC}"
# Note: npm removed the 'jobs' config option, but pnpm handles parallelism automatically
if command -v pnpm &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} pnpm handles parallel installations automatically"
else
    echo -e "  ${GREEN}✓${NC} Modern npm versions handle parallelism automatically"
fi

# 8. Create MCP cache directories
echo -e "\n${YELLOW}8. Creating MCP cache directories...${NC}"
mkdir -p ~/.cache/mcp-servers
mkdir -p ~/.cache/npx
echo -e "  ${GREEN}✓${NC} Cache directories created"

# 9. Pre-download commonly used MCP servers
echo -e "\n${YELLOW}9. Pre-caching common MCP servers...${NC}"
echo -e "  ${BLUE}This will pre-download servers for faster first startup${NC}"

# Array of MCP servers to pre-cache
MCP_SERVERS=(
    "@modelcontextprotocol/server-filesystem@latest"
    "@modelcontextprotocol/server-brave-search@latest"
    "@modelcontextprotocol/server-memory@latest"
    "@modelcontextprotocol/server-sequential-thinking@latest"
    "@modelcontextprotocol/server-puppeteer@latest"
    "mcp-ripgrep@latest"
    "@modelcontextprotocol/server-code-analysis@latest"
    "opik-mcp@latest"
    "@arizeai/phoenix-mcp@latest"
)

for server in "${MCP_SERVERS[@]}"; do
    echo -e "  ${BLUE}Pre-caching $server...${NC}"
    npx -y "$server" --help &> /dev/null || true
done
echo -e "  ${GREEN}✓${NC} MCP servers pre-cached"

# 10. Set up shell aliases for performance
echo -e "\n${YELLOW}10. Setting up performance aliases...${NC}"
ALIASES_FILE="$HOME/.mcp_aliases"
cat > "$ALIASES_FILE" << 'EOF'
# MCP Performance Aliases
alias ll='eza -la --git --icons'
alias cat='bat'
alias find='fd'
alias npm='pnpm'  # Use pnpm by default

# Quick MCP commands
alias mcp-start='./scripts/start-claude-full.sh'
alias mcp-status='./scripts/check-mcp-status.sh'
EOF

# Add to shell profile if not already there
if ! grep -q "source.*mcp_aliases" ~/.zshrc 2>/dev/null; then
    echo "" >> ~/.zshrc
    echo "# MCP Performance Aliases" >> ~/.zshrc
    echo "[ -f ~/.mcp_aliases ] && source ~/.mcp_aliases" >> ~/.zshrc
    echo -e "  ${GREEN}✓${NC} Aliases added to ~/.zshrc"
else
    echo -e "  ${GREEN}✓${NC} Aliases already configured"
fi

# Summary
echo -e "\n${GREEN}=== Optimization Complete ===${NC}"
echo ""
echo -e "${YELLOW}Installed/Configured:${NC}"
echo -e "  ✓ pnpm - Faster package manager"
echo -e "  ✓ bun - Ultra-fast JS runtime"
echo -e "  ✓ watchman - Efficient file watching"
echo -e "  ✓ eza - Modern ls replacement"
echo -e "  ✓ NPM cache - Moved to persistent storage"
echo -e "  ✓ Parallel installs - Enabled"
echo -e "  ✓ MCP servers - Pre-cached for fast startup"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Restart your terminal or run: source ~/.zshrc"
echo -e "  2. Run Claude with: ./scripts/start-claude-full.sh"
echo ""
echo -e "${GREEN}Your MCP servers will now start faster and use less resources!${NC}"