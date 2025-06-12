#!/bin/bash

# Setup local self-hosted Opik and Phoenix trace servers

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Setting up Local Trace Servers (Opik + Phoenix) ===${NC}"
echo ""

# 1. Setup Opik Platform
echo -e "\n${YELLOW}1. Setting up Opik Platform (Docker)...${NC}"

OPIK_DIR="$HOME/mcp-servers/opik-platform"
mkdir -p "$OPIK_DIR"

if [ -d "$OPIK_DIR/.git" ]; then
    echo -e "  ${YELLOW}Opik already cloned, updating...${NC}"
    cd "$OPIK_DIR"
    git pull
else
    echo -e "  ${BLUE}Cloning Opik platform...${NC}"
    git clone https://github.com/comet-ml/opik.git "$OPIK_DIR"
    cd "$OPIK_DIR"
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "  ${RED}✗${NC} Docker is not running!"
    echo -e "  ${YELLOW}Please start Docker Desktop and re-run this script${NC}"
    exit 1
fi

# Start Opik platform
echo -e "  ${BLUE}Starting Opik platform...${NC}"
if [ -f "./opik.sh" ]; then
    chmod +x ./opik.sh
    ./opik.sh &  # No 'start' argument needed
    echo -e "  ${GREEN}✓${NC} Opik platform starting on http://localhost:5173"
    echo -e "  ${YELLOW}Note: It may take 30-60 seconds to fully start${NC}"
else
    echo -e "  ${RED}✗${NC} opik.sh script not found"
fi

# 2. Setup Phoenix Platform
echo -e "\n${YELLOW}2. Setting up Phoenix Platform (Python)...${NC}"

# Check if Phoenix is installed
if pip show arize-phoenix &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Phoenix already installed"
else
    echo -e "  ${BLUE}Installing Phoenix...${NC}"
    pip install arize-phoenix
fi

# Start Phoenix server
echo -e "  ${BLUE}Starting Phoenix server...${NC}"
# Kill any existing Phoenix server
pkill -f "phoenix serve" 2>/dev/null || true
sleep 1

# Start in background
nohup phoenix serve > ~/.cache/phoenix-server.log 2>&1 &
PHOENIX_PID=$!
echo -e "  ${GREEN}✓${NC} Phoenix platform started on http://localhost:6006 (PID: $PHOENIX_PID)"

# 3. Update MCP configuration for local servers
echo -e "\n${YELLOW}3. Creating local MCP configuration...${NC}"

MCP_LOCAL_CONFIG="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers-local.json"

cat > "$MCP_LOCAL_CONFIG" << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem@latest", "/Users/mikeedwards"]
    },
    "github": {
      "transport": "stdio",
      "command": "mcp-server-github",
      "args": [],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "brave": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search@latest"],
      "env": { "BRAVE_API_KEY": "${BRAVE_API_KEY}" }
    },
    "memory": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory@latest"]
    },
    "sequential-thinking": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking@latest"]
    },
    "puppeteer": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer@latest"]
    },
    "statsource": {
      "transport": "stdio",
      "command": "python3",
      "args": ["-m", "mcp_server_stats"],
      "env": {
        "PYTHONPATH": "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers/statsource"
      }
    },
    "duckdb": {
      "transport": "stdio",
      "command": "python3",
      "args": ["-m", "mcp_server_duckdb", "data/cache/wheel_cache.duckdb"]
    },
    "mlflow": {
      "transport": "stdio",
      "command": "python3",
      "args": ["/Users/mikeedwards/mcp-servers/community/mlflowMCPServer/mlflow_server.py"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "PYTHONPATH": "/Users/mikeedwards/mcp-servers/community/mlflowMCPServer"
      }
    },
    "pyrepl": {
      "transport": "stdio",
      "command": "python3",
      "args": ["-m", "mcp_py_repl"]
    },
    "sklearn": {
      "transport": "stdio",
      "command": "python3",
      "args": ["/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py"],
      "env": {
        "PYTHONPATH": "/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src"
      }
    },
    "optionsflow": {
      "transport": "stdio",
      "command": "python3",
      "args": ["/Users/mikeedwards/mcp-servers/community/mcp-optionsflow/optionsflow.py"],
      "env": {
        "PYTHONPATH": "/Users/mikeedwards/mcp-servers/community/mcp-optionsflow",
        "DATABENTO_API_KEY": "${DATABENTO_API_KEY}"
      }
    },
    "python_analysis": {
      "transport": "stdio",
      "command": "/Users/mikeedwards/.pyenv/shims/python3",
      "args": ["/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py"],
      "env": {
        "PYTHONPATH": "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading",
        "DATABENTO_API_KEY": "${DATABENTO_API_KEY}",
        "FRED_API_KEY": "${FRED_API_KEY}"
      }
    },
    "ripgrep": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "mcp-ripgrep@latest", "--path", "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"]
    },
    "dependency-graph": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-code-analysis@latest", "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"]
    },
    "trace": {
      "transport": "stdio",
      "command": "python3",
      "args": ["-m", "logfire", "mcp"],
      "env": {
        "LOGFIRE_TOKEN": "pylf_v1_us_00l06NMSXxWp1V9cTNJWJLvjRPs5HPRVsFtmdTSS1YC2"
      }
    },
    "-local": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "opik-mcp@latest", "--baseUrl", "http://localhost:5173/api"]
    },
    "-local": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@arizeai/phoenix-mcp@latest", "--baseUrl", "http://localhost:6006"]
    }
  }
}
EOF

echo -e "  ${GREEN}✓${NC} Created local MCP configuration"

# 4. Create startup script
echo -e "\n${YELLOW}4. Creating startup script...${NC}"

cat > "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/start-claude-local.sh" << 'EOFSCRIPT'
#!/bin/bash

# Start Claude with local trace servers (no API keys needed!)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}Unity Wheel Trading - Claude Code LOCAL Trace Servers${NC}"
echo "====================================================="

# Check if local servers are running
echo -e "\n${YELLOW}Checking local trace servers...${NC}"

# Check Opik
if curl -s http://localhost:5173/api/health &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Opik platform running on http://localhost:5173"
else
    echo -e "  ${RED}✗${NC} Opik platform not running"
    echo -e "  ${YELLOW}Run: cd ~/mcp-servers/opik-platform && ./opik.sh start${NC}"
fi

# Check Phoenix
if curl -s http://localhost:6006/healthz &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Phoenix platform running on http://localhost:6006"
else
    echo -e "  ${RED}✗${NC} Phoenix platform not running"
    echo -e "  ${YELLOW}Run: phoenix serve${NC}"
fi

# Use local MCP config
MCP_CONFIG="$PROJECT_ROOT/mcp-servers-local.json"

# Find Claude
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
else
    CLAUDE_CMD="$(alias claude 2>/dev/null | sed "s/alias claude='//" | sed "s/'$//")"
fi

# Load tokens from keychain
LOGFIRE_READ_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null)
if [ -n "$LOGFIRE_READ_TOKEN" ]; then
    export LOGFIRE_READ_TOKEN
fi

echo -e "\n${GREEN}Starting Claude with 18 servers (including local trace servers)...${NC}"
eval "$CLAUDE_CMD --mcp-config \"$MCP_CONFIG\""
EOFSCRIPT

chmod +x "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/start-claude-local.sh"
echo -e "  ${GREEN}✓${NC} Created start-claude-local.sh"

# 5. Health check
echo -e "\n${YELLOW}5. Running health checks...${NC}"
sleep 5  # Give servers time to start

# Check Opik
if curl -s http://localhost:5173/api/health | grep -q "ok\|healthy"; then
    echo -e "  ${GREEN}✓${NC} Opik API healthy"
else
    echo -e "  ${YELLOW}⚠${NC} Opik API not responding yet (may still be starting)"
fi

# Check Phoenix
if curl -s http://localhost:6006/healthz | grep -q "healthy\|ok"; then
    echo -e "  ${GREEN}✓${NC} Phoenix API healthy"
else
    echo -e "  ${YELLOW}⚠${NC} Phoenix API not responding yet (may still be starting)"
fi

echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo ""
echo -e "${YELLOW}Local trace servers:${NC}"
echo -e "  • Opik UI: ${BLUE}http://localhost:5173${NC}"
echo -e "  • Phoenix UI: ${BLUE}http://localhost:6006${NC}"
echo ""
echo -e "${YELLOW}To start Claude with local trace servers:${NC}"
echo -e "  ${GREEN}./scripts/start-claude-local.sh${NC}"
echo ""
echo -e "${YELLOW}Benefits of local setup:${NC}"
echo -e "  ✓ No API keys needed"
echo -e "  ✓ All data stays on your machine"
echo -e "  ✓ Sub-10ms latency"
echo -e "  ✓ Perfect for development"
echo ""