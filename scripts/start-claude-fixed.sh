#!/bin/bash

# Start Claude Code with MCP servers configured for Unity Wheel Trading
# This script starts 13 MCP servers: 6 Node.js-based and 7 Python-based

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}Unity Wheel Trading - Claude Code MCP Launcher${NC}"
echo "================================================"

# Use the full MCP config with 13 servers
MCP_CONFIG="$PROJECT_ROOT/mcp-servers.json"
if [ ! -f "$MCP_CONFIG" ]; then
    echo -e "${RED}Error: MCP configuration file not found at $MCP_CONFIG${NC}"
    exit 1
fi

# Claude command location - check multiple locations
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
else
    # Claude might be aliased
    CLAUDE_CMD="$(alias claude 2>/dev/null | sed "s/alias claude='//" | sed "s/'$//")"
    if [ -z "$CLAUDE_CMD" ] || [ ! -f "$CLAUDE_CMD" ]; then
        echo -e "${RED}Error: Claude command not found${NC}"
        echo "Please ensure Claude Code is installed"
        echo "Try running: which claude"
        exit 1
    fi
fi

# Load Logfire token from keychain
echo -e "\n${YELLOW}Loading tokens from keychain...${NC}"
LOGFIRE_READ_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null)
if [ -n "$LOGFIRE_READ_TOKEN" ]; then
    export LOGFIRE_READ_TOKEN
    echo -e "  ${GREEN}✓${NC} Loaded LOGFIRE_READ_TOKEN from keychain"
else
    echo -e "  ${YELLOW}⚠${NC} LOGFIRE_READ_TOKEN not found in keychain"
fi

# Check required environment variables
echo -e "\n${YELLOW}Checking environment variables...${NC}"
MISSING_VARS=()

if [ -z "$GITHUB_TOKEN" ]; then
    MISSING_VARS+=("GITHUB_TOKEN")
fi
if [ -z "$BRAVE_API_KEY" ]; then
    MISSING_VARS+=("BRAVE_API_KEY")
fi
if [ -z "$DATABENTO_API_KEY" ]; then
    MISSING_VARS+=("DATABENTO_API_KEY")
fi
if [ -z "$FRED_API_KEY" ]; then
    MISSING_VARS+=("FRED_API_KEY")
fi

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}Warning: The following environment variables are not set:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo -e "${YELLOW}Some MCP servers may not function properly.${NC}"
    echo -e "Continue anyway? (y/n): \c"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ All required environment variables are set${NC}"
fi

# List of MCP servers that will be started
echo -e "\n${YELLOW}MCP Servers to be initialized:${NC}"
echo "  ${GREEN}Node.js-based servers:${NC}"
echo "    1. filesystem     - File system access"
echo "    2. brave          - Web search"
echo "    3. memory         - Persistent memory"
echo "    4. sequential-thinking - Chain of thought"
echo "    5. puppeteer      - Web automation"
echo ""
echo "  ${GREEN}Python-based servers:${NC}"
echo "    6. github         - GitHub integration"
echo "    7. statsource     - Statistical data source"
echo "    8. duckdb         - Database queries"
echo "    9. mlflow         - ML experiment tracking"
echo "    10. pyrepl        - Python REPL"
echo "    11. sklearn       - Scikit-learn integration"
echo "    12. optionsflow   - Options trading data"
echo "    13. python_analysis - Custom Unity Wheel analysis"

# Quick validation of Python servers
echo -e "\n${YELLOW}Validating Python server paths...${NC}"
PYTHON_SERVERS=(
    "/Users/mikeedwards/mcp-servers/community/mlflowMCPServer/mlflow_server.py"
    "/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py"
    "/Users/mikeedwards/mcp-servers/community/mcp-optionsflow/optionsflow.py"
    "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py"
)

ALL_VALID=true
for server in "${PYTHON_SERVERS[@]}"; do
    if [ -f "$server" ]; then
        echo -e "  ${GREEN}✓${NC} Found: $(basename "$server")"
    else
        echo -e "  ${RED}✗${NC} Missing: $server"
        ALL_VALID=false
    fi
done

# Check DuckDB file
DB_FILE="$PROJECT_ROOT/data/cache/wheel_cache.duckdb"
if [ -f "$DB_FILE" ]; then
    echo -e "  ${GREEN}✓${NC} DuckDB file exists"
else
    echo -e "  ${RED}✗${NC} DuckDB file missing: $DB_FILE"
    ALL_VALID=false
fi

if [ "$ALL_VALID" = false ]; then
    echo -e "\n${RED}Some validation checks failed. Continue anyway? (y/n):${NC} \c"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start Claude with the MCP configuration
echo -e "\n${GREEN}Starting Claude Code with 13 MCP servers...${NC}"
echo -e "Config: ${YELLOW}$MCP_CONFIG${NC}"
echo -e "\n${YELLOW}Launching Claude Code...${NC}\n"

# Export any additional environment variables that might be needed
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Launch Claude with the MCP configuration
echo -e "Starting with command: ${YELLOW}$CLAUDE_CMD${NC}"
eval "$CLAUDE_CMD --mcp-config \"$MCP_CONFIG\""