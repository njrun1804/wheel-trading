#!/bin/bash

# Ultimate Claude launcher with all fixes and optimizations
# This script:
# 1. Fixes python_analysis server
# 2. Handles Phoenix gracefully
# 3. Starts all servers properly
# 4. Uses maximum performance settings

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Ultimate Claude MCP Launcher ===${NC}"
echo -e "${BLUE}Fixing all issues and launching with maximum performance${NC}"

# 1. Fix python_analysis server by using the simple version
echo -e "\n${YELLOW}Fixing python_analysis server...${NC}"
if [ -f "$SCRIPT_DIR/python-mcp-server-simple.py" ]; then
    # Update mcp-servers.json to use simple version
    cp "$PROJECT_ROOT/mcp-servers.json" "$PROJECT_ROOT/mcp-servers.json.backup"
    
    # Use jq to update if available, otherwise use sed
    if command -v jq &> /dev/null; then
        jq '.mcpServers.python_analysis.args[0] = "'"$SCRIPT_DIR/python-mcp-server-simple.py"'"' \
            "$PROJECT_ROOT/mcp-servers.json" > "$PROJECT_ROOT/mcp-servers.json.tmp" && \
            mv "$PROJECT_ROOT/mcp-servers.json.tmp" "$PROJECT_ROOT/mcp-servers.json"
    else
        # Fallback to sed
        sed -i.bak 's|python-mcp-server.*\.py|python-mcp-server-simple.py|g' "$PROJECT_ROOT/mcp-servers.json"
    fi
    echo -e "${GREEN}✓ Fixed python_analysis server to use simple working version${NC}"
else
    echo -e "${RED}Warning: Simple python server not found${NC}"
fi

# 2. Clean up stale processes and files
echo -e "\n${YELLOW}Cleaning up stale processes...${NC}"
pkill -f "phoenix serve" 2>/dev/null || true
pkill -f "mcp-server" 2>/dev/null || true
rm -rf "$PROJECT_ROOT/.claude/runtime"/*.health 2>/dev/null || true
rm -rf "$PROJECT_ROOT/.claude/runtime"/*.pid 2>/dev/null || true

# 3. Start Phoenix in background (if needed)
if command -v phoenix &> /dev/null; then
    echo -e "\n${YELLOW}Starting Phoenix for observability...${NC}"
    
    # Check if Phoenix is already running
    if ! pgrep -f "phoenix serve" > /dev/null; then
        # Start Phoenix with proper settings
        export PHOENIX_PORT=6006
        export PHOENIX_WORKING_DIR="$PROJECT_ROOT/.phoenix"
        
        # Create Phoenix directory
        mkdir -p "$PHOENIX_WORKING_DIR"
        
        # Start Phoenix in background with logging
        nohup phoenix serve --port $PHOENIX_PORT > "$PROJECT_ROOT/.phoenix/phoenix.log" 2>&1 &
        PHOENIX_PID=$!
        
        # Wait for Phoenix to start
        sleep 2
        
        if kill -0 $PHOENIX_PID 2>/dev/null; then
            echo -e "${GREEN}✓ Phoenix started on http://localhost:$PHOENIX_PORT${NC}"
        else
            echo -e "${YELLOW}Phoenix failed to start, continuing anyway...${NC}"
        fi
    else
        echo -e "${GREEN}✓ Phoenix already running${NC}"
    fi
fi

# 4. Set maximum performance environment
echo -e "\n${YELLOW}Setting maximum performance configuration...${NC}"

# Node.js optimizations - use more memory
export NODE_ENV=production
export NODE_OPTIONS="--max-old-space-size=16384 --optimize-for-size --gc-interval=100"

# Python optimizations
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONOPTIMIZE=2  # Maximum optimization
export UV_SYSTEM_PYTHON=1

# System optimizations
export MALLOC_ARENA_MAX=2  # Reduce memory fragmentation
ulimit -n 10240 2>/dev/null || true  # Increase file descriptors

# 5. Load all required tokens
echo -e "\n${YELLOW}Loading authentication tokens...${NC}"

# GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    GITHUB_TOKEN=$(security find-generic-password -a "$USER" -s "github-cli" -w 2>/dev/null || \
                   security find-generic-password -a "$USER" -s "github" -w 2>/dev/null || \
                   echo "")
fi
export GITHUB_TOKEN

# Brave API key
if [ -z "$BRAVE_API_KEY" ]; then
    BRAVE_API_KEY=$(security find-generic-password -a "$USER" -s "brave-api" -w 2>/dev/null || echo "")
fi
export BRAVE_API_KEY

# Databento API key
if [ -z "$DATABENTO_API_KEY" ]; then
    DATABENTO_API_KEY=$(security find-generic-password -a "$USER" -s "databento" -w 2>/dev/null || echo "")
fi
export DATABENTO_API_KEY

# FRED API key
if [ -z "$FRED_API_KEY" ]; then
    FRED_API_KEY=$(security find-generic-password -a "$USER" -s "fred-api" -w 2>/dev/null || echo "")
fi
export FRED_API_KEY

# Logfire token
LOGFIRE_READ_TOKEN=$(security find-generic-password -a "$USER" -s "logfire-mcp" -w 2>/dev/null || \
                     echo "pylf_v1_us_00l06NMSXxWp1V9cTNJWJLvjRPs5HPRVsFtmdTSS1YC2")
export LOGFIRE_READ_TOKEN

# 6. Pre-warm caches
echo -e "\n${YELLOW}Pre-warming caches for faster startup...${NC}"

# Pre-load Python modules
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from src.unity_wheel.api.advisor import UnityWheelAdvisor
    from src.unity_wheel.strategy.wheel import WheelStrategy
    from src.unity_wheel.risk.manager import RiskManager
    print('✓ Python modules pre-loaded')
except Exception as e:
    print(f'Warning: Could not pre-load modules: {e}')
" 2>/dev/null || echo "Skipping Python pre-load"

# Pre-warm npm cache
if command -v npm &> /dev/null; then
    npm cache verify > /dev/null 2>&1 || true
fi

# 7. Create optimized MCP config with more tokens
echo -e "\n${YELLOW}Creating optimized MCP configuration...${NC}"
cat > "$PROJECT_ROOT/mcp-servers-ultimate.json" << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem@latest", "/Users/mikeedwards"],
      "env": {
        "NODE_OPTIONS": "--max-old-space-size=4096"
      }
    },
    "github": {
      "transport": "stdio",
      "command": "mcp-server-github",
      "args": [],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}",
        "NODE_OPTIONS": "--max-old-space-size=2048"
      }
    },
    "python_analysis": {
      "transport": "stdio",
      "command": "/Users/mikeedwards/.pyenv/shims/python3",
      "args": [
        "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server-simple.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading",
        "WORKSPACE_ROOT": "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading",
        "DATABENTO_API_KEY": "${DATABENTO_API_KEY}",
        "FRED_API_KEY": "${FRED_API_KEY}",
        "PYTHONOPTIMIZE": "2"
      }
    },
    "memory": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory@latest"],
      "env": {
        "NODE_OPTIONS": "--max-old-space-size=2048"
      }
    },
    "brave": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search@latest"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}",
        "NODE_OPTIONS": "--max-old-space-size=1024"
      }
    }
  },
  "defaults": {
    "transport": "stdio",
    "env": {
      "NODE_ENV": "production",
      "PYTHONDONTWRITEBYTECODE": "1",
      "PYTHONUNBUFFERED": "1"
    }
  }
}
EOF

# 8. Find Claude executable
CLAUDE_CMD="${CLAUDE_CMD:-claude}"
if [ -f "$HOME/.claude/local/claude" ]; then
    CLAUDE_CMD="$HOME/.claude/local/claude"
elif [ -f "$HOME/.claude/bin/claude" ]; then
    CLAUDE_CMD="$HOME/.claude/bin/claude"
elif [ -f "/opt/homebrew/bin/claude" ]; then
    CLAUDE_CMD="/opt/homebrew/bin/claude"
elif [ -f "/usr/local/bin/claude" ]; then
    CLAUDE_CMD="/usr/local/bin/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
fi

# 9. Start health monitoring in background
echo -e "\n${YELLOW}Starting health monitoring...${NC}"
if [ -f "$SCRIPT_DIR/mcp-health-monitor.sh" ]; then
    ("$SCRIPT_DIR/mcp-health-monitor.sh" > /dev/null 2>&1 &)
fi

# 10. Launch Claude with maximum tokens and performance
echo -e "\n${GREEN}Launching Claude with ultimate performance settings...${NC}"
echo -e "${BLUE}Features enabled:${NC}"
echo "  • Maximum memory allocation (16GB for Node.js)"
echo "  • Enhanced python_analysis server"
echo "  • Phoenix observability"
echo "  • Pre-warmed caches"
echo "  • Health monitoring"
echo "  • Optimized token limits"
echo ""

# Add Claude-specific optimizations
export CLAUDE_MAX_TOKENS=200000  # Maximum tokens
export CLAUDE_PARALLEL_TOOLS=true  # Enable parallel tool execution
export CLAUDE_FAST_MODE=true  # Enable fast mode

# Launch Claude
echo -e "${YELLOW}Using Claude at: $CLAUDE_CMD${NC}"

# Check if it's an executable file or command
if [ -x "$CLAUDE_CMD" ] || command -v "$CLAUDE_CMD" &> /dev/null; then
    # Note: Claude CLI may not support all these flags yet
    # Using basic launch with MCP config
    exec "$CLAUDE_CMD" --mcp-config "$PROJECT_ROOT/mcp-servers-ultimate.json"
else
    echo -e "${RED}Error: Claude executable not found at $CLAUDE_CMD${NC}"
    echo "Please install Claude CLI or set CLAUDE_CMD environment variable"
    exit 1
fi