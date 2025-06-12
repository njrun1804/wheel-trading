#!/bin/bash
# Robust MCP server startup script

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🚀 Starting MCP servers..."

# Workspace root
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_ROOT"

# Create runtime directory
RUNTIME_DIR="$WORKSPACE_ROOT/.claude/runtime"
mkdir -p "$RUNTIME_DIR"

# Load environment variables
if [ -f "$WORKSPACE_ROOT/.env" ]; then
    export $(cat "$WORKSPACE_ROOT/.env" | grep -v '^#' | xargs)
fi

# Function to check if a process is running
is_running() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    kill -0 "$pid" 2>/dev/null
}

# Function to start a server
start_server() {
    local name=$1
    local command=$2
    shift 2
    local args=("$@")
    
    echo -n "Starting $name..."
    
    # Check if already running
    local pidfile="$RUNTIME_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if is_running "$pid"; then
            echo -e " ${YELLOW}[ALREADY RUNNING]${NC}"
            return 0
        fi
    fi
    
    # Start the server
    local logfile="$RUNTIME_DIR/$name.log"
    
    # Run in background with proper signal handling
    (
        exec "$command" "${args[@]}" > "$logfile" 2>&1
    ) &
    
    local pid=$!
    echo "$pid" > "$pidfile"
    
    # Wait a moment to check if it started successfully
    sleep 1
    
    if is_running "$pid"; then
        echo -e " ${GREEN}[OK]${NC} (PID: $pid)"
        
        # Create health file
        cat > "$RUNTIME_DIR/$name.health" << EOF
{
    "server_name": "$name",
    "pid": $pid,
    "timestamp": "$(date -Iseconds)",
    "status": "healthy",
    "command": "$command",
    "log_file": "$logfile"
}
EOF
    else
        echo -e " ${RED}[FAILED]${NC}"
        echo "Check log: $logfile"
        return 1
    fi
}

# Start critical servers first
echo "Starting critical servers..."

# Python analysis server (most important for trading)
start_server "python_analysis"     "/Users/mikeedwards/.pyenv/shims/python3"     "$WORKSPACE_ROOT/scripts/python-mcp-server.py"

# Trace server for observability
start_server "trace"     "/Users/mikeedwards/.pyenv/shims/python3"     "$WORKSPACE_ROOT/scripts/trace-mcp-server.py"

# Start other servers
echo -e "\nStarting additional servers..."

# Ripgrep for code search
start_server "ripgrep"     "/Users/mikeedwards/.pyenv/shims/python3"     "$WORKSPACE_ROOT/scripts/ripgrep-mcp.py"

# Dependency graph
start_server "dependency_graph"     "/Users/mikeedwards/.pyenv/shims/python3"     "$WORKSPACE_ROOT/scripts/dependency-graph-mcp.py"

# Summary
echo -e "\n${GREEN}=== Startup Complete ===${NC}"
echo "Runtime directory: $RUNTIME_DIR"
echo "To monitor: $WORKSPACE_ROOT/scripts/mcp-health-monitor.py --watch"
echo "To stop: $WORKSPACE_ROOT/scripts/stop-mcp-servers.sh"
