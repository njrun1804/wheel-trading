#!/bin/bash
# Graceful MCP server shutdown script

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "🛑 Stopping MCP servers..."

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$WORKSPACE_ROOT/.claude/runtime"

if [ ! -d "$RUNTIME_DIR" ]; then
    echo "No runtime directory found - no servers running"
    exit 0
fi

# Function to stop a server
stop_server() {
    local pidfile=$1
    local name=$(basename "$pidfile" .pid)
    
    if [ ! -f "$pidfile" ]; then
        return 0
    fi
    
    local pid=$(cat "$pidfile")
    echo -n "Stopping $name (PID: $pid)..."
    
    if kill -0 "$pid" 2>/dev/null; then
        # Send SIGTERM for graceful shutdown
        kill -TERM "$pid" 2>/dev/null
        
        # Wait up to 5 seconds for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && [ $count -lt 5 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null
            echo -e " ${RED}[FORCE KILLED]${NC}"
        else
            echo -e " ${GREEN}[STOPPED]${NC}"
        fi
    else
        echo -e " ${GREEN}[NOT RUNNING]${NC}"
    fi
    
    # Clean up files
    rm -f "$pidfile"
    rm -f "${pidfile%.pid}.health"
}

# Stop all servers
for pidfile in "$RUNTIME_DIR"/*.pid; do
    if [ -f "$pidfile" ]; then
        stop_server "$pidfile"
    fi
done

echo -e "\n${GREEN}All servers stopped${NC}"

# Optional: Clean up runtime directory
read -p "Clean up runtime directory? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$RUNTIME_DIR"
    echo "Runtime directory cleaned"
fi
