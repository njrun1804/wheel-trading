#!/bin/bash
# Meta System & Jarvis2 Startup Script
# Run this at system startup to ensure all components are running

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${GREEN}${BOLD}🚀 Meta System & Jarvis2 Startup${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Function to check if process is running
is_running() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # Running
        else
            rm -f "$pid_file"  # Cleanup stale PID file
            return 1  # Not running
        fi
    fi
    return 1  # PID file doesn't exist
}

# Check and start Meta Daemon
echo -e "\n${BLUE}🔍 Checking Meta Daemon...${NC}"
if is_running "meta_daemon.pid"; then
    meta_pid=$(cat meta_daemon.pid)
    echo -e "${GREEN}✅ Meta Daemon already running (PID: $meta_pid)${NC}"
else
    echo -e "${YELLOW}🚀 Starting Meta Daemon...${NC}"
    python meta_daemon.py --background &
    meta_daemon_pid=$!
    echo $meta_daemon_pid > meta_daemon.pid
    sleep 2
    
    if is_running "meta_daemon.pid"; then
        echo -e "${GREEN}✅ Meta Daemon started (PID: $meta_daemon_pid)${NC}"
    else
        echo -e "${RED}❌ Meta Daemon failed to start${NC}"
    fi
fi

# Check and start Jarvis2 (Optional - interactive mode only)
echo -e "\n${BLUE}🔍 Checking Jarvis2...${NC}"
if is_running "jarvis2.pid"; then
    jarvis_pid=$(cat jarvis2.pid)
    echo -e "${GREEN}✅ Jarvis2 already running (PID: $jarvis_pid)${NC}"
else
    echo -e "${YELLOW}ℹ️ Jarvis2 runs in interactive mode only${NC}"
    echo -e "${YELLOW}   Use: python jarvis2_unified.py interactive${NC}"
    # Create empty PID file to indicate "service not applicable"
    echo "N/A" > jarvis2.pid
fi

# Optional: Start Unified Meta System for continuous monitoring
if [[ "${1:-}" == "--with-unified-meta" ]]; then
    echo -e "\n${BLUE}🔍 Checking Unified Meta System...${NC}"
    if is_running "unified_meta.pid"; then
        unified_pid=$(cat unified_meta.pid)
        echo -e "${GREEN}✅ Unified Meta System already running (PID: $unified_pid)${NC}"
    else
        echo -e "${YELLOW}🚀 Starting Unified Meta System...${NC}"
        python unified_meta_system.py > unified_meta.log 2>&1 &
        unified_pid=$!
        echo $unified_pid > unified_meta.pid
        sleep 3
        
        if is_running "unified_meta.pid"; then
            echo -e "${GREEN}✅ Unified Meta System started (PID: $unified_pid)${NC}"
        else
            echo -e "${RED}❌ Unified Meta System failed to start${NC}"
        fi
    fi
fi

# Health check
echo -e "\n${BLUE}🏥 Health Check...${NC}"
python check_system_status.py

# Show monitoring commands
echo -e "\n${BLUE}📊 Monitoring Commands:${NC}"
echo "  tail -f meta_daemon.log           # Meta daemon logs"
echo "  tail -f unified_meta.log          # Unified meta logs (if started)"
echo "  python check_system_status.py     # Quick status check"
echo "  python run.py --diagnose          # Trading system diagnostics"

echo -e "\n${GREEN}🎉 Startup complete!${NC}"