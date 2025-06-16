#!/bin/bash
# Stop Memory Management Systems Script

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo -e "${YELLOW}Stopping Memory Management Systems${NC}"
echo "=========================================="

# Function to stop a process gracefully
stop_process() {
    local name="$1"
    local pid_file="$2"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${BLUE}Stopping $name (PID: $pid)...${NC}"
            
            # Try graceful termination first
            kill -TERM $pid
            
            # Wait up to 10 seconds for graceful shutdown
            local count=0
            while [ $count -lt 10 ] && ps -p $pid > /dev/null 2>&1; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${YELLOW}Force killing $name...${NC}"
                kill -KILL $pid
                sleep 1
            fi
            
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${RED}Failed to stop $name${NC}"
                return 1
            else
                echo -e "${GREEN}$name stopped successfully${NC}"
                rm -f "$pid_file"
                return 0
            fi
        else
            echo -e "${YELLOW}$name was not running${NC}"
            rm -f "$pid_file"
            return 0
        fi
    else
        echo -e "${YELLOW}$name PID file not found${NC}"
        return 0
    fi
}

# Stop Memory Monitor
stop_process "Memory Monitor" "pids/memory_monitor.pid"

# Stop Process Manager
stop_process "Process Manager" "pids/process_manager.pid"

# Remove cron jobs
echo -e "${BLUE}Removing automated monitoring...${NC}"
if crontab -l 2>/dev/null | grep -q "memory_cleanup_emergency.py"; then
    # Remove the cron jobs
    crontab -l 2>/dev/null | grep -v "memory_cleanup_emergency.py" | grep -v "system_service_optimizer.py" | crontab -
    echo -e "${GREEN}Automated monitoring removed${NC}"
else
    echo -e "${YELLOW}No automated monitoring found${NC}"
fi

# Clean up any remaining processes
echo -e "${BLUE}Cleaning up remaining processes...${NC}"

# Find and kill any remaining memory management processes
for process_name in "memory_monitor_daemon.py" "process_manager.py"; do
    pids=$(ps aux | grep "$process_name" | grep -v grep | awk '{print $2}')
    for pid in $pids; do
        if [ ! -z "$pid" ]; then
            echo -e "${YELLOW}Found orphaned process: $process_name (PID: $pid)${NC}"
            kill -TERM $pid 2>/dev/null || kill -KILL $pid 2>/dev/null
        fi
    done
done

# Clean up PID directory
if [ -d "pids" ]; then
    rm -rf pids/*.pid 2>/dev/null
fi

echo ""
echo -e "${GREEN}Memory Management Systems stopped${NC}"
echo ""

# Show final memory status
echo "FINAL MEMORY STATUS:"
python3 -c "
import psutil
from datetime import datetime

vm = psutil.virtual_memory()
print(f'  Available Memory: {vm.available / 1024 / 1024:.0f} MB')
print(f'  Memory Usage: {vm.percent:.1f}%')
print(f'  Timestamp: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
" 2>/dev/null || echo "  Unable to get memory status"

echo ""
echo "Log files preserved in logs/ directory"
echo "To restart: ./start_memory_systems.sh"
echo "=========================================="