#!/bin/bash
# Quick Process Cleanup - Immediate resource management
# Targets fileproviderd and excess Claude instances

echo "=== QUICK PROCESS CLEANUP ==="
echo "Timestamp: $(date)"
echo ""

# Function to log actions
log_action() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Check current high CPU processes
log_action "Identifying high CPU processes..."
HIGH_CPU_PROCS=$(/bin/ps -axo pid,pcpu,comm --sort=-pcpu | awk '$2 > 50 {print $0}' | head -5)

if [ ! -z "$HIGH_CPU_PROCS" ]; then
    log_action "High CPU processes found:"
    echo "$HIGH_CPU_PROCS"
    echo ""
else
    log_action "No high CPU processes detected"
fi

# Handle fileproviderd specifically
FILEPROVIDERD_PID=$(pgrep fileproviderd)
if [ ! -z "$FILEPROVIDERD_PID" ]; then
    FILEPROVIDERD_CPU=$(/bin/ps -o pcpu -p $FILEPROVIDERD_PID | tail -1 | xargs)
    log_action "fileproviderd (PID: $FILEPROVIDERD_PID) CPU usage: ${FILEPROVIDERD_CPU}%"
    
    if (( $(echo "$FILEPROVIDERD_CPU > 80" | bc -l) )); then
        log_action "fileproviderd CPU usage too high, attempting to renice..."
        if sudo renice 10 $FILEPROVIDERD_PID; then
            log_action "Successfully reniced fileproviderd"
        else
            log_action "Failed to renice fileproviderd"
        fi
    fi
else
    log_action "fileproviderd not found"
fi

# Handle Claude instances
log_action "Checking Claude instances..."
CLAUDE_PIDS=$(pgrep -f claude | head -10)  # Limit to prevent issues

if [ ! -z "$CLAUDE_PIDS" ]; then
    CLAUDE_COUNT=$(echo "$CLAUDE_PIDS" | wc -l)
    log_action "Found $CLAUDE_COUNT Claude instances"
    
    # Show Claude process details
    echo "Claude processes:"
    for pid in $CLAUDE_PIDS; do
        PROC_INFO=$(/bin/ps -o pid,pcpu,pmem,time,comm -p $pid | tail -1)
        echo "  $PROC_INFO"
    done
    echo ""
    
    # If more than 2 Claude instances, kill the highest memory users
    if [ $CLAUDE_COUNT -gt 2 ]; then
        log_action "Too many Claude instances ($CLAUDE_COUNT), terminating excess..."
        
        # Get Claude processes sorted by memory usage (highest first)
        CLAUDE_TO_KILL=$(for pid in $CLAUDE_PIDS; do
            mem=$(/bin/ps -o pmem -p $pid | tail -1 | xargs)
            echo "$mem $pid"
        done | sort -nr | tail -n +3 | awk '{print $2}')
        
        for pid in $CLAUDE_TO_KILL; do
            log_action "Terminating Claude PID $pid..."
            if kill -TERM $pid; then
                sleep 3
                # Check if still running
                if kill -0 $pid 2>/dev/null; then
                    log_action "Claude PID $pid didn't terminate, using SIGKILL"
                    kill -KILL $pid
                fi
                log_action "Claude PID $pid terminated"
            else
                log_action "Failed to terminate Claude PID $pid"
            fi
        done
    else
        log_action "Claude instance count acceptable ($CLAUDE_COUNT)"
    fi
else
    log_action "No Claude instances found"
fi

# Check memory pressure
log_action "Checking memory pressure..."
if command -v memory_pressure >/dev/null 2>&1; then
    MEMORY_STATUS=$(memory_pressure | grep "System-wide memory free percentage" | awk '{print $5}' | tr -d '%')
    if [ ! -z "$MEMORY_STATUS" ]; then
        log_action "System memory free: ${MEMORY_STATUS}%"
        if [ $MEMORY_STATUS -lt 20 ]; then
            log_action "WARNING: Low memory condition detected"
        fi
    fi
fi

# Final system check
log_action "Final system status:"
TOP_PROCS=$(/bin/ps -axo pid,pcpu,pmem,comm --sort=-pcpu | head -6)
echo "$TOP_PROCS"

echo ""
log_action "Cleanup completed"
echo "=== END CLEANUP ==="