#!/bin/bash
# MCP Health Monitor Service

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONITOR_SCRIPT="$WORKSPACE_ROOT/scripts/mcp-health-monitor.py"
PID_FILE="$WORKSPACE_ROOT/.claude/monitor.pid"

case "$1" in
    start)
        echo "Starting MCP health monitor..."
        python3 "$MONITOR_SCRIPT" --watch --interval 30 > /dev/null 2>&1 &
        echo $! > "$PID_FILE"
        echo "Monitor started with PID $(cat $PID_FILE)"
        ;;
    stop)
        if [ -f "$PID_FILE" ]; then
            kill $(cat "$PID_FILE") 2>/dev/null
            rm -f "$PID_FILE"
            echo "Monitor stopped"
        else
            echo "Monitor not running"
        fi
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "Monitor running (PID: $(cat $PID_FILE))"
        else
            echo "Monitor not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
