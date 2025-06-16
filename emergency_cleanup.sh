#!/bin/bash

echo "=== Emergency Meta Process Cleanup ==="

# Target PIDs from PID files
PIDS="39124 84182 77238 77250"

echo "Attempting to terminate PIDs: $PIDS"

# Try graceful shutdown first
for pid in $PIDS; do
    if kill -0 $pid 2>/dev/null; then
        echo "Sending TERM signal to PID $pid"
        kill $pid 2>/dev/null
    else
        echo "PID $pid not running"
    fi
done

sleep 3

# Force kill any remaining
for pid in $PIDS; do
    if kill -0 $pid 2>/dev/null; then
        echo "Force killing PID $pid"
        kill -9 $pid 2>/dev/null
    fi
done

# Kill any processes matching meta patterns
echo "Killing any remaining meta processes..."
pkill -f "meta.*py" 2>/dev/null || true
pkill -f "meta_daemon" 2>/dev/null || true
pkill -f "meta_system" 2>/dev/null || true
pkill -f "memory_monitor" 2>/dev/null || true

# Remove PID files
echo "Removing PID files..."
rm -f meta_system.pid meta_daemon.pid 2>/dev/null || true
rm -f pids/memory_monitor.pid pids/process_manager.pid 2>/dev/null || true

echo "=== Cleanup Complete ==="