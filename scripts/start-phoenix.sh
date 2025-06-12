#!/usr/bin/env bash
# Start Phoenix observability in background

set -euo pipefail

# Check if already running
if curl -s http://localhost:6006/health >/dev/null 2>&1; then
    echo "Phoenix is already running on port 6006"
    exit 0
fi

echo "Starting Phoenix observability..."

# Start in background and detach
nohup phoenix serve > ~/Library/Logs/phoenix.log 2>&1 &
PHOENIX_PID=$!

# Wait for it to start
sleep 3

if curl -s http://localhost:6006/health >/dev/null 2>&1; then
    echo "✓ Phoenix started successfully (PID: $PHOENIX_PID)"
    echo "View at: http://localhost:6006"
    echo "Logs at: ~/Library/Logs/phoenix.log"
else
    echo "✗ Phoenix failed to start"
    echo "Check logs: tail -f ~/Library/Logs/phoenix.log"
    exit 1
fi