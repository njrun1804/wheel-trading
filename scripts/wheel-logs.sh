#!/bin/bash
# Dynamic log viewer for Wheel Trading
# Shows live logs, test output, and system metrics in split panes

# Function to find and tail log files
start_log_viewer() {
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Start system monitoring in background
    echo "📊 Starting Wheel Trading Log Monitor..."
    echo "=================================="
    echo ""
    
    # Use tmux for persistent split-screen if available
    if command -v tmux &> /dev/null; then
        # Create or attach to wheel-logs session
        tmux new-session -d -s wheel-logs 2>/dev/null || true
        
        # Main pane - follow all logs
        tmux send-keys -t wheel-logs:0.0 "tail -f logs/*.log 2>/dev/null | grep -v '^$' || echo 'Waiting for logs...'" C-m
        
        # Split horizontally for test output
        tmux split-window -v -t wheel-logs:0 -p 30
        tmux send-keys -t wheel-logs:0.1 "watch -n 1 'echo \"🧪 Test Status:\"; pytest --tb=no -q 2>&1 | tail -20'" C-m
        
        # Split vertically for metrics
        tmux split-window -h -t wheel-logs:0.1 -p 50
        tmux send-keys -t wheel-logs:0.2 "watch -n 1 'echo \"⚡ System Metrics:\"; top -l 1 -n 10 | head -20'" C-m
        
        # Attach to session
        tmux attach-session -t wheel-logs
    else
        # Fallback to simple tail with multiplexing
        echo "💡 Install tmux for better log viewing: brew install tmux"
        echo ""
        echo "Tailing logs (Ctrl+C to stop)..."
        echo ""
        
        # Use multitail if available
        if command -v multitail &> /dev/null; then
            multitail -i logs/*.log -i logs/test_*.log
        else
            # Simple tail fallback
            tail -f logs/*.log 2>/dev/null || echo "No logs found. They will appear here when generated."
        fi
    fi
}

# Handle different modes
case "${1:-live}" in
    "live")
        start_log_viewer
        ;;
    "test")
        # Watch test output specifically
        echo "🧪 Watching test output..."
        if [[ -f "pytest.ini" ]]; then
            pytest --tb=short -v --log-cli-level=INFO 2>&1 | tee logs/test_$(date +%Y%m%d_%H%M%S).log
        else
            echo "No pytest.ini found"
        fi
        ;;
    "trading")
        # Watch trading-specific logs
        echo "📈 Watching trading logs..."
        tail -f logs/*trading*.log logs/*wheel*.log 2>/dev/null || echo "No trading logs yet"
        ;;
    "errors")
        # Show only errors
        echo "❌ Watching for errors..."
        tail -f logs/*.log 2>/dev/null | grep -E "(ERROR|FAIL|Exception|Traceback)" --color=always
        ;;
    *)
        echo "Usage: wheel-logs [live|test|trading|errors]"
        echo "  live    - Show all logs in split screen (default)"
        echo "  test    - Run and watch tests"
        echo "  trading - Watch trading-specific logs"
        echo "  errors  - Show only errors"
        ;;
esac