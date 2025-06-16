#!/bin/bash

echo "🛑 Stopping all system daemons and services..."

# Kill all python monitoring processes
echo "Stopping Python monitoring processes..."
pkill -f "memory_monitor_daemon.py"
pkill -f "service_watchdog.py" 
pkill -f "core4_manager.py"
pkill -f "process_manager.py"
pkill -f "system_service_optimizer.py"

# Kill any remaining background processes
echo "Stopping other background processes..."
pkill -f "service_optimizer.sh"
pkill -f "service_monitor.sh"

# Wait a moment
sleep 2

# Check what's still running
echo "Checking for remaining processes..."
ps aux | grep -E "(memory_monitor|service_watchdog|core4_manager|process_manager)" | grep -v grep

echo "✅ All daemons stopped"
echo ""
echo "🚀 You can now run the unified system instead:"
echo "   python3 unified_system_manager.py"
echo "   or"
echo "   python3 trading_system_with_optimization.py"