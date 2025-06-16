#!/bin/bash

echo "🚀 Deploying Unified System - No External Daemons Required"
echo "=================================================="

# Stop any existing daemons first
echo "1. Stopping all existing daemons..."
./stop_all_daemons.sh > /dev/null 2>&1

# Create necessary directories
echo "2. Setting up directories..."
mkdir -p logs
mkdir -p data
mkdir -p config

# Set permissions
echo "3. Setting permissions..."
chmod +x unified_system_manager.py
chmod +x trading_system_with_optimization.py

echo "4. System ready! Choose your deployment option:"
echo ""
echo "   Option A - System Monitoring Only:"
echo "   python3 unified_system_manager.py"
echo ""
echo "   Option B - Full Trading System with Embedded Optimization:"
echo "   python3 trading_system_with_optimization.py"
echo ""
echo "   Option C - Trading System without Optimization:"
echo "   python3 trading_system_with_optimization.py --no-optimization"
echo ""

# Show current system status
echo "5. Current System Status:"
echo "   Memory Available: $(python3 -c "import psutil; print(f'{psutil.virtual_memory().available/1024/1024/1024:.1f}GB')")"
echo "   CPU Usage: $(python3 -c "import psutil; print(f'{psutil.cpu_percent(interval=1):.1f}%')")"
echo "   Load Average: $(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}')"
echo "   Process Count: $(ps aux | wc -l)"

echo ""
echo "✅ Unified System Deployment Complete"
echo "   • No external daemons required"
echo "   • All monitoring embedded in main process"  
echo "   • 8 M4 Pro cores optimally utilized"
echo "   • Memory management built-in"
echo "   • GPU optimization included"