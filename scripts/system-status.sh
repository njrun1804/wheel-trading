#!/bin/bash
# Comprehensive system status for Wheel Trading

echo "🚀 WHEEL TRADING SYSTEM STATUS"
echo "=============================="
echo ""

# Hardware
echo "💻 Hardware (M4 Pro):"
echo "  CPU: $(sysctl -n hw.ncpu) cores ($(sysctl -n hw.physicalcpu) physical)"
echo "  Memory: $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))GB"
echo "  GPU: 16 cores (Metal)"
echo ""

# Environment
echo "🔧 Environment:"
echo "  Shell: $(echo $SHELL)"
echo "  Python: $(python --version 2>&1 | cut -d' ' -f2)"
echo "  Conda: $(conda info --envs 2>/dev/null | grep '*' | awk '{print $1}' || echo 'not active')"
echo "  Working dir: $(pwd)"
echo ""

# Process Status
echo "📊 Running Processes:"
jarvis_procs=$(ps aux | grep -E "jarvis" | grep -v grep | wc -l | tr -d ' ')
mcp_procs=$(ps aux | grep -E "mcp.*server" | grep -v grep | wc -l | tr -d ' ')
python_procs=$(ps aux | grep -E "python.*unity" | grep -v grep | wc -l | tr -d ' ')
echo "  Jarvis2: $jarvis_procs processes"
echo "  MCP servers: $mcp_procs active"
echo "  Unity Python: $python_procs running"
echo ""

# Database
echo "💾 Database Status:"
for db in data/wheel_trading_master.duckdb data/wheel_trading_optimized.duckdb; do
    if [[ -f "$db" ]]; then
        size=$(du -h "$db" | awk '{print $1}')
        echo "  $(basename "$db"): $size"
    fi
done
echo ""

# Logs
echo "📝 Recent Activity:"
if [[ -d logs ]]; then
    log_count=$(find logs -name "*.log" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [[ $log_count -gt 0 ]]; then
        echo "  Log files: $log_count"
        latest=$(ls -t logs/*.log 2>/dev/null | head -1)
        if [[ -n "$latest" ]]; then
            echo "  Latest: $(basename "$latest")"
            errors=$(tail -100 "$latest" 2>/dev/null | grep -c "ERROR" || echo "0")
            echo "  Errors in last 100 lines: $errors"
        fi
    else
        echo "  No log files found"
    fi
else
    echo "  Logs directory not found"
fi
echo ""

# Quick Health Check
echo "🏥 Quick Health Check:"
errors=0

# Check Python imports
if python -c "import unity_wheel" 2>/dev/null; then
    echo "  ✅ Unity wheel module OK"
else
    echo "  ❌ Unity wheel module FAIL"
    ((errors++))
fi

# Check database connectivity
if python -c "import duckdb; duckdb.connect(':memory:')" 2>/dev/null; then
    echo "  ✅ DuckDB OK"
else
    echo "  ❌ DuckDB FAIL"
    ((errors++))
fi

# Check API keys
if [[ -n "$DATABENTO_API_KEY" ]] && [[ "$DATABENTO_API_KEY" != "your_databento_key_here" ]]; then
    echo "  ✅ Databento API configured"
else
    echo "  ⚠️  Databento API not configured"
fi

echo ""
echo "=============================="
if [[ $errors -eq 0 ]]; then
    echo "✅ System is operational"
else
    echo "⚠️  System has $errors issue(s)"
fi
echo ""
echo "Run 'wheel-help' for available commands"