#!/bin/bash
# Wheel Trading Environment Doctor
# Diagnoses common issues and verifies setup

echo "🏥 Wheel Trading Environment Doctor"
echo "=================================="
echo ""

# Track issues
ISSUES=0

# Function to check and report
check() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    echo -n "Checking $name... "
    
    result=$(eval "$command" 2>&1)
    if [[ -n "$expected" ]]; then
        if [[ "$result" == *"$expected"* ]]; then
            echo "✅ OK"
        else
            echo "❌ FAIL (expected: $expected, got: $result)"
            ((ISSUES++))
        fi
    else
        if [[ $? -eq 0 ]]; then
            echo "✅ OK ($result)"
        else
            echo "❌ FAIL"
            ((ISSUES++))
        fi
    fi
}

# Environment checks
echo "📍 Environment Variables:"
check "WHEEL_TRADING_ROOT" "echo \$WHEEL_TRADING_ROOT" "$PWD"
check "PYTHONPATH includes src" "echo \$PYTHONPATH | grep -q 'src' && echo 'yes'" "yes"
check "Hardware acceleration" "echo \$CLAUDE_HARDWARE_ACCEL" "1"
check "CPU threads" "echo \$OMP_NUM_THREADS" "12"
check "Jarvis2 enabled" "echo \$JARVIS2_ENABLED" "1"
echo ""

# Python environment
echo "🐍 Python Environment:"
check "Python executable" "which python | grep -q 'conda' && echo 'conda' || echo 'system'"
check "Conda environment" "conda info --envs | grep '*' | awk '{print \$1}'"
check "Python version" "python --version | awk '{print \$2}' | cut -d. -f1,2"
echo ""

# Dependencies
echo "📦 Dependencies:"
check "DuckDB module" "python -c 'import duckdb; print(duckdb.__version__)'"
check "Databento module" "python -c 'import databento; print(\"installed\")'"
check "Unity wheel module" "python -c 'import unity_wheel; print(\"installed\")'"
check "Accelerated tools" "python -c 'from unity_wheel.accelerated_tools import *; print(\"available\")'"
echo ""

# Database checks
echo "💾 Database:"
if [[ -f "data/wheel_trading_master.duckdb" ]]; then
    size=$(du -h data/wheel_trading_master.duckdb | awk '{print $1}')
    echo "✅ Database exists (size: $size)"
    
    # Try to connect
    python -c "
import duckdb
try:
    conn = duckdb.connect('data/wheel_trading_master.duckdb', read_only=True)
    tables = conn.execute('SHOW TABLES').fetchall()
    print(f'✅ Database accessible ({len(tables)} tables)')
    conn.close()
except Exception as e:
    print(f'❌ Database error: {e}')
" 2>&1
else
    echo "❌ Database not found at data/wheel_trading_master.duckdb"
    ((ISSUES++))
fi
echo ""

# File permissions
echo "🔐 Permissions:"
check "Scripts executable" "ls -la scripts/*.sh | grep -v '^-rwx' | wc -l | grep -q '^0$' && echo 'all executable'"
check "Jarvis2 executable" "test -x jarvis2.py && echo 'yes'" "yes"
echo ""

# MCP servers
echo "🔌 MCP Servers:"
if [[ -f "mcp-servers.json" ]]; then
    essential_servers=$(python -c "
import json
with open('mcp-servers.json') as f:
    servers = json.load(f)
    essential = [s for s in servers.get('mcpServers', {}).keys()]
    print(len(essential))
" 2>&1)
    echo "✅ MCP configuration found ($essential_servers servers configured)"
else
    echo "❌ MCP configuration missing"
    ((ISSUES++))
fi
echo ""

# Performance check
echo "⚡ Performance:"
check "Metal GPU available" "python -c 'import torch; print(\"yes\" if torch.backends.mps.is_available() else \"no\")'" "yes"
check "CPU cores detected" "python -c 'import os; print(os.cpu_count())'" "12"
echo ""

# Summary
echo "=================================="
if [[ $ISSUES -eq 0 ]]; then
    echo "✅ All checks passed! Environment is healthy."
else
    echo "❌ Found $ISSUES issue(s). Please fix them for optimal performance."
fi
echo ""

# Quick fixes
if [[ $ISSUES -gt 0 ]]; then
    echo "💡 Quick fixes:"
    echo "  - Re-source your shell: source ~/.zshrc"
    echo "  - Activate conda: conda activate wheel"
    echo "  - Install missing deps: pip install -r requirements.txt"
    echo "  - Fix permissions: chmod +x scripts/*.sh jarvis2.py"
fi