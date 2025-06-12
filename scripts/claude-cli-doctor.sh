#!/usr/bin/env bash
# Claude CLI Doctor - Comprehensive diagnostic tool

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Claude CLI Doctor ===${NC}"
echo "Running comprehensive diagnostics..."
echo ""

# Output file
DOCTOR_LOG="claude-doctor-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$DOCTOR_LOG")
exec 2>&1

# 1. System Information
echo -e "${GREEN}1. System Information${NC}"
echo "========================"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "OS: $(uname -s) $(uname -r)"
echo "CPU: $(sysctl -n hw.ncpu 2>/dev/null || nproc) cores"
echo "Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 " GB"}' || free -h | grep Mem | awk '{print $2}')"
echo "Serial: KXQ93HN7DP"
echo ""

# 2. Claude CLI Check
echo -e "${GREEN}2. Claude CLI Status${NC}"
echo "==================="
if command -v claude &> /dev/null; then
    echo -e "${GREEN}✓${NC} Claude CLI found: $(which claude)"
    claude --version 2>&1 || echo -e "${RED}✗${NC} Failed to get version"
else
    echo -e "${RED}✗${NC} Claude CLI not found in PATH"
fi
echo ""

# 3. Port Scan
echo -e "${GREEN}3. Port Availability${NC}"
echo "===================="
echo "Checking common MCP ports..."
for port in 4318 5001 6006 8080; do
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${YELLOW}⚠${NC} Port $port is in use"
        lsof -i :$port 2>/dev/null | grep LISTEN | head -1 || true
    else
        echo -e "${GREEN}✓${NC} Port $port is available"
    fi
done
echo ""

# 4. Version Checks
echo -e "${GREEN}4. Tool Versions${NC}"
echo "================"
# Python
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} Python: $(python3 --version)"
else
    echo -e "${RED}✗${NC} Python3 not found"
fi

# Node.js
if command -v node &> /dev/null; then
    echo -e "${GREEN}✓${NC} Node.js: $(node --version)"
else
    echo -e "${RED}✗${NC} Node.js not found"
fi

# DuckDB
if command -v duckdb &> /dev/null; then
    echo -e "${GREEN}✓${NC} DuckDB: $(duckdb --version 2>&1 | head -1)"
else
    echo -e "${YELLOW}⚠${NC} DuckDB CLI not found (Python module may still work)"
fi

# Ripgrep
if command -v rg &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ripgrep: $(rg --version | head -1)"
else
    echo -e "${RED}✗${NC} Ripgrep not found"
fi
echo ""

# 5. MCP Server Status
echo -e "${GREEN}5. MCP Server Status${NC}"
echo "==================="
if command -v mcp-health &> /dev/null; then
    mcp-health | grep -E "✓|✗" | head -10
else
    echo "MCP health check not available"
fi

# Check PID lock
if [ -f "/tmp/claude_mcp_locks/claude_mcp.lock" ]; then
    echo -e "${YELLOW}⚠${NC} MCP lock file exists"
    cat /tmp/claude_mcp_locks/claude_mcp.lock 2>/dev/null || true
fi
echo ""

# 6. Environment Variables
echo -e "${GREEN}6. Environment Check${NC}"
echo "==================="
env_vars=(
    "CLAUDE_CODE_THINKING_BUDGET_TOKENS"
    "NODE_OPTIONS"
    "PYTHONPATH"
    "MCP_ROOT"
    "OTEL_EXPORTER_OTLP_ENDPOINT"
)

for var in "${env_vars[@]}"; do
    if [ -n "${!var:-}" ]; then
        echo -e "${GREEN}✓${NC} $var = ${!var}"
    else
        echo -e "${YELLOW}⚠${NC} $var not set"
    fi
done
echo ""

# 7. File Descriptor Limits
echo -e "${GREEN}7. Resource Limits${NC}"
echo "================="
echo "File descriptors: $(ulimit -n)"
echo "Max processes: $(ulimit -u)"
echo ""

# 8. Token Budget Dry Run
echo -e "${GREEN}8. Token Budget Estimation${NC}"
echo "========================="
if [ -d "src" ]; then
    py_files=$(find src -name "*.py" 2>/dev/null | wc -l)
    total_lines=$(find src -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    echo "Python files in src/: $py_files"
    echo "Total lines: ${total_lines:-0}"
    
    # Rough token estimate (1 line ≈ 10 tokens)
    if [ -n "$total_lines" ] && [ "$total_lines" -gt 0 ]; then
        estimated_tokens=$((total_lines * 10))
        echo "Estimated tokens: ~$estimated_tokens"
        
        # Check against typical budgets
        if [ $estimated_tokens -gt 100000 ]; then
            echo -e "${YELLOW}⚠${NC} Large codebase - consider using .claudeignore"
        else
            echo -e "${GREEN}✓${NC} Token usage within typical budgets"
        fi
    fi
fi
echo ""

# 9. Performance Tests
echo -e "${GREEN}9. Performance Benchmarks${NC}"
echo "========================"

# Ripgrep performance
if command -v rg &> /dev/null && [ -d "src" ]; then
    echo -n "Ripgrep scan time: "
    start=$(date +%s%N)
    rg "import" src --count-matches > /dev/null 2>&1
    end=$(date +%s%N)
    duration=$(( (end - start) / 1000000 ))
    
    if [ $duration -lt 500 ]; then
        echo -e "${GREEN}✓${NC} ${duration}ms (< 500ms SLO)"
    else
        echo -e "${YELLOW}⚠${NC} ${duration}ms (> 500ms SLO)"
    fi
fi

# File open test
echo -n "File descriptor test: "
fd_test=$(mktemp -d)
touch "$fd_test"/test_{1..100}.txt 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Can open 100 files"
else
    echo -e "${RED}✗${NC} Failed to open 100 files"
fi
rm -rf "$fd_test"
echo ""

# 10. Common Issues Check
echo -e "${GREEN}10. Common Issues${NC}"
echo "================"

# Check .claudeignore
if [ -f ".claudeignore" ]; then
    echo -e "${GREEN}✓${NC} .claudeignore exists"
    excluded=$(grep -v "^#" .claudeignore | grep -v "^$" | wc -l)
    echo "  Exclusion rules: $excluded"
else
    echo -e "${YELLOW}⚠${NC} No .claudeignore file - all files will be scanned"
fi

# Check node_modules
if [ -d "node_modules" ]; then
    size=$(du -sh node_modules 2>/dev/null | cut -f1)
    echo -e "${YELLOW}⚠${NC} node_modules exists ($size) - ensure it's in .claudeignore"
fi

# Check Python cache
pycache_count=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)
if [ $pycache_count -gt 0 ]; then
    echo -e "${YELLOW}⚠${NC} Found $pycache_count __pycache__ directories"
fi
echo ""

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo "Diagnostic log saved to: $DOCTOR_LOG"
echo ""
echo "To share this report:"
echo "  cat $DOCTOR_LOG | pbcopy  # Copy to clipboard"
echo "  # Then paste in issue report"
echo ""

# Exit with appropriate code
if grep -q "✗" "$DOCTOR_LOG"; then
    echo -e "${RED}Some issues detected - review log for details${NC}"
    exit 1
else
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
fi