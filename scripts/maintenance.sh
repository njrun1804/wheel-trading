#!/bin/bash
# Periodic maintenance tasks for Unity Wheel Trading Bot

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🔧 Unity Wheel Trading Bot - Maintenance Tasks"
echo "=============================================="
echo "Started at: $(date)"
echo ""

# 1. Clean old cache files
echo "🗑️  Cleaning old cache files..."
CACHE_COUNT=$(find "$PROJECT_ROOT" -name "*.cache" -mtime +7 -type f 2>/dev/null | wc -l)
if [ "$CACHE_COUNT" -gt 0 ]; then
    find "$PROJECT_ROOT" -name "*.cache" -mtime +7 -delete
    echo "   Removed $CACHE_COUNT old cache files"
else
    echo "   No old cache files found"
fi

# 2. Clean old export files
echo ""
echo "📁 Cleaning old export files..."
if [ -d "$PROJECT_ROOT/exports" ]; then
    OLD_EXPORTS=$(find "$PROJECT_ROOT/exports" -type f -mtime +30 2>/dev/null | wc -l)
    if [ "$OLD_EXPORTS" -gt 0 ]; then
        find "$PROJECT_ROOT/exports" -type f -mtime +30 -delete
        echo "   Removed $OLD_EXPORTS old export files"
    else
        echo "   No old export files found"
    fi
fi

# 3. Rotate log files
echo ""
echo "📜 Rotating log files..."
LOG_DIR="$PROJECT_ROOT/logs"
if [ -d "$LOG_DIR" ]; then
    for log in "$LOG_DIR"/*.log; do
        if [ -f "$log" ]; then
            SIZE=$(stat -f%z "$log" 2>/dev/null || stat -c%s "$log" 2>/dev/null || echo 0)
            if [ "$SIZE" -gt 10485760 ]; then  # 10MB
                mv "$log" "${log}.$(date +%Y%m%d_%H%M%S)"
                echo "   Rotated: $(basename "$log")"
                # Keep only last 5 rotated logs
                ls -t "${log}".* 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
            fi
        fi
    done
fi

# 4. Clean Python cache
echo ""
echo "🐍 Cleaning Python cache..."
PYCACHE_COUNT=$(find "$PROJECT_ROOT" -type d -name "__pycache__" 2>/dev/null | wc -l)
if [ "$PYCACHE_COUNT" -gt 0 ]; then
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "   Removed $PYCACHE_COUNT __pycache__ directories"
fi

# 5. Vacuum SQLite databases
echo ""
echo "🗄️  Optimizing databases..."
for db in "$PROJECT_ROOT"/*.db "$PROJECT_ROOT/exports"/*.db; do
    if [ -f "$db" ]; then
        python -c "
import sqlite3
conn = sqlite3.connect('$db')
conn.execute('VACUUM')
conn.close()
print('   Optimized: $(basename "$db")')
" 2>/dev/null || echo "   Failed to optimize: $(basename "$db")"
    fi
done

# 6. Generate maintenance report
echo ""
echo "📊 Maintenance summary:"
echo "   - Cache files cleaned: $CACHE_COUNT"
echo "   - Export files cleaned: ${OLD_EXPORTS:-0}"
echo "   - Python cache cleaned: $PYCACHE_COUNT"

echo ""
echo "=============================================="
echo "✅ Maintenance tasks completed at: $(date)"
