#!/bin/bash
# Quick recovery script - run immediately after terminal restart

set -e

echo "=== Emergency File Descriptor Recovery ==="

# 1. Increase limits first
ulimit -n 4096
echo "✓ Increased FD limit to 4096"

# 2. Kill hanging processes
echo "Killing hanging processes..."
pkill -f "meta_daemon" 2>/dev/null || true
pkill -f "meta_system" 2>/dev/null || true  
pkill -f "meta_monitoring" 2>/dev/null || true
kill -9 39124 2>/dev/null || true
kill -9 84182 2>/dev/null || true
echo "✓ Processes killed"

# 3. Clean database lock files
echo "Cleaning database files..."
find . -name "*.db-wal" -delete 2>/dev/null || true
find . -name "*.db-shm" -delete 2>/dev/null || true
find . -name "*.pid" -delete 2>/dev/null || true
echo "✓ Database files cleaned"

# 4. Verify recovery
echo "=== System Status ==="
echo "FD Limit: $(ulimit -n)"
echo "Open files: $(lsof 2>/dev/null | wc -l || echo 'unknown')"
echo "Python processes: $(ps aux | grep -c python || echo '0')"

echo "=== Recovery Complete ==="
echo "You can now run normal commands"