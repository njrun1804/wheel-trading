#!/bin/bash

echo "🚨 EMERGENCY FILE DESCRIPTOR CLEANUP"
echo "===================================="

# Step 1: Kill all related processes
echo "🛑 Stopping all related processes..."

# Kill by PID files
for pidfile in *.pid; do
    if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "  Killing process $pid from $pidfile"
            kill -TERM "$pid" 2>/dev/null
            sleep 1
            kill -KILL "$pid" 2>/dev/null
        fi
        rm -f "$pidfile"
    fi
done

# Kill by process name patterns
process_patterns=(
    "meta_daemon"
    "meta_monitoring"
    "meta_watcher"
    "einstein"
    "bolt"
    "jarvis2"
    "memory_monitor"
    "system_monitor"
    "gpu_monitor"
    "thermal_monitor"
    "trading_system"
)

for pattern in "${process_patterns[@]}"; do
    pids=$(pgrep -f "$pattern" 2>/dev/null)
    if [[ -n "$pids" ]]; then
        echo "  Killing $pattern processes: $pids"
        pkill -TERM -f "$pattern" 2>/dev/null
        sleep 1
        pkill -KILL -f "$pattern" 2>/dev/null
    fi
done

# Step 2: Close database connections by removing WAL/SHM files
echo "🔒 Cleaning up database files..."

# Remove WAL files
wal_files=(*.db-wal)
if [[ -f "${wal_files[0]}" ]]; then
    echo "  Removing WAL files..."
    rm -f *.db-wal
fi

# Remove SHM files
shm_files=(*.db-shm)
if [[ -f "${shm_files[0]}" ]]; then
    echo "  Removing SHM files..."
    rm -f *.db-shm
fi

# Remove lock files
lock_files=(*.lock)
if [[ -f "${lock_files[0]}" ]]; then
    echo "  Removing lock files..."
    rm -f *.lock
fi

# Step 3: Clean up temporary files
echo "🧹 Cleaning up temporary files..."
rm -f *.tmp *.temp *~ .#* *.socket 2>/dev/null

# Step 4: Force SQLite checkpoint on remaining databases
echo "💾 Forcing database checkpoints..."
for db in *.db; do
    if [[ -f "$db" ]]; then
        echo "  Checkpointing $db..."
        sqlite3 "$db" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true
    fi
done

# Step 5: Check file descriptor usage
echo "📊 Checking file descriptors..."
if command -v lsof >/dev/null 2>&1; then
    fd_count=$(lsof -p $$ 2>/dev/null | wc -l)
    echo "  Current process file descriptors: $fd_count"
else
    echo "  lsof not available"
fi

# Check ulimit
echo "  File descriptor limit: $(ulimit -n)"

# Step 6: Try to increase limits
echo "📈 Attempting to increase file descriptor limit..."
ulimit -n 65536 2>/dev/null && echo "  ✅ Increased to 65536" || echo "  ❌ Could not increase"

echo ""
echo "✅ Emergency cleanup complete!"
echo ""
echo "💡 Next steps:"
echo "  1. Try running basic commands to test"
echo "  2. If still having issues, consider system restart"
echo "  3. Review database connection patterns in code"
echo "  4. Add proper cleanup methods to classes"
echo ""
echo "🔍 To investigate remaining issues:"
echo "  python process_investigation.py"