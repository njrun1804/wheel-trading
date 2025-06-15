#!/bin/bash
# Einstein Stop Script
# Gracefully stop Einstein services and clean up resources

echo "🛑 Stopping Einstein Unified Indexing System"
echo "=============================================="

# Function to log and display
log_and_echo() {
    echo "$1"
    if [ -f ".einstein/logs/shutdown.log" ]; then
        echo "$1" >> ".einstein/logs/shutdown.log"
    fi
}

# Create shutdown log
mkdir -p .einstein/logs
echo "$(date): Einstein shutdown initiated" > ".einstein/logs/shutdown.log"

# Step 1: Stop file watcher services
log_and_echo "👁️ Stopping file watcher services..."

python3 -c "
import asyncio
import signal
import os
from pathlib import Path

async def stop_file_watcher():
    try:
        from einstein.unified_index import get_einstein_hub
        
        hub = get_einstein_hub()
        
        # Stop file watching if running
        if hasattr(hub, 'stop_file_watching'):
            await hub.stop_file_watching()
            print('✅ File watcher stopped')
        
        # Cleanup any running processes
        if hasattr(hub, 'cleanup'):
            await hub.cleanup()
            print('✅ Resources cleaned up')
            
        return True
        
    except Exception as e:
        print(f'⚠️ Cleanup warning: {e}')
        return False

asyncio.run(stop_file_watcher())
" 2>&1

# Step 2: Stop any meta processes (if running)
log_and_echo "🧠 Checking for meta processes..."

META_PIDS=$(ps aux | grep -E "(meta_daemon|meta_coordinator|meta_prime)" | grep -v grep | awk '{print $2}' | tr '\n' ' ')

if [ ! -z "$META_PIDS" ]; then
    log_and_echo "   Found meta processes: $META_PIDS"
    log_and_echo "   Stopping meta processes..."
    echo $META_PIDS | xargs kill -TERM 2>/dev/null || true
    sleep 2
    
    # Force kill if still running
    REMAINING=$(ps aux | grep -E "(meta_daemon|meta_coordinator|meta_prime)" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
    if [ ! -z "$REMAINING" ]; then
        log_and_echo "   Force stopping remaining processes: $REMAINING"
        echo $REMAINING | xargs kill -9 2>/dev/null || true
    fi
    log_and_echo "✅ Meta processes stopped"
else
    log_and_echo "   No meta processes found"
fi

# Step 3: Clean up shared memory and locks
log_and_echo "🧹 Cleaning up shared resources..."

python3 -c "
import os
import shutil
from pathlib import Path

# Clean up temporary files
temp_patterns = [
    '/tmp/einstein_*',
    '/tmp/duckdb_*',
    '/dev/shm/einstein_*'
]

cleaned = 0
for pattern in temp_patterns:
    import glob
    for path in glob.glob(pattern):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
            cleaned += 1
        except:
            pass

if cleaned > 0:
    print(f'✅ Cleaned {cleaned} temporary files')
else:
    print('✅ No temporary files to clean')

# Clean up lock files
lock_files = [
    '.einstein/index.lock',
    '.einstein/faiss.lock',
    '.einstein/db.lock'
]

for lock_file in lock_files:
    if Path(lock_file).exists():
        Path(lock_file).unlink()
        print(f'✅ Removed lock file: {lock_file}')
"

# Step 4: Update status
log_and_echo "📝 Updating status..."

if [ -f ".einstein/status.json" ]; then
    python3 -c "
import json
import time

try:
    with open('.einstein/status.json', 'r') as f:
        status = json.load(f)
    
    status['status'] = 'stopped'
    status['shutdown_time'] = time.time()
    
    with open('.einstein/status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    print('✅ Status updated to stopped')
    
except Exception as e:
    print(f'⚠️ Could not update status: {e}')
"
else
    log_and_echo "   No status file found"
fi

# Step 5: Final cleanup
log_and_echo "🗑️ Final cleanup..."

# Remove PID files if they exist
for pid_file in .einstein/*.pid; do
    if [ -f "$pid_file" ]; then
        rm "$pid_file"
        log_and_echo "   Removed PID file: $pid_file"
    fi
done

# Compress old log files
if [ -d ".einstein/logs" ]; then
    find .einstein/logs -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null || true
    log_and_echo "   Compressed old log files"
fi

log_and_echo ""
log_and_echo "✅ Einstein shutdown complete!"
log_and_echo ""
log_and_echo "📊 Final status:"

if [ -f ".einstein/status.json" ]; then
    python3 -c "
import json
with open('.einstein/status.json') as f:
    data = json.load(f)
print(f'   Status: {data.get(\"status\", \"unknown\")}')
print(f'   Last indexed: {data.get(\"files_indexed\", 0)} files')
print(f'   Index preserved: {data.get(\"index_size_mb\", 0):.2f} MB')
"
fi

log_and_echo ""
log_and_echo "💡 To restart Einstein:"
log_and_echo "   ./start_einstein.sh"
log_and_echo ""
log_and_echo "🗂️ Index data preserved in .einstein/ directory"