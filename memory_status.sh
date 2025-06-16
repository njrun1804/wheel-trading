#!/bin/bash
# Memory Management System Status Script

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=========================================="
echo -e "${CYAN}Memory Management System Status${NC}"
echo "=========================================="

# Get current memory status using Python
python3 -c "
import psutil
import json
from datetime import datetime

vm = psutil.virtual_memory()
swap = psutil.swap_memory()

print(f'Timestamp: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print()
print('MEMORY STATUS:')
print(f'  Total RAM: {vm.total / 1024 / 1024 / 1024:.1f} GB')
print(f'  Available: {vm.available / 1024 / 1024:.0f} MB')
print(f'  Used: {vm.percent:.1f}%')

if vm.available < 500 * 1024 * 1024:
    status = 'CRITICAL'
    color = '\033[0;31m'
elif vm.available < 1024 * 1024 * 1024:
    status = 'WARNING'
    color = '\033[1;33m'
else:
    status = 'OPTIMAL'
    color = '\033[0;32m'

print(f'  Status: {color}{status}\033[0m')
print()
print('SWAP STATUS:')
print(f'  Total: {swap.total / 1024 / 1024 / 1024:.1f} GB')
print(f'  Used: {swap.percent:.1f}%')
"

echo ""
echo "SYSTEM SERVICES:"

# Check if memory monitor is running
if [ -f "pids/memory_monitor.pid" ]; then
    pid=$(cat pids/memory_monitor.pid)
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "  Memory Monitor: ${GREEN}RUNNING${NC} (PID: $pid)"
    else
        echo -e "  Memory Monitor: ${RED}STOPPED${NC}"
    fi
else
    echo -e "  Memory Monitor: ${RED}NOT DEPLOYED${NC}"
fi

# Check if process manager is running
if [ -f "pids/process_manager.pid" ]; then
    pid=$(cat pids/process_manager.pid)
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "  Process Manager: ${GREEN}RUNNING${NC} (PID: $pid)"
    else
        echo -e "  Process Manager: ${RED}STOPPED${NC}"
    fi
else
    echo -e "  Process Manager: ${RED}NOT DEPLOYED${NC}"
fi

# Check cron jobs
if crontab -l 2>/dev/null | grep -q "memory_cleanup_emergency.py"; then
    echo -e "  Automated Cleanup: ${GREEN}SCHEDULED${NC}"
else
    echo -e "  Automated Cleanup: ${YELLOW}NOT SCHEDULED${NC}"
fi

echo ""
echo "TOP MEMORY CONSUMERS:"

# Show top 10 memory consuming processes
python3 -c "
import psutil

processes = []
for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
    try:
        proc_info = proc.info
        memory_mb = proc_info['memory_info'].rss / 1024 / 1024
        processes.append((proc_info['pid'], proc_info['name'], memory_mb))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

processes.sort(key=lambda x: x[2], reverse=True)

for pid, name, memory_mb in processes[:10]:
    print(f'  {name:<20} (PID: {pid:<6}): {memory_mb:>6.1f} MB')
"

echo ""
echo "RECENT LOG ENTRIES:"

# Show recent memory monitor log entries
if [ -f "logs/memory_monitor.log" ]; then
    echo -e "${BLUE}Memory Monitor (last 5 entries):${NC}"
    tail -5 logs/memory_monitor.log | sed 's/^/  /'
else
    echo -e "${YELLOW}  No memory monitor log found${NC}"
fi

echo ""

# Show recent process manager log entries
if [ -f "logs/process_manager.log" ]; then
    echo -e "${BLUE}Process Manager (last 5 entries):${NC}"
    tail -5 logs/process_manager.log | sed 's/^/  /'
else
    echo -e "${YELLOW}  No process manager log found${NC}"
fi

echo ""
echo "SYSTEM LOAD:"

# Show system load
uptime

echo ""
echo "RECOMMENDATIONS:"

# Generate recommendations based on current status
python3 -c "
import psutil

vm = psutil.virtual_memory()
load_avg = open('/proc/loadavg').read().split()[0] if os.path.exists('/proc/loadavg') else '0'

recommendations = []

if vm.available < 500 * 1024 * 1024:
    recommendations.append('🚨 CRITICAL: Run emergency cleanup immediately')
    recommendations.append('   Command: python3 memory_cleanup_emergency.py')
elif vm.available < 1024 * 1024 * 1024:
    recommendations.append('⚠️  WARNING: Consider running cleanup')
    recommendations.append('   Command: python3 memory_cleanup_emergency.py')

if vm.percent > 90:
    recommendations.append('💾 High memory usage - close unnecessary applications')

# Check for memory leaks (processes using > 2GB)
high_memory_processes = []
for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
    try:
        memory_mb = proc.info['memory_info'].rss / 1024 / 1024
        if memory_mb > 2000:
            high_memory_processes.append((proc.info['name'], memory_mb))
    except:
        continue

if high_memory_processes:
    recommendations.append('🔍 Large memory consumers detected:')
    for name, memory_mb in high_memory_processes[:3]:
        recommendations.append(f'   {name}: {memory_mb:.0f}MB')

if not recommendations:
    recommendations.append('✅ System memory status is optimal')

for rec in recommendations:
    print(f'  {rec}')
" 2>/dev/null || echo "  ✅ System appears to be running normally"

echo ""
echo "=========================================="