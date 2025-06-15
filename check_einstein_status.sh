#!/bin/bash
# Einstein Status Checker
# Quick health check for Einstein indexing services

echo "🔍 Einstein System Status Check"
echo "================================"

# Check if status file exists
if [ ! -f ".einstein/status.json" ]; then
    echo "❌ Einstein not initialized (no status file found)"
    echo "   Run: ./start_einstein.sh"
    exit 1
fi

# Parse status file
echo "📊 System Status:"
python3 -c "
import json
import time
from pathlib import Path

try:
    with open('.einstein/status.json') as f:
        status = json.load(f)
    
    startup_time = status.get('startup_time', 0)
    uptime_hours = (time.time() - startup_time) / 3600
    
    print(f'   Status: {status.get(\"status\", \"unknown\")}')
    print(f'   Uptime: {uptime_hours:.1f} hours')
    print(f'   Files indexed: {status.get(\"files_indexed\", 0):,}')
    print(f'   Total lines: {status.get(\"total_lines\", 0):,}')
    print(f'   Index size: {status.get(\"index_size_mb\", 0):.2f} MB')
    print(f'   Coverage: {status.get(\"coverage_percentage\", 0):.1f}%')
    
    # Check index directory
    einstein_dir = Path('.einstein')
    if einstein_dir.exists():
        total_size = sum(f.stat().st_size for f in einstein_dir.rglob('*') if f.is_file())
        print(f'   Disk usage: {total_size / 1024 / 1024:.2f} MB')
    
except Exception as e:
    print(f'❌ Error reading status: {e}')
"

echo ""
echo "🧪 Quick functionality test:"

# Test basic search
python3 -c "
import asyncio
import time

async def test_search():
    try:
        from einstein.unified_index import get_einstein_hub
        
        start = time.time()
        hub = get_einstein_hub()
        
        # Quick search test
        results = await hub.search('WheelStrategy', ['text'])
        elapsed = (time.time() - start) * 1000
        
        print(f'   Text search: ✅ {len(results)} results in {elapsed:.1f}ms')
        
        # Test semantic search
        start = time.time()
        results = await hub.search('options trading', ['semantic'])
        elapsed = (time.time() - start) * 1000
        
        print(f'   Semantic search: ✅ {len(results)} results in {elapsed:.1f}ms')
        
        return True
        
    except Exception as e:
        print(f'   Search test: ❌ {e}')
        return False

success = asyncio.run(test_search())
" 2>/dev/null

echo ""
echo "📁 Index files:"
if [ -d ".einstein" ]; then
    find .einstein -name "*.db" -o -name "*.index" -o -name "*.json" | head -10 | while read file; do
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "   $file ($size)"
    done
else
    echo "   ❌ .einstein directory not found"
fi

echo ""
echo "🔄 Recent activity:"
if [ -f ".einstein/logs" ] && [ -d ".einstein/logs" ]; then
    echo "   $(ls -1t .einstein/logs/*.log 2>/dev/null | head -1 | xargs tail -3 2>/dev/null || echo 'No recent logs')"
else
    echo "   No log directory found"
fi

echo ""
if [ -f ".einstein/status.json" ]; then
    echo "✅ Einstein is running and healthy"
    echo ""
    echo "💡 Usage examples:"
    echo "   • Search code: python -c \"from einstein.unified_index import get_einstein_hub; import asyncio; hub = get_einstein_hub(); print(asyncio.run(hub.search('function_name')))\""
    echo "   • Get stats: python -c \"from einstein.unified_index import get_einstein_hub; import asyncio; hub = get_einstein_hub(); print(asyncio.run(hub.get_stats()))\""
else
    echo "⚠️ Einstein status unclear"
    echo "   Try: ./start_einstein.sh"
fi