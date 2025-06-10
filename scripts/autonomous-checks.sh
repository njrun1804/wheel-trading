#!/bin/bash
# Autonomous system checks and maintenance

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🤖 Unity Wheel Trading Bot - Autonomous Checks"
echo "=============================================="

# 1. Run diagnostics
echo ""
echo "📊 Running system diagnostics..."
python run.py --diagnose --format json > /tmp/diagnostics.json
if [ $? -eq 0 ]; then
    echo "✅ Diagnostics passed"
else
    echo "❌ Diagnostics failed - check logs"
    exit 1
fi

# 2. Check performance metrics
echo ""
echo "⏱️  Checking performance metrics..."
python run.py --performance --format json > /tmp/performance.json
# Extract key metrics
if command -v jq &> /dev/null; then
    SLOW_OPS=$(jq '.slow_operations | length' /tmp/performance.json)
    if [ "$SLOW_OPS" -gt 0 ]; then
        echo "⚠️  Warning: $SLOW_OPS slow operations detected"
    else
        echo "✅ Performance within SLA"
    fi
fi

# 3. Validate configuration
echo ""
echo "⚙️  Validating configuration..."
python -c "
from src.config import get_config_loader
loader = get_config_loader()
report = loader.generate_health_report()
print('✅ Configuration valid' if report else '❌ Configuration issues')
"

# 4. Check feature flags
echo ""
echo "🚦 Checking feature flags..."
python -c "
from src.unity_wheel.utils import get_feature_flags
flags = get_feature_flags()
report = flags.get_status_report()
degraded = sum(1 for f in report['features'].values() if f['status'] == 'degraded')
if degraded > 0:
    print(f'⚠️  {degraded} features degraded')
else:
    print('✅ All features operational')
"

# 5. Export metrics for monitoring
echo ""
echo "📈 Exporting metrics..."
python run.py --export-metrics --format json > /tmp/export_result.json
echo "✅ Metrics exported to exports/"

# 6. Check for stale cache
echo ""
echo "🗑️  Checking cache..."
find "$PROJECT_ROOT" -name "*.cache" -mtime +7 -exec rm {} \; 2>/dev/null || true
echo "✅ Cache cleaned"

# 7. Validate test coverage (if in dev mode)
if [ "${DEV_MODE:-false}" = "true" ]; then
    echo ""
    echo "🧪 Running tests..."
    python -m pytest tests/test_autonomous_flow.py -v
fi

echo ""
echo "=============================================="
echo "✅ Autonomous checks complete"
echo ""

# Generate summary report
python -c "
import json
from datetime import datetime

# Load results
with open('/tmp/diagnostics.json', 'r') as f:
    diag = json.load(f)
with open('/tmp/performance.json', 'r') as f:
    perf = json.load(f)

print(f'📊 Summary Report - {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print(f'   System Health: {diag[\"summary\"][\"critical_passed\"]}')
print(f'   Total Operations: {perf[\"total_measurements\"]}')
print(f'   Uptime: {perf[\"uptime_hours\"]:.1f} hours')
"
