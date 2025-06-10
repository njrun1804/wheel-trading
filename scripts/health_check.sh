#!/bin/bash
# On-demand health check script for wheel trading bot

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
EXPORT_METRICS=${EXPORT_METRICS:-false}
LOG_FILE="${PROJECT_ROOT}/logs/health_check.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_health() {
    log "Running system health check..."

    # Run diagnostics
    if python "$PROJECT_ROOT/run_on_demand.py" --diagnose 2>&1; then
        log "✅ System diagnostics passed"
        return 0
    else
        log "❌ System diagnostics failed!"
        return 1
    fi
}

check_storage() {
    log "Checking storage health..."

    # Check storage statistics
    python "$PROJECT_ROOT/run_on_demand.py" --storage-stats > /tmp/storage_stats.json 2>&1

    # Parse results if jq is available
    if command -v jq &> /dev/null; then
        DB_SIZE=$(jq '.db_size_mb' /tmp/storage_stats.json 2>/dev/null || echo "0")
        if (( $(echo "$DB_SIZE > 1000" | bc -l) )); then
            log "⚠️  Storage warning: Database size is ${DB_SIZE}MB (>1GB)"
            log "   Consider running cleanup: python -c 'from src.unity_wheel.storage import Storage; import asyncio; s = Storage(); asyncio.run(s.cleanup_old_data())'"
        else
            log "✅ Storage healthy: ${DB_SIZE}MB used"
        fi
    else
        log "✅ Storage check complete (install jq for detailed stats)"
    fi
}

check_cache() {
    log "Checking cache efficiency..."

    python -c "
import asyncio
from src.unity_wheel.storage import Storage

async def check():
    storage = Storage()
    await storage.initialize()
    stats = await storage.get_storage_stats()

    # Calculate cache age
    for table in ['option_chains', 'position_snapshots']:
        key = f'{table}_oldest_days'
        if key in stats:
            days = stats[key]
            if days > 30:
                print(f'⚠️  Old data in {table}: {days} days')
            else:
                print(f'✅ {table} data age: {days} days')

    print(f'Cache size: {stats[\"db_size_mb\"]:.1f} MB')

asyncio.run(check())
" 2>&1 || log "Failed to check cache statistics"
}

check_credentials() {
    log "Checking API credentials..."

    python -c "
from src.unity_wheel.secrets import SecretManager

manager = SecretManager()

# Check for required credentials
required = ['schwab', 'databento', 'ofred']
missing = []

for service in required:
    try:
        creds = manager.get_secret(f'{service}_credentials')
        if creds:
            print(f'✅ {service.capitalize()} credentials found')
        else:
            missing.append(service)
    except Exception as exc:
        missing.append(service)
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.error("Credential check failed for %s: %s", service, exc)

if missing:
    print(f'❌ Missing credentials: {\", \".join(missing)}')
    print('   Run: python scripts/setup-secrets.py')
else:
    print('✅ All credentials configured')
" 2>&1 || log "Failed to check credentials"
}

cleanup_old_data() {
    log "Cleaning up old data..."

    # Remove old cache files
    find "$PROJECT_ROOT" -name "*.cache" -mtime +7 -delete 2>/dev/null || true

    # Remove old export files
    find "$PROJECT_ROOT/exports" -name "*.json" -mtime +30 -delete 2>/dev/null || true
    find "$PROJECT_ROOT/exports" -name "*.csv" -mtime +30 -delete 2>/dev/null || true

    log "✅ Cleanup complete"
}

export_metrics() {
    if [ "$EXPORT_METRICS" = "true" ]; then
        log "Exporting metrics..."
        timestamp=$(date +%Y%m%d_%H%M%S)
        mkdir -p "$PROJECT_ROOT/exports"

        # Export storage stats
        python "$PROJECT_ROOT/run_on_demand.py" --storage-stats > "$PROJECT_ROOT/exports/storage_${timestamp}.json" 2>&1

        log "✅ Metrics exported to exports/storage_${timestamp}.json"
    fi
}

# Main execution
log "=== Unity Wheel Bot Health Check ==="
log "Running comprehensive system check..."

# Run all checks
FAILED=0

check_health || FAILED=1
check_storage
check_cache
check_credentials
export_metrics
cleanup_old_data

# Summary
log ""
log "=== Health Check Summary ==="

if [ $FAILED -eq 0 ]; then
    log "✅ All systems operational"
    log "Ready to generate wheel strategy recommendations"
else
    log "❌ Some issues detected - review logs above"
fi

log ""
log "To get a trading recommendation, run:"
log "  python run_on_demand.py --portfolio 100000"

exit $FAILED
