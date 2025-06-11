#!/bin/bash
# Automated data refresh script for wheel trading system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_ROOT" || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Log file with timestamp
LOG_FILE="logs/data_refresh_$(date +%Y%m%d).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check command success
check_status() {
    if [ $1 -eq 0 ]; then
        log "✅ $2 completed successfully"
    else
        log "❌ $2 failed with exit code $1"
        ERRORS=$((ERRORS + 1))
    fi
}

# Initialize error counter
ERRORS=0

log "🔄 Starting data refresh"
log "Project root: $PROJECT_ROOT"

# 1. Check environment
log "🔍 Checking environment..."
if [ -z "$SCHWAB_CLIENT_ID" ] || [ -z "$SCHWAB_CLIENT_SECRET" ]; then
    log "⚠️  Warning: Schwab credentials not set in environment"
fi

# 2. Update Unity prices
log "📈 Updating Unity price data..."
python tools/analysis/pull_unity_prices.py >> "$LOG_FILE" 2>&1
check_status $? "Unity price update"

# 3. Fetch current options chain
log "📋 Fetching options chain..."
if [ -f "tools/data/fetch_unity_options.py" ]; then
    python tools/data/fetch_unity_options.py >> "$LOG_FILE" 2>&1
    check_status $? "Options chain fetch"
else
    log "⚠️  Options fetch script not found, skipping"
fi

# 4. Update positions snapshot
log "💼 Updating positions..."
if [ -n "$SCHWAB_CLIENT_ID" ]; then
    log "ℹ️  Legacy ingestion step removed; skipping position snapshot"
else
    log "⚠️  Skipping position update - no Schwab credentials"
fi

# 5. Refresh FRED data (only if older than 1 day)
log "📊 Checking economic data..."
FRED_AGE=$(python -c "
import duckdb
import os
from datetime import datetime, timedelta
db_path = os.path.expanduser('~/.wheel_trading/cache/wheel_cache.duckdb')
if os.path.exists(db_path):
    conn = duckdb.connect(db_path, read_only=True)
    try:
        result = conn.execute(\"\"\"
            SELECT CURRENT_DATE - MAX(observation_date) as days_old
            FROM fred_observations
        \"\"\").fetchone()
        print(result[0] if result and result[0] else 999)
    except:
        print(999)
    finally:
        conn.close()
else:
    print(999)
" 2>/dev/null)

if [ "$FRED_AGE" -gt 1 ]; then
    log "📊 Updating economic data (${FRED_AGE} days old)..."
    python tools/analysis/pull_fred_data_efficient.py >> "$LOG_FILE" 2>&1
    check_status $? "FRED data update"
else
    log "✅ Economic data is current (${FRED_AGE} days old)"
fi

# 6. Clean up old cache entries
log "🧹 Cleaning old cache entries..."
python -c "
from src.unity_wheel.storage.duckdb_cache import DuckDBCache
cache = DuckDBCache()
deleted = cache.cleanup_old_entries(days=30)
print(f'Deleted {deleted} old cache entries')
" >> "$LOG_FILE" 2>&1
check_status $? "Cache cleanup"

# 7. Run diagnostics
log "🏥 Running health check..."
python run.py --diagnose >> "$LOG_FILE" 2>&1
check_status $? "Health check"

# 8. Generate summary
log "📊 Generating data summary..."
python -c "
import duckdb
import os
from datetime import datetime
from src.config.loader import get_config
config = get_config()
ticker = config.unity.ticker

db_path = os.path.expanduser('~/.wheel_trading/cache/wheel_cache.duckdb')
if os.path.exists(db_path):
    conn = duckdb.connect(db_path, read_only=True)
    try:
        # Get data counts
        unity_prices = conn.execute(f'SELECT COUNT(*) FROM price_history WHERE symbol = \"{ticker}\"').fetchone()[0]
        options = conn.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_name = \"options_data\"').fetchone()[0]
        if options > 0:
            options_count = conn.execute(f'SELECT COUNT(*) FROM options_data WHERE underlying = \"{ticker}\"').fetchone()[0]
        else:
            options_count = 0

        # Get latest dates
        latest_price = conn.execute(f'SELECT MAX(date) FROM price_history WHERE symbol = \"{ticker}\"').fetchone()[0]

        print(f'Data Summary:')
        print(f'  {ticker} prices: {unity_prices} records')
        print(f'  {ticker} options: {options_count} contracts')
        print(f'  Latest price date: {latest_price}')
    finally:
        conn.close()
" >> "$LOG_FILE" 2>&1

# 9. Check for errors and notify
if [ $ERRORS -gt 0 ]; then
    log "⚠️  Data refresh completed with $ERRORS errors"
    echo -e "${YELLOW}Data refresh completed with errors. Check $LOG_FILE for details.${NC}"

    # Show recent errors
    echo -e "\n${RED}Recent errors:${NC}"
    grep -E "(ERROR|FAIL|Error|Failed|❌)" "$LOG_FILE" | tail -5
else
    log "✅ Data refresh completed successfully"
    echo -e "${GREEN}Data refresh completed successfully!${NC}"
fi

# 10. Show data freshness
echo -e "\n📊 Current Data Status:"
python -c "
import duckdb
import os
from datetime import datetime
from src.config.loader import get_config
config = get_config()
ticker = config.unity.ticker

db_path = os.path.expanduser('~/.wheel_trading/cache/wheel_cache.duckdb')
if os.path.exists(db_path):
    conn = duckdb.connect(db_path, read_only=True)
    try:
        # Unity prices
        result = conn.execute(f\"\"\"
            SELECT MAX(date) as latest,
                   CURRENT_DATE - MAX(date) as days_old,
                   COUNT(*) as records
            FROM price_history WHERE symbol = '{ticker}'
        \"\"\").fetchone()
        if result[0]:
            status = '✅' if result[1] <= 1 else '⚠️'
            print(f'{status} {ticker} prices: {result[0]} ({result[1]} days old, {result[2]} records)')

        # Options data
        if conn.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_name = \"options_data\"').fetchone()[0] > 0:
            result = conn.execute(f\"\"\"
                SELECT MAX(timestamp) as latest,
                       COUNT(*) as records
                FROM options_data WHERE underlying = '{ticker}'
            \"\"\").fetchone()
            if result[0]:
                latest = datetime.fromisoformat(str(result[0]))
                days_old = (datetime.now() - latest).days
                status = '✅' if days_old <= 1 else '⚠️'
                print(f'{status} {ticker} options: {latest.strftime(\"%Y-%m-%d %H:%M\")} ({days_old} days old, {result[1]} records)')
            else:
                print(f'❌ {ticker} options: No data')

        # FRED data
        result = conn.execute(\"\"\"
            SELECT MAX(observation_date) as latest,
                   CURRENT_DATE - MAX(observation_date) as days_old
            FROM fred_observations
        \"\"\").fetchone()
        if result[0]:
            status = '✅' if result[1] <= 7 else '⚠️'
            print(f'{status} Economic data: {result[0]} ({result[1]} days old)')

    finally:
        conn.close()
"

log "🏁 Data refresh process completed"
exit $ERRORS
