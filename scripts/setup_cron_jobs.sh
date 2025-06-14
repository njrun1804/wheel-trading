#!/bin/bash
# Setup cron jobs for automated data collection

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the absolute path to the project
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH="$(which python3)"

echo -e "${GREEN}Unity Wheel Trading - Data Collection Setup${NC}"
echo "================================================"
echo "Project directory: $PROJECT_DIR"
echo "Python path: $PYTHON_PATH"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Function to add cron job if it doesn't exist
add_cron_job() {
    local schedule="$1"
    local command="$2"
    local description="$3"
    
    # Check if the job already exists
    if crontab -l 2>/dev/null | grep -q "$command"; then
        echo -e "${YELLOW}⚠️  $description already scheduled${NC}"
    else
        # Add the job
        (crontab -l 2>/dev/null; echo "# $description"; echo "$schedule $command") | crontab -
        echo -e "${GREEN}✅ Added: $description${NC}"
    fi
}

# Function to create wrapper scripts
create_wrapper_script() {
    local script_name="$1"
    local wrapper_name="$2"
    local wrapper_path="$PROJECT_DIR/scripts/$wrapper_name"
    
    cat > "$wrapper_path" << EOF
#!/bin/bash
# Wrapper script for $script_name
# Ensures proper environment and error handling

# Load environment variables if .env exists
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Change to project directory
cd "$PROJECT_DIR"

# Run the script with proper Python path
exec "$PYTHON_PATH" "$PROJECT_DIR/scripts/$script_name" "\$@"
EOF
    
    chmod +x "$wrapper_path"
    echo -e "${GREEN}✅ Created wrapper: $wrapper_name${NC}"
}

# Create wrapper scripts
echo -e "\n${GREEN}Creating wrapper scripts...${NC}"
create_wrapper_script "collect_eod_production.py" "eod_wrapper.sh"
create_wrapper_script "collect_intraday.py" "intraday_wrapper.sh"

# Setup cron jobs
echo -e "\n${GREEN}Setting up cron jobs...${NC}"

# EOD Collection - 4:30 PM ET every weekday
# Note: Adjust time based on your server timezone
add_cron_job \
    "30 16 * * 1-5" \
    "$PROJECT_DIR/scripts/eod_wrapper.sh >> $PROJECT_DIR/logs/eod_collection.log 2>&1" \
    "EOD Data Collection (4:30 PM ET weekdays)"

# Intraday Collection - Every 15 minutes during market hours
# 9:30 AM - 4:00 PM ET = 9-16 hours
add_cron_job \
    "*/15 9-15 * * 1-5" \
    "$PROJECT_DIR/scripts/intraday_wrapper.sh --once >> $PROJECT_DIR/logs/intraday_collection.log 2>&1" \
    "Intraday Data Collection (every 15 min during market)"

# Log rotation - Daily at midnight
add_cron_job \
    "0 0 * * *" \
    "$PROJECT_DIR/scripts/rotate_logs.sh >> $PROJECT_DIR/logs/maintenance.log 2>&1" \
    "Daily log rotation"

# Create log rotation script
cat > "$PROJECT_DIR/scripts/rotate_logs.sh" << 'EOF'
#!/bin/bash
# Rotate logs daily, keep 30 days

LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../logs" && pwd)"
RETENTION_DAYS=30

# Function to rotate a log file
rotate_log() {
    local log_file="$1"
    local base_name=$(basename "$log_file" .log)
    local date_suffix=$(date +%Y%m%d)
    
    if [ -f "$log_file" ]; then
        # Move current log to dated version
        mv "$log_file" "${LOG_DIR}/${base_name}_${date_suffix}.log"
        
        # Compress logs older than 1 day
        find "$LOG_DIR" -name "${base_name}_*.log" -mtime +1 -exec gzip {} \;
        
        # Remove logs older than retention period
        find "$LOG_DIR" -name "${base_name}_*.log.gz" -mtime +$RETENTION_DAYS -delete
        
        # Create new empty log file
        touch "$log_file"
    fi
}

# Rotate each log file
rotate_log "$LOG_DIR/eod_collection.log"
rotate_log "$LOG_DIR/intraday_collection.log"
rotate_log "$LOG_DIR/maintenance.log"

echo "Log rotation completed at $(date)"
EOF

chmod +x "$PROJECT_DIR/scripts/rotate_logs.sh"

# Show current crontab
echo -e "\n${GREEN}Current cron jobs:${NC}"
crontab -l 2>/dev/null | grep -E "(wheel-trading|Unity)" || echo "No Unity wheel trading jobs found"

# Create systemd service (optional, for systems using systemd)
if command -v systemctl &> /dev/null; then
    echo -e "\n${YELLOW}Optional: Create systemd service for more reliable scheduling?${NC}"
    echo "This provides better logging and restart capabilities than cron."
    read -p "Create systemd service? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Create service file
        sudo tee /etc/systemd/system/unity-eod-collection.service > /dev/null << EOF
[Unit]
Description=Unity EOD Data Collection
After=network.target

[Service]
Type=oneshot
ExecStart=$PYTHON_PATH $PROJECT_DIR/scripts/collect_eod_production.py
WorkingDirectory=$PROJECT_DIR
StandardOutput=append:$PROJECT_DIR/logs/eod_collection.log
StandardError=append:$PROJECT_DIR/logs/eod_collection.log
Environment="PATH=/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
EOF

        # Create timer
        sudo tee /etc/systemd/system/unity-eod-collection.timer > /dev/null << EOF
[Unit]
Description=Run Unity EOD Collection at 4:30 PM ET weekdays
Requires=unity-eod-collection.service

[Timer]
OnCalendar=Mon-Fri 16:30:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

        # Enable and start timer
        sudo systemctl daemon-reload
        sudo systemctl enable unity-eod-collection.timer
        sudo systemctl start unity-eod-collection.timer
        
        echo -e "${GREEN}✅ Systemd service created and enabled${NC}"
        echo "View status: systemctl status unity-eod-collection.timer"
    fi
fi

# Create test script
cat > "$PROJECT_DIR/scripts/test_collection.sh" << EOF
#!/bin/bash
# Test data collection scripts

echo "Testing EOD collection..."
cd "$PROJECT_DIR"
"$PYTHON_PATH" scripts/collect_eod_production.py

echo -e "\nTesting intraday collection..."
"$PYTHON_PATH" scripts/collect_intraday.py --once

echo -e "\nChecking database..."
echo "SELECT COUNT(*) as options_count FROM options.contracts WHERE symbol='U';" | \\
    duckdb "$PROJECT_DIR/data/wheel_trading_optimized.duckdb"
EOF

chmod +x "$PROJECT_DIR/scripts/test_collection.sh"

echo -e "\n${GREEN}Setup complete!${NC}"
echo "================================================"
echo "✅ Cron jobs scheduled"
echo "✅ Log rotation configured"
echo "✅ Wrapper scripts created"
echo ""
echo "Next steps:"
echo "1. Test collection: ./scripts/test_collection.sh"
echo "2. View logs: tail -f logs/eod_collection.log"
echo "3. Check cron: crontab -l"
echo "4. Monitor: watch 'tail -20 logs/*.log'"
echo ""
echo -e "${YELLOW}Note: Adjust cron times if your server is not in ET timezone${NC}"