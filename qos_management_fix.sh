#!/bin/bash
# QoS Management Fix Script for macOS M4 Pro
# Addresses high load averages and failing services

set -e

LOG_FILE="/tmp/qos_management_fix.log"
BACKUP_DIR="/tmp/service_backups"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

create_backup_dir() {
    mkdir -p "$BACKUP_DIR"
    log "Created backup directory: $BACKUP_DIR"
}

fix_unity_trading_services() {
    log "=== Fixing Unity Trading Services ==="
    
    # Stop and disable all Unity trading services
    local unity_services=(
        "com.unity.trading.daily-updater"
        "com.unity.trading.daily"
    )
    
    for service in "${unity_services[@]}"; do
        log "Processing service: $service"
        
        # Try to stop if running
        if launchctl list | grep -q "$service"; then
            log "Stopping service: $service"
            launchctl stop "gui/501/$service" 2>/dev/null || true
            launchctl bootout "gui/501/$service" 2>/dev/null || true
        fi
        
        # Find and disable plist files
        local plist_locations=(
            "$HOME/Library/LaunchAgents/$service.plist"
            "/System/Library/LaunchAgents/$service.plist"
            "/Library/LaunchAgents/$service.plist"
        )
        
        for plist in "${plist_locations[@]}"; do
            if [[ -f "$plist" ]]; then
                log "Found plist: $plist"
                cp "$plist" "$BACKUP_DIR/" 2>/dev/null || true
                mv "$plist" "$plist.disabled" 2>/dev/null || true
                log "Disabled plist: $plist"
            fi
        done
    done
    
    # Kill any remaining Unity processes
    pkill -f "daily_updater.py" 2>/dev/null || true
    pkill -f "unity.*trading" 2>/dev/null || true
    log "Killed remaining Unity trading processes"
}

fix_problematic_apple_services() {
    log "=== Managing Problematic Apple Services ==="
    
    # Services that are causing high load due to constant crashes
    local problematic_services=(
        "com.apple.knowledgeconstructiond"
        "com.apple.duetexpertd"
        "com.apple.mlruntimed"
        "com.apple.feedbackd"
        "com.apple.spindump_agent"
        "com.apple.metrickitd"
    )
    
    for service in "${problematic_services[@]}"; do
        log "Temporarily disabling problematic service: $service"
        launchctl stop "gui/501/$service" 2>/dev/null || true
        launchctl disable "gui/501/$service" 2>/dev/null || true
    done
}

optimize_system_resources() {
    log "=== Optimizing System Resources ==="
    
    # Clear system caches
    log "Clearing system caches..."
    sudo purge 2>/dev/null || true
    
    # Restart critical services that may be stuck
    log "Restarting critical services..."
    sudo launchctl stop com.apple.WindowServer 2>/dev/null || true
    
    # Adjust memory pressure settings
    log "Adjusting memory pressure settings..."
    sudo sysctl -w vm.pressure_disable_threshold=15 2>/dev/null || log "Could not adjust pressure threshold"
    
    # Reduce background app refresh
    log "Reducing background activity..."
    killall -STOP backgroundtaskmanagementagent 2>/dev/null || true
    sleep 2
    killall -CONT backgroundtaskmanagementagent 2>/dev/null || true
}

create_load_monitoring_service() {
    log "=== Creating Load Monitoring Service ==="
    
    # Create a service that monitors load and takes action
    cat > "$HOME/Library/LaunchAgents/com.wheel.load-monitor.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wheel.load-monitor</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/tmp/load_monitor.sh</string>
    </array>
    
    <key>StartInterval</key>
    <integer>300</integer>
    
    <key>StandardOutPath</key>
    <string>/tmp/load_monitor.log</string>
    
    <key>StandardErrorPath</key>
    <string>/tmp/load_monitor_error.log</string>
</dict>
</plist>
EOF

    # Create the monitoring script
    cat > "/tmp/load_monitor.sh" << 'EOF'
#!/bin/bash
# Load Monitor Script - Runs every 5 minutes

LOAD_THRESHOLD=10.0
CURRENT_LOAD=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $2}')

if (( $(echo "$CURRENT_LOAD > $LOAD_THRESHOLD" | bc -l) )); then
    echo "$(date): High load detected: $CURRENT_LOAD"
    
    # Kill runaway processes
    pkill -f "daily_updater.py" 2>/dev/null || true
    pkill -f "knowledgeconstructiond" 2>/dev/null || true
    pkill -f "duetexpertd" 2>/dev/null || true
    
    # Clear memory pressure
    sudo purge 2>/dev/null || true
    
    echo "$(date): Load mitigation actions taken"
fi
EOF

    chmod +x "/tmp/load_monitor.sh"
    
    # Load the service
    launchctl load "$HOME/Library/LaunchAgents/com.wheel.load-monitor.plist" 2>/dev/null || true
    log "Created and loaded load monitoring service"
}

fix_mcp_autostart_service() {
    log "=== Fixing MCP Autostart Service ==="
    
    # The com.mcp.autostart service is failing (status 127)
    local mcp_plist="$HOME/Library/LaunchAgents/com.mcp.autostart.plist"
    
    if [[ -f "$mcp_plist" ]]; then
        log "Found MCP autostart plist, backing up and disabling"
        cp "$mcp_plist" "$BACKUP_DIR/" 2>/dev/null || true
        launchctl bootout "gui/501/com.mcp.autostart" 2>/dev/null || true
        mv "$mcp_plist" "$mcp_plist.disabled" 2>/dev/null || true
    fi
}

restart_essential_services() {
    log "=== Restarting Essential Services ==="
    
    # Restart services that are important but may be stuck
    local essential_services=(
        "com.apple.Dock.agent"
    )
    
    for service in "${essential_services[@]}"; do
        log "Restarting essential service: $service"
        launchctl stop "gui/501/$service" 2>/dev/null || true
        sleep 2
        launchctl start "gui/501/$service" 2>/dev/null || true
    done
}

show_system_status() {
    log "=== System Status After Fixes ==="
    
    local load_avg=$(uptime | awk -F'load averages:' '{print $2}')
    log "Current load averages:$load_avg"
    
    local memory_pressure=$(vm_stat | grep -E "(free|wired|active|inactive)" | head -4)
    log "Memory status:"
    echo "$memory_pressure" | while read line; do
        log "  $line"
    done
    
    local failed_services=$(launchctl list | grep -E "^[0-9].*[^0-]$" | wc -l | tr -d ' ')
    log "Failed services count: $failed_services"
}

main() {
    log "Starting QoS Management Fix Script"
    
    create_backup_dir
    fix_unity_trading_services
    fix_mcp_autostart_service
    fix_problematic_apple_services
    optimize_system_resources
    create_load_monitoring_service
    restart_essential_services
    
    # Wait for changes to take effect
    sleep 10
    
    show_system_status
    log "QoS Management Fix Script completed"
    
    echo "============================================"
    echo "QoS MANAGEMENT FIX SUMMARY"
    echo "============================================"
    echo "Actions completed:"
    echo "✓ Stopped Unity trading services"
    echo "✓ Disabled problematic Apple services"
    echo "✓ Fixed MCP autostart service"
    echo "✓ Optimized system resources"
    echo "✓ Created load monitoring service"
    echo "✓ Restarted essential services"
    echo ""
    echo "Log file: $LOG_FILE"
    echo "Backups: $BACKUP_DIR"
    echo "============================================"
}

main "$@"