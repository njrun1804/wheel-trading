#!/bin/bash

# Service Health Monitor and Auto-Remediation Script
# Monitors failing services and implements automated fixes

set -euo pipefail

# Configuration
LOG_FILE="/tmp/service_health.log"
FAILED_SERVICES_LOG="/tmp/failed_services.log"
MAX_LOAD_THRESHOLD=10.0
HIGH_CPU_THRESHOLD=30.0
RESTART_COOLDOWN=300  # 5 minutes between restarts

# Logging function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Get current load average
get_load_average() {
    uptime | awk -F'load averages: ' '{print $2}' | awk '{print $1}'
}

# Get failed services (non-zero exit codes)
get_failed_services() {
    launchctl list | grep -E '\-[0-9]+\s' | awk '{print $3}' > "$FAILED_SERVICES_LOG"
    cat "$FAILED_SERVICES_LOG"
}

# Get high CPU processes
get_high_cpu_processes() {
    # Use different approach since top command varies
    ps -Ao pid,pcpu,comm | awk -v threshold="$HIGH_CPU_THRESHOLD" '$2 > threshold {print $1, $2, $3}'
}

# Restart a failed service safely
restart_service() {
    local service_name="$1"
    local cooldown_file="/tmp/restart_${service_name//\./_}"
    
    # Check cooldown period
    if [[ -f "$cooldown_file" ]]; then
        local last_restart=$(stat -f %m "$cooldown_file" 2>/dev/null || echo 0)
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_restart))
        
        if [[ $time_diff -lt $RESTART_COOLDOWN ]]; then
            log_message "Service $service_name is in cooldown period, skipping restart"
            return 1
        fi
    fi
    
    log_message "Attempting to restart service: $service_name"
    
    # Try to unload first, then load
    if launchctl unload -w "/System/Library/LaunchDaemons/${service_name}.plist" 2>/dev/null || 
       launchctl unload -w "/System/Library/LaunchAgents/${service_name}.plist" 2>/dev/null ||
       launchctl unload -w "/Library/LaunchDaemons/${service_name}.plist" 2>/dev/null ||
       launchctl unload -w "/Library/LaunchAgents/${service_name}.plist" 2>/dev/null; then
        sleep 2
        if launchctl load -w "/System/Library/LaunchDaemons/${service_name}.plist" 2>/dev/null ||
           launchctl load -w "/System/Library/LaunchAgents/${service_name}.plist" 2>/dev/null ||
           launchctl load -w "/Library/LaunchDaemons/${service_name}.plist" 2>/dev/null ||
           launchctl load -w "/Library/LaunchAgents/${service_name}.plist" 2>/dev/null; then
            log_message "Successfully restarted service: $service_name"
            touch "$cooldown_file"
            return 0
        fi
    fi
    
    # Alternative: try bootstrapping
    if launchctl bootstrap system "/System/Library/LaunchDaemons/${service_name}.plist" 2>/dev/null ||
       launchctl bootstrap gui/$(id -u) "/System/Library/LaunchAgents/${service_name}.plist" 2>/dev/null; then
        log_message "Successfully bootstrapped service: $service_name"
        touch "$cooldown_file"
        return 0
    fi
    
    log_message "Failed to restart service: $service_name"
    return 1
}

# Kill high CPU processes if they're runaway
handle_high_cpu_processes() {
    log_message "Checking for high CPU processes..."
    
    while IFS=' ' read -r pid cpu_percent command; do
        if [[ -n "$pid" && "$cpu_percent" > "$HIGH_CPU_THRESHOLD" ]]; then
            log_message "High CPU process detected: PID $pid ($command) using ${cpu_percent}% CPU"
            
            # Be selective about what we kill - avoid system critical processes
            case "$command" in
                *python*|*node*|*java*|*ruby*)
                    log_message "Attempting to terminate runaway process: $pid ($command)"
                    if kill -TERM "$pid" 2>/dev/null; then
                        sleep 5
                        if kill -0 "$pid" 2>/dev/null; then
                            log_message "Process $pid still running, sending KILL signal"
                            kill -KILL "$pid" 2>/dev/null || true
                        fi
                        log_message "Terminated high CPU process: $pid"
                    fi
                    ;;
                *)
                    log_message "Skipping system process: $pid ($command)"
                    ;;
            esac
        fi
    done < <(get_high_cpu_processes)
}

# Clear various system caches
clear_system_caches() {
    log_message "Clearing system caches to reduce load..."
    
    # DNS cache
    sudo dscacheutil -flushcache 2>/dev/null || true
    
    # Directory services cache
    sudo killall -HUP mDNSResponder 2>/dev/null || true
    
    # Clear user caches (safely)
    find ~/Library/Caches -name "*.cache" -type f -mtime +1 -delete 2>/dev/null || true
    
    log_message "System cache clearing completed"
}

# Generate service health report
generate_health_report() {
    local current_load=$(get_load_average)
    local failed_count=$(get_failed_services | wc -l | tr -d ' ')
    local total_services=$(launchctl list | wc -l | tr -d ' ')
    
    cat << EOF > "/tmp/service_health_report.txt"
=== SYSTEM SERVICE HEALTH REPORT ===
Generated: $(date)

System Load: $current_load
Failed Services: $failed_count / $total_services
Hardware: $(system_profiler SPHardwareDataType | grep "Model Name" | cut -d: -f2 | xargs)
Memory: $(vm_stat | grep "Pages free" | awk '{print $3 * 16384 / 1024 / 1024}' | cut -d. -f1)MB free

=== FAILED SERVICES ===
$(get_failed_services | head -20)

=== HIGH CPU PROCESSES ===
$(get_high_cpu_processes | head -10)

=== MEMORY STATS ===
$(vm_stat | head -10)
EOF

    log_message "Health report generated at /tmp/service_health_report.txt"
}

# Main monitoring function
monitor_services() {
    log_message "Starting service health monitoring..."
    
    local current_load=$(get_load_average)
    log_message "Current system load: $current_load"
    
    # Check if load is critically high
    if (( $(echo "$current_load > $MAX_LOAD_THRESHOLD" | bc -l) )); then
        log_message "HIGH LOAD DETECTED ($current_load > $MAX_LOAD_THRESHOLD) - Taking remedial action"
        
        # Handle high CPU processes first
        handle_high_cpu_processes
        
        # Clear system caches
        clear_system_caches
        
        # Try restarting a few critical failed services
        local restart_count=0
        while IFS= read -r service && [[ $restart_count -lt 5 ]]; do
            if [[ -n "$service" ]]; then
                restart_service "$service" && ((restart_count++))
            fi
        done < <(get_failed_services | head -5)
    else
        log_message "System load is acceptable ($current_load)"
    fi
    
    # Always generate a health report
    generate_health_report
    
    log_message "Service monitoring cycle completed"
}

# Cleanup function
cleanup() {
    log_message "Service health monitor shutting down"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
case "${1:-monitor}" in
    "monitor")
        monitor_services
        ;;
    "report")
        generate_health_report
        cat "/tmp/service_health_report.txt"
        ;;
    "continuous")
        log_message "Starting continuous monitoring (every 60 seconds)..."
        while true; do
            monitor_services
            sleep 60
        done
        ;;
    *)
        echo "Usage: $0 [monitor|report|continuous]"
        echo "  monitor    - Run one-time health check (default)"
        echo "  report     - Generate and display health report"
        echo "  continuous - Run continuous monitoring"
        exit 1
        ;;
esac