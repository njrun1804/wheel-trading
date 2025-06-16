#!/bin/bash

# Service Optimizer - Manages service startup order and dependencies
# Optimizes system boot and service resource allocation

set -euo pipefail

# Configuration
LOG_FILE="/tmp/service_optimizer.log"
RESOURCE_LIMITS_FILE="/tmp/service_resource_limits.conf"
DEPENDENCY_MAP_FILE="/tmp/service_dependencies.json"
OPTIMIZATION_RESULTS="/tmp/optimization_results.txt"

# Service priority groups (higher number = higher priority)
declare -A SERVICE_PRIORITIES=(
    # Core system services (highest priority)
    ["com.apple.launchd"]=100
    ["com.apple.kernel"]=100
    ["com.apple.WindowServer"]=90
    
    # Network and connectivity
    ["com.apple.networkd"]=80
    ["com.apple.mDNSResponder"]=80
    ["com.apple.wifi"]=75
    
    # Security and authentication
    ["com.apple.securityd"]=85
    ["com.apple.authd"]=85
    
    # File system and storage
    ["com.apple.fskd"]=70
    ["com.apple.diskmanagementd"]=70
    
    # User interface
    ["com.apple.Dock"]=60
    ["com.apple.Finder"]=60
    
    # Background services (lower priority)
    ["com.apple.bird"]=30  # iCloud Drive (often problematic)
    ["com.apple.cloudd"]=30  # CloudKit
    ["com.apple.photoanalysisd"]=20
    ["com.apple.spotlightd"]=40
)

# Resource limits for problematic services (CPU percentage, Memory MB)
declare -A RESOURCE_LIMITS=(
    ["com.apple.bird"]="20,1024"
    ["com.apple.cloudd"]="15,512"
    ["com.apple.photoanalysisd"]="10,2048"
    ["com.apple.spotlightd"]="25,1024"
    ["com.apple.mlruntimed"]="10,512"
)

# Logging function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Get service startup time
get_service_startup_time() {
    local service_name="$1"
    # This is a simplified estimation - actual startup time tracking would require more complex monitoring
    local base_time=5  # Base startup time in seconds
    local priority=${SERVICE_PRIORITIES[$service_name]:-50}
    
    # Higher priority services get more time allowance
    echo $((base_time + (priority / 10)))
}

# Create service dependency map
create_dependency_map() {
    log_message "Creating service dependency map..."
    
    cat > "$DEPENDENCY_MAP_FILE" << 'EOF'
{
  "dependencies": {
    "com.apple.WindowServer": ["com.apple.launchd"],
    "com.apple.Dock": ["com.apple.WindowServer"],
    "com.apple.Finder": ["com.apple.WindowServer"],
    "com.apple.networkd": ["com.apple.launchd"],
    "com.apple.mDNSResponder": ["com.apple.networkd"],
    "com.apple.wifi": ["com.apple.networkd"],
    "com.apple.bird": ["com.apple.networkd", "com.apple.securityd"],
    "com.apple.cloudd": ["com.apple.networkd", "com.apple.securityd"],
    "com.apple.photoanalysisd": ["com.apple.WindowServer"],
    "com.apple.spotlightd": ["com.apple.fskd"]
  },
  "startup_groups": {
    "phase1_critical": ["com.apple.launchd", "com.apple.kernel", "com.apple.securityd"],
    "phase2_core": ["com.apple.WindowServer", "com.apple.networkd", "com.apple.fskd"],
    "phase3_network": ["com.apple.mDNSResponder", "com.apple.wifi"],
    "phase4_ui": ["com.apple.Dock", "com.apple.Finder"],
    "phase5_background": ["com.apple.bird", "com.apple.cloudd", "com.apple.photoanalysisd", "com.apple.spotlightd"]
  }
}
EOF
    
    log_message "Service dependency map created at $DEPENDENCY_MAP_FILE"
}

# Generate resource limit configuration
generate_resource_limits() {
    log_message "Generating service resource limits..."
    
    cat > "$RESOURCE_LIMITS_FILE" << 'EOF'
# Service Resource Limits Configuration
# Format: service_name:cpu_percent:memory_mb

EOF
    
    for service in "${!RESOURCE_LIMITS[@]}"; do
        local limits="${RESOURCE_LIMITS[$service]}"
        local cpu_limit="${limits%,*}"
        local memory_limit="${limits#*,}"
        
        echo "${service}:${cpu_limit}:${memory_limit}" >> "$RESOURCE_LIMITS_FILE"
        log_message "Set resource limits for $service: CPU ${cpu_limit}%, Memory ${memory_limit}MB"
    done
    
    log_message "Resource limits configuration saved to $RESOURCE_LIMITS_FILE"
}

# Apply CPU throttling to a service
apply_cpu_throttling() {
    local service_name="$1"
    local cpu_limit="$2"
    
    # Find the service PID
    local pid=$(launchctl list | grep "$service_name" | awk '{print $1}' | grep -v '^-$' | head -1)
    
    if [[ -n "$pid" && "$pid" != "-" ]]; then
        log_message "Applying CPU throttling to $service_name (PID: $pid) at ${cpu_limit}%"
        
        # Use cpulimit if available, otherwise use nice/renice
        if command -v cpulimit &> /dev/null; then
            cpulimit --pid="$pid" --limit="$cpu_limit" --background &
            echo $! > "/tmp/cpulimit_${service_name}.pid"
        else
            # Fallback to process priority adjustment
            local nice_value=$((20 - (cpu_limit / 5)))  # Convert CPU% to nice value
            renice "$nice_value" "$pid" 2>/dev/null || true
        fi
        
        log_message "CPU throttling applied to $service_name"
    else
        log_message "Could not find PID for service $service_name"
    fi
}

# Apply memory limits to a service
apply_memory_limits() {
    local service_name="$1"
    local memory_limit_mb="$2"
    
    # Find the service PID
    local pid=$(launchctl list | grep "$service_name" | awk '{print $1}' | grep -v '^-$' | head -1)
    
    if [[ -n "$pid" && "$pid" != "-" ]]; then
        log_message "Monitoring memory usage for $service_name (PID: $pid) limit: ${memory_limit_mb}MB"
        
        # Create a memory monitor script for this service
        cat > "/tmp/memory_monitor_${service_name}.sh" << EOL
#!/bin/bash
while true; do
    if ps -p $pid > /dev/null 2>&1; then
        memory_kb=\$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
        if [[ -n "\$memory_kb" ]]; then
            memory_mb=\$((memory_kb / 1024))
            if [[ \$memory_mb -gt $memory_limit_mb ]]; then
                echo "[$(date)] WARNING: $service_name using \${memory_mb}MB > ${memory_limit_mb}MB limit" >> "$LOG_FILE"
                # Could implement memory pressure relief here if needed
            fi
        fi
    else
        break
    fi
    sleep 30
done
EOL
        
        chmod +x "/tmp/memory_monitor_${service_name}.sh"
        "/tmp/memory_monitor_${service_name}.sh" &
        echo $! > "/tmp/memory_monitor_${service_name}.pid"
        
        log_message "Memory monitoring started for $service_name"
    else
        log_message "Could not find PID for service $service_name"
    fi
}

# Apply all resource limits
apply_resource_limits() {
    log_message "Applying resource limits to services..."
    
    for service in "${!RESOURCE_LIMITS[@]}"; do
        local limits="${RESOURCE_LIMITS[$service]}"
        local cpu_limit="${limits%,*}"
        local memory_limit="${limits#*,}"
        
        apply_cpu_throttling "$service" "$cpu_limit"
        apply_memory_limits "$service" "$memory_limit"
    done
    
    log_message "Resource limits application completed"
}

# Optimize service startup order
optimize_startup_order() {
    log_message "Analyzing and optimizing service startup order..."
    
    # Create optimized LaunchDaemon configuration
    local optimization_dir="/tmp/optimized_launch_config"
    mkdir -p "$optimization_dir"
    
    # Generate startup delay configuration based on priorities
    cat > "${optimization_dir}/startup_delays.conf" << 'EOF'
# Optimized service startup delays (seconds)
# Lower priority services start later to reduce boot load

EOF
    
    for service in "${!SERVICE_PRIORITIES[@]}"; do
        local priority="${SERVICE_PRIORITIES[$service]}"
        local delay=$((100 - priority))  # Higher priority = lower delay
        
        if [[ $delay -gt 0 ]]; then
            echo "${service}:${delay}" >> "${optimization_dir}/startup_delays.conf"
            log_message "Set startup delay for $service: ${delay}s (priority: $priority)"
        fi
    done
    
    log_message "Startup order optimization configuration created"
}

# Monitor service performance after optimization
monitor_optimization_results() {
    log_message "Monitoring optimization results..."
    
    local start_time=$(date +%s)
    local monitoring_duration=300  # 5 minutes
    
    {
        echo "=== SERVICE OPTIMIZATION RESULTS ==="
        echo "Monitoring started: $(date)"
        echo "Duration: ${monitoring_duration} seconds"
        echo ""
        
        echo "=== INITIAL SYSTEM STATE ==="
        echo "Load Average: $(uptime | awk -F'load averages: ' '{print $2}')"
        echo "Memory Usage: $(vm_stat | grep 'Pages free' | awk '{print $3 * 16384 / 1024 / 1024 " MB"}')"
        echo "Failed Services: $(launchctl list | grep -c '\-[0-9]')"
        echo ""
        
        # Monitor for the specified duration
        local end_time=$((start_time + monitoring_duration))
        while [[ $(date +%s) -lt $end_time ]]; do
            local current_load=$(uptime | awk -F'load averages: ' '{print $2}' | awk '{print $1}')
            local failed_services=$(launchctl list | grep -c '\-[0-9]')
            
            echo "[$(date '+%H:%M:%S')] Load: $current_load, Failed Services: $failed_services"
            sleep 30
        done
        
        echo ""
        echo "=== FINAL SYSTEM STATE ==="
        echo "Load Average: $(uptime | awk -F'load averages: ' '{print $2}')"
        echo "Memory Usage: $(vm_stat | grep 'Pages free' | awk '{print $3 * 16384 / 1024 / 1024 " MB"}')"
        echo "Failed Services: $(launchctl list | grep -c '\-[0-9]')"
        echo ""
        echo "Monitoring completed: $(date)"
        
    } > "$OPTIMIZATION_RESULTS"
    
    log_message "Optimization monitoring completed. Results saved to $OPTIMIZATION_RESULTS"
}

# Clean up optimization processes
cleanup_optimization() {
    log_message "Cleaning up optimization processes..."
    
    # Stop CPU limiting processes
    for pidfile in /tmp/cpulimit_*.pid; do
        if [[ -f "$pidfile" ]]; then
            local pid=$(cat "$pidfile")
            kill "$pid" 2>/dev/null || true
            rm -f "$pidfile"
        fi
    done
    
    # Stop memory monitoring processes
    for pidfile in /tmp/memory_monitor_*.pid; do
        if [[ -f "$pidfile" ]]; then
            local pid=$(cat "$pidfile")
            kill "$pid" 2>/dev/null || true
            rm -f "$pidfile"
        fi
    done
    
    # Remove temporary monitoring scripts
    rm -f /tmp/memory_monitor_*.sh
    
    log_message "Cleanup completed"
}

# Generate comprehensive optimization report
generate_optimization_report() {
    log_message "Generating comprehensive optimization report..."
    
    local report_file="/tmp/service_optimization_report.txt"
    
    {
        echo "=== COMPREHENSIVE SERVICE OPTIMIZATION REPORT ==="
        echo "Generated: $(date)"
        echo "Hardware: $(system_profiler SPHardwareDataType | grep 'Model Name' | cut -d: -f2 | xargs)"
        echo "macOS Version: $(sw_vers -productVersion)"
        echo ""
        
        echo "=== OPTIMIZATION STRATEGIES APPLIED ==="
        echo "1. Service Priority Classification"
        echo "2. Resource Limits (CPU/Memory)"
        echo "3. Startup Order Optimization"
        echo "4. Dependency Management"
        echo ""
        
        echo "=== CURRENT SYSTEM STATUS ==="
        echo "Load Average: $(uptime | awk -F'load averages: ' '{print $2}')"
        echo "Total Services: $(launchctl list | wc -l | tr -d ' ')"
        echo "Failed Services: $(launchctl list | grep -c '\-[0-9]')"
        echo "Running Services: $(launchctl list | grep -v '\-' | grep -c '[0-9]')"
        echo ""
        
        echo "=== RESOURCE-LIMITED SERVICES ==="
        for service in "${!RESOURCE_LIMITS[@]}"; do
            local limits="${RESOURCE_LIMITS[$service]}"
            local cpu_limit="${limits%,*}"
            local memory_limit="${limits#*,}"
            echo "  $service: CPU ${cpu_limit}%, Memory ${memory_limit}MB"
        done
        echo ""
        
        echo "=== TOP FAILED SERVICES ==="
        launchctl list | grep '\-[0-9]' | head -10 | while read -r line; do
            echo "  $line"
        done
        echo ""
        
        echo "=== RECOMMENDATIONS ==="
        echo "1. Monitor services in cooldown period before restart"
        echo "2. Review failed services for manual intervention"
        echo "3. Consider disabling non-essential background services"
        echo "4. Implement regular cache clearing schedule"
        echo "5. Monitor system load trends over time"
        echo ""
        
        if [[ -f "$OPTIMIZATION_RESULTS" ]]; then
            echo "=== OPTIMIZATION MONITORING RESULTS ==="
            cat "$OPTIMIZATION_RESULTS"
        fi
        
    } > "$report_file"
    
    log_message "Comprehensive optimization report generated at $report_file"
    cat "$report_file"
}

# Main optimization workflow
main() {
    local action="${1:-optimize}"
    
    case "$action" in
        "optimize")
            log_message "Starting comprehensive service optimization..."
            create_dependency_map
            generate_resource_limits
            optimize_startup_order
            apply_resource_limits
            monitor_optimization_results &
            sleep 2
            generate_optimization_report
            ;;
        "apply-limits")
            log_message "Applying resource limits only..."
            apply_resource_limits
            ;;
        "monitor")
            log_message "Starting optimization monitoring..."
            monitor_optimization_results
            ;;
        "report")
            log_message "Generating optimization report..."
            generate_optimization_report
            ;;
        "cleanup")
            log_message "Cleaning up optimization processes..."
            cleanup_optimization
            ;;
        *)
            echo "Usage: $0 [optimize|apply-limits|monitor|report|cleanup]"
            echo "  optimize     - Run full optimization (default)"
            echo "  apply-limits - Apply resource limits only"
            echo "  monitor      - Monitor optimization results"
            echo "  report       - Generate optimization report"
            echo "  cleanup      - Clean up optimization processes"
            exit 1
            ;;
    esac
}

# Cleanup on exit
trap cleanup_optimization EXIT

# Run main function
main "$@"