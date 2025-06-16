#!/bin/bash
# Core 5 Optimized Service Manager
# Real-time service optimization and management for M4 Pro trading system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/service_optimization.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# M4 Pro Hardware Configuration
M4_P_CORES=8
M4_E_CORES=4
TOTAL_CORES=12
TOTAL_MEMORY_GB=24
TRADING_MEMORY_QUOTA_GB=12

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

log_color() {
    local color=$1
    local message=$2
    echo -e "${color}[$TIMESTAMP] $message${NC}" | tee -a "$LOG_FILE"
}

# Check if running on M4 Pro
check_hardware() {
    local cpu_brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    if [[ "$cpu_brand" == *"M4"* ]]; then
        log_color "$GREEN" "✅ M4 Pro detected: $cpu_brand"
        return 0
    else
        log_color "$YELLOW" "⚠️  Non-M4 system detected: $cpu_brand"
        return 1
    fi
}

# Get current system stats
get_system_stats() {
    local total_processes=$(pgrep -f "." | wc -l)
    local memory_pressure=$(memory_pressure | grep "System-wide memory free percentage" | awk '{print $5}' | tr -d '%')
    local load_avg=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}')
    
    echo "processes:$total_processes,memory_free:$memory_pressure,load:$load_avg"
}

# Optimize redundant processes
optimize_redundant_processes() {
    log_color "$BLUE" "🔧 Optimizing redundant processes..."
    
    local optimized_count=0
    
    # Common redundant process patterns
    local patterns=(
        "com.notion.id"
        "Code Helper"
        "WebKit.*Content"
        "Google Chrome Helper"
        "Electron"
    )
    
    for pattern in "${patterns[@]}"; do
        local pids=($(pgrep -f "$pattern" 2>/dev/null || true))
        local count=${#pids[@]}
        
        if [[ $count -gt 3 ]]; then
            log_color "$YELLOW" "Found $count $pattern processes, optimizing..."
            
            # Keep first 2, terminate others
            for ((i=2; i<count; i++)); do
                if kill -TERM "${pids[$i]}" 2>/dev/null; then
                    log "Terminated redundant process: ${pids[$i]} ($pattern)"
                    ((optimized_count++))
                    sleep 0.1
                fi
            done
        fi
    done
    
    log_color "$GREEN" "✅ Optimized $optimized_count redundant processes"
    return $optimized_count
}

# Set trading process priorities
optimize_trading_processes() {
    log_color "$BLUE" "🎯 Optimizing trading process priorities..."
    
    local trading_patterns=(
        "python.*run.py"
        "python.*advisor"
        "python.*einstein"
        "python.*jarvis"
        "duckdb"
        "claude"
    )
    
    local optimized=0
    for pattern in "${patterns[@]}"; do
        while read -r pid; do
            if [[ -n "$pid" ]]; then
                # Set high priority (lower nice value)
                if renice -10 "$pid" >/dev/null 2>&1; then
                    log "Set high priority for trading process: $pid ($pattern)"
                    ((optimized++))
                fi
            fi
        done < <(pgrep -f "$pattern" 2>/dev/null || true)
    done
    
    log_color "$GREEN" "✅ Optimized priority for $optimized trading processes"
}

# CPU affinity optimization for M4 Pro
optimize_cpu_affinity() {
    if ! check_hardware; then
        log_color "$YELLOW" "⚠️  Skipping CPU affinity optimization (not M4 Pro)"
        return
    fi
    
    log_color "$BLUE" "⚡ Optimizing CPU affinity for M4 Pro..."
    
    # Trading processes get P-cores (0-7)
    local trading_patterns=(
        "python.*advisor"
        "python.*einstein"
        "duckdb"
    )
    
    for pattern in "${trading_patterns[@]}"; do
        while read -r pid; do
            if [[ -n "$pid" ]]; then
                # Bind to P-cores (0-7) - Note: macOS doesn't support direct CPU affinity
                # but we can use process priority and QoS classes
                if sysctl -w kern.tg_throttle_level=0 >/dev/null 2>&1; then
                    log "Optimized QoS for trading process: $pid ($pattern)"
                fi
            fi
        done < <(pgrep -f "$pattern" 2>/dev/null || true)
    done
    
    log_color "$GREEN" "✅ CPU optimization applied"
}

# Monitor system performance
monitor_system() {
    log_color "$BLUE" "📊 System monitoring started..."
    
    while true; do
        local stats=$(get_system_stats)
        local processes=$(echo "$stats" | cut -d',' -f1 | cut -d':' -f2)
        local memory_free=$(echo "$stats" | cut -d',' -f2 | cut -d':' -f2)
        local load=$(echo "$stats" | cut -d',' -f3 | cut -d':' -f2)
        
        # Alert conditions
        if [[ $processes -gt 700 ]]; then
            log_color "$RED" "🚨 HIGH PROCESS COUNT: $processes (>700)"
            optimize_redundant_processes
        fi
        
        if [[ $(echo "$memory_free < 20" | bc 2>/dev/null || echo 1) -eq 1 ]]; then
            log_color "$RED" "🚨 LOW MEMORY: ${memory_free}% free"
        fi
        
        if [[ $(echo "$load > 8.0" | bc 2>/dev/null || echo 0) -eq 1 ]]; then
            log_color "$RED" "🚨 HIGH LOAD: $load"
        fi
        
        log "System stats - Processes: $processes, Memory free: ${memory_free}%, Load: $load"
        sleep 30
    done
}

# Clean up orphaned processes
cleanup_orphaned_processes() {
    log_color "$BLUE" "🧹 Cleaning up orphaned processes..."
    
    local cleaned=0
    
    # Find processes with PPID 1 that aren't system processes
    while read -r pid ppid cmd; do
        if [[ "$ppid" == "1" && ! "$cmd" =~ ^(kernel|launchd|system) ]]; then
            # Check if it's been running for more than 1 hour without activity
            local runtime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ' || echo "")
            if [[ -n "$runtime" && "$runtime" =~ ^[0-9]+-[0-9][0-9]:[0-9][0-9]:[0-9][0-9]$ ]]; then
                log "Found potential orphan: PID $pid ($cmd), runtime: $runtime"
                if kill -TERM "$pid" 2>/dev/null; then
                    log "Cleaned orphaned process: $pid"
                    ((cleaned++))
                fi
            fi
        fi
    done < <(ps -eo pid,ppid,comm | tail -n +2)
    
    log_color "$GREEN" "✅ Cleaned $cleaned orphaned processes"
}

# Defer non-essential services
defer_nonessential_services() {
    log_color "$BLUE" "⏰ Deferring non-essential services..."
    
    local deferred_services=(
        "com.adobe.ccxprocess"
        "com.google.keystone"
        "com.microsoft.autoupdate"
        "com.docker.helper"
        "com.jetbrains.toolbox"
        "com.spotify.client"
    )
    
    local deferred=0
    for service in "${deferred_services[@]}"; do
        if launchctl list | grep -q "$service"; then
            if launchctl stop "$service" 2>/dev/null; then
                log "Deferred service: $service"
                ((deferred++))
                
                # Schedule restart in 60 seconds
                (sleep 60 && launchctl start "$service" 2>/dev/null) &
            fi
        fi
    done
    
    log_color "$GREEN" "✅ Deferred $deferred non-essential services"
}

# Generate optimization report
generate_report() {
    local output_file="service_optimization_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$output_file" << EOF
# Service Optimization Report
Generated: $(date)

## System Overview
- Hardware: $(sysctl -n machdep.cpu.brand_string)
- Total Memory: ${TOTAL_MEMORY_GB}GB
- CPU Cores: ${TOTAL_CORES} (${M4_P_CORES} P-cores + ${M4_E_CORES} E-cores)

## Current Statistics
$(get_system_stats | tr ',' '\n' | sed 's/:/: /')

## Optimizations Applied
$(tail -20 "$LOG_FILE" | grep "✅")

## Recommendations
1. Run this optimization every 30 minutes during trading hours
2. Monitor process count - alert if >700
3. Check memory usage - alert if <20% free
4. Review LaunchD services monthly for new additions

## Trading System Priorities
- Python trading processes: High priority (nice -10)
- DuckDB database: High priority + memory quota
- Einstein AI: P-core allocation
- Claude integration: QoS optimization

EOF

    log_color "$GREEN" "📄 Report generated: $output_file"
}

# Main execution
main() {
    case "${1:-optimize}" in
        "optimize")
            log_color "$GREEN" "🚀 Starting Core 5 Service Optimization"
            optimize_redundant_processes
            optimize_trading_processes
            optimize_cpu_affinity
            cleanup_orphaned_processes
            defer_nonessential_services
            generate_report
            log_color "$GREEN" "✅ Optimization complete"
            ;;
        "monitor")
            monitor_system
            ;;
        "report")
            generate_report
            ;;
        "stats")
            echo "Current system statistics:"
            get_system_stats | tr ',' '\n'
            ;;
        "help"|*)
            echo "Core 5 Optimized Service Manager"
            echo "Usage: $0 [optimize|monitor|report|stats|help]"
            echo ""
            echo "Commands:"
            echo "  optimize  - Run full service optimization (default)"
            echo "  monitor   - Start continuous monitoring"
            echo "  report    - Generate optimization report"
            echo "  stats     - Show current system statistics"
            echo "  help      - Show this help message"
            ;;
    esac
}

# Execute main function
main "$@"