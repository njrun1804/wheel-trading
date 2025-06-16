#!/bin/bash
# Core 5 Service Monitor
# Continuous monitoring and optimization for trading system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/service_monitor.log"
PID_FILE="$SCRIPT_DIR/service_monitor.pid"

# Trading focus mode - prioritize trading processes
TRADING_FOCUS=false

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_alert() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ALERT:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

# Check if already running
check_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Service monitor already running (PID: $pid)"
            exit 1
        else
            rm -f "$PID_FILE"
        fi
    fi
}

# Create PID file
create_pid_file() {
    echo $$ > "$PID_FILE"
}

# Cleanup on exit
cleanup() {
    rm -f "$PID_FILE"
    log "Service monitor stopped"
}

trap cleanup EXIT

# Get system metrics
get_metrics() {
    local processes=$(pgrep -f "." | wc -l)
    local load_avg=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}')
    local memory_pressure=$(memory_pressure 2>/dev/null | grep "free percentage" | awk '{print $5}' | tr -d '%' || echo "50")
    
    echo "$processes,$load_avg,$memory_pressure"
}

# Check for trading processes
check_trading_processes() {
    local trading_patterns=(
        "python.*run.py"
        "python.*advisor"
        "python.*einstein"
        "duckdb"
        "claude"
    )
    
    local trading_active=0
    for pattern in "${trading_patterns[@]}"; do
        if pgrep -f "$pattern" >/dev/null 2>&1; then
            ((trading_active++))
        fi
    done
    
    echo $trading_active
}

# Monitor and alert
monitor_system() {
    local alert_count=0
    local optimization_count=0
    
    while true; do
        local metrics=$(get_metrics)
        local processes=$(echo "$metrics" | cut -d',' -f1)
        local load=$(echo "$metrics" | cut -d',' -f2)
        local memory_free=$(echo "$metrics" | cut -d',' -f3)
        local trading_active=$(check_trading_processes)
        
        # Process count alerts
        if [[ $processes -gt 700 ]]; then
            log_alert "High process count: $processes (>700)"
            "$SCRIPT_DIR/optimized_service_manager.sh" optimize &
            ((alert_count++))
        elif [[ $processes -gt 650 ]]; then
            log_warning "Elevated process count: $processes"
        fi
        
        # Memory alerts
        if [[ $memory_free -lt 15 ]]; then
            log_alert "Critical memory: ${memory_free}% free"
            sudo purge 2>/dev/null || true
            ((alert_count++))
        elif [[ $memory_free -lt 25 ]]; then
            log_warning "Low memory: ${memory_free}% free"
        fi
        
        # Load alerts  
        if [[ $(echo "$load > 10.0" | bc 2>/dev/null || echo 0) -eq 1 ]]; then
            log_alert "High system load: $load"
            ((alert_count++))
        elif [[ $(echo "$load > 8.0" | bc 2>/dev/null || echo 0) -eq 1 ]]; then
            log_warning "Elevated system load: $load"
        fi
        
        ((optimization_count++))
        
        # Log periodic status
        if [[ $((optimization_count % 20)) -eq 0 ]]; then  # Every 10 minutes
            log "Status - Processes: $processes, Load: $load, Memory free: ${memory_free}%, Trading: $trading_active active"
        fi
        
        # Auto-optimize every 30 minutes
        if [[ $((optimization_count % 60)) -eq 0 ]]; then
            log "Running scheduled optimization..."
            "$SCRIPT_DIR/optimized_service_manager.sh" optimize &
        fi
        
        sleep 30
    done
}

# Main execution
main() {
    case "${1:-monitor}" in
        "monitor")
            log "🚀 Starting Core 5 Service Monitor"
            check_running
            create_pid_file
            monitor_system
            ;;
        "--trading-focus")
            TRADING_FOCUS=true
            log "🎯 Trading focus mode enabled"
            check_running
            create_pid_file
            monitor_system
            ;;
        "status")
            local metrics=$(get_metrics)
            echo "System Status:"
            echo "  Processes: $(echo "$metrics" | cut -d',' -f1)"
            echo "  Load: $(echo "$metrics" | cut -d',' -f2)"
            echo "  Memory Free: $(echo "$metrics" | cut -d',' -f3)%"
            echo "  Trading Processes: $(check_trading_processes)"
            ;;
        "stop")
            if [[ -f "$PID_FILE" ]]; then
                local pid=$(cat "$PID_FILE")
                if kill "$pid" 2>/dev/null; then
                    log "Service monitor stopped (PID: $pid)"
                    rm -f "$PID_FILE"
                else
                    log "No running monitor found"
                fi
            else
                log "No PID file found"
            fi
            ;;
        "help"|*)
            echo "Core 5 Service Monitor"
            echo "Usage: $0 [monitor|--trading-focus|status|report|stop|help]"
            echo ""
            echo "Commands:"
            echo "  monitor         - Start monitoring (default)"
            echo "  --trading-focus - Start with trading optimization focus"
            echo "  status          - Show current system status"
            echo "  stop            - Stop running monitor"
            echo "  help            - Show this help message"
            ;;
    esac
}

main "$@"

