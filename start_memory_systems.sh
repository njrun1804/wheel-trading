#!/bin/bash
# Memory Management System Deployment Script
# Starts all memory monitoring and optimization components

set -e

echo "=========================================="
echo "Memory Management System Deployment"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check for required Python packages
    python3 -c "import psutil" 2>/dev/null || {
        log_warning "psutil not installed, installing..."
        pip3 install psutil
    }
    
    log_success "Dependencies verified"
}

# Create necessary directories
setup_directories() {
    log "Setting up directories..."
    
    mkdir -p logs
    mkdir -p logs/reports
    mkdir -p pids
    
    log_success "Directories created"
}

# Install psutil if needed
install_psutil() {
    if ! python3 -c "import psutil" &> /dev/null; then
        log "Installing psutil..."
        pip3 install psutil || {
            log_error "Failed to install psutil"
            exit 1
        }
        log_success "psutil installed"
    fi
}

# Start memory monitor daemon
start_memory_monitor() {
    log "Starting Memory Monitor Daemon..."
    
    if [ -f "pids/memory_monitor.pid" ]; then
        local pid=$(cat pids/memory_monitor.pid)
        if ps -p $pid > /dev/null 2>&1; then
            log_warning "Memory Monitor already running (PID: $pid)"
            return 0
        fi
    fi
    
    python3 memory_monitor_daemon.py &
    local monitor_pid=$!
    echo $monitor_pid > pids/memory_monitor.pid
    
    sleep 2
    if ps -p $monitor_pid > /dev/null 2>&1; then
        log_success "Memory Monitor started (PID: $monitor_pid)"
    else
        log_error "Failed to start Memory Monitor"
        return 1
    fi
}

# Start process manager
start_process_manager() {
    log "Starting Process Manager..."
    
    if [ -f "pids/process_manager.pid" ]; then
        local pid=$(cat pids/process_manager.pid)
        if ps -p $pid > /dev/null 2>&1; then
            log_warning "Process Manager already running (PID: $pid)"
            return 0
        fi
    fi
    
    python3 process_manager.py --daemon &
    local manager_pid=$!
    echo $manager_pid > pids/process_manager.pid
    
    sleep 2
    if ps -p $manager_pid > /dev/null 2>&1; then
        log_success "Process Manager started (PID: $manager_pid)"
    else
        log_error "Failed to start Process Manager"
        return 1
    fi
}

# Optimize system services
optimize_services() {
    log "Running initial system optimization..."
    
    python3 system_service_optimizer.py --optimize
    
    if [ $? -eq 0 ]; then
        log_success "System optimization completed"
    else
        log_warning "System optimization had issues"
    fi
}

# Create monitoring cron job
setup_cron() {
    log "Setting up automated monitoring..."
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "memory_cleanup_emergency.py"; then
        log_warning "Cron job already exists"
        return 0
    fi
    
    # Add cron job for emergency cleanup (every 5 minutes)
    (crontab -l 2>/dev/null; echo "*/5 * * * * cd $(pwd) && python3 memory_cleanup_emergency.py --status > /dev/null 2>&1") | crontab -
    
    # Add cron job for system optimization (every hour)
    (crontab -l 2>/dev/null; echo "0 * * * * cd $(pwd) && python3 system_service_optimizer.py --optimize > /dev/null 2>&1") | crontab -
    
    log_success "Automated monitoring scheduled"
}

# Perform immediate emergency cleanup if needed
emergency_check() {
    log "Performing emergency memory check..."
    
    # Get current memory status
    local available_mb=$(python3 -c "
import psutil
vm = psutil.virtual_memory()
print(int(vm.available / 1024 / 1024))
")
    
    if [ "$available_mb" -lt 1000 ]; then
        log_warning "Low memory detected ($available_mb MB), running emergency cleanup..."
        python3 memory_cleanup_emergency.py
    else
        log_success "Memory status OK ($available_mb MB available)"
    fi
}

# Verify all systems are running
verify_systems() {
    log "Verifying system status..."
    
    local all_good=true
    
    # Check memory monitor
    if [ -f "pids/memory_monitor.pid" ]; then
        local pid=$(cat pids/memory_monitor.pid)
        if ps -p $pid > /dev/null 2>&1; then
            log_success "Memory Monitor running (PID: $pid)"
        else
            log_error "Memory Monitor not running"
            all_good=false
        fi
    else
        log_error "Memory Monitor PID file not found"
        all_good=false
    fi
    
    # Check process manager
    if [ -f "pids/process_manager.pid" ]; then
        local pid=$(cat pids/process_manager.pid)
        if ps -p $pid > /dev/null 2>&1; then
            log_success "Process Manager running (PID: $pid)"
        else
            log_error "Process Manager not running"
            all_good=false
        fi
    else
        log_error "Process Manager PID file not found"
        all_good=false
    fi
    
    if [ "$all_good" = true ]; then
        log_success "All systems operational"
        return 0
    else
        log_error "Some systems are not running properly"
        return 1
    fi
}

# Generate status report
generate_status_report() {
    log "Generating status report..."
    
    python3 -c "
import json
from datetime import datetime
import psutil

# Get memory status
vm = psutil.virtual_memory()
swap = psutil.swap_memory()

status = {
    'timestamp': datetime.now().isoformat(),
    'memory': {
        'total_gb': vm.total / 1024 / 1024 / 1024,
        'available_mb': vm.available / 1024 / 1024,
        'used_percent': vm.percent,
        'status': 'critical' if vm.available < 500*1024*1024 else 'warning' if vm.available < 1024*1024*1024 else 'optimal'
    },
    'swap': {
        'total_gb': swap.total / 1024 / 1024 / 1024,
        'used_percent': swap.percent
    },
    'deployment_status': 'success'
}

print(json.dumps(status, indent=2))
" > logs/deployment_status.json
    
    log_success "Status report saved to logs/deployment_status.json"
}

# Main execution
main() {
    log "Starting Memory Management System deployment..."
    
    # Run all setup steps
    check_dependencies
    install_psutil
    setup_directories
    emergency_check
    
    # Start services
    start_memory_monitor
    start_process_manager
    
    # Optimize system
    optimize_services
    
    # Setup automation
    setup_cron
    
    # Verify everything is working
    verify_systems
    
    # Generate status report
    generate_status_report
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Memory Management System Deployed${NC}"
    echo "=========================================="
    echo ""
    echo "Active Components:"
    echo "  • Memory Monitor Daemon"
    echo "  • Process Manager"
    echo "  • System Service Optimizer"
    echo "  • Emergency Cleanup (automated)"
    echo ""
    echo "Log Files:"
    echo "  • logs/memory_monitor.log"
    echo "  • logs/process_manager.log"
    echo "  • logs/service_optimization.log"
    echo ""
    echo "Control Commands:"
    echo "  • ./memory_status.sh        - Check status"
    echo "  • ./stop_memory_systems.sh  - Stop all systems"
    echo "  • python3 memory_cleanup_emergency.py --cleanup  - Emergency cleanup"
    echo ""
    echo "System is now protected against memory crises!"
}

# Handle command line arguments
case "${1:-}" in
    --status)
        verify_systems
        ;;
    --emergency)
        emergency_check
        ;;
    --stop)
        exec ./stop_memory_systems.sh
        ;;
    *)
        main
        ;;
esac