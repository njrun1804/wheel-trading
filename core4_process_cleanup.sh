#!/bin/bash
# Core 4 Process Cleanup Script
# Advanced process cleanup and management utilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/core4_cleanup.log"
CONFIG_FILE="${SCRIPT_DIR}/core4_config.json"
LOCK_FILE="/tmp/core4_cleanup.lock"

# Thresholds
CPU_THRESHOLD=80
MEMORY_THRESHOLD=85
PROCESS_MEMORY_THRESHOLD=1024  # MB
MIN_PROCESS_AGE=300  # 5 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

# Check if another instance is running
check_lock() {
    if [[ -f "${LOCK_FILE}" ]]; then
        local pid=$(cat "${LOCK_FILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            log "WARN" "Another instance is running (PID: ${pid})"
            exit 1
        else
            log "INFO" "Removing stale lock file"
            rm -f "${LOCK_FILE}"
        fi
    fi
    echo $$ > "${LOCK_FILE}"
    trap 'rm -f "${LOCK_FILE}"; exit' INT TERM EXIT
}

# Get system resource usage
get_system_resources() {
    local cpu_usage
    local memory_usage
    local load_avg
    
    # Get CPU usage (5 second average)
    cpu_usage=$(top -l 2 -n 0 -F | tail -1 | awk '{print $3}' | sed 's/%//')
    
    # Get memory usage
    memory_usage=$(vm_stat | awk '
        /Pages free/ { free = $3 }
        /Pages active/ { active = $3 }
        /Pages inactive/ { inactive = $3 }
        /Pages wired/ { wired = $3 }
        /Pages speculative/ { spec = $3 }
        END {
            gsub(/\./, "", free); gsub(/\./, "", active); 
            gsub(/\./, "", inactive); gsub(/\./, "", wired); gsub(/\./, "", spec);
            total = free + active + inactive + wired + spec;
            used = active + inactive + wired;
            if (total > 0) print int((used/total) * 100);
            else print 0;
        }')
    
    # Get load average
    load_avg=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}')
    
    echo "${cpu_usage:-0} ${memory_usage:-0} ${load_avg:-0}"
}

# Get top processes by CPU
get_top_cpu_processes() {
    local limit=${1:-10}
    /bin/ps -ax -o pid,ppid,pcpu,pmem,comm | sort -k3 -nr | head -n $((limit + 1)) | tail -n +2
}

# Get top processes by memory
get_top_memory_processes() {
    local limit=${1:-10}
    /bin/ps -ax -o pid,ppid,pcpu,pmem,comm | sort -k4 -nr | head -n $((limit + 1)) | tail -n +2
}

# Check if process is critical/protected
is_protected_process() {
    local pid=$1
    local name=$2
    
    # Critical system processes
    local protected_names=(
        "kernel_task" "launchd" "WindowServer" "loginwindow"
        "SystemUIServer" "Dock" "Finder" "ssh" "sshd"
        "Activity Monitor" "claude" "python" "bash" "zsh"
        "Terminal" "WezTerm" "iTerm"
    )
    
    # Check by name
    for protected in "${protected_names[@]}"; do
        if [[ "${name,,}" == *"${protected,,}"* ]]; then
            return 0  # Protected
        fi
    done
    
    # Check if it's our parent process
    if [[ $pid -eq $PPID ]] || [[ $pid -eq $$ ]]; then
        return 0  # Protected
    fi
    
    # Check if it's a system process (low PID)
    if [[ $pid -lt 100 ]]; then
        return 0  # Protected
    fi
    
    return 1  # Not protected
}

# Get process info
get_process_info() {
    local pid=$1
    
    if ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi
    
    local info
    info=$(/bin/ps -p "$pid" -o pid,ppid,pcpu,pmem,vsz,rss,etime,comm 2>/dev/null | tail -n +2)
    
    if [[ -z "$info" ]]; then
        return 1
    fi
    
    echo "$info"
}

# Kill process safely
kill_process_safe() {
    local pid=$1
    local name=$2
    local force=${3:-false}
    
    if is_protected_process "$pid" "$name"; then
        log "WARN" "Refusing to kill protected process: $pid ($name)"
        return 1
    fi
    
    log "INFO" "Attempting to terminate process: $pid ($name)"
    
    # Try graceful termination first
    if kill -TERM "$pid" 2>/dev/null; then
        sleep 3
        
        # Check if process is still running
        if kill -0 "$pid" 2>/dev/null; then
            if [[ "$force" == "true" ]]; then
                log "WARN" "Force killing process: $pid ($name)"
                kill -KILL "$pid" 2>/dev/null || true
                sleep 1
            else
                log "WARN" "Process $pid did not terminate gracefully"
                return 1
            fi
        fi
        
        # Verify termination
        if ! kill -0 "$pid" 2>/dev/null; then
            log "INFO" "Successfully terminated process: $pid ($name)"
            return 0
        else
            log "ERROR" "Failed to terminate process: $pid ($name)"
            return 1
        fi
    else
        log "ERROR" "Could not send signal to process: $pid ($name)"
        return 1
    fi
}

# Find and clean zombie processes
cleanup_zombies() {
    log "INFO" "Searching for zombie processes..."
    
    local zombies
    zombies=$(/bin/ps -ax -o pid,ppid,stat,comm | awk '$3 ~ /Z/ {print $1, $2, $4}')
    
    if [[ -z "$zombies" ]]; then
        log "INFO" "No zombie processes found"
        return 0
    fi
    
    local zombie_count=0
    while IFS=' ' read -r pid ppid name; do
        [[ -z "$pid" ]] && continue
        
        log "WARN" "Found zombie process: $pid (parent: $ppid, name: $name)"
        
        # Try to terminate the parent process
        if [[ $ppid -ne 1 ]] && ! is_protected_process "$ppid" ""; then
            log "INFO" "Attempting to terminate parent process: $ppid"
            if kill_process_safe "$ppid" "parent_of_zombie" false; then
                ((zombie_count++))
            fi
        fi
        
    done <<< "$zombies"
    
    log "INFO" "Cleaned up $zombie_count zombie processes"
    return $zombie_count
}

# Find stuck/runaway processes
find_stuck_processes() {
    log "INFO" "Searching for stuck/runaway processes..."
    
    local stuck_processes=()
    local current_time=$(date +%s)
    
    # Get processes with high CPU usage
    while IFS=' ' read -r pid ppid cpu mem name; do
        [[ -z "$pid" ]] || [[ "$pid" == "PID" ]] && continue
        
        # Skip if CPU usage is below threshold
        cpu_int=$(echo "$cpu" | cut -d. -f1)
        [[ $cpu_int -lt $CPU_THRESHOLD ]] && continue
        
        # Get process start time
        local start_time
        start_time=$(stat -f %B "/proc/$pid" 2>/dev/null || echo "$current_time")
        local age=$((current_time - start_time))
        
        # Skip recently started processes
        [[ $age -lt $MIN_PROCESS_AGE ]] && continue
        
        # Skip protected processes
        is_protected_process "$pid" "$name" && continue
        
        stuck_processes+=("$pid $ppid $cpu $mem $name")
        log "WARN" "Found stuck process: PID=$pid CPU=${cpu}% MEM=${mem}% NAME=$name AGE=${age}s"
        
    done < <(get_top_cpu_processes 20)
    
    printf '%s\n' "${stuck_processes[@]}"
}

# Find memory hogs
find_memory_hogs() {
    log "INFO" "Searching for memory-intensive processes..."
    
    local memory_hogs=()
    
    while IFS=' ' read -r pid ppid cpu mem name; do
        [[ -z "$pid" ]] || [[ "$pid" == "PID" ]] && continue
        
        # Calculate memory in MB (approximation)
        local mem_mb
        mem_mb=$(echo "$mem * 24 / 100" | bc -l 2>/dev/null | cut -d. -f1)
        [[ -z "$mem_mb" ]] && mem_mb=0
        
        # Skip if memory usage is below threshold
        [[ $mem_mb -lt $PROCESS_MEMORY_THRESHOLD ]] && continue
        
        # Skip protected processes
        is_protected_process "$pid" "$name" && continue
        
        memory_hogs+=("$pid $ppid $cpu $mem $name ${mem_mb}MB")
        log "WARN" "Found memory hog: PID=$pid MEM=${mem}% (~${mem_mb}MB) NAME=$name"
        
    done < <(get_top_memory_processes 20)
    
    printf '%s\n' "${memory_hogs[@]}"
}

# Automatic cleanup based on system resources
auto_cleanup() {
    local cpu_usage memory_usage load_avg
    read -r cpu_usage memory_usage load_avg < <(get_system_resources)
    
    log "INFO" "System resources: CPU=${cpu_usage}% Memory=${memory_usage}% Load=${load_avg}"
    
    local cleanup_needed=false
    local processes_killed=0
    
    # Check if cleanup is needed
    if (( $(echo "$cpu_usage > $CPU_THRESHOLD" | bc -l) )); then
        log "WARN" "High CPU usage detected: ${cpu_usage}%"
        cleanup_needed=true
    fi
    
    if (( memory_usage > MEMORY_THRESHOLD )); then
        log "WARN" "High memory usage detected: ${memory_usage}%"
        cleanup_needed=true
    fi
    
    if [[ "$cleanup_needed" == "false" ]]; then
        log "INFO" "System resources within normal limits"
        return 0
    fi
    
    # Clean up stuck processes
    local stuck_processes
    mapfile -t stuck_processes < <(find_stuck_processes)
    
    for process_info in "${stuck_processes[@]}"; do
        [[ -z "$process_info" ]] && continue
        
        local pid name
        pid=$(echo "$process_info" | awk '{print $1}')
        name=$(echo "$process_info" | awk '{print $5}')
        
        if kill_process_safe "$pid" "$name" true; then
            ((processes_killed++))
        fi
        
        # Don't kill too many processes at once
        [[ $processes_killed -ge 3 ]] && break
    done
    
    # Clean up memory hogs if memory usage is critical
    if (( memory_usage > 90 )); then
        local memory_hogs
        mapfile -t memory_hogs < <(find_memory_hogs)
        
        for process_info in "${memory_hogs[@]}"; do
            [[ -z "$process_info" ]] && continue
            
            local pid name
            pid=$(echo "$process_info" | awk '{print $1}')
            name=$(echo "$process_info" | awk '{print $5}')
            
            if kill_process_safe "$pid" "$name" true; then
                ((processes_killed++))
            fi
            
            # Limit cleanup to avoid system instability
            [[ $processes_killed -ge 5 ]] && break
        done
    fi
    
    log "INFO" "Auto cleanup completed: $processes_killed processes terminated"
    return $processes_killed
}

# Set process priorities
optimize_priorities() {
    log "INFO" "Optimizing process priorities..."
    
    local optimized=0
    
    # Lower priority for high CPU processes
    while IFS=' ' read -r pid ppid cpu mem name; do
        [[ -z "$pid" ]] || [[ "$pid" == "PID" ]] && continue
        
        local cpu_int
        cpu_int=$(echo "$cpu" | cut -d. -f1)
        [[ $cpu_int -lt 50 ]] && continue
        
        # Skip protected processes
        is_protected_process "$pid" "$name" && continue
        
        # Get current priority
        local current_nice
        current_nice=$(ps -o nice= -p "$pid" 2>/dev/null | tr -d ' ')
        [[ -z "$current_nice" ]] && continue
        
        # Lower priority if not already low
        if [[ $current_nice -lt 10 ]]; then
            if renice +5 "$pid" >/dev/null 2>&1; then
                log "INFO" "Lowered priority for high CPU process: $pid ($name) nice: $current_nice -> $((current_nice + 5))"
                ((optimized++))
            fi
        fi
        
    done < <(get_top_cpu_processes 10)
    
    # Raise priority for important processes
    local important_processes=("claude" "python" "Terminal" "WezTerm")
    
    for proc_name in "${important_processes[@]}"; do
        local pids
        pids=$(pgrep -i "$proc_name" 2>/dev/null || true)
        
        for pid in $pids; do
            [[ -z "$pid" ]] && continue
            
            local current_nice
            current_nice=$(ps -o nice= -p "$pid" 2>/dev/null | tr -d ' ')
            [[ -z "$current_nice" ]] && continue
            
            # Raise priority if not already high
            if [[ $current_nice -gt -5 ]]; then
                if renice -5 "$pid" >/dev/null 2>&1; then
                    log "INFO" "Raised priority for important process: $pid ($proc_name) nice: $current_nice -> -5"
                    ((optimized++))
                fi
            fi
        done
    done
    
    log "INFO" "Priority optimization completed: $optimized processes adjusted"
    return $optimized
}

# Generate monitoring report
generate_report() {
    local report_file="${SCRIPT_DIR}/core4_report_$(date +%Y%m%d_%H%M%S).json"
    local cpu_usage memory_usage load_avg
    read -r cpu_usage memory_usage load_avg < <(get_system_resources)
    
    # Get process counts
    local total_processes zombie_count
    total_processes=$(ps ax | wc -l)
    zombie_count=$(ps ax -o stat | grep -c Z || echo 0)
    
    # Create JSON report
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "system_resources": {
        "cpu_percent": $cpu_usage,
        "memory_percent": $memory_usage,
        "load_average": $load_avg,
        "total_processes": $total_processes,
        "zombie_processes": $zombie_count
    },
    "top_cpu_processes": [
$(get_top_cpu_processes 5 | awk '{printf "        {\"pid\": %s, \"cpu\": %s, \"memory\": %s, \"name\": \"%s\"},\n", $1, $3, $4, $5}' | sed '$s/,$//')
    ],
    "top_memory_processes": [
$(get_top_memory_processes 5 | awk '{printf "        {\"pid\": %s, \"cpu\": %s, \"memory\": %s, \"name\": \"%s\"},\n", $1, $3, $4, $5}' | sed '$s/,$//')
    ]
}
EOF
    
    echo "$report_file"
}

# Display help
show_help() {
    cat << EOF
Core 4 Process Cleanup Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -m, --monitor           Show current system status
    -c, --cleanup           Run automatic cleanup
    -z, --zombies           Clean up zombie processes only
    -s, --stuck             Find and display stuck processes
    -k, --kill PID          Kill specific process
    -p, --priorities        Optimize process priorities
    -r, --report            Generate monitoring report
    --cpu-threshold N       Set CPU threshold (default: $CPU_THRESHOLD)
    --memory-threshold N    Set memory threshold (default: $MEMORY_THRESHOLD)
    --force                 Enable force killing
    
EXAMPLES:
    $0 --monitor            # Show current system status
    $0 --cleanup            # Run automatic cleanup
    $0 --kill 12345         # Kill process with PID 12345
    $0 --report             # Generate JSON report
    
EOF
}

# Main function
main() {
    local action="monitor"
    local force_kill=false
    local target_pid=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -m|--monitor)
                action="monitor"
                shift
                ;;
            -c|--cleanup)
                action="cleanup"
                shift
                ;;
            -z|--zombies)
                action="zombies"
                shift
                ;;
            -s|--stuck)
                action="stuck"
                shift
                ;;
            -k|--kill)
                action="kill"
                target_pid="$2"
                shift 2
                ;;
            -p|--priorities)
                action="priorities"
                shift
                ;;
            -r|--report)
                action="report"
                shift
                ;;
            --cpu-threshold)
                CPU_THRESHOLD="$2"
                shift 2
                ;;
            --memory-threshold)
                MEMORY_THRESHOLD="$2"
                shift 2
                ;;
            --force)
                force_kill=true
                shift
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check for required tools
    for tool in ps kill renice bc; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log "ERROR" "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Execute requested action
    case "$action" in
        monitor)
            echo -e "${BLUE}Core 4 Process Monitor${NC}"
            echo "=========================="
            
            local cpu_usage memory_usage load_avg
            read -r cpu_usage memory_usage load_avg < <(get_system_resources)
            
            echo -e "System Resources:"
            echo -e "  CPU Usage: ${cpu_usage}%"
            echo -e "  Memory Usage: ${memory_usage}%"
            echo -e "  Load Average: ${load_avg}"
            echo
            
            echo -e "${YELLOW}Top CPU Processes:${NC}"
            get_top_cpu_processes 10 | awk 'NR<=10 {printf "  %2d. PID %-6s CPU: %5s%% MEM: %5s%% %s\n", NR, $1, $3, $4, $5}'
            echo
            
            echo -e "${RED}Top Memory Processes:${NC}"
            get_top_memory_processes 10 | awk 'NR<=10 {printf "  %2d. PID %-6s MEM: %5s%% CPU: %5s%% %s\n", NR, $1, $4, $3, $5}'
            ;;
            
        cleanup)
            check_lock
            log "INFO" "Starting automatic cleanup"
            auto_cleanup
            cleanup_zombies
            ;;
            
        zombies)
            check_lock
            cleanup_zombies
            ;;
            
        stuck)
            echo -e "${YELLOW}Stuck/Runaway Processes:${NC}"
            find_stuck_processes | while IFS=' ' read -r pid ppid cpu mem name; do
                [[ -z "$pid" ]] && continue
                echo -e "  PID $pid - CPU: ${cpu}% MEM: ${mem}% - $name"
            done
            ;;
            
        kill)
            if [[ -z "$target_pid" ]]; then
                log "ERROR" "PID required for kill action"
                exit 1
            fi
            
            check_lock
            local proc_info
            proc_info=$(get_process_info "$target_pid")
            
            if [[ -z "$proc_info" ]]; then
                log "ERROR" "Process $target_pid not found"
                exit 1
            fi
            
            local name
            name=$(echo "$proc_info" | awk '{print $8}')
            kill_process_safe "$target_pid" "$name" "$force_kill"
            ;;
            
        priorities)
            check_lock
            optimize_priorities
            ;;
            
        report)
            local report_file
            report_file=$(generate_report)
            echo "Report generated: $report_file"
            cat "$report_file"
            ;;
    esac
}

# Run main function
main "$@"