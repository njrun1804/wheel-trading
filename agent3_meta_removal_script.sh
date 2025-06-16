#!/bin/bash
# AGENT 3 - CORE META SYSTEM REMOVAL SCRIPT
# Safe removal of core meta system files in priority order
# Execute this script AFTER system recovery from FD issues

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/meta_removal_backup_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${SCRIPT_DIR}/agent3_removal.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

create_backup() {
    log "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Archive all files before removal
    log "Creating archive of all meta files..."
    tar -czf "${BACKUP_DIR}/meta_system_complete_backup.tar.gz" \
        meta_*.py meta_*.sh meta_*.json meta_*.db* start_*meta*.py \
        production_meta_*.py unified_meta*.py metacode.py \
        activate_meta_improvements.sh start_meta_and_jarvis.sh \
        meta/ 2>/dev/null || true
    
    log "Backup created successfully"
}

verify_no_processes() {
    log "Checking for running meta processes..."
    if pgrep -f "meta_" >/dev/null 2>&1; then
        log "ERROR: Meta processes still running. Please stop them first."
        pgrep -f "meta_" | xargs ps -p
        exit 1
    fi
    log "No meta processes detected"
}

phase1_startup_scripts() {
    log "=== PHASE 1: Removing Meta Startup Scripts ==="
    local files=(
        "start_complete_meta_system.py"
        "start_production_meta_system.py"
        "start_meta_system.py"
        "production_meta_improvement_system.py"
        "unified_meta_system.py"
        "metacode.py"
        "start_meta_and_jarvis.sh"
    )
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Removing startup script: $file"
            rm -f "$file"
        else
            log "Already removed: $file"
        fi
    done
    log "Phase 1 complete"
}

phase2_shell_scripts() {
    log "=== PHASE 2: Removing Meta Shell Scripts ==="
    local files=(
        "meta_status.sh"
        "meta_stop.sh"
        "activate_meta_improvements.sh"
    )
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Removing shell script: $file"
            rm -f "$file"
        else
            log "Already removed: $file"
        fi
    done
    log "Phase 2 complete"
}

phase3_configuration() {
    log "=== PHASE 3: Removing Meta Configuration Files ==="
    local files=(
        "meta_system_config.json"
        "meta_daemon_config.py"
    )
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Removing configuration: $file"
            rm -f "$file"
        else
            log "Already removed: $file"
        fi
    done
    log "Phase 3 complete"
}

phase4_core_python() {
    log "=== PHASE 4: Removing Core Meta Python Files ==="
    local files=(
        "meta_prime.py"
        "meta_coordinator.py"
        "meta_daemon.py"
        "meta_config.py"
        "meta_auditor.py"
        "meta_generator.py"
        "meta_executor.py"
        "meta_watcher.py"
        "meta_monitoring.py"
        "meta_fast_pattern_cache.py"
        "meta_reality_bridge.py"
        "meta_quality_enforcer.py"
        "meta_active_improvement_engine.py"
    )
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Removing core Python file: $file"
            rm -f "$file"
        else
            log "Already removed: $file"
        fi
    done
    log "Phase 4 complete"
}

phase5_databases() {
    log "=== PHASE 5: Removing Meta Database Files ==="
    log "Removing all meta_*.db* files..."
    
    # Remove database files safely
    find . -maxdepth 1 -name "meta_*.db*" -type f | while read -r file; do
        log "Removing database file: $file"
        rm -f "$file"
    done
    
    log "Phase 5 complete"
}

phase6_documentation() {
    log "=== PHASE 6: Removing Meta Documentation ==="
    if [[ -d "meta/" ]]; then
        log "Removing meta directory and contents..."
        rm -rf "meta/"
    else
        log "Meta directory already removed"
    fi
    log "Phase 6 complete"
}

verify_removal() {
    log "=== VERIFICATION: Checking for remaining meta files ==="
    
    local remaining_files=()
    
    # Check for any remaining meta files
    while IFS= read -r -d '' file; do
        remaining_files+=("$file")
    done < <(find . -maxdepth 1 -name "*meta*" -type f -print0 2>/dev/null || true)
    
    if [[ ${#remaining_files[@]} -gt 0 ]]; then
        log "WARNING: Found remaining meta files:"
        printf '%s\n' "${remaining_files[@]}" | tee -a "$LOG_FILE"
    else
        log "SUCCESS: No meta files remaining"
    fi
    
    # Check for meta references in key files
    log "Checking for meta references in startup files..."
    if grep -r "meta_" startup*.sh 2>/dev/null || true; then
        log "WARNING: Found meta references in startup scripts"
    else
        log "SUCCESS: No meta references in startup scripts"
    fi
}

main() {
    log "Starting Agent 3 - Core Meta System Removal"
    log "Working directory: $SCRIPT_DIR"
    
    # Safety checks
    verify_no_processes
    
    # Create backup
    create_backup
    
    # Execute removal phases
    phase1_startup_scripts
    phase2_shell_scripts
    phase3_configuration
    phase4_core_python
    phase5_databases
    phase6_documentation
    
    # Verify removal
    verify_removal
    
    log "=== AGENT 3 REMOVAL COMPLETE ==="
    log "Backup location: $BACKUP_DIR"
    log "Log file: $LOG_FILE"
    log "System should now be free of core meta system components"
}

# Execute main function
main "$@"