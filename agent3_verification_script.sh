#!/bin/bash
# AGENT 3 - META SYSTEM VERIFICATION SCRIPT
# Comprehensive verification that no meta system bootstrap code remains

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/agent3_verification.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

check_processes() {
    log "=== CHECKING FOR RUNNING META PROCESSES ==="
    
    if pgrep -f "meta_" >/dev/null 2>&1; then
        log "ERROR: Meta processes still running:"
        pgrep -f "meta_" | xargs ps -p | tee -a "$LOG_FILE"
        return 1
    else
        log "✓ No meta processes running"
        return 0
    fi
}

check_files() {
    log "=== CHECKING FOR REMAINING META FILES ==="
    
    local meta_files=()
    
    # Find any remaining meta files
    while IFS= read -r -d '' file; do
        meta_files+=("$file")
    done < <(find . -maxdepth 1 -name "*meta*" -type f -print0 2>/dev/null || true)
    
    if [[ ${#meta_files[@]} -gt 0 ]]; then
        log "WARNING: Found remaining meta files:"
        printf '%s\n' "${meta_files[@]}" | tee -a "$LOG_FILE"
        return 1
    else
        log "✓ No meta files found"
        return 0
    fi
}

check_startup_scripts() {
    log "=== CHECKING STARTUP SCRIPTS FOR META REFERENCES ==="
    
    local startup_files=(
        "startup.sh"
        "startup_enhanced.sh"
        "startup_unified.sh"
        "optimized_startup.sh"
        "launch_optimized.sh"
        "quick_recovery.sh"
    )
    
    local found_references=false
    
    for file in "${startup_files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Checking $file for meta references..."
            if grep -q "meta_" "$file" 2>/dev/null; then
                log "WARNING: Found meta references in $file:"
                grep -n "meta_" "$file" | tee -a "$LOG_FILE"
                found_references=true
            else
                log "✓ No meta references in $file"
            fi
        fi
    done
    
    if $found_references; then
        return 1
    else
        log "✓ No meta references in startup scripts"
        return 0
    fi
}

check_python_imports() {
    log "=== CHECKING PYTHON FILES FOR META IMPORTS ==="
    
    local found_imports=false
    
    # Check main Python files for meta imports
    local python_files=(
        "run.py"
        "jarvis.py"
        "jarvis2.py"
        "orchestrate.py"
    )
    
    for file in "${python_files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Checking $file for meta imports..."
            if grep -q "import.*meta" "$file" 2>/dev/null || grep -q "from.*meta" "$file" 2>/dev/null; then
                log "WARNING: Found meta imports in $file:"
                grep -n -E "(import.*meta|from.*meta)" "$file" | tee -a "$LOG_FILE"
                found_imports=true
            else
                log "✓ No meta imports in $file"
            fi
        fi
    done
    
    # Check src directory for meta imports
    if [[ -d "src" ]]; then
        log "Checking src/ directory for meta imports..."
        if find src -name "*.py" -exec grep -l -E "(import.*meta|from.*meta)" {} \; 2>/dev/null | grep -v __pycache__ | head -5; then
            log "WARNING: Found meta imports in src/ directory"
            found_imports=true
        else
            log "✓ No meta imports in src/ directory"
        fi
    fi
    
    if $found_imports; then
        return 1
    else
        log "✓ No meta imports found"
        return 0
    fi
}

check_cron_jobs() {
    log "=== CHECKING FOR META CRON JOBS ==="
    
    if crontab -l 2>/dev/null | grep -q "meta"; then
        log "WARNING: Found meta-related cron jobs:"
        crontab -l 2>/dev/null | grep "meta" | tee -a "$LOG_FILE"
        return 1
    else
        log "✓ No meta cron jobs found"
        return 0
    fi
}

check_environment() {
    log "=== CHECKING ENVIRONMENT VARIABLES ==="
    
    local meta_env_vars=()
    
    # Check for meta-related environment variables
    while IFS= read -r var; do
        if [[ "$var" == *META* ]]; then
            meta_env_vars+=("$var")
        fi
    done < <(env | grep -i meta || true)
    
    if [[ ${#meta_env_vars[@]} -gt 0 ]]; then
        log "WARNING: Found meta environment variables:"
        printf '%s\n' "${meta_env_vars[@]}" | tee -a "$LOG_FILE"
        return 1
    else
        log "✓ No meta environment variables found"
        return 0
    fi
}

check_database_connections() {
    log "=== CHECKING FOR META DATABASE CONNECTIONS ==="
    
    # Check if any processes are holding meta database files
    if lsof 2>/dev/null | grep -q "meta.*\.db" || true; then
        log "WARNING: Found processes with meta database connections:"
        lsof 2>/dev/null | grep "meta.*\.db" | tee -a "$LOG_FILE" || true
        return 1
    else
        log "✓ No meta database connections found"
        return 0
    fi
}

check_git_status() {
    log "=== CHECKING GIT STATUS ==="
    
    # Check if meta files are still tracked in git
    if git ls-files | grep -q "meta" 2>/dev/null; then
        log "INFO: Meta files still tracked in git (this is expected):"
        git ls-files | grep "meta" | head -10 | tee -a "$LOG_FILE"
        log "These will be removed from git tracking in the commit"
    else
        log "✓ No meta files in git tracking"
    fi
    
    return 0
}

main() {
    log "Starting Agent 3 - Meta System Verification"
    log "Working directory: $SCRIPT_DIR"
    
    local overall_status=0
    
    # Run all verification checks
    check_processes || overall_status=1
    check_files || overall_status=1
    check_startup_scripts || overall_status=1
    check_python_imports || overall_status=1
    check_cron_jobs || overall_status=1
    check_environment || overall_status=1
    check_database_connections || overall_status=1
    check_git_status || overall_status=1
    
    log "=== VERIFICATION SUMMARY ==="
    if [[ $overall_status -eq 0 ]]; then
        log "✓ SUCCESS: System appears to be free of meta system bootstrap code"
        log "✓ Safe to proceed with normal operations"
    else
        log "⚠ WARNING: Some meta system components may still be present"
        log "⚠ Review the warnings above and take appropriate action"
    fi
    
    log "Verification complete. Log saved to: $LOG_FILE"
    return $overall_status
}

# Execute main function
main "$@"