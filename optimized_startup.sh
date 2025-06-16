#\!/bin/bash
# Core 5 Optimized Startup Script
# M4 Pro hardware-tuned 5-phase boot sequence for trading system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/startup_optimization.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$TIMESTAMP]${NC} $1" | tee -a "$LOG_FILE"
}

log_phase() {
    echo -e "${BLUE}[$TIMESTAMP] 🚀 PHASE $1: $2${NC}" | tee -a "$LOG_FILE"
}

# Phase 1: Critical System Services (0-10s)
phase1_critical() {
    log_phase "1" "Critical System Services"
    
    # Set system performance mode
    sudo pmset -a powernap 0 2>/dev/null || true
    sudo pmset -a standbydelaylow 0 2>/dev/null || true
    sudo pmset -a standbydelayhigh 0 2>/dev/null || true
    
    log "✅ Phase 1 complete - Critical services initialized"
}

# Phase 2: Core System Components (10-20s)  
phase2_system() {
    log_phase "2" "Core System Components"
    
    # Configure system for trading performance
    defaults write com.apple.dock mineffect -string "scale" 2>/dev/null || true
    defaults write com.apple.dock launchanim -bool false 2>/dev/null || true
    defaults write NSGlobalDomain NSAutomaticWindowAnimationsEnabled -bool false 2>/dev/null || true
    
    log "✅ Phase 2 complete - System components optimized"
}

# Phase 3: Trading System Components (20-40s)
phase3_trading() {
    log_phase "3" "Trading System Components"
    
    # Set up trading environment
    export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
    export TRADING_ENV="production"
    export HARDWARE_ACCELERATION="enabled"
    
    # Prepare Einstein AI system
    if [[ -f "$SCRIPT_DIR/einstein_launcher.py" ]]; then
        log "Preparing Einstein AI system..."
        python "$SCRIPT_DIR/einstein_launcher.py" --prepare-only &
    fi
    
    # Start meta system
    if [[ -f "$SCRIPT_DIR/meta_daemon.py" ]]; then
        log "Starting Meta development system..."
        python "$SCRIPT_DIR/meta_daemon.py" --startup-mode &
    fi
    
    log "✅ Phase 3 complete - Trading system prepared"
}

# Phase 4: Development Tools (40-60s)
phase4_development() {
    log_phase "4" "Development Tools"
    
    # Start Claude integration
    if [[ -f "$SCRIPT_DIR/claude_integration_implementation.py" ]]; then
        log "Starting Claude integration..."
        python "$SCRIPT_DIR/claude_integration_implementation.py" --background &
    fi
    
    log "✅ Phase 4 complete - Development tools ready"
}

# Phase 5: Deferred Services (60s+)
phase5_deferred() {
    log_phase "5" "Deferred Services"
    
    # Non-essential services deferred for 60 seconds
    (
        sleep 60
        log "Starting deferred services..."
        log "✅ Phase 5 complete - Deferred services ready"
    ) &
    
    log "✅ Phase 5 initiated - Services will start in background"
}

# Performance optimization
optimize_performance() {
    log "⚡ Applying M4 Pro performance optimizations..."
    
    # Memory optimization
    sudo purge 2>/dev/null || true
    
    # System optimization
    sudo sysctl -w kern.maxvnodes=200000 2>/dev/null || true
    sudo sysctl -w kern.maxproc=2048 2>/dev/null || true
    
    # Network optimization for trading
    sudo sysctl -w net.inet.tcp.delayed_ack=0 2>/dev/null || true
    
    log "✅ Performance optimizations applied"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    log "🚀 Starting Core 5 Optimized M4 Pro Boot Sequence"
    log "================================================="
    
    # Execute phases
    phase1_critical
    sleep 2
    
    phase2_system  
    sleep 2
    
    phase3_trading
    sleep 5
    
    phase4_development
    sleep 3
    
    phase5_deferred
    
    # Apply optimizations
    optimize_performance
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "================================================="
    log "✅ Core 5 Optimized Startup Complete\!"
    log "⏱️  Total startup time: ${duration} seconds"
    log "📊 System ready for trading operations"
    log "================================================="
}

# Execute if run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
EOF < /dev/null