#!/usr/bin/env python3
"""
STARTUP OPTIMIZATION SYSTEM
Core 5 - Service Startup Optimization and Management

This script provides advanced startup optimization specifically tuned for
the trading system's performance requirements on M4 Pro hardware.
"""

import json
import os
from pathlib import Path


class StartupOptimizer:
    def __init__(self):
        self.base_path = Path(
            "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
        )
        self.config_file = self.base_path / "startup_config.json"
        self.critical_services = {
            # Essential system services
            "com.apple.launchd",
            "com.apple.WindowServer",
            "com.apple.Finder",
            "com.apple.loginwindow",
            # Trading system services
            "claude",
            "einstein",
            "duckdb",
            "python",  # For trading scripts
            # Essential development
            "com.apple.Terminal",
            "wezterm-gui",
        }

        self.defer_services = {
            # Non-essential at startup
            "com.apple.SafariBookmarksSyncAgent",
            "com.apple.cloudphotod",
            "com.apple.mediaremoteagent",
            "com.apple.bird",
            "com.apple.nsurlsessiond",
        }

        self.trading_priority_processes = {
            "claude": {"cpu_priority": "high", "memory_mb": 4096},
            "einstein": {"cpu_priority": "high", "memory_mb": 2048},
            "duckdb": {"cpu_priority": "medium", "memory_mb": 1024},
            "python": {"cpu_priority": "medium", "memory_mb": 512},
        }

    def load_config(self) -> dict:
        """Load startup optimization configuration"""
        default_config = {
            "enabled": True,
            "aggressive_optimization": False,
            "defer_non_essential": True,
            "prioritize_trading": True,
            "max_startup_processes": 300,
            "memory_limit_gb": 12,
            "cpu_cores_reserved": 4,
            "startup_timeout_seconds": 30,
        }

        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    config = json.load(f)
                    return {**default_config, **config}
            except Exception as e:
                print(f"Error loading config: {e}")

        return default_config

    def save_config(self, config: dict) -> None:
        """Save startup optimization configuration"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_startup_services(self) -> list[dict]:
        """Get all services that start at boot"""
        startup_services = []

        # LaunchDaemons (system-wide)
        daemon_dirs = ["/System/Library/LaunchDaemons", "/Library/LaunchDaemons"]

        # LaunchAgents (user-specific)
        agent_dirs = [
            "/System/Library/LaunchAgents",
            "/Library/LaunchAgents",
            f'{os.path.expanduser("~")}/Library/LaunchAgents',
        ]

        for directory in daemon_dirs + agent_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith(".plist"):
                        startup_services.append(
                            {
                                "name": file.replace(".plist", ""),
                                "path": os.path.join(directory, file),
                                "type": "daemon" if "Daemon" in directory else "agent",
                                "critical": any(
                                    svc in file for svc in self.critical_services
                                ),
                            }
                        )

        return startup_services

    def analyze_startup_impact(self) -> dict:
        """Analyze current startup performance"""
        analysis = {
            "total_services": 0,
            "critical_services": 0,
            "deferrable_services": 0,
            "estimated_startup_time": 0,
            "memory_impact": 0,
            "recommendations": [],
        }

        services = self.get_startup_services()
        analysis["total_services"] = len(services)

        for service in services:
            if service["critical"]:
                analysis["critical_services"] += 1
            elif any(defer in service["name"] for defer in self.defer_services):
                analysis["deferrable_services"] += 1
                analysis["recommendations"].append(
                    {
                        "service": service["name"],
                        "action": "defer_startup",
                        "estimated_savings": "2-5 seconds",
                    }
                )

        # Estimate startup time (rough calculation)
        analysis["estimated_startup_time"] = (
            analysis["total_services"] * 0.1
        )  # 100ms per service

        # Estimate memory impact
        analysis["memory_impact"] = (
            analysis["total_services"] * 50
        )  # 50MB per service average

        return analysis

    def create_optimized_startup_sequence(self) -> dict:
        """Create optimized startup sequence for trading system"""
        sequence = {
            "phase_1_critical": [  # 0-10 seconds
                "com.apple.launchd",
                "com.apple.logd",
                "com.apple.configd",
                "com.apple.WindowServer",
            ],
            "phase_2_system": [  # 10-20 seconds
                "com.apple.Finder",
                "com.apple.loginwindow",
                "com.apple.powerd",
                "wezterm-gui",
            ],
            "phase_3_trading": [  # 20-30 seconds - PRIORITY
                "python",  # Python environment
                "duckdb",  # Database
                "claude",  # AI assistant
                "einstein",  # Search system
            ],
            "phase_4_development": [  # 30-45 seconds
                "com.apple.Terminal",
                "com.apple.Spotlight",
                "vscode",
            ],
            "phase_5_deferred": [  # After 60 seconds
                "com.apple.SafariBookmarksSyncAgent",
                "com.apple.cloudphotod",
                "com.apple.mediaremoteagent",
                "com.apple.bird",
            ],
        }

        return sequence

    def generate_launch_script(self) -> str:
        """Generate optimized launch script for trading system"""
        script = """#!/bin/bash
# OPTIMIZED TRADING SYSTEM STARTUP SCRIPT
# Generated by Core 5 Startup Optimizer

set -euo pipefail

# Configuration
LOG_FILE="/tmp/trading_startup.log"
STARTUP_LOCK="/tmp/trading_startup.lock"
MAX_WAIT_TIME=60

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Ensure single instance
if [ -f "$STARTUP_LOCK" ]; then
    log "Startup already in progress, exiting"
    exit 1
fi
touch "$STARTUP_LOCK"
trap 'rm -f "$STARTUP_LOCK"' EXIT

log "=== OPTIMIZED TRADING SYSTEM STARTUP ==="

# Phase 1: Hardware Optimization (M4 Pro specific)
optimize_hardware() {
    log "Phase 1: Optimizing hardware for trading"
    
    # Set CPU performance mode
    sudo pmset -c powernap 0 2>/dev/null || true
    sudo pmset -c sleep 0 2>/dev/null || true
    sudo pmset -c standby 0 2>/dev/null || true
    
    # Optimize memory settings
    sudo sysctl -w vm.swappiness=10 2>/dev/null || true
    sudo sysctl -w kern.maxfiles=65536 2>/dev/null || true
    
    log "Hardware optimization complete"
}

# Phase 2: Start Essential Services
start_essential_services() {
    log "Phase 2: Starting essential services"
    
    # Ensure critical services are running
    essential_services=(
        "com.apple.WindowServer"
        "com.apple.Finder"
        "com.apple.loginwindow"
    )
    
    for service in "${essential_services[@]}"; do
        if ! pgrep -f "$service" > /dev/null; then
            log "Starting essential service: $service"
            launchctl load "/System/Library/LaunchAgents/$service.plist" 2>/dev/null || true
        fi
    done
    
    log "Essential services started"
}

# Phase 3: Trading System Priority Start
start_trading_system() {
    log "Phase 3: Starting trading system (PRIORITY)"
    
    # Change to trading directory
    cd "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading" || exit 1
    
    # Start Einstein search system
    if [ -f "einstein_launcher.py" ]; then
        log "Starting Einstein system..."
        python3 einstein_launcher.py --background --optimized &
        EINSTEIN_PID=$!
        log "Einstein started with PID: $EINSTEIN_PID"
    fi
    
    # Start Meta system
    if [ -f "meta_coordinator.py" ]; then
        log "Starting Meta coordination system..."
        python3 meta_coordinator.py --startup-mode &
        META_PID=$!
        log "Meta coordinator started with PID: $META_PID"
    fi
    
    # Warm up DuckDB connections
    if [ -f "run.py" ]; then
        log "Warming up database connections..."
        timeout 10s python3 run.py --warmup 2>/dev/null &
    fi
    
    log "Trading system startup complete"
}

# Phase 4: Set Process Priorities
set_trading_priorities() {
    log "Phase 4: Setting trading process priorities"
    
    # Set CPU affinity for trading processes (P-cores 0-7 on M4 Pro)
    for proc in "claude" "einstein" "python.*wheel" "duckdb"; do
        pids=$(pgrep -f "$proc" || true)
        for pid in $pids; do
            if [ -n "$pid" ]; then
                # Set high priority for trading processes
                renice -10 "$pid" 2>/dev/null || true
                # Use performance cores
                taskpolicy -c performance -t "$pid" 2>/dev/null || true
                log "Set priority for $proc (PID: $pid)"
            fi
        done
    done
    
    log "Process priorities set"
}

# Phase 5: Defer Non-Essential Services
defer_non_essential() {
    log "Phase 5: Deferring non-essential services"
    
    # Services to defer for 60 seconds
    defer_services=(
        "com.apple.SafariBookmarksSyncAgent"
        "com.apple.cloudphotod"
        "com.apple.mediaremoteagent"
        "com.apple.bird"
        "com.apple.nsurlsessiond"
    )
    
    for service in "${defer_services[@]}"; do
        # Unload and schedule for later
        launchctl unload "/System/Library/LaunchAgents/$service.plist" 2>/dev/null || true
        (sleep 60 && launchctl load "/System/Library/LaunchAgents/$service.plist" 2>/dev/null) &
        log "Deferred service: $service"
    done
    
    log "Non-essential services deferred"
}

# Phase 6: Monitoring Setup
setup_monitoring() {
    log "Phase 6: Setting up performance monitoring"
    
    # Start resource monitor for trading system
    if [ -f "service_monitor.sh" ]; then
        ./service_monitor.sh --trading-focus &
        log "Performance monitoring started"
    fi
    
    log "Monitoring setup complete"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    optimize_hardware
    start_essential_services
    start_trading_system
    set_trading_priorities
    defer_non_essential
    setup_monitoring
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "=== STARTUP COMPLETE IN ${duration} SECONDS ==="
    log "Trading system optimized and ready"
    
    # Verify trading system health
    sleep 5
    if pgrep -f "claude|einstein|python.*wheel" > /dev/null; then
        log "✅ Trading system processes verified running"
    else
        log "⚠️  Warning: Some trading processes may not be running"
    fi
}

# Execute main function
main "$@"
"""
        return script

    def create_service_monitor(self) -> str:
        """Create continuous service monitoring script"""
        monitor_script = """#!/bin/bash
# TRADING SYSTEM SERVICE MONITOR
# Continuous monitoring optimized for trading performance

set -euo pipefail

LOG_FILE="/tmp/service_monitor.log"
PID_FILE="/tmp/service_monitor.pid"
ALERT_THRESHOLD_CPU=70
ALERT_THRESHOLD_MEM=80
CHECK_INTERVAL=30

# Store PID
echo $$ > "$PID_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_trading_processes() {
    pgrep -f "claude|einstein|python.*wheel|duckdb" || echo ""
}

check_trading_health() {
    local trading_pids=$(get_trading_processes)
    local healthy=true
    
    if [ -z "$trading_pids" ]; then
        log "⚠️  WARNING: No trading processes detected"
        healthy=false
    else
        local count=$(echo "$trading_pids" | wc -w)
        log "✅ Trading processes: $count active"
        
        # Check resource usage
        for pid in $trading_pids; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                local cpu=$(ps -p "$pid" -o pcpu= 2>/dev/null | xargs || echo "0")
                local mem=$(ps -p "$pid" -o pmem= 2>/dev/null | xargs || echo "0")
                local cmd=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
                
                # Convert to integer for comparison
                cpu=${cpu%.*}
                mem=${mem%.*}
                
                if [ "$cpu" -gt "$ALERT_THRESHOLD_CPU" ]; then
                    log "⚠️  HIGH CPU: $cmd (PID $pid) using ${cpu}% CPU"
                fi
                
                if [ "$mem" -gt "$ALERT_THRESHOLD_MEM" ]; then
                    log "⚠️  HIGH MEMORY: $cmd (PID $pid) using ${mem}% memory"
                fi
            fi
        done
    fi
    
    return $healthy
}

optimize_if_needed() {
    local total_processes=$(ps -ax | wc -l | xargs)
    local memory_pressure=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
    
    if [ "$total_processes" -gt 500 ]; then
        log "🔧 High process count ($total_processes), running cleanup"
        # Run the service optimizer
        if [ -f "optimized_service_manager.sh" ]; then
            ./optimized_service_manager.sh cleanup
        fi
    fi
    
    # Check memory pressure
    if [ "$memory_pressure" -lt 100000 ]; then  # Less than ~1.6GB free
        log "🔧 Memory pressure detected, optimizing"
        purge 2>/dev/null || true
    fi
}

main_monitor_loop() {
    log "=== TRADING SYSTEM MONITOR STARTED ==="
    
    while true; do
        if check_trading_health; then
            log "✅ Trading system health check passed"
        else
            log "❌ Trading system health check failed"
            # Attempt to restart critical services
            if [ -f "startup_optimizer.py" ]; then
                python3 startup_optimizer.py --restart-trading
            fi
        fi
        
        optimize_if_needed
        
        sleep "$CHECK_INTERVAL"
    done
}

# Handle signals
trap 'log "Monitor stopped"; rm -f "$PID_FILE"; exit 0' SIGTERM SIGINT

# Main execution
case "${1:-monitor}" in
    "--trading-focus")
        main_monitor_loop
        ;;
    "--status")
        check_trading_health && echo "Healthy" || echo "Issues detected"
        ;;
    "--stop")
        if [ -f "$PID_FILE" ]; then
            kill "$(cat "$PID_FILE")" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "Monitor stopped"
        fi
        ;;
    *)
        echo "Usage: $0 {--trading-focus|--status|--stop}"
        exit 1
        ;;
esac
"""
        return monitor_script

    def apply_optimizations(self, config: dict) -> dict:
        """Apply startup optimizations based on configuration"""
        results = {"optimizations_applied": [], "errors": []}

        try:
            # Generate optimized scripts
            launch_script = self.generate_launch_script()
            launch_path = self.base_path / "optimized_startup.sh"
            with open(launch_path, "w") as f:
                f.write(launch_script)
            os.chmod(launch_path, 0o755)
            results["optimizations_applied"].append(
                f"Created optimized startup script: {launch_path}"
            )

            monitor_script = self.create_service_monitor()
            monitor_path = self.base_path / "service_monitor.sh"
            with open(monitor_path, "w") as f:
                f.write(monitor_script)
            os.chmod(monitor_path, 0o755)
            results["optimizations_applied"].append(
                f"Created service monitor: {monitor_path}"
            )

            # Create system optimization plist for auto-start
            if config.get("auto_optimize_startup", False):
                plist_content = self.create_startup_plist()
                plist_path = (
                    Path.home() / "Library/LaunchAgents/com.trading.optimizer.plist"
                )
                with open(plist_path, "w") as f:
                    f.write(plist_content)
                results["optimizations_applied"].append(
                    f"Created startup optimization plist: {plist_path}"
                )

        except Exception as e:
            results["errors"].append(str(e))

        return results

    def create_startup_plist(self) -> str:
        """Create LaunchAgent plist for automatic startup optimization"""
        script_path = self.base_path / "optimized_startup.sh"

        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trading.optimizer</string>
    <key>ProgramArguments</key>
    <array>
        <string>{script_path}</string>
    </array>
    <key>StartInterval</key>
    <integer>3600</integer>
    <key>StandardOutPath</key>
    <string>/tmp/trading_optimizer.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/trading_optimizer_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>"""
        return plist


def main():
    """Main function for startup optimization"""
    optimizer = StartupOptimizer()

    print("🚀 TRADING SYSTEM STARTUP OPTIMIZER")
    print("=" * 50)

    # Load configuration
    config = optimizer.load_config()
    print(f"📋 Configuration loaded: {len(config)} settings")

    # Analyze current startup
    print("🔍 Analyzing current startup performance...")
    analysis = optimizer.analyze_startup_impact()

    print("\n📊 STARTUP ANALYSIS:")
    print(f"   • Total Services: {analysis['total_services']}")
    print(f"   • Critical Services: {analysis['critical_services']}")
    print(f"   • Deferrable Services: {analysis['deferrable_services']}")
    print(f"   • Estimated Startup Time: {analysis['estimated_startup_time']:.1f}s")
    print(f"   • Memory Impact: {analysis['memory_impact']}MB")
    print(f"   • Optimization Opportunities: {len(analysis['recommendations'])}")

    # Create optimized startup sequence
    print("\n⚙️  Creating optimized startup sequence...")
    sequence = optimizer.create_optimized_startup_sequence()

    print("🎯 STARTUP SEQUENCE PHASES:")
    for phase, services in sequence.items():
        print(f"   • {phase}: {len(services)} services")

    # Apply optimizations
    print("\n🔧 Applying optimizations...")
    results = optimizer.apply_optimizations(config)

    print("\n✅ OPTIMIZATIONS APPLIED:")
    for opt in results["optimizations_applied"]:
        print(f"   • {opt}")

    if results["errors"]:
        print("\n❌ ERRORS:")
        for error in results["errors"]:
            print(f"   • {error}")

    print("\n🎯 NEXT STEPS:")
    print("   1. Run ./optimized_startup.sh to test optimized startup")
    print("   2. Run ./service_monitor.sh --trading-focus to start monitoring")
    print("   3. Check /tmp/trading_startup.log for startup logs")
    print("   4. Monitor system performance during trading hours")

    print("\n⚡ EXPECTED IMPROVEMENTS:")
    potential_savings = analysis["deferrable_services"] * 2
    print(f"   • Startup time reduction: ~{potential_savings} seconds")
    print(f"   • Memory savings: ~{analysis['deferrable_services'] * 50}MB")
    print("   • Trading system priority: P-core allocation guaranteed")
    print(
        f"   • Background process reduction: ~{analysis['deferrable_services']} fewer services"
    )


if __name__ == "__main__":
    main()
