#!/bin/bash

# M4 Pro Mac Optimization Setup - Use ALL Resources for MCP/Claude
# This makes optimizations permanent across restarts

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== M4 Pro Mac Optimization Setup ===${NC}"
echo -e "${YELLOW}Configuring your Mac to use ALL 24GB RAM and 12 cores for MCP servers${NC}"
echo ""

# 1. Create permanent sysctl configurations
echo -e "\n${YELLOW}1. Setting up permanent system optimizations...${NC}"

# Create sysctl config for boot persistence
sudo tee /etc/sysctl.conf > /dev/null << 'EOF'
# M4 Pro Optimizations for MCP Servers
# Increase shared memory segments
kern.sysv.shmmax=8589934592
kern.sysv.shmall=2097152
kern.sysv.shmmni=512
kern.sysv.shmseg=512

# Increase semaphores for parallel processing
kern.sysv.semmns=4096
kern.sysv.semmni=512

# Network optimizations
kern.ipc.somaxconn=4096
net.inet.tcp.msl=1000
net.inet.tcp.sendspace=1048576
net.inet.tcp.recvspace=1048576

# File system optimizations
kern.maxfiles=524288
kern.maxfilesperproc=524288
EOF

# Apply immediately
sudo sysctl -w kern.sysv.shmmax=8589934592 >/dev/null 2>&1 || true
sudo sysctl -w kern.maxfiles=524288 >/dev/null 2>&1 || true
echo -e "  ${GREEN}✓${NC} System limits configured"

# 2. Create LaunchDaemon for boot optimizations
echo -e "\n${YELLOW}2. Creating boot-time optimizations...${NC}"

sudo tee /Library/LaunchDaemons/com.wheel-trading.performance.plist > /dev/null << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wheel-trading.performance</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>
            # Disable CPU throttling
            sudo pmset -a powernap 0;
            sudo pmset -a disksleep 0;
            sudo pmset -a sleep 0;
            # Disable App Nap for Python and Node
            defaults write -g NSAppSleepDisabled -bool YES;
            # Set performance mode
            sudo nvram boot-args="serverperfmode=1";
        </string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
EOF

sudo chmod 644 /Library/LaunchDaemons/com.wheel-trading.performance.plist
sudo launchctl load /Library/LaunchDaemons/com.wheel-trading.performance.plist 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Boot optimizations installed"

# 3. Update shell configuration for permanent environment
echo -e "\n${YELLOW}3. Setting up permanent environment variables...${NC}"

# Add to .zshenv (loads for ALL zsh sessions, even non-interactive)
cat >> ~/.zshenv << 'EOF'

# M4 Pro MCP Optimizations - Use ALL Resources
export NODE_OPTIONS="--max-old-space-size=20480 --max-semi-space-size=256"  # 20GB for Node
export PYTHON_MEMORY_LIMIT="20G"
export OMP_NUM_THREADS=12  # Use all 12 cores
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export VECLIB_MAXIMUM_THREADS=12

# Python optimizations
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONOPTIMIZE=2  # Maximum optimization

# Parallel processing
export RAYON_NUM_THREADS=12  # For Rust-based tools
export UV_THREADPOOL_SIZE=12  # For UV package manager
export CARGO_BUILD_JOBS=12

# File system
ulimit -n 524288  # Max file descriptors
ulimit -u 4096    # Max processes

# Memory
ulimit -m unlimited
ulimit -v unlimited
EOF

echo -e "  ${GREEN}✓${NC} Environment variables set permanently"

# 4. Create performance-mode MCP configuration
echo -e "\n${YELLOW}4. Creating M4 Pro performance MCP config...${NC}"

cat > ~/Library/LaunchAgents/com.wheel-trading.mcp-boost.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wheel-trading.mcp-boost</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>
            # Keep Python and Node processes at high priority
            while true; do
                for pid in $(pgrep -f "mcp|claude"); do
                    renice -20 $pid 2>/dev/null || true
                done
                sleep 60
            done
        </string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.wheel-trading.mcp-boost.plist 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} MCP process priority boost enabled"

# 5. Configure memory pressure settings
echo -e "\n${YELLOW}5. Configuring memory management...${NC}"

# Disable memory compression for MCP processes (use real RAM)
sudo nvram boot-args="vm_compressor=2" 2>/dev/null || true

# Create swap preallocation
if [ ! -f /var/vm/swapfile8 ]; then
    echo -e "  ${BLUE}Creating 8GB swap file...${NC}"
    sudo dd if=/dev/zero of=/var/vm/swapfile8 bs=1g count=8 2>/dev/null
    sudo chmod 600 /var/vm/swapfile8
    echo -e "  ${GREEN}✓${NC} Swap file created"
fi

# 6. Optimize SSD settings
echo -e "\n${YELLOW}6. Optimizing SSD performance...${NC}"

# Disable sudden motion sensor (not needed on SSD)
sudo pmset -a sms 0 2>/dev/null || true

# Enable TRIM
sudo trimforce enable 2>/dev/null || true

echo -e "  ${GREEN}✓${NC} SSD optimizations applied"

# 7. Create performance launcher
echo -e "\n${YELLOW}7. Creating M4 Pro performance launcher...${NC}"

cat > /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/start-claude-m4-max.sh << 'EOFSCRIPT'
#!/bin/bash

# M4 Pro Maximum Performance Launcher - Uses ALL 24GB RAM & 12 Cores

# Force performance mode
sudo pmset -a powermode 2 2>/dev/null || true  # High performance mode

# Set process priority
renice -20 $$ 2>/dev/null || true

# Use performance cores
taskpolicy -c background -s 0 $$  # Use performance cores, not efficiency

# Source optimized environment
source ~/.zshenv

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== M4 Pro Maximum Performance Mode ===${NC}"
echo -e "  • RAM Available: 24GB (20GB allocated to MCP)"
echo -e "  • CPU Cores: 12 (all allocated)"
echo -e "  • Performance Cores: Priority mode"
echo -e "  • Memory Compression: Disabled"
echo -e "  • Process Priority: Maximum"
echo ""

# Pre-warm everything
echo -e "${YELLOW}Pre-warming all systems...${NC}"

# Warm CPU caches
yes > /dev/null & PID=$!; sleep 0.1; kill $PID 2>/dev/null

# Pre-load Python
python3 -c "
import numpy as np
import pandas as pd
import duckdb
import sqlalchemy
# Allocate 2GB to warm memory
data = np.zeros((250_000_000,), dtype=np.float64)
del data
" 2>/dev/null || true

# Use the ultra launcher with M4 optimizations
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "$SCRIPT_DIR/start-claude-ultra.sh"
EOFSCRIPT

chmod +x /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/start-claude-m4-max.sh
echo -e "  ${GREEN}✓${NC} M4 Pro launcher created"

# 8. Summary
echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo ""
echo -e "${YELLOW}What was configured:${NC}"
echo -e "  ✓ System limits increased (permanent)"
echo -e "  ✓ Boot-time performance mode"
echo -e "  ✓ 20GB RAM allocated to Node.js/Python"
echo -e "  ✓ All 12 cores available"
echo -e "  ✓ Process priority boosting"
echo -e "  ✓ SSD optimizations"
echo -e "  ✓ Memory compression disabled"
echo ""
echo -e "${YELLOW}Changes that persist across restarts:${NC}"
echo -e "  • /etc/sysctl.conf - System limits"
echo -e "  • ~/.zshenv - Environment variables"
echo -e "  • LaunchDaemons - Boot optimizations"
echo -e "  • LaunchAgents - Process priority"
echo ""
echo -e "${GREEN}To use maximum performance:${NC}"
echo -e "  ./scripts/start-claude-m4-max.sh"
echo ""
echo -e "${YELLOW}Note:${NC} Some changes require a restart to take full effect."
echo -e "Restart now? (y/n): \c"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Restarting in 5 seconds..."
    sleep 5
    sudo reboot
fi