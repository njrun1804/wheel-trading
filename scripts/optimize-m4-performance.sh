#!/bin/bash
# M4 Pro Performance Optimization Script for Wheel Trading

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 M4 Pro Performance Optimizer for Wheel Trading${NC}"
echo "================================================"
echo

# Function to set process priority
set_high_priority() {
    local pid=$1
    local name=$2
    if [ -n "$pid" ]; then
        renice -n -10 -p "$pid" 2>/dev/null && \
            echo -e "${GREEN}✓${NC} Set high priority for $name (PID: $pid)" || \
            echo -e "${YELLOW}!${NC} Could not set priority for $name (needs sudo)"
    fi
}

# 1. Disable Spotlight indexing for wheel-trading data
echo -e "${BLUE}1. Optimizing Spotlight indexing...${NC}"
WHEEL_PATH="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
if [ -d "$WHEEL_PATH/data" ]; then
    touch "$WHEEL_PATH/data/.metadata_never_index"
    echo -e "${GREEN}✓${NC} Disabled Spotlight indexing for data directory"
fi
if [ -d "$WHEEL_PATH/logs" ]; then
    touch "$WHEEL_PATH/logs/.metadata_never_index"
    echo -e "${GREEN}✓${NC} Disabled Spotlight indexing for logs directory"
fi

# 2. Configure memory pressure settings
echo -e "\n${BLUE}2. Configuring memory settings...${NC}"
# Set vm pressure settings for better performance
sudo sysctl -w vm.compressor_mode=2 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Set memory compressor to performance mode" || \
    echo -e "${YELLOW}!${NC} Could not set memory compressor (needs sudo)"

# 3. Optimize network settings for Databento/FRED
echo -e "\n${BLUE}3. Optimizing network settings...${NC}"
# Increase TCP buffer sizes for better API performance
sudo sysctl -w net.inet.tcp.sendspace=131072 2>/dev/null
sudo sysctl -w net.inet.tcp.recvspace=131072 2>/dev/null
sudo sysctl -w kern.ipc.maxsockbuf=8388608 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Optimized TCP buffer sizes" || \
    echo -e "${YELLOW}!${NC} Could not optimize network (needs sudo)"

# 4. Set CPU performance mode
echo -e "\n${BLUE}4. Setting CPU performance mode...${NC}"
# Enable performance mode
sudo pmset -a powermode 2 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Set CPU to high performance mode" || \
    echo -e "${YELLOW}!${NC} Could not set CPU mode (needs sudo)"

# Disable CPU throttling during development
sudo pmset -a disablesleep 1 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Disabled sleep during development" || \
    echo -e "${YELLOW}!${NC} Could not disable sleep (needs sudo)"

# 5. Configure file system caching
echo -e "\n${BLUE}5. Optimizing file system...${NC}"
# Increase vnodes for better file caching
sudo sysctl -w kern.maxvnodes=600000 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Increased vnode cache" || \
    echo -e "${YELLOW}!${NC} Could not increase vnode cache (needs sudo)"

# 6. Python-specific optimizations
echo -e "\n${BLUE}6. Python optimizations...${NC}"
# Clear Python cache
find "$WHEEL_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo -e "${GREEN}✓${NC} Cleared Python cache"

# Pre-compile Python files
python -m compileall -q "$WHEEL_PATH/src" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Pre-compiled Python modules" || \
    echo -e "${YELLOW}!${NC} Could not pre-compile modules"

# 7. DuckDB optimization
echo -e "\n${BLUE}7. DuckDB optimization...${NC}"
# Set DuckDB memory limit to use more RAM
export DUCKDB_MEMORY_LIMIT="8GB"
echo -e "${GREEN}✓${NC} Set DuckDB memory limit to 8GB"

# 8. Process priority optimization
echo -e "\n${BLUE}8. Setting process priorities...${NC}"
# Find and prioritize key processes
PYTHON_PID=$(pgrep -f "python.*wheel" | head -1)
set_high_priority "$PYTHON_PID" "Python (wheel trading)"

DUCKDB_PID=$(pgrep -f "duckdb" | head -1)
set_high_priority "$DUCKDB_PID" "DuckDB"

# 9. Create RAM disk for temporary files
echo -e "\n${BLUE}9. Creating RAM disk for temp files...${NC}"
RAMDISK_SIZE=2048  # 2GB in MB
RAMDISK_PATH="/Volumes/WheelTradingRAM"

if [ ! -d "$RAMDISK_PATH" ]; then
    RAMDISK_SECTORS=$((RAMDISK_SIZE * 2048))
    DISK_ID=$(hdiutil attach -nomount ram://$RAMDISK_SECTORS)
    diskutil erasevolume HFS+ "WheelTradingRAM" $DISK_ID
    echo -e "${GREEN}✓${NC} Created 2GB RAM disk at $RAMDISK_PATH"
    
    # Create temp directories
    mkdir -p "$RAMDISK_PATH/cache"
    mkdir -p "$RAMDISK_PATH/tmp"
    
    # Set environment variables
    export WHEEL_CACHE_DIR="$RAMDISK_PATH/cache"
    export TMPDIR="$RAMDISK_PATH/tmp"
else
    echo -e "${GREEN}✓${NC} RAM disk already exists at $RAMDISK_PATH"
fi

# 10. TG Pro integration
echo -e "\n${BLUE}10. TG Pro thermal management...${NC}"
if [ -d "/Applications/TG Pro.app" ]; then
    # Create TG Pro rule for wheel trading
    cat > ~/Library/Application\ Support/TG\ Pro/wheel_trading_rule.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>name</key>
    <string>Wheel Trading High Performance</string>
    <key>enabled</key>
    <true/>
    <key>conditions</key>
    <array>
        <dict>
            <key>type</key>
            <string>process</string>
            <key>processName</key>
            <string>python</string>
            <key>contains</key>
            <string>wheel</string>
        </dict>
    </array>
    <key>fanSettings</key>
    <dict>
        <key>minimumSpeed</key>
        <integer>3000</integer>
        <key>targetTemperature</key>
        <integer>65</integer>
    </dict>
</dict>
</plist>
EOF
    echo -e "${GREEN}✓${NC} Created TG Pro rule for thermal management"
else
    echo -e "${YELLOW}!${NC} TG Pro not found - manual thermal management recommended"
fi

# 11. System integrity protection notice
echo -e "\n${YELLOW}📌 Note:${NC} Some optimizations require sudo access."
echo "For permanent settings, add to /etc/sysctl.conf"

# 12. Performance monitoring
echo -e "\n${BLUE}Performance Monitoring Commands:${NC}"
echo "- CPU usage: top -pid \$(pgrep -f 'python.*wheel')"
echo "- Memory: vm_stat"
echo "- Disk I/O: iostat -w 1"
echo "- Network: nettop -P \$(pgrep -f 'python.*wheel')"
echo "- GPU: sudo powermetrics --samplers gpu_power"

# Save current settings
echo -e "\n${BLUE}Saving optimization settings...${NC}"
cat > "$WHEEL_PATH/.performance_settings" << EOF
# M4 Pro Performance Settings for Wheel Trading
# Generated: $(date)

# Memory
export DUCKDB_MEMORY_LIMIT="8GB"
export WHEEL_CACHE_DIR="$RAMDISK_PATH/cache"
export TMPDIR="$RAMDISK_PATH/tmp"

# CPU
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# Python
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# Metal/GPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
export METAL_DEVICE_WRAPPER_TYPE=1
export USE_GPU_ACCELERATION=true
EOF

echo -e "${GREEN}✓${NC} Saved settings to .performance_settings"
echo
echo -e "${GREEN}🎉 M4 Pro optimization complete!${NC}"
echo "Source the settings: source $WHEEL_PATH/.performance_settings"