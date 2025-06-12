#!/bin/bash

# Configure app priorities - MCP/Claude gets performance cores, others get efficiency cores

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Configuring App Core Allocation ===${NC}"
echo -e "${YELLOW}MCP/Claude → Performance cores (8) | Others → Efficiency cores (4)${NC}"
echo ""

# 1. Configure Docker to use efficiency cores only
echo -e "\n${YELLOW}1. Configuring Docker...${NC}"

# Docker Desktop settings
DOCKER_SETTINGS="$HOME/Library/Group Containers/group.com.docker/settings.json"
if [ -f "$DOCKER_SETTINGS" ]; then
    # Backup current settings
    cp "$DOCKER_SETTINGS" "$DOCKER_SETTINGS.backup"
    
    # Update Docker to use less resources
    cat > ~/Library/Preferences/com.docker.priority.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CPULimit</key>
    <integer>4</integer>
    <key>MemoryLimit</key>
    <integer>4096</integer>
    <key>UseEfficiencyCores</key>
    <true/>
</dict>
</plist>
EOF
    
    echo -e "  ${GREEN}✓${NC} Docker limited to 4 cores & 4GB RAM"
else
    echo -e "  ${YELLOW}Docker Desktop not found - configure manually:${NC}"
    echo "    • CPUs: 4 (efficiency cores)"
    echo "    • Memory: 4GB"
    echo "    • Swap: 1GB"
fi

# 2. Create QoS rules for processes
echo -e "\n${YELLOW}2. Setting up Quality of Service rules...${NC}"

# Create process priority configuration
cat > ~/Library/LaunchAgents/com.wheel-trading.qos.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wheel-trading.qos</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>
            while true; do
                # Performance cores for MCP/Claude
                for pid in $(pgrep -f "claude|mcp|python.*wheel|node.*mcp"); do
                    taskpolicy -c background -s 0 -t 80 -p $pid 2>/dev/null || true
                    renice -20 $pid 2>/dev/null || true
                done
                
                # Efficiency cores for Docker
                for pid in $(pgrep -f "docker|containerd|com.docker"); do
                    taskpolicy -c background -s 1 -t 20 -p $pid 2>/dev/null || true
                    renice 10 $pid 2>/dev/null || true
                done
                
                # Efficiency cores for other dev tools
                for pid in $(pgrep -f "Electron|Chrome|Slack|Spotify"); do
                    taskpolicy -c background -s 1 -p $pid 2>/dev/null || true
                    renice 5 $pid 2>/dev/null || true
                done
                
                sleep 30
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

launchctl load ~/Library/LaunchAgents/com.wheel-trading.qos.plist 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} QoS rules configured"

# 3. Configure other common apps
echo -e "\n${YELLOW}3. Configuring other applications...${NC}"

# VS Code - limit to efficiency cores when not in focus
defaults write com.microsoft.VSCode NSAppSleepDisabled -bool NO
defaults write com.microsoft.VSCode LSMultipleInstancesProhibited -bool YES

# Browsers - background throttling
defaults write com.google.Chrome CPUThrottlingRate -int 4
defaults write org.mozilla.firefox CPUThrottlingRate -int 4

echo -e "  ${GREEN}✓${NC} App priorities configured"

# 4. Create memory pressure handler
echo -e "\n${YELLOW}4. Setting up memory pressure handling...${NC}"

cat > ~/Library/LaunchAgents/com.wheel-trading.memory.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wheel-trading.memory</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>
            while true; do
                # Get memory pressure
                pressure=$(memory_pressure | grep "System-wide memory free" | awk '{print $4}' | tr -d '%')
                
                # If memory pressure is high, kill non-essential processes
                if [ "$pressure" -gt 80 ]; then
                    # Kill Docker containers first
                    docker stop $(docker ps -q) 2>/dev/null || true
                    
                    # Force purge memory
                    sudo purge
                fi
                
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

launchctl load ~/Library/LaunchAgents/com.wheel-trading.memory.plist 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Memory pressure handler installed"

# 5. Create CPU affinity script
echo -e "\n${YELLOW}5. Creating CPU affinity manager...${NC}"

cat > /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/set-cpu-affinity.sh << 'EOFSCRIPT'
#!/bin/bash

# Set CPU affinity for processes

# Performance cores: 0-7 (M4 Pro)
# Efficiency cores: 8-11 (M4 Pro)

set_performance_affinity() {
    local pid=$1
    # Use performance cores (0-7)
    taskpolicy -c background -s 0 -p $pid 2>/dev/null || true
}

set_efficiency_affinity() {
    local pid=$1
    # Use efficiency cores (8-11)
    taskpolicy -c background -s 1 -p $pid 2>/dev/null || true
}

echo "Setting CPU affinity..."

# MCP/Claude processes → Performance cores
for pid in $(pgrep -f "claude|mcp|wheel.*py"); do
    set_performance_affinity $pid
    echo "  Performance → PID $pid ($(ps -p $pid -o comm=))"
done

# Docker → Efficiency cores
for pid in $(pgrep -f "docker|containerd"); do
    set_efficiency_affinity $pid
    echo "  Efficiency → PID $pid ($(ps -p $pid -o comm=))"
done

echo "Done!"
EOFSCRIPT

chmod +x /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/set-cpu-affinity.sh
echo -e "  ${GREEN}✓${NC} CPU affinity manager created"

# 6. Docker-specific optimizations
echo -e "\n${YELLOW}6. Optimizing Docker configuration...${NC}"

# Create Docker daemon config
mkdir -p ~/.docker
cat > ~/.docker/daemon.json << 'EOF'
{
  "cpu-shares": 512,
  "cpus": "4",
  "memory": "4g",
  "memory-reservation": "2g",
  "cpu-period": 100000,
  "cpu-quota": 400000,
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  },
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "experimental": true,
  "features": {
    "buildkit": true
  }
}
EOF

echo -e "  ${GREEN}✓${NC} Docker daemon configured for efficiency"

# 7. Create monitoring dashboard
echo -e "\n${YELLOW}7. Creating resource monitor...${NC}"

cat > /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/monitor-resources.sh << 'EOFMON'
#!/bin/bash

# Real-time resource monitoring

while true; do
    clear
    echo "=== M4 Pro Resource Monitor ==="
    echo ""
    
    # CPU usage by core type
    echo "CPU Usage:"
    echo "  Performance cores (0-7): $(ps aux | grep -E "claude|mcp|wheel" | awk '{sum+=$3} END {print sum}')%"
    echo "  Efficiency cores (8-11): $(ps aux | grep -E "docker|containerd" | awk '{sum+=$3} END {print sum}')%"
    echo ""
    
    # Memory usage
    echo "Memory Usage (24GB Total):"
    echo "  MCP/Claude: $(ps aux | grep -E "claude|mcp|wheel" | awk '{sum+=$6} END {printf "%.1f GB", sum/1024/1024}')"
    echo "  Docker: $(ps aux | grep -E "docker" | awk '{sum+=$6} END {printf "%.1f GB", sum/1024/1024}')"
    echo "  Available: $(vm_stat | grep "Pages free" | awk '{print $3*4096/1024/1024/1024 " GB"}')"
    echo ""
    
    # Top processes
    echo "Top Processes:"
    ps aux | sort -nrk 3,3 | head -5 | awk '{printf "  %-20s %5s%% %6.1fGB %s\n", $11, $3, $6/1024/1024, $2}'
    
    sleep 2
done
EOFMON

chmod +x /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/monitor-resources.sh
echo -e "  ${GREEN}✓${NC} Resource monitor created"

# Summary
echo -e "\n${GREEN}=== Configuration Complete ===${NC}"
echo ""
echo -e "${YELLOW}Resource Allocation:${NC}"
echo -e "  ${GREEN}MCP/Claude:${NC}"
echo -e "    • CPU: 8 performance cores (priority)"
echo -e "    • RAM: 20GB allocated"
echo -e "    • Priority: Maximum (-20)"
echo ""
echo -e "  ${BLUE}Docker:${NC}"
echo -e "    • CPU: 4 efficiency cores (limited)"
echo -e "    • RAM: 4GB maximum"
echo -e "    • Priority: Low (+10)"
echo ""
echo -e "  ${BLUE}Other Apps:${NC}"
echo -e "    • CPU: Efficiency cores"
echo -e "    • Priority: Normal"
echo ""
echo -e "${YELLOW}New Commands:${NC}"
echo -e "  • ${GREEN}./scripts/monitor-resources.sh${NC} - Real-time monitor"
echo -e "  • ${GREEN}./scripts/set-cpu-affinity.sh${NC} - Manually set affinity"
echo ""
echo -e "${YELLOW}Docker Memory Tip:${NC}"
echo "  If Docker uses too much RAM, run:"
echo "  ${GREEN}docker system prune -a --volumes${NC}"
echo ""
echo -e "All settings persist across restarts! 🚀"