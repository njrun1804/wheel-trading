#!/bin/bash
# System Optimization Fix Script
# Addresses critical issues found in system analysis

echo "🔧 Starting System Optimization Fixes..."

# 1. CRITICAL: Disable failing trading service
echo "📌 Step 1: Disabling failing trading service..."
launchctl unload ~/Library/LaunchAgents/com.unity.trading.daily-updater.plist 2>/dev/null || echo "Service already unloaded"

# 2. CRITICAL: Disable dangerous memory management service
echo "📌 Step 2: Disabling dangerous memory management service..."
launchctl unload ~/Library/LaunchAgents/com.wheel-trading.memory.plist 2>/dev/null || echo "Service already unloaded"

# 3. OPTIMIZE: Reduce QoS polling frequency
echo "📌 Step 3: Optimizing QoS service..."
if [ -f ~/Library/LaunchAgents/com.wheel-trading.qos.plist ]; then
    # Unload current service
    launchctl unload ~/Library/LaunchAgents/com.wheel-trading.qos.plist 2>/dev/null
    
    # Create backup
    cp ~/Library/LaunchAgents/com.wheel-trading.qos.plist ~/Library/LaunchAgents/com.wheel-trading.qos.plist.backup
    
    # Update with safer configuration
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
                # Check if processes exist before modifying them
                if pgrep -f "claude|mcp|python.*wheel|node.*mcp" > /dev/null; then
                    for pid in $(pgrep -f "claude|mcp|python.*wheel|node.*mcp"); do
                        # Check if process still exists before applying taskpolicy
                        if kill -0 "$pid" 2>/dev/null; then
                            taskpolicy -c background -s 0 -t 80 -p $pid 2>/dev/null || true
                            renice -10 $pid 2>/dev/null || true  # Less aggressive renice
                        fi
                    done
                fi
                
                # Reduced frequency and gentler management for other processes
                if pgrep -f "docker|containerd|com.docker" > /dev/null; then
                    for pid in $(pgrep -f "docker|containerd|com.docker"); do
                        if kill -0 "$pid" 2>/dev/null; then
                            taskpolicy -c background -s 1 -t 20 -p $pid 2>/dev/null || true
                            renice 5 $pid 2>/dev/null || true  # Less aggressive renice
                        fi
                    done
                fi
                
                sleep 60  # Reduced frequency from 30 to 60 seconds
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
    
    # Reload with new configuration
    launchctl load ~/Library/LaunchAgents/com.wheel-trading.qos.plist
    echo "✅ QoS service updated with safer configuration"
else
    echo "⚠️  QoS service file not found"
fi

# 4. CHECK: Display current service status
echo "📌 Step 4: Checking service status..."
echo "Active trading-related services:"
launchctl list | grep -E "(com\.unity|com\.wheel|trading)" || echo "No trading services currently active"

echo "Failed services:"
failed_count=$(launchctl list | grep -E "\-[0-9]" | wc -l)
echo "Total services with exit codes: $failed_count"

# 5. MONITOR: Create monitoring script
echo "📌 Step 5: Creating monitoring script..."
cat > ~/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/monitor_system_health.sh << 'EOF'
#!/bin/bash
# System Health Monitor
echo "=== System Health Report ==="
echo "Date: $(date)"
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
echo "Active Services: $(launchctl list | grep -v "^-" | wc -l)"
echo "Failed Services: $(launchctl list | grep -E "\-[0-9]" | wc -l)"
echo "Memory Free: $(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')  pages"
echo "Top CPU Processes:"
ps aux | sort -nrk 3,3 | head -5
echo "=========================="
EOF

chmod +x ~/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/monitor_system_health.sh

echo "✅ System optimization fixes completed!"
echo ""
echo "📊 SUMMARY:"
echo "- Disabled failing com.unity.trading.daily-updater service"
echo "- Disabled dangerous memory management service (sudo purge)"
echo "- Updated QoS service with safer configuration and reduced polling"
echo "- Created system health monitoring script"
echo ""
echo "🔄 NEXT STEPS:"
echo "1. Run the monitoring script: ./monitor_system_health.sh"
echo "2. Monitor system load over the next hour"
echo "3. Check for improved terminal stability"
echo ""
echo "📋 If issues persist, review the full analysis:"
echo "   - /Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/SYSTEM_OPTIMIZATION_ANALYSIS.md"