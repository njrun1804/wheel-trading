#!/bin/bash

# Setup Advanced Observability for Claude Code + VS Code

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Advanced Observability Setup ===${NC}"
echo ""

# 1. Install OpenTelemetry Collector (if available)
echo -e "${YELLOW}1. Installing OpenTelemetry components...${NC}"

# Check if logfire-otel-collector exists in brew
if brew list --cask | grep -q "logfire-otel-collector"; then
    echo -e "  ${GREEN}✓${NC} logfire-otel-collector already installed"
else
    echo -e "  ${BLUE}Checking for logfire-otel-collector...${NC}"
    brew search logfire-otel-collector 2>/dev/null || echo -e "  ${YELLOW}⚠${NC} logfire-otel-collector not found in Homebrew"
fi

# Install Python OpenTelemetry packages
echo -e "  ${BLUE}Installing Python telemetry packages...${NC}"
pip install opik logfire phoenix-opentelemetry opentelemetry-distro opentelemetry-exporter-otlp

echo -e "  ${GREEN}✓${NC} Python telemetry packages installed"

# 2. Setup OpenTelemetry environment variables
echo -e "\n${YELLOW}2. Configuring OpenTelemetry environment...${NC}"

OTEL_CONFIG="
# OpenTelemetry Configuration for Claude Code
export OTEL_EXPORTER_OTLP_ENDPOINT='http://127.0.0.1:4318'
export OTEL_PYTHON_LOG_LEVEL='INFO'
export OTEL_SERVICE_NAME='wheel-trading-bot'
export OTEL_TRACES_EXPORTER='otlp'
export OTEL_METRICS_EXPORTER='otlp'
export OTEL_LOGS_EXPORTER='otlp'
"

# Check if already in .zshrc
if grep -q "OTEL_EXPORTER_OTLP_ENDPOINT" ~/.zshrc; then
    echo -e "  ${GREEN}✓${NC} OpenTelemetry already configured in .zshrc"
else
    echo -e "  ${BLUE}Adding to .zshrc...${NC}"
    echo "$OTEL_CONFIG" >> ~/.zshrc
    echo -e "  ${GREEN}✓${NC} Added OpenTelemetry config to .zshrc"
fi

# 3. Create VS Code tasks.json
echo -e "\n${YELLOW}3. Creating VS Code integration...${NC}"

VSCODE_DIR="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/.vscode"
mkdir -p "$VSCODE_DIR"

cat > "$VSCODE_DIR/tasks.json" << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Claude Code with Telemetry",
            "type": "shell",
            "command": "claude",
            "args": [
                "--mcp-config", "${workspaceFolder}/mcp-servers-final.json",
                "--verbose"
            ],
            "problemMatcher": [],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": true,
                "panel": "new"
            }
        },
        {
            "label": "Start Local Trace Servers",
            "type": "shell",
            "command": "${workspaceFolder}/scripts/setup-local-trace-servers.sh",
            "problemMatcher": [],
            "group": "build"
        },
        {
            "label": "Check MCP Status",
            "type": "shell",
            "command": "${workspaceFolder}/scripts/check-mcp-status.sh",
            "problemMatcher": [],
            "group": "test"
        }
    ]
}
EOF
echo -e "  ${GREEN}✓${NC} Created VS Code tasks.json"

# 4. Create VS Code extensions.json
cat > "$VSCODE_DIR/extensions.json" << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.debugpy",
        "redhat.vscode-yaml",
        "ms-azuretools.vscode-docker"
    ]
}
EOF
echo -e "  ${GREEN}✓${NC} Created VS Code extension recommendations"

# 5. Create LaunchAgent for OTEL collector (if using standard otel-collector)
echo -e "\n${YELLOW}4. Creating LaunchAgent for collector...${NC}"

LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_AGENT_DIR"

cat > "$LAUNCH_AGENT_DIR/com.wheel-trading.otel-collector.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wheel-trading.otel-collector</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/otelcol</string>
        <string>--config</string>
        <string>$HOME/.config/otel-collector/config.yaml</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/otel-collector.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/otel-collector.error.log</string>
</dict>
</plist>
EOF

echo -e "  ${GREEN}✓${NC} Created LaunchAgent (not loaded yet)"

# 6. Create OTEL collector config
echo -e "\n${YELLOW}5. Creating OTEL collector configuration...${NC}"

OTEL_CONFIG_DIR="$HOME/.config/otel-collector"
mkdir -p "$OTEL_CONFIG_DIR"

cat > "$OTEL_CONFIG_DIR/config.yaml" << 'EOF'
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 127.0.0.1:4317
      http:
        endpoint: 127.0.0.1:4318

processors:
  batch:

exporters:
  # Export to Logfire
  otlp/logfire:
    endpoint: "otel.logfire.io:443"
    headers:
      "Authorization": "Bearer ${LOGFIRE_TOKEN}"
  
  # Export to local Phoenix
  otlp/phoenix:
    endpoint: "127.0.0.1:6006"
    tls:
      insecure: true
  
  # Console debugging
  logging:
    loglevel: info

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/logfire, otlp/phoenix, logging]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/logfire, otlp/phoenix]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/logfire, logging]
EOF

echo -e "  ${GREEN}✓${NC} Created OTEL collector config"

# Summary
echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo ""
echo -e "${YELLOW}What was configured:${NC}"
echo -e "  ✓ OpenTelemetry Python packages"
echo -e "  ✓ Environment variables in .zshrc"
echo -e "  ✓ VS Code tasks for Claude integration"
echo -e "  ✓ LaunchAgent for collector (optional)"
echo -e "  ✓ OTEL collector configuration"
echo ""
echo -e "${YELLOW}To activate:${NC}"
echo -e "  1. Restart terminal or run: ${GREEN}source ~/.zshrc${NC}"
echo -e "  2. In VS Code: ${GREEN}Cmd+Shift+P → Tasks: Run Task → Claude Code with Telemetry${NC}"
echo -e "  3. (Optional) Load collector: ${GREEN}launchctl load ~/Library/LaunchAgents/com.wheel-trading.otel-collector.plist${NC}"
echo ""
echo -e "${YELLOW}Benefits:${NC}"
echo -e "  • Automatic telemetry collection from Python code"
echo -e "  • VS Code integration for quick launches"
echo -e "  • Persistent collector that survives reboots"
echo -e "  • Exports to both Logfire and local Phoenix"
echo ""