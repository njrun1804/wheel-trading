#!/bin/bash
# Setup Phoenix tracing for MCP servers

echo "🔥 Setting up Phoenix tracing..."

# Check if Phoenix is installed
if ! pip show arize-phoenix > /dev/null 2>&1; then
    echo "Installing Phoenix..."
    pip install arize-phoenix
fi

# Create Phoenix config directory
PHOENIX_DIR="$HOME/.phoenix"
mkdir -p "$PHOENIX_DIR"

# Create Phoenix config
cat > "$PHOENIX_DIR/config.yaml" << EOF
# Phoenix Configuration
host: 0.0.0.0
port: 6006
storage:
  type: sqlite
  path: $PHOENIX_DIR/phoenix.db
telemetry:
  enabled: true
EOF

# Create systemd service (if on Linux/macOS with systemd)
if command -v systemctl > /dev/null 2>&1; then
    echo "Creating systemd service..."
    cat > /tmp/phoenix.service << EOF
[Unit]
Description=Phoenix Observability Platform
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME
ExecStart=/usr/bin/python -m phoenix.server
Restart=always
Environment="PHOENIX_CONFIG=$PHOENIX_DIR/config.yaml"

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv /tmp/phoenix.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable phoenix
    sudo systemctl start phoenix
else
    echo "Systemd not available - creating start script instead"
    cat > "$PHOENIX_DIR/start-phoenix.sh" << 'EOF'
#!/bin/bash
# Start Phoenix in background
nohup python -m phoenix.server > $HOME/.phoenix/phoenix.log 2>&1 &
echo $! > $HOME/.phoenix/phoenix.pid
echo "Phoenix started with PID $(cat $HOME/.phoenix/phoenix.pid)"
EOF
    chmod +x "$PHOENIX_DIR/start-phoenix.sh"
fi

echo "✅ Phoenix tracing setup complete!"
echo "Access Phoenix UI at: http://localhost:6006"
