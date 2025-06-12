#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Ensuring MCP Servers Work Everywhere ===${NC}"

# 1. Install all Python MCP packages globally
echo -e "\n${YELLOW}Installing Python MCP packages globally...${NC}"
/Users/mikeedwards/.pyenv/shims/pip install --upgrade \
    mcp \
    mcp-server-github \
    mcp-server-stats \
    mcp-server-duckdb \
    mcp-py-repl \
    mlflow \
    scikit-learn \
    pandas \
    numpy

# Also install in system Python if different
if command -v python3 &> /dev/null; then
    python3 -m pip install --upgrade \
        mcp \
        mcp-server-github \
        mcp-server-stats \
        mcp-server-duckdb \
        mcp-py-repl 2>/dev/null || true
fi

# 2. Create symlinks for Python scripts
echo -e "\n${YELLOW}Creating global MCP server symlinks...${NC}"
sudo mkdir -p /usr/local/bin/mcp-servers

# Link Python MCP scripts
sudo ln -sf "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py" /usr/local/bin/mcp-servers/python-analysis
sudo ln -sf "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-mcp-server.py" /usr/local/bin/mcp-servers/trace

# 3. Create wrapper scripts for community servers
echo -e "\n${YELLOW}Creating wrapper scripts for community servers...${NC}"

# MLflow wrapper
cat > /tmp/mlflow-mcp-wrapper.sh << 'EOF'
#!/bin/bash
cd /Users/mikeedwards/mcp-servers/community/mlflowMCPServer
/Users/mikeedwards/.pyenv/shims/python3 mlflow_server.py "$@"
EOF
sudo mv /tmp/mlflow-mcp-wrapper.sh /usr/local/bin/mcp-servers/mlflow
sudo chmod +x /usr/local/bin/mcp-servers/mlflow

# Sklearn wrapper
cat > /tmp/sklearn-mcp-wrapper.sh << 'EOF'
#!/bin/bash
cd /Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn
PYTHONPATH=src /Users/mikeedwards/.pyenv/shims/python3 src/mcp_server_scikit_learn/server.py "$@"
EOF
sudo mv /tmp/sklearn-mcp-wrapper.sh /usr/local/bin/mcp-servers/sklearn
sudo chmod +x /usr/local/bin/mcp-servers/sklearn

# Optionsflow wrapper
cat > /tmp/optionsflow-mcp-wrapper.sh << 'EOF'
#!/bin/bash
cd /Users/mikeedwards/mcp-servers/community/mcp-optionsflow
/Users/mikeedwards/.pyenv/shims/python3 optionsflow.py "$@"
EOF
sudo mv /tmp/optionsflow-mcp-wrapper.sh /usr/local/bin/mcp-servers/optionsflow
sudo chmod +x /usr/local/bin/mcp-servers/optionsflow

# 4. Update mcp-servers.json to use simpler paths
echo -e "\n${YELLOW}Updating MCP configuration for reliability...${NC}"
python3 << 'EOF'
import json
import os

config_path = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Simplify Python server commands
updates = {
    "mlflow": {
        "command": "/usr/local/bin/mcp-servers/mlflow",
        "args": []
    },
    "sklearn": {
        "command": "/usr/local/bin/mcp-servers/sklearn", 
        "args": []
    },
    "optionsflow": {
        "command": "/usr/local/bin/mcp-servers/optionsflow",
        "args": []
    },
    "python_analysis": {
        "command": "/usr/local/bin/mcp-servers/python-analysis",
        "args": []
    },
    "trace": {
        "command": "/usr/local/bin/mcp-servers/trace",
        "args": []
    }
}

for server, update in updates.items():
    if server in config["mcpServers"]:
        config["mcpServers"][server]["command"] = update["command"]
        config["mcpServers"][server]["args"] = update["args"]

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
    f.write('\n')

print("✓ Configuration updated")
EOF

# 5. Pre-install NPM packages globally
echo -e "\n${YELLOW}Installing Node.js MCP servers globally...${NC}"
npm install -g \
    @modelcontextprotocol/server-filesystem \
    @modelcontextprotocol/server-brave-search \
    @modelcontextprotocol/server-memory \
    @modelcontextprotocol/server-sequential-thinking \
    @modelcontextprotocol/server-puppeteer \
    @modelcontextprotocol/server-ripgrep \
    @modelcontextprotocol/server-code-analysis 2>/dev/null || true

# 6. Create a universal MCP config
echo -e "\n${YELLOW}Creating universal MCP configuration...${NC}"
cp mcp-servers.json ~/.mcp-servers-universal.json

echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo -e "MCP servers are now configured to work in any Claude session."
echo -e "\nTo use in a new Claude session:"
echo -e "  ${YELLOW}claude --mcp-config ~/.mcp-servers-universal.json${NC}"
echo -e "\nOr copy the config:"
echo -e "  ${YELLOW}cp ~/.mcp-servers-universal.json [your-project]/mcp-servers.json${NC}"