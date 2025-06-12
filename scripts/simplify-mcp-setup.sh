#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Simplifying MCP Setup ===${NC}"

# 1. Create minimal configuration
echo -e "\n${YELLOW}1. Creating minimal MCP configuration...${NC}"
cat > mcp-servers-minimal.json << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem@latest", "/Users/mikeedwards"]
    },
    "github": {
      "transport": "stdio", 
      "command": "mcp-server-github",
      "args": [],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "python_analysis": {
      "transport": "stdio",
      "command": "/Users/mikeedwards/.pyenv/shims/python3",
      "args": [
        "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py"
      ]
    }
  }
}
EOF

# 2. Create single launcher script
echo -e "\n${YELLOW}2. Creating unified launcher...${NC}"
cat > start-claude.sh << 'EOF'
#!/bin/bash

# Simple Claude launcher with minimal MCP servers

set -e

# Find Claude command
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
else
    echo "Error: Claude command not found"
    exit 1
fi

# Configuration options
CONFIG_FILE="${1:-mcp-servers-minimal.json}"
DEBUG_MODE="${MCP_DEBUG:-0}"

# Show what we're doing
echo "Starting Claude with MCP servers"
echo "Configuration: $CONFIG_FILE"
[ "$DEBUG_MODE" = "1" ] && echo "Debug mode: ON"

# Launch Claude
if [ "$DEBUG_MODE" = "1" ]; then
    MCP_DEBUG=1 $CLAUDE_CMD --mcp-config "$CONFIG_FILE"
else
    $CLAUDE_CMD --mcp-config "$CONFIG_FILE"
fi
EOF
chmod +x start-claude.sh

# 3. Create MCP doctor script
echo -e "\n${YELLOW}3. Creating MCP diagnostic tool...${NC}"
cat > mcp-doctor.py << 'EOF'
#!/usr/bin/env python3
"""Simple MCP server diagnostic tool"""

import json
import subprocess
import shutil
import os

def check_server(name, config):
    """Test if a server can start and respond to init"""
    cmd = config["command"]
    args = config.get("args", [])
    
    # Check if command exists
    if cmd == "npx":
        # NPX packages download on demand, assume OK
        return True, "NPX package (downloads on first use)"
    
    if not shutil.which(cmd):
        return False, f"Command not found: {cmd}"
    
    # For Python scripts, check if file exists
    if cmd.endswith("python3") and args:
        script = args[0]
        if not os.path.exists(script):
            return False, f"Script not found: {script}"
    
    # Try to run with init message
    full_cmd = [cmd] + args
    init_msg = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"doctor","version":"1.0.0"}}}'
    
    try:
        result = subprocess.run(
            full_cmd,
            input=init_msg,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if '"result"' in result.stdout:
            return True, "Server responds correctly"
        else:
            return False, f"Invalid response: {result.stderr[:100]}"
            
    except subprocess.TimeoutExpired:
        return False, "Server timeout (5s)"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"

def main():
    """Check all configured MCP servers"""
    print("MCP Server Health Check\n")
    
    # Load configuration
    with open("mcp-servers-minimal.json") as f:
        config = json.load(f)
    
    servers = config["mcpServers"]
    working = 0
    total = len(servers)
    
    for name, server_config in servers.items():
        ok, message = check_server(name, server_config)
        status = "✓" if ok else "✗"
        color = "\033[0;32m" if ok else "\033[0;31m"
        nc = "\033[0m"
        
        print(f"{color}{status}{nc} {name}: {message}")
        if ok:
            working += 1
    
    print(f"\nSummary: {working}/{total} servers ready")
    
    if working < total:
        print("\nTroubleshooting:")
        print("1. Install missing commands (npm install -g ...)")
        print("2. Check file paths in configuration")
        print("3. Set required environment variables")
        print("4. Run with MCP_DEBUG=1 for more info")

if __name__ == "__main__":
    main()
EOF
chmod +x mcp-doctor.py

# 4. Create living documentation
echo -e "\n${YELLOW}4. Creating documentation...${NC}"
cat > MCP_SETUP.md << 'EOF'
# MCP Server Setup

## Philosophy
- Start minimal, add only what you need
- Each server should work independently  
- Clear error messages over silent failures

## Core Servers (3)

### filesystem
- **Purpose**: File operations
- **Test**: `npx @modelcontextprotocol/server-filesystem@latest --help`
- **Troubleshoot**: Requires npm/npx installed

### github
- **Purpose**: GitHub operations
- **Test**: `mcp-server-github --help`
- **Troubleshoot**: 
  - Install: `npm install -g @modelcontextprotocol/server-github`
  - Set: `export GITHUB_TOKEN=your_token`

### python_analysis
- **Purpose**: Trading bot analysis
- **Test**: `python3 scripts/python-mcp-server.py --test`
- **Troubleshoot**: Requires `pip install mcp`

## Usage

```bash
# Start with minimal servers
./start-claude.sh

# Debug mode
MCP_DEBUG=1 ./start-claude.sh

# Check server health
./mcp-doctor.py

# Use full configuration if needed
./start-claude.sh mcp-servers.json
```

## Adding New Servers

Only add a server if you:
1. Actually need its functionality
2. Have tested it works standalone
3. Understand its dependencies

## Troubleshooting

Run `./mcp-doctor.py` first. It will tell you exactly what's wrong.
EOF

# 5. Clean up old scripts (optional)
echo -e "\n${YELLOW}5. Old scripts to remove (manual action):${NC}"
echo "Consider removing these redundant scripts:"
ls scripts/start-claude-*.sh scripts/fix-*.sh scripts/setup-*.sh 2>/dev/null | head -20

echo -e "\n${GREEN}=== Simplification Complete ===${NC}"
echo ""
echo "New setup:"
echo "  • 3 core MCP servers (down from 17-19)"
echo "  • 1 launcher script (down from 10+)" 
echo "  • 1 diagnostic tool (replaces multiple fix scripts)"
echo "  • Clear documentation"
echo ""
echo "Usage:"
echo "  ./start-claude.sh           # Start with minimal servers"
echo "  ./mcp-doctor.py            # Check server health"
echo "  cat MCP_SETUP.md           # Read documentation"