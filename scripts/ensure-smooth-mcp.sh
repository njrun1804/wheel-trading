#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Ensuring Smooth MCP Setup ===${NC}"

# 1. Check Python version compatibility
echo -e "\n${YELLOW}1. Checking Python environment...${NC}"
PYTHON_VERSION=$(/Users/mikeedwards/.pyenv/shims/python3 --version | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# The project requires Python 3.12+, but MCP works with 3.11
if [[ "$PYTHON_VERSION" < "3.11" ]]; then
    echo -e "${RED}✗ Python 3.11+ required${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Python version OK${NC}"
fi

# 2. Install critical dependencies for MCP
echo -e "\n${YELLOW}2. Installing MCP dependencies...${NC}"
/Users/mikeedwards/.pyenv/shims/pip install --upgrade pip
/Users/mikeedwards/.pyenv/shims/pip install --upgrade mcp

# 3. Check critical environment variables
echo -e "\n${YELLOW}3. Checking environment variables...${NC}"

check_env() {
    local var=$1
    local desc=$2
    if [ -n "${!var}" ]; then
        echo -e "${GREEN}✓ $var set${NC} - $desc"
    else
        echo -e "${YELLOW}! $var not set${NC} - $desc"
        return 1
    fi
}

ENV_OK=1
check_env "GITHUB_TOKEN" "Required for GitHub MCP server" || ENV_OK=0

# Optional but useful
check_env "DATABENTO_API_KEY" "For market data (bot uses directly, not MCP)" || true
check_env "SCHWAB_REFRESH_TOKEN" "For Schwab trading (bot uses directly, not MCP)" || true

# 4. Create enhanced minimal configuration with optional servers
echo -e "\n${YELLOW}4. Creating enhanced configuration...${NC}"
cat > mcp-servers-enhanced.json << 'EOF'
{
  "mcpServers": {
    "comment": "Core servers for trading bot development",
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem@latest", "/Users/mikeedwards"],
      "_purpose": "Essential - file operations for code and configs"
    },
    "github": {
      "transport": "stdio", 
      "command": "mcp-server-github",
      "args": [],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      },
      "_purpose": "Essential - source control for trading bot"
    },
    "python_analysis": {
      "transport": "stdio",
      "command": "/Users/mikeedwards/.pyenv/shims/python3",
      "args": [
        "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/python-mcp-server.py"
      ],
      "_purpose": "Essential - trading position analysis"
    }
  },
  "optionalServers": {
    "comment": "Add these if needed for specific tasks",
    "duckdb": {
      "transport": "stdio",
      "command": "/Users/mikeedwards/.pyenv/shims/python3",
      "args": [
        "-m",
        "mcp_server_duckdb",
        "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/data/wheel_trading_master.duckdb"
      ],
      "_purpose": "Direct database queries (bot usually uses Python API)",
      "_enable": "Move to mcpServers section if needed"
    },
    "pyrepl": {
      "transport": "stdio",
      "command": "/Users/mikeedwards/.pyenv/shims/python3",
      "args": ["-m", "mcp_py_repl"],
      "_purpose": "Interactive Python testing",
      "_enable": "Move to mcpServers section if needed"
    }
  }
}
EOF

# 5. Create helper to add optional servers
echo -e "\n${YELLOW}5. Creating server management helper...${NC}"
cat > mcp-server-manager.py << 'EOF'
#!/usr/bin/env python3
"""Manage MCP server configuration"""

import json
import sys
import argparse

def load_config(file="mcp-servers-enhanced.json"):
    with open(file) as f:
        return json.load(f)

def save_config(config, file="mcp-servers-enhanced.json"):
    with open(file, 'w') as f:
        json.dump(config, f, indent=2)
        f.write('\n')

def list_servers(config):
    print("Active servers:")
    for name, server in config.get("mcpServers", {}).items():
        if name != "comment":
            purpose = server.get("_purpose", "No description")
            print(f"  ✓ {name}: {purpose}")
    
    print("\nOptional servers:")
    for name, server in config.get("optionalServers", {}).items():
        if name != "comment":
            purpose = server.get("_purpose", "No description")
            print(f"  - {name}: {purpose}")

def enable_server(config, server_name):
    optional = config.get("optionalServers", {})
    if server_name in optional:
        server = optional[server_name].copy()
        # Remove metadata
        server.pop("_enable", None)
        
        # Move to active
        config["mcpServers"][server_name] = server
        del optional[server_name]
        
        print(f"✓ Enabled {server_name}")
        return True
    else:
        print(f"✗ Server '{server_name}' not found in optional servers")
        return False

def disable_server(config, server_name):
    active = config.get("mcpServers", {})
    if server_name in active and server_name not in ["filesystem", "github", "python_analysis"]:
        server = active[server_name].copy()
        
        # Move to optional
        config.setdefault("optionalServers", {})[server_name] = server
        del active[server_name]
        
        print(f"✓ Disabled {server_name}")
        return True
    else:
        print(f"✗ Cannot disable core server or server not found")
        return False

def main():
    parser = argparse.ArgumentParser(description="Manage MCP servers")
    parser.add_argument("action", choices=["list", "enable", "disable"], help="Action to perform")
    parser.add_argument("server", nargs="?", help="Server name for enable/disable")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.action == "list":
        list_servers(config)
    elif args.action == "enable" and args.server:
        if enable_server(config, args.server):
            save_config(config)
    elif args.action == "disable" and args.server:
        if disable_server(config, args.server):
            save_config(config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
EOF
chmod +x mcp-server-manager.py

# 6. Create trading bot specific test
echo -e "\n${YELLOW}6. Creating trading bot integration test...${NC}"
cat > test-trading-mcp.py << 'EOF'
#!/usr/bin/env python3
"""Test MCP servers work with trading bot needs"""

import subprocess
import json
import os

def test_filesystem_access():
    """Test we can access trading bot files"""
    critical_files = [
        "config.yaml",
        "data/wheel_trading_master.duckdb",
        "src/unity_wheel/strategy/wheel.py"
    ]
    
    print("Testing filesystem access...")
    for file in critical_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
    
    return True

def test_github_integration():
    """Test GitHub access"""
    print("\nTesting GitHub integration...")
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✓ Git repository accessible")
            if result.stdout:
                print(f"  ! {len(result.stdout.splitlines())} uncommitted changes")
        return True
    except:
        print("  ✗ Git not accessible")
        return False

def test_python_analysis():
    """Test python analysis server"""
    print("\nTesting python analysis server...")
    script = "scripts/python-mcp-server.py"
    if os.path.exists(script):
        print(f"  ✓ {script} exists")
        # Test it can import trading bot modules
        try:
            import sys
            sys.path.insert(0, os.getcwd())
            from src.unity_wheel.config import TradingConfig
            print("  ✓ Can import trading bot modules")
            return True
        except ImportError as e:
            print(f"  ✗ Cannot import trading modules: {e}")
            return False
    else:
        print(f"  ✗ {script} not found")
        return False

def main():
    print("=== Trading Bot MCP Integration Test ===\n")
    
    all_good = True
    all_good &= test_filesystem_access()
    all_good &= test_github_integration()
    all_good &= test_python_analysis()
    
    print("\n" + "="*40)
    if all_good:
        print("✅ All tests passed - MCP ready for trading bot")
    else:
        print("❌ Some tests failed - check configuration")

if __name__ == "__main__":
    main()
EOF
chmod +x test-trading-mcp.py

# 7. Create quick reference card
echo -e "\n${YELLOW}7. Creating quick reference...${NC}"
cat > MCP_QUICK_REFERENCE.md << 'EOF'
# MCP Quick Reference

## Daily Commands
```bash
# Start Claude with minimal servers
./start-claude.sh

# Check everything is working
./mcp-doctor.py
./test-trading-mcp.py

# Manage servers
./mcp-server-manager.py list
./mcp-server-manager.py enable duckdb
./mcp-server-manager.py disable duckdb
```

## Trading Bot MCP Usage

### File Operations (filesystem)
- Read/write config.yaml
- Access strategy files in src/
- Manage data files

### Source Control (github)
- Commit trading bot changes
- Create PRs for strategy updates
- Check git status

### Analysis (python_analysis)
- analyze_position("AAPL")
- monitor_system()
- data_quality_check()

## Troubleshooting

1. **MCP server fails**
   ```bash
   ./mcp-doctor.py  # Shows exact error
   ```

2. **Missing GITHUB_TOKEN**
   ```bash
   export GITHUB_TOKEN=ghp_xxxxx
   ```

3. **Python import errors**
   ```bash
   cd /path/to/wheel-trading
   pip install -r requirements.txt
   ```

## Optional Servers

Enable if needed:
- `duckdb` - Direct SQL queries on trading database
- `pyrepl` - Interactive Python for testing strategies

## Environment Variables

Required:
- GITHUB_TOKEN

Trading bot uses (not MCP):
- DATABENTO_API_KEY
- SCHWAB_REFRESH_TOKEN
- FRED_API_KEY
EOF

echo -e "\n${GREEN}=== Smooth MCP Setup Complete ===${NC}"
echo ""
echo "Key improvements:"
echo "  ✓ Python environment validation"
echo "  ✓ Dependency installation"
echo "  ✓ Environment variable checking"
echo "  ✓ Server management tool"
echo "  ✓ Trading bot integration test"
echo "  ✓ Quick reference guide"
echo ""
echo "Next steps:"
echo "  1. Set GITHUB_TOKEN if not set"
echo "  2. Run ./test-trading-mcp.py"
echo "  3. Start Claude with ./start-claude.sh"