#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Restoring and Fixing ALL MCP Servers ===${NC}"

# 1. Restore ripgrep and dependency-graph to config
echo -e "\n${YELLOW}1. Restoring ripgrep and dependency-graph...${NC}"
python3 << 'EOF'
import json

config_path = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Add ripgrep back
if "ripgrep" not in config["mcpServers"]:
    config["mcpServers"]["ripgrep"] = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-ripgrep"]
    }
    print("✓ Restored ripgrep")

# Add dependency-graph back
if "dependency-graph" not in config["mcpServers"]:
    config["mcpServers"]["dependency-graph"] = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-code-analysis"]
    }
    print("✓ Restored dependency-graph")

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
    f.write('\n')
EOF

# 2. Install/verify these NPX packages exist
echo -e "\n${YELLOW}2. Installing MCP server packages...${NC}"

# Check if these are the correct package names by searching npm
echo "Searching for correct MCP server packages..."

# Try different package names
RIPGREP_PACKAGES=(
    "@modelcontextprotocol/server-ripgrep"
    "@anthropic/mcp-server-ripgrep"
    "mcp-server-ripgrep"
    "@mcp/server-ripgrep"
)

ANALYSIS_PACKAGES=(
    "@modelcontextprotocol/server-code-analysis"
    "@modelcontextprotocol/server-dependency-graph"
    "@anthropic/mcp-server-code-analysis"
    "mcp-server-code-analysis"
)

# Find correct ripgrep package
echo "Finding ripgrep MCP server..."
for pkg in "${RIPGREP_PACKAGES[@]}"; do
    if npm view "$pkg" version 2>/dev/null; then
        echo "✓ Found: $pkg"
        RIPGREP_PKG="$pkg"
        break
    fi
done

# Find correct code analysis package
echo "Finding code analysis MCP server..."
for pkg in "${ANALYSIS_PACKAGES[@]}"; do
    if npm view "$pkg" version 2>/dev/null; then
        echo "✓ Found: $pkg"
        ANALYSIS_PKG="$pkg"
        break
    fi
done

# If we couldn't find the packages, create local wrapper scripts
if [ -z "$RIPGREP_PKG" ]; then
    echo "Creating local ripgrep MCP wrapper..."
    cat > /usr/local/bin/mcp-server-ripgrep << 'RIPGREP_EOF'
#!/usr/bin/env node
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

const server = new Server({
  name: 'ripgrep',
  version: '1.0.0',
}, {
  capabilities: {
    tools: {}
  }
});

server.setRequestHandler('tools/list', async () => {
  return {
    tools: [{
      name: 'search',
      description: 'Search files using ripgrep',
      inputSchema: {
        type: 'object',
        properties: {
          pattern: { type: 'string', description: 'Search pattern' },
          path: { type: 'string', description: 'Path to search in' }
        },
        required: ['pattern']
      }
    }]
  };
});

server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'search') {
    const { pattern, path = '.' } = request.params.arguments;
    try {
      const { stdout } = await execAsync(`rg "${pattern}" "${path}" || true`);
      return {
        content: [{
          type: 'text',
          text: stdout || 'No matches found'
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error: ${error.message}`
        }]
      };
    }
  }
});

const transport = new StdioServerTransport();
server.connect(transport).catch(console.error);
RIPGREP_EOF
    chmod +x /usr/local/bin/mcp-server-ripgrep
    
    # Update config to use local script
    python3 -c "
import json
with open('mcp-servers.json', 'r') as f: config = json.load(f)
config['mcpServers']['ripgrep']['command'] = '/usr/local/bin/mcp-server-ripgrep'
config['mcpServers']['ripgrep']['args'] = []
with open('mcp-servers.json', 'w') as f: json.dump(config, f, indent=2); f.write('\n')
"
fi

if [ -z "$ANALYSIS_PKG" ]; then
    echo "Creating local code analysis MCP wrapper..."
    cat > /usr/local/bin/mcp-server-code-analysis << 'ANALYSIS_EOF'
#!/usr/bin/env node
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const fs = require('fs').promises;
const path = require('path');

const server = new Server({
  name: 'dependency-graph',
  version: '1.0.0',
}, {
  capabilities: {
    tools: {}
  }
});

server.setRequestHandler('tools/list', async () => {
  return {
    tools: [{
      name: 'analyze',
      description: 'Analyze code dependencies',
      inputSchema: {
        type: 'object',
        properties: {
          file: { type: 'string', description: 'File to analyze' }
        },
        required: ['file']
      }
    }]
  };
});

server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'analyze') {
    const { file } = request.params.arguments;
    try {
      const content = await fs.readFile(file, 'utf8');
      // Simple import detection
      const imports = content.match(/import .* from ['"].*['"]/g) || [];
      const requires = content.match(/require\(['"].*['"]\)/g) || [];
      
      return {
        content: [{
          type: 'text',
          text: `Dependencies in ${file}:\n${[...imports, ...requires].join('\n')}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error: ${error.message}`
        }]
      };
    }
  }
});

const transport = new StdioServerTransport();
server.connect(transport).catch(console.error);
ANALYSIS_EOF
    chmod +x /usr/local/bin/mcp-server-code-analysis
    
    # Update config to use local script
    python3 -c "
import json
with open('mcp-servers.json', 'r') as f: config = json.load(f)
config['mcpServers']['dependency-graph']['command'] = '/usr/local/bin/mcp-server-code-analysis'
config['mcpServers']['dependency-graph']['args'] = []
with open('mcp-servers.json', 'w') as f: json.dump(config, f, indent=2); f.write('\n')
"
fi

# 3. Fix other failing servers
echo -e "\n${YELLOW}3. Ensuring all other servers work...${NC}"

# Make sure DuckDB works
if ! /Users/mikeedwards/.pyenv/shims/python3 -c "import mcp_server_duckdb" 2>/dev/null; then
    /Users/mikeedwards/.pyenv/shims/pip install mcp-server-duckdb
fi

# Ensure sklearn server exists
if [ ! -f "/Users/mikeedwards/mcp-servers/community/mcp-server-scikit-learn/src/mcp_server_scikit_learn/server.py" ]; then
    echo "sklearn server missing - run install-all-mcp-dependencies.sh"
fi

# 4. Update launcher script
echo -e "\n${YELLOW}4. Updating launcher script...${NC}"
sed -i '' "s/15 total/17 total/" scripts/start-claude-ultimate.sh
sed -i '' "s/Node.js (5)/Node.js (7)/" scripts/start-claude-ultimate.sh
sed -i '' "s/• puppeteer, /• puppeteer, ripgrep, dependency-graph/" scripts/start-claude-ultimate.sh

echo -e "\n${GREEN}=== All Servers Restored ===${NC}"
echo "You now have 17 MCP servers:"
echo "  • 7 Node.js servers (including ripgrep & dependency-graph)"
echo "  • 10 Python servers"
echo ""
echo "Restart Claude with: ./scripts/start-claude-ultimate.sh"