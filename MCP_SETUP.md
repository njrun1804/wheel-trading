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
