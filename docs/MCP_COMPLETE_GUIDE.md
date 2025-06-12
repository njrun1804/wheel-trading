# MCP Complete Setup Guide

This guide documents the complete MCP (Model Context Protocol) setup for the wheel-trading project.

## Overview

The MCP setup has been comprehensively fixed to address:
- Asyncio compatibility issues with FastMCP
- Phoenix tracing integration
- Robust startup/shutdown procedures
- Health monitoring
- Proper error handling

## Fixed Issues

### 1. Asyncio Compatibility
- **Problem**: `asyncio.run(mcp.run())` causes "already running" errors
- **Solution**: Direct call to `mcp.run()` - FastMCP handles the event loop internally

### 2. Phoenix Tracing
- **Setup**: Run `./scripts/setup-phoenix-tracing.sh`
- **Access**: http://localhost:6006
- **Integration**: All MCP servers now emit traces

### 3. Startup/Shutdown
- **Start**: `./scripts/start-mcp-servers.sh`
- **Stop**: `./scripts/stop-mcp-servers.sh`
- **Monitor**: `./scripts/mcp-monitor-service.sh start`

## MCP Servers

### Critical Servers
1. **python_analysis** - Trading analysis and monitoring
2. **trace** - Observability and debugging
3. **ripgrep** - Fast code search
4. **dependency_graph** - Code dependency analysis

### Usage Examples

```bash
# Start all servers
./scripts/start-mcp-servers.sh

# Monitor health
./scripts/mcp-health-monitor.py --watch

# Check specific server
cat .claude/runtime/python_analysis.health

# Stop all servers
./scripts/stop-mcp-servers.sh
```

## Creating New MCP Servers

Use the template at `scripts/mcp-server-template.py`:

```python
#!/usr/bin/env python3
from mcp.server import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def my_tool(input: str) -> str:
    return f"Processed: {input}"

if __name__ == "__main__":
    mcp.run()  # No asyncio.run() needed!
```

## Troubleshooting

### Server won't start
1. Check logs: `tail -f .claude/runtime/<server>.log`
2. Verify dependencies: `pip install mcp`
3. Check port conflicts

### Asyncio errors
- Ensure NO `asyncio.run()` calls in MCP servers
- Use the template as reference

### Phoenix not working
1. Check if running: `curl http://localhost:6006/health`
2. Restart: `python -m phoenix.server`
3. Check logs: `~/.phoenix/phoenix.log`

## Best Practices

1. **Always use absolute paths** in configurations
2. **Implement healthz endpoint** in every server
3. **Log errors properly** for debugging
4. **Clean shutdown** - handle SIGTERM gracefully
5. **Monitor regularly** - use the health monitor service

## Environment Variables

Required variables:
- `DATABENTO_API_KEY` - For market data
- `FRED_API_KEY` - For economic data
- `GITHUB_TOKEN` - For GitHub integration
- `BRAVE_API_KEY` - For web search

## Maintenance

### Daily
- Check health monitor for issues
- Review error logs

### Weekly
- Clean up old logs: `find .claude/runtime -name "*.log" -mtime +7 -delete`
- Update dependencies: `pip install -U mcp`

### Monthly
- Review and optimize slow operations via Phoenix
- Update MCP server configurations as needed
