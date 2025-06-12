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
