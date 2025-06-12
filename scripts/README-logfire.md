# Logfire MCP Server Setup

This directory contains scripts to securely manage Logfire tokens using macOS Keychain.

## Initial Setup

1. **Store your Logfire token in Keychain** (already done):
   ```bash
   security add-generic-password -a "$USER" -s "logfire-mcp" -w "YOUR_LOGFIRE_TOKEN" -U
   ```

2. **Configure the Logfire MCP server**:
   ```bash
   ./scripts/setup-logfire-mcp.sh
   ```
   This will:
   - Retrieve the token from Keychain
   - Add the Logfire MCP server to `mcp-servers.json`
   - Test the connection

## Daily Usage

To use the Logfire token in your shell sessions:
```bash
source ./scripts/setup-logfire-env.sh
```

This will export `LOGFIRE_READ_TOKEN` to your environment.

## Available Scripts

- **keychain-helper.sh**: Generic helper for storing/retrieving secrets
  ```bash
  ./scripts/keychain-helper.sh get logfire-mcp
  ./scripts/keychain-helper.sh export LOGFIRE_READ_TOKEN logfire-mcp
  ```

- **setup-logfire-env.sh**: Sets up environment variables
  ```bash
  source ./scripts/setup-logfire-env.sh
  ```

- **setup-logfire-mcp.sh**: Configures MCP server
  ```bash
  ./scripts/setup-logfire-mcp.sh
  ```

## Security Notes

- Tokens are stored securely in macOS Keychain
- Never commit tokens to Git
- The MCP configuration file contains the actual token (required by MCP)
- Use environment variables when possible

## Logfire MCP Server Features

The Logfire MCP server provides:
- Access to OpenTelemetry traces and metrics
- Exception analysis from traces
- Custom SQL queries via Logfire APIs
- Integration with AI tools for debugging

## Requirements

- macOS (for Keychain support)
- `jq` for JSON processing
- `uv` for running the Logfire MCP server
- Python 3.12+ (handled by `uvx`)

## Troubleshooting

If the token retrieval fails:
```bash
# Check if token exists
security find-generic-password -a "$USER" -s "logfire-mcp"

# Delete and re-add if needed
security delete-generic-password -a "$USER" -s "logfire-mcp"
security add-generic-password -a "$USER" -s "logfire-mcp" -w "YOUR_TOKEN" -U
```