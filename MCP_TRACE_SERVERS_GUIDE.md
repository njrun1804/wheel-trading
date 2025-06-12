# MCP Trace Servers Guide - Local vs Cloud

## Your 3 Trace Server Options

### 1. 🔥 **Logfire** (Cloud - Already Configured)
- **Status**: ✅ Ready (token in keychain)
- **Port**: N/A (cloud service)
- **Best for**: Traditional app logs, metrics, Python-focused
- **Cost**: Free tier available

### 2. 🎯 **Opik** (Local or Cloud)
- **Local**: http://localhost:5173 (self-hosted, no API key)
- **Cloud**: comet.com/opik (needs OPIK_API_KEY)
- **Best for**: LLM traces, prompt engineering, eval metrics
- **Cost**: Local = FREE, Cloud = paid tiers

### 3. 🔥 **Phoenix** (Local or Cloud)
- **Local**: http://localhost:6006 (self-hosted, no API key)
- **Cloud**: app.phoenix.arize.com (needs PHOENIX_API_KEY)
- **Best for**: OpenTelemetry traces, token usage, latency analysis
- **Cost**: Local = FREE, Cloud = paid tiers

## Quick Start Commands

### Option A: Local Trace Servers (Recommended for Dev)
```bash
# One-time setup
./scripts/setup-local-trace-servers.sh

# Start Claude with local servers (no API keys!)
./scripts/start-claude-local.sh
```

### Option B: Cloud Trace Servers
```bash
# Set API keys (if you have them)
export OPIK_API_KEY="your-key"
export PHOENIX_API_KEY="your-key"

# Start Claude with cloud servers
./scripts/start-claude-full.sh
```

### Option C: Original Setup (13 servers)
```bash
# Just the basics + Logfire
./scripts/start-claude-fixed.sh
```

## When to Use What

### Use LOCAL trace servers when:
- 🏠 Developing locally
- 🔒 Data can't leave your machine
- 💰 Want zero costs
- ⚡ Need sub-10ms latency
- 🧪 Testing/experimenting

### Use CLOUD trace servers when:
- 👥 Working with a team
- 📊 Need persistent history
- 🔐 Want managed auth/RBAC
- 📈 Running in production
- 🌍 Need remote access

## Current Status

| Server | Local Available | Cloud Available | Status |
|--------|----------------|-----------------|---------|
| Logfire | ❌ | ✅ | Cloud configured |
| Opik | ✅ | ⚠️ | Local ready, cloud needs key |
| Phoenix | ✅ | ⚠️ | Local ready, cloud needs key |

## Architecture

```
Your Code → Claude → MCP Servers → Trace Platforms
                          ↓
                   • trace (Logfire Cloud)
                   • trace-opik-local (Opik Docker)
                   • trace-phoenix-local (Phoenix Python)
```

## Trace Server Comparison

| Feature | Logfire | Opik | Phoenix |
|---------|---------|------|---------|
| Focus | App observability | LLM observability | OpenTelemetry |
| Best for | Python apps | Prompt engineering | Token/latency analysis |
| Local option | ❌ | ✅ Docker | ✅ Python |
| Setup complexity | Low (cloud) | Medium (Docker) | Low (pip) |
| Resource usage | None (cloud) | High (Postgres) | Low (SQLite) |

## Troubleshooting

### Opik not starting?
```bash
cd ~/mcp-servers/opik-platform
docker compose logs
./opik.sh restart
```

### Phoenix not starting?
```bash
pkill -f "phoenix serve"
phoenix serve
# Check logs at ~/.cache/phoenix-server.log
```

### Port conflicts?
- Opik: 5173 (UI), 5432 (Postgres)
- Phoenix: 6006 (UI + API)
- Change ports in docker-compose.yml or phoenix config

## Next Steps

1. **Try local first**: Zero config, instant gratification
2. **Get API keys later**: When you need team features
3. **Mix and match**: Use Logfire cloud + local Opik/Phoenix

The beauty is you can switch between local/cloud anytime just by changing which startup script you use!