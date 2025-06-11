# GitHub Copilot / Codex Instructions

This is a wheel trading bot for personal use. Key guidelines:

## Architecture
- Single-user system running locally on macOS
- Manual position entry (no broker integration)
- DuckDB for local data storage (`data/wheel_trading_master.duckdb`)
- No cloud deployment needed

## Code Style
- Python 3.11+ with type hints
- Follow existing patterns in codebase
- No comments unless explaining complex math
- Prefer clarity over cleverness

## Key Components
1. **Trading Logic**: `/src/unity_wheel/strategy/wheel.py`
2. **Risk Management**: `/src/unity_wheel/risk/`
3. **Options Math**: `/src/unity_wheel/math/options.py`
4. **API Layer**: `/src/unity_wheel/api/advisor.py`

## Testing
- Always write tests for new features
- Run `pytest -v -m "not slow"` for quick tests
- Mock external API calls

## Entry Points
- `run.py` - Main CLI entry point
- `src/unity_wheel/api/advisor.py` - Programmatic API

## Data Flow
1. Market data → Databento → DuckDB
2. Positions → Manual entry → Analysis
3. Recommendations → Console/API output

## Security
- Never commit secrets
- Use environment variables for any credentials
- Manual position management only
