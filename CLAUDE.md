# CLAUDE.md - Unity Wheel Trading Bot v3.0

Optimized for Claude Code CLI. Minimal context, maximum efficiency.

## Quick Commands
```bash
python run.py -p 100000          # Get recommendation
python run.py --diagnose         # System health
pytest -v -m "not slow"          # Fast tests
```

## Critical Paths
- Entry: `run.py` → `src/unity_wheel/api/advisor.py`
- Data: `data/wheel_trading_master.duckdb` (single source)
- Config: `config.yaml` + env vars

## Key Values
Unity (U) | Delta: 0.30 | Max position: 100% | Min confidence: 0.30

## MCP Servers (Active)
1. `fs` - File system access
2. `github` - Repository management
3. `python_analysis` - Real-time trading analysis

## Development Workflow
1. **Local Only** - No cloud deployment
2. **GitHub** - Source control (`njrun1804/wheel-trading`)
3. **Testing** - Always test before commits

## Code Style
- No comments unless complex math
- Type hints everywhere
- Follow existing patterns
- Prefer editing over creating

## Key Components
- Trading: `src/unity_wheel/strategy/wheel.py`
- Risk: `src/unity_wheel/risk/`
- Math: `src/unity_wheel/math/options.py`
- API: `src/unity_wheel/api/`

## When You Need More
- Architecture: `docs/ARCHITECTURE.md`
- Commands: `docs/QUICK_REFERENCE.md`
- Original guide: `CLAUDE-BACKUP-FULL.md`

## Autonomous Development
- GitHub integration configured
- Clear API boundaries
- All auth handled automatically
- MCP servers simplified

# Important
- Do exactly what's asked
- Never create unnecessary files
- Test changes locally first
