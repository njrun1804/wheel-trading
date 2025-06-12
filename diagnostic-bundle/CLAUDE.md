# Claude Playbook – Trading-Bot Repo

## Workflow
1. `dependency_graph.search_code_fuzzy(<term>)`  
2. `ripgrep.search` if step 1 empty or regex-style query  
3. `filesystem.read` surrounding lines  
4. `python_analysis.*` domain checks  
5. `trace_phoenix.*` for runtime / perf issues  
6. Persist interim notes in `memory.save()`  
7. Ask before edits > 30 LOC

## Pre-commit gate
Run `dependency_graph.detect_cycles()`; block commit if cycles > 0.

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

## MCP Servers (Tiered)
### Essential (autostart)
- `filesystem` - Core file I/O
- `github` - PR / commit workflow
- `dependency_graph` - 2-5 ms symbol graph
- `ripgrep` - Fallback literal / regex search
- `memory` - Retains long-chain reasoning
- `sequential-thinking` - Structured multi-step planner
- `python_analysis` - Wheel trading analysis
- `trace_phoenix` - Rich OTLP spans

### On-demand
- `pyrepl` - Scratch calculations
- `duckdb` - Heavy SQL on options db
- `statsource` - Macro-series lookups
- `brave`, `puppeteer` - Scraping / screenshots
- `mlflow` - Model grid-searches

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
