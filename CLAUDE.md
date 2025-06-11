# CLAUDE.md - Unity Wheel Trading Bot v2.2

Minimal context file. For details, read referenced docs below.

## Quick Commands
```bash
python run.py -p 100000          # Get recommendation
python run.py --diagnose         # System health
pytest -v -m "not slow"          # Fast tests
```

## Key Files
- `advisor.py:106` - Main logic
- `wheel.py:153` - Strike selection
- `options.py:746` - Math
- `config.yaml` - Settings

## Reference Docs
When you need details, read these:
- `docs/QUICK_REFERENCE.md` - Full commands & workflows
- `docs/ARCHITECTURE.md` - System design
- `docs/DATABENTO_UNITY_GUIDE.md` - Data integration
- `CLAUDE-BACKUP-FULL.md` - Original 949-line reference

## Critical Values
Unity (U) | Delta: 0.30 | Max position: 100% | Min confidence: 0.30

## MCP Toolkits & Retrieval Policy

You have six toolkits:

1. `mcp__fs__*` – my whole local drive
2. `mcp__vscode__*` – the open workspace (preferred for edits)
3. `mcp__python_analysis__*` – **real-time trading analysis**
4. `mcp__github__*` – public repos & issues
5. `mcp__web__*` – live news & the wider web
6. `mcp__wiki__*` – encyclopaedia facts

**Priority:** 1 → 2 → 3 → 4 → 5 → 6.
**Never paste >2,000 tokens from any single file; summarise instead.**

## Python Analysis MCP Capabilities

- `analyze_position(params)` – Real-time position analysis with Greeks
- `monitor_system()` – Live system performance monitoring
- `data_quality_check()` – Validate data freshness and integrity
- Direct access to project modules and mathematical functions

## Large Repo Optimizations

- Output limited to 8k tokens for faster responses
- Enhanced mathematical analysis tools (statsmodels, scipy, sympy)
- Performance profiling (line_profiler, memory_profiler, snakeviz)
- Data validation (pandera, pydantic, jsonschema)
- Advanced visualizations (plotly, dash, seaborn)
