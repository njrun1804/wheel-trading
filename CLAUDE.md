# Claude Playbook – Trading-Bot Repo

## 🔧 META-SYSTEM DEVELOPMENT ENVIRONMENT
This codebase includes a meta-programming system that provides development workflow automation and code quality monitoring.

### 📁 Meta System Documentation
- **README**: `META_SYSTEM_README.md` - Complete overview and usage guide
- **Capabilities**: `META_CAPABILITIES.md` - Detailed what it does vs. doesn't do
- **Build Instructions**: `/meta/BUILD_INSTRUCTIONS.md` - Setup guide for meta components
- **Integration Guide**: `/meta/META_INTEGRATION.md` - How to integrate meta-awareness into code

### 🔄 Meta Development Workflow
1. **Observe** → File change monitoring and event logging
2. **Analyze** → Pattern detection from development activity
3. **Quality Check** → Code validation using predefined rules
4. **Generate** → Template-based code improvements
5. **Execute** → Safe file modifications with backups

## ⚡ HARDWARE OPTIMIZATION
System is optimized for M4 Pro: 8 P-cores + 4 E-cores with parallel processing.

### NEW: Hardware-Accelerated Local Tools (10-30x Faster!)
Replace slow MCP servers with blazing-fast local implementations:

```python
# 🔍 Ripgrep Turbo - 30x faster search
from unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
rg = get_ripgrep_turbo()
results = await rg.parallel_search(["TODO", "FIXME"], "src")  # Uses all 12 cores

# 🕸️ Dependency Graph - 12x faster with GPU
from unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
graph = get_dependency_graph()
await graph.build_graph()  # Parallel AST parsing
symbols = await graph.find_symbol("WheelStrategy")

# 🐍 Python Analysis - 173x faster
from unity_wheel.accelerated_tools.python_analysis_turbo import get_python_analyzer
analyzer = get_python_analyzer()
analysis = await analyzer.analyze_directory("src")  # MLX GPU acceleration

# 🦆 DuckDB Native - No MCP overhead
from unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo
db = get_duckdb_turbo("data/trading.db")
df = await db.query_to_pandas("SELECT * FROM options")  # 24 parallel connections

# 📊 Unified Tracing - All backends
from unity_wheel.accelerated_tools.trace_simple import get_trace_turbo
tracer = get_trace_turbo()
async with tracer.trace_span("operation") as span:
    # Your code here
    pass

# 🛠️ Python Helpers - Combined tools
from unity_wheel.accelerated_tools.python_helpers_turbo import get_code_helper
helper = get_code_helper()
sig = await helper.get_function_signature("module.py", "function_name")
```

### Quick Commands - Hardware Accelerated + Meta-Aware
```bash
# Start meta development environment
python meta_coordinator.py --dev-mode &

# Start file monitoring with quality checks
python meta_daemon.py --watch-path . &

# Run one-time quality audit
python meta_auditor.py --validate

# Start complete integrated system
python start_complete_meta_system.py

# Test accelerated tools
python test_all_accelerated_tools.py
```

### Performance Metrics
- **Search operations**: 23ms (was 150ms with MCP)
- **Dependency graph**: 3.2s for entire codebase (was 6s)
- **Python analysis**: 15ms per file (was 2.6s)
- **DuckDB queries**: 14ms (was 100ms)
- **Memory usage**: 80% reduction

### Automatic Acceleration
When you use these tools, they automatically run in parallel:
- **Grep**: Searches using all 12 CPU cores
- **Glob**: File matching parallelized
- **Read**: Multiple files read simultaneously  
- **Task**: Complex operations distributed
- **Bash**: Compound commands run in parallel

The system provides:
- 12 CPU cores (M4 Pro) running at maximum
- 19.2GB RAM allocated (80% of 24GB)
- Metal GPU acceleration (20 cores)
- <5ms filesystem operations
- Automatic parallelization of all operations

## Workflow
1. **Meta-First**: Use meta system for all development (`/meta/BUILD_INSTRUCTIONS.md`)
2. `dependency_graph.search_code_fuzzy(<term>)`  
3. `ripgrep.search` if step 2 empty or regex-style query  
4. `filesystem.read` surrounding lines  
5. `python_analysis.*` domain checks  
6. `trace_phoenix.*` for runtime / perf issues  
7. **Meta Integration**: Apply meta-awareness patterns (`/meta/META_INTEGRATION_STRATEGY.md`)
8. Persist interim notes in `memory.save()`  
9. Ask before edits > 30 LOC

## Pre-commit gate
Run `dependency_graph.detect_cycles()`; block commit if cycles > 0.

## Quick Commands
```bash
# Start Claude with all optimizations
./startup.sh                     # Launch with M4 Pro optimizations + orchestrator

# Trading analysis
python run.py -p 100000          # Get recommendation
python run.py --diagnose         # System health

# Complex code analysis (NEW)
./orchestrate "optimize all trading functions"

# Testing
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
