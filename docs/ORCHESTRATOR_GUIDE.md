# MCP Orchestrator Guide

## Overview

The MCP Orchestrator is a sophisticated system for coordinating multiple MCP (Model Context Protocol) servers to perform complex code transformations efficiently and reliably. It transforms natural language commands into deterministic, multi-phase operations that can refactor code, run tests, and create pull requests in under 90 seconds.

## Architecture

### Core Components

#### 1. MCPOrchestrator (`orchestrator.py`)
The main orchestration engine that:
- Parses natural language commands
- Generates and validates execution plans
- Coordinates 7-phase execution flow
- Manages retry logic and error recovery
- Enforces resource constraints (memory, tokens)

#### 2. SliceCache (`slice_cache.py`)
High-performance caching system that:
- Stores code slices with SHA-1 keys
- Supports vector embeddings for similarity search
- Implements LRU eviction for memory efficiency
- Uses SQLite with WAL mode for persistence
- Provides sub-millisecond lookups for cached content

#### 3. MemoryPressureMonitor (`pressure.py`)
Real-time memory monitoring that:
- Samples RSS/total ratio every 250ms
- Triggers backoff when approaching 70% threshold
- Tracks pressure events and history
- Suggests garbage collection timing
- Prevents OOM kills during intensive operations

### 7-Phase Execution Model

```
┌─────────┐    ┌─────────┐    ┌──────────────┐    ┌────────┐
│   MAP   │ -> │  LOGIC  │ -> │ MONTE CARLO  │ -> │  PLAN  │
└─────────┘    └─────────┘    └──────────────┘    └────────┘
                                                        ↓
┌─────────┐    ┌─────────┐    ┌──────────┐         ┌──────────┐
│ REVIEW  │ <- │ EXECUTE │ <- │ OPTIMIZE │    <-   │          │
└─────────┘    └─────────┘    └──────────┘         └──────────┘
```

1. **MAP Phase**: Discovers relevant code using parallel searches
2. **LOGIC Phase**: Analyzes call graphs and prunes candidates
3. **MONTE CARLO Phase**: Runs risk simulations (15s cap)
4. **PLAN Phase**: Generates JSON DAG execution plan
5. **OPTIMIZE Phase**: Parameter optimization via DuckDB/PyREPL
6. **EXECUTE Phase**: Applies transformations and runs tests
7. **REVIEW Phase**: Validates results with Phoenix tracing

## Usage

### Basic Command Execution

```python
from src.unity_wheel.orchestrator import MCPOrchestrator

# Initialize orchestrator
orchestrator = MCPOrchestrator("/path/to/workspace")
await orchestrator.initialize()

# Execute natural language command
result = await orchestrator.execute_command(
    "Refactor the risk management module to use async patterns"
)

# Check results
if result["success"]:
    print(f"Completed in {result['duration_ms']}ms")
    print(f"Total tokens used: {result['total_tokens']}")
```

### Using Execution Plans

```python
from src.unity_wheel.orchestrator import ExecutionPlan

# Load a pre-defined plan
plan = ExecutionPlan.from_json("refactor_plan.json")

# Execute with custom plan
orchestrator.current_plan = plan
result = await orchestrator.execute_command(plan.command)
```

### Memory Pressure Handling

```python
# Monitor memory pressure
monitor = orchestrator.memory_monitor
if monitor.is_pressure_high():
    # Wait for pressure to decrease
    if monitor.wait_for_low_pressure(timeout=30):
        # Safe to proceed
        pass

# Get memory statistics
stats = monitor.get_stats()
print(f"Current memory ratio: {stats['current_ratio']}")
print(f"Peak memory: {stats['peak_memory_mb']}MB")
```

### Cache Management

```python
# Access slice cache directly
cache = orchestrator.slice_cache

# Store analysis results
await cache.store(
    slice_hash="abc123",
    data={"file": "module.py", "content": "..."},
    vector=embedding_vector
)

# Search for similar code
similar = await cache.search_similar(query_vector, top_k=5)

# Get cache statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']}")
```

## Configuration

### Environment Variables

```bash
# Token limits
export CLAUDE_CODE_THINKING_BUDGET_TOKENS=4096
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096

# Parallelism
export CLAUDE_CODE_PARALLELISM=8

# Memory threshold (optional, default 0.70)
export ORCHESTRATOR_MEMORY_THRESHOLD=0.75

# Cache size (optional, default 512MB)
export ORCHESTRATOR_CACHE_SIZE_MB=1024
```

### MCP Server Requirements

Essential servers that must be running:
- `filesystem` - File operations
- `ripgrep` - Code search
- `dependency_graph` - Code analysis
- `memory` - State persistence
- `sequential-thinking` - Planning
- `python_analysis` - Domain logic

Optional servers for enhanced functionality:
- `trace_phoenix` - Observability
- `duckdb` - Heavy analytics
- `pyrepl` - Dynamic execution

## Performance Considerations

### Resource Limits
- **Memory**: Maintains RSS < 70% of system RAM
- **Tokens**: 4,096 per phase, 28,672 total budget
- **Time**: 90-second target for end-to-end execution
- **Retries**: Maximum 3 attempts per phase

### Optimization Tips

1. **Pre-warm the cache** for frequently accessed code:
   ```python
   await orchestrator.slice_cache.cleanup_old_entries(days=30)
   ```

2. **Monitor pressure events** to identify bottlenecks:
   ```python
   history = monitor.get_pressure_history(seconds=300)
   high_pressure_count = sum(1 for h in history if h["high_pressure"])
   ```

3. **Use appropriate phase configurations**:
   ```python
   plan.phases = [
       {"phase": "map", "targets": ["ripgrep"]},  # Skip dependency_graph if not needed
       {"phase": "monte_carlo", "duration_limit_ms": 5000},  # Reduce MC time
   ]
   ```

## Error Handling

The orchestrator implements comprehensive error handling:

1. **Phase-level retries**: Each phase can retry up to 3 times
2. **Memory backoff**: Automatic delay when pressure is high
3. **Graceful degradation**: Continue with partial results
4. **Detailed error tracking**: All failures logged with context

Example error handling:

```python
result = await orchestrator.execute_command("complex refactor")

if not result["success"]:
    # Find failed phase
    failed = next(p for p in result["phases"] if not p["success"])
    print(f"Failed at {failed['phase']}: {failed['error']}")
    print(f"Retries attempted: {failed['retries']}")
```

## Integration with CI/CD

The orchestrator is designed for automated workflows:

```yaml
# .github/workflows/orchestrated-refactor.yml
name: Orchestrated Refactor
on:
  issue_comment:
    types: [created]

jobs:
  refactor:
    if: contains(github.event.comment.body, '/refactor')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Orchestrator
        run: |
          python -m src.unity_wheel.orchestrator \
            --command "${{ github.event.comment.body }}" \
            --output pr.json
      - name: Create PR
        if: success()
        uses: peter-evans/create-pull-request@v5
```

## Troubleshooting

### Common Issues

1. **High memory pressure**
   - Reduce `batch_size` in optimization phase
   - Lower `max_memory_entries` in cache
   - Enable swap if available

2. **Token limit exceeded**
   - Simplify command language
   - Break into smaller operations
   - Adjust phase-specific limits

3. **Slow execution**
   - Check cache hit rates
   - Verify MCP server health
   - Profile individual phases

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Get phase-by-phase timing
for phase_result in result["phases"]:
    print(f"{phase_result['phase']}: {phase_result['duration_ms']}ms")
```

## Future Enhancements

Planned improvements include:
- GPU acceleration for vector operations
- Distributed execution across multiple machines
- Advanced caching with learned eviction policies
- Real-time progress streaming
- Integration with more MCP servers