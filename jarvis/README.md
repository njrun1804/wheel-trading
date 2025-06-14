# Jarvis - Meta-Coder for Claude Code CLI

Jarvis is a streamlined meta-coder specifically designed to help Claude Code understand and execute complex tasks on your M4 Pro Mac. It's the successor to the orchestrator, keeping the best ideas while leveraging hardware-accelerated tools.

## Key Features

- **Simplified 4-Phase Execution**: DISCOVER → ANALYZE → IMPLEMENT → VERIFY
- **Hardware Accelerated**: Uses all 12 CPU cores + Metal GPU
- **No MCP Dependencies**: 10-30x faster with direct file I/O
- **Smart Strategy Selection**: Automatically chooses the best approach
- **Optional MCTS**: Monte Carlo Tree Search for complex optimizations

## Usage

### Command Line
```bash
# Execute a task
./jarvis.py "optimize all trading functions for performance"

# Explain approach without executing
./jarvis.py --explain "refactor the WheelStrategy class"

# Interactive mode
./jarvis.py --interactive

# Verbose output
./jarvis.py --verbose "find and fix all type errors"
```

### Programmatic
```python
from jarvis import Jarvis

jarvis = Jarvis()
result = await jarvis.assist("optimize database queries")
```

## How It Works

### 1. DISCOVER Phase
- Uses hardware-accelerated ripgrep (all 12 cores)
- Builds dependency graphs with GPU assistance
- Finds all relevant code in milliseconds

### 2. ANALYZE Phase
- Parallel Python AST analysis
- Automatic strategy selection
- Complexity assessment

### 3. IMPLEMENT Phase
- Strategy-specific implementation
- Optional MCTS for optimization tasks
- Full hardware utilization

### 4. VERIFY Phase
- Test execution
- Result validation
- Performance metrics

## Strategies

Jarvis automatically selects from these strategies based on your query:

- **Optimization**: Performance improvements, uses MCTS
- **Refactoring**: Code restructuring, dependency-aware
- **Testing**: Test generation and coverage
- **Debugging**: Issue identification and fixes
- **Analysis**: Code understanding and documentation
- **Generation**: Creating new features

## Hardware Utilization

Optimized for M4 Pro Mac:
- **CPU**: 12 cores (8 performance + 4 efficiency)
- **GPU**: Metal acceleration (20 cores)
- **Memory**: 19.2GB allocated (80% of 24GB)

## Comparison to Orchestrator

| Feature | Orchestrator | Jarvis |
|---------|-------------|---------|
| Phases | 7 complex phases | 4 streamlined phases |
| MCP Servers | 18+ (slow, timeouts) | 0 (direct I/O) |
| Performance | 0.1-0.9% CPU usage | 100% CPU when needed |
| GPU Usage | Created but unused | Actually uses Metal |
| Complexity | Over-engineered | Just right |
| Speed | 1000x slower | Optimized for M4 Pro |

## Examples

### Find and Optimize Slow Functions
```bash
./jarvis.py "find functions that take over 100ms and optimize them"
```

### Refactor with Confidence
```bash
./jarvis.py "rename WheelStrategy to OptionsWheelStrategy everywhere"
```

### Add Comprehensive Tests
```bash
./jarvis.py "add unit tests for all trading analysis functions"
```

### Interactive Analysis
```bash
./jarvis.py -i
jarvis> explain the dependency structure of analytics module
jarvis> optimize all database queries for batch processing
jarvis> quit
```

## Configuration

Create a custom config:
```python
from jarvis import JarvisConfig

config = JarvisConfig(
    workspace_root="/path/to/project",
    use_mcts=True,
    max_mcts_simulations=2000,
    hardware_mode="maximum",  # or "balanced", "eco"
    trace_enabled=True,
    verbose=True
)

jarvis = Jarvis(config)
```

## Architecture

```
jarvis/
├── core/
│   ├── jarvis.py         # Main coordinator
│   └── phases.py         # Phase execution engine
├── strategies/
│   └── strategy_selector.py  # Smart strategy selection
├── analysis/
│   └── mcts_lite.py      # Simplified MCTS for optimization
└── README.md
```

## Future Enhancements

- Learning from past executions
- Custom strategy plugins
- Integration with Claude Code's context
- Performance prediction models

## Retiring the Orchestrator

Once you're satisfied with Jarvis:
1. Test thoroughly with your common tasks
2. Compare performance metrics
3. Remove the `/orchestration` directory
4. Update any scripts that referenced orchestrator

Jarvis is designed to be the orchestrator done right - focused, fast, and actually using your hardware!