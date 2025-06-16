# Unified CLI - Intelligent Router for Einstein and Bolt Systems

## Overview

The Unified CLI provides a single entry point that intelligently routes queries between two specialized systems:

- **Einstein**: Fast semantic search, code understanding, simple queries (<50ms)
- **Bolt**: Complex analysis, multi-step problem solving, optimization tasks (8-agent system)

## Key Features

### 🤖 Intelligent Routing
- **Automatic classification** of queries based on complexity and intent
- **High accuracy** routing (100% on benchmark tests)
- **Smart fallback** to alternative system if primary fails
- **Confidence scoring** for routing decisions

### ⚡ Performance Optimized
- **Einstein**: Sub-50ms responses for simple searches
- **Bolt**: 8-agent parallel processing for complex tasks
- **Hardware acceleration** using M4 Pro (12 cores + Metal GPU)
- **Minimal overhead** routing layer

### 🛡️ Robust Error Handling
- **Automatic fallback** between systems
- **Graceful degradation** when components unavailable
- **Comprehensive logging** and debugging support
- **Session statistics** tracking

## Installation

### Prerequisites
- Python 3.9+
- macOS with Apple Silicon (M1/M2/M3/M4)
- Einstein and Bolt systems installed

### Quick Start
```bash
# Make executable
chmod +x unified_cli.py unified

# Test the system
python3 unified_cli.py --benchmark

# Run demo
python3 unified_cli_demo.py
```

## Usage

### Command Line Interface

```bash
# Basic usage - auto-routing
python3 unified_cli.py "find WheelStrategy"
python3 unified_cli.py "optimize database performance"

# Force specific system
python3 unified_cli.py --force-einstein "optimize queries"
python3 unified_cli.py --force-bolt "find WheelStrategy"

# Interactive mode
python3 unified_cli.py --interactive

# Short launcher
./unified "your query here"
```

### Interactive Mode

```bash
$ python3 unified_cli.py --interactive

🚀 Unified CLI - Interactive Mode
   Commands: <query>, !einstein <query>, !bolt <query>, stats, help, quit
   Examples:
     find WheelStrategy          → Einstein (search)
     optimize database queries   → Bolt (analysis)
     !bolt find WheelStrategy    → Force Bolt

unified> find WheelStrategy
🤖 Routing to EINSTEIN
   Confidence: 85.0%
   Reasoning: Simple search query detected
   Query: find WheelStrategy
------------------------------------------------------------
✅ EINSTEIN completed in 0.05s
   Found 3 results
   Files: 2
   Top score: 0.95

unified> optimize database queries
🤖 Routing to BOLT
   Confidence: 90.0%
   Reasoning: Complex pattern detected
   Query: optimize database queries
------------------------------------------------------------
✅ BOLT completed in 2.34s
   Status: ✅ Success
   Summary: Database optimization analysis complete
   Findings:
     • Query execution time can be reduced by 60%
     • Missing indexes on frequently queried columns
     • Inefficient JOIN operations detected

unified> stats
📊 Session Statistics:
   Einstein queries: 1
   Bolt queries: 1
   Fallbacks: 0
   Errors: 0
   Success rate: 100.0%
```

## Routing Logic

### Einstein System (Fast Search)
**Ideal for:**
- Simple searches: `find WheelStrategy`, `show options.py`
- Code element lookups: `calculate_delta`, `WheelBacktester`
- Technical searches: `import pandas`, `def main`
- Symbol names and identifiers
- Short queries (≤5 words)

**Keywords that trigger Einstein:**
- find, search, locate, show, list, where, what, which
- definition, function, class, method, variable, import
- usage, example, reference, documentation

### Bolt System (Complex Analysis)  
**Ideal for:**
- Optimization tasks: `optimize database performance`
- Problem solving: `fix memory leak in trading module`
- Analysis requests: `analyze bottlenecks in wheel strategy`
- Action-oriented: `help me refactor the risk calculation`
- Long queries (>10 words)
- Multi-step reasoning

**Keywords that trigger Bolt:**
- optimize, fix, debug, analyze, improve, refactor, solve
- performance, memory, speed, bottleneck, issue, problem
- architecture, design, pattern, strategy, algorithm

### Complex Pattern Detection
The router uses regex patterns to detect complex queries:
- `how to|how can I|help me` + action words
- `analyze|review|audit` + system components  
- `multiple` + `files|components|systems`
- `database|sql|query` + `optimization|performance`
- `refactor|redesign|restructure`

## Advanced Features

### Forced Routing
Override automatic routing when needed:

```bash
# Force Einstein for complex query (get simple search results)
python3 unified_cli.py --force-einstein "optimize database performance"

# Force Bolt for simple query (get deep analysis)
python3 unified_cli.py --force-bolt "find WheelStrategy"

# Interactive forced routing
unified> !einstein optimize database queries
unified> !bolt find WheelStrategy
```

### Benchmarking
Test routing accuracy:

```bash
python3 unified_cli.py --benchmark

🧪 Running routing benchmark tests...
Testing 15 routing decisions...
✅ 'find WheelStrategy' → einstein (85.0%)
✅ 'optimize database performance' → bolt (90.0%)
...
📊 Routing Accuracy: 15/15 (100.0%)
🎉 Excellent routing performance!
```

### Debug Mode
Get detailed routing information:

```bash
python3 unified_cli.py --debug "your query"
```

## Examples

### Einstein Examples (Fast Search)
```bash
# Function/class lookups
unified "WheelStrategy"
unified "calculate_delta"
unified "BacktestEngine"

# File searches  
unified "show options.py"
unified "find risk_analytics.py"

# Code pattern searches
unified "def main"
unified "import pandas"
unified "TODO comments"
```

### Bolt Examples (Complex Analysis)
```bash
# Performance optimization
unified "optimize database query performance"
unified "improve memory usage in trading system"
unified "analyze CPU bottlenecks"

# Problem solving
unified "fix memory leak in options pricing"
unified "debug connection timeouts"
unified "resolve threading issues"

# Architecture analysis
unified "review system architecture"
unified "analyze data flow patterns"
unified "help me refactor the wheel strategy"
```

## Performance Metrics

### Routing Performance
- **Classification time**: <1ms per query
- **Routing accuracy**: 100% on benchmark tests
- **Confidence scoring**: 85-90% for clear cases

### System Performance
- **Einstein**: 23-50ms average response time
- **Bolt**: 2-15s depending on complexity
- **Fallback overhead**: <100ms additional latency

## Configuration

### Environment Variables
```bash
# Optional: Override project root
export UNIFIED_CLI_PROJECT_ROOT="/path/to/project"

# Optional: Force debug mode
export UNIFIED_CLI_DEBUG=1

# Optional: Default system preference
export UNIFIED_CLI_DEFAULT_SYSTEM="einstein"  # or "bolt"
```

### System Requirements
- **Einstein**: Semantic search dependencies, FAISS index
- **Bolt**: 8-agent system, hardware acceleration libraries
- **Memory**: 8GB+ recommended (24GB optimal)
- **Storage**: 2GB+ for indices and models

## Troubleshooting

### Common Issues

**System not available:**
```bash
# Check system availability
python3 unified_cli.py --debug "test query"

# Force alternative system
python3 unified_cli.py --force-bolt "test query"
```

**Routing seems incorrect:**
```bash
# Check routing logic
python3 unified_cli.py --benchmark

# Use forced routing
unified> !einstein your query
unified> !bolt your query
```

**Performance issues:**
```bash
# Check system health
python3 unified_cli.py --debug --interactive

# Monitor system stats
unified> stats
```

### Debug Information
```bash
# Enable debug logging
python3 unified_cli.py --debug "your query"

# Check routing decision
unified> help
unified> stats
```

## API Reference

### QueryRouter Class
```python
from unified_cli import QueryRouter

router = QueryRouter()
system, confidence, reasoning = router.classify_query("find WheelStrategy")
# Returns: ("einstein", 0.85, "Simple search query detected")
```

### UnifiedCLI Class
```python
from unified_cli import UnifiedCLI

cli = UnifiedCLI()
result = await cli.route_query("optimize database performance")
# Returns: {"system": "bolt", "success": True, ...}
```

## Integration

### Programmatic Usage
```python
import asyncio
from unified_cli import UnifiedCLI

async def main():
    cli = UnifiedCLI()
    
    # Route query automatically
    result = await cli.route_query("find WheelStrategy")
    
    # Force specific system
    result = await cli.route_query("optimize queries", force_system="bolt")
    
    print(result)

asyncio.run(main())
```

### Shell Integration
```bash
# Add to .bashrc or .zshrc
alias uc='./unified'
alias ucb='./unified --force-bolt'
alias uce='./unified --force-einstein'
alias uci='./unified --interactive'
```

## Contributing

### Adding New Routing Rules
1. Edit `QueryRouter.classify_query()` method
2. Add keywords to `EINSTEIN_KEYWORDS` or `BOLT_KEYWORDS`
3. Add complex patterns to `COMPLEX_PATTERNS`
4. Update test cases in `run_benchmark()`
5. Test with `python3 unified_cli.py --benchmark`

### Performance Optimization
1. Profile routing decisions: `python3 -m cProfile unified_cli.py`
2. Optimize pattern matching for speed
3. Cache routing decisions for repeated queries
4. Monitor system performance metrics

## License

See project LICENSE file.

## Support

For questions and issues:
1. Check troubleshooting section above
2. Run benchmark tests
3. Enable debug mode for detailed logging
4. Review routing logic and examples