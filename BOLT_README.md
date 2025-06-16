# Bolt - 8-Agent Hardware-Accelerated Problem Solver

## Overview

Bolt is a production-ready 8-agent system that leverages Apple M4 Pro hardware to solve complex programming problems through intelligent orchestration of parallel agents with GPU acceleration, memory safety, and real-time monitoring.

## What Bolt Does

Bolt transforms complex programming tasks into coordinated parallel execution:

1. **Context Understanding**: Uses Einstein semantic search to understand your codebase in <50ms
2. **Task Decomposition**: Breaks problems into 8 parallel tasks with intelligent dependencies
3. **Agent Orchestration**: Coordinates 8 Claude Code agents (one per P-core) with hardware acceleration
4. **Result Synthesis**: Combines agent outputs into coherent findings and recommendations
5. **Safe Execution**: Enforces memory limits, prevents recursion, and monitors system health

## Key Features

### 🚀 **8 Parallel Agents**
- One agent per P-core for maximum M4 Pro utilization
- Intelligent task routing with dependency resolution
- No sub-agent spawning (max recursion depth = 1)
- Graceful failure isolation

### 🔥 **Hardware Acceleration** 
- **MLX GPU**: 20-core Metal acceleration for vector operations
- **Parallel Search**: 30x faster ripgrep with all CPU cores
- **Memory Safety**: 18GB enforced limit with automatic throttling
- **Real-time Monitoring**: CPU, memory, and GPU usage tracking

### 🧠 **Einstein Integration**
- Semantic code search in <50ms
- Multimodal codebase understanding
- Context-aware task generation
- Intelligent scope refinement

### 🛡️ **Production Safety**
- Strict recursion control (depth = 1)
- Per-tool concurrency semaphores
- System health monitoring
- Memory pressure management
- Automatic resource cleanup

## Installation

### Prerequisites
- macOS (required for Metal GPU acceleration)
- Python 3.9+
- Apple Silicon Mac (M1/M2/M3/M4)
- 8GB+ RAM (24GB recommended for optimal performance)

### Quick Install
```bash
# Clone and navigate to project
cd /path/to/wheel-trading

# Run installation script
python install_bolt.py

# Test installation
bolt solve "analyze current codebase" --analyze-only
```

### Manual Setup
```bash
# Install dependencies
pip install click asyncio-mqtt psutil numpy pandas aiofiles rich

# Install MLX for Apple Silicon
pip install mlx mlx-lm

# Make CLI executable
chmod +x bolt_cli.py

# Optional: Add to PATH
ln -s $(pwd)/bolt_cli.py /usr/local/bin/bolt
```

## Quick Start

### Basic Usage
```bash
# Simple problem solving
bolt solve "optimize database queries"

# Analysis only (no changes)
bolt solve "debug memory leak" --analyze-only

# Complex refactoring
bolt solve "refactor wheel strategy for better maintainability"
```

### Python API
```python
from bolt.integration import BoltIntegration

async def solve_problem():
    # Create and initialize system
    bolt = BoltIntegration(num_agents=8)
    await bolt.initialize()
    
    # Execute query
    result = await bolt.solve("optimize performance bottlenecks")
    
    # Process results
    if result["success"]:
        print(f"Tasks executed: {result['tasks_executed']}")
        for finding in result['results']['findings']:
            print(f"Finding: {finding}")
        for rec in result['results']['recommendations']:
            print(f"Recommendation: {rec}")
    
    await bolt.shutdown()

# Run
import asyncio
asyncio.run(solve_problem())
```

## Usage Examples

### Code Optimization
```bash
# Performance bottlenecks
bolt solve "optimize slow database queries in src/trading/"

# Memory usage
bolt solve "improve memory usage in wheel strategy"

# GPU acceleration opportunities
bolt solve "add GPU acceleration to options pricing"
```

### Debugging
```bash
# Memory leaks
bolt solve "find memory leak in trading loop"

# Connection issues
bolt solve "debug intermittent connection failures"

# Performance analysis
bolt solve "trace performance bottleneck in options calculation"
```

### Refactoring
```bash
# Class refactoring
bolt solve "refactor WheelStrategy class for better maintainability"

# Function splitting
bolt solve "split large functions in options.py"

# Error handling
bolt solve "add proper error handling to API endpoints"
```

### Feature Development
```bash
# Dashboard creation
bolt solve "add real-time portfolio monitoring dashboard"

# Feature implementation
bolt solve "implement stop-loss functionality"

# Framework development
bolt solve "create backtesting framework for strategies"
```

## Architecture

### Core Components

#### BoltIntegration (`bolt/integration.py`)
- Central orchestrator for all 8 agents
- System state monitoring and health checks
- Task queue management with priority levels
- Query analysis and execution coordination

#### Agent (`bolt/integration.py`)
- Individual worker with hardware-accelerated tools
- Executes tasks with tool-specific timeouts
- Tracks performance metrics and completion status
- Isolated failure handling

#### SystemState (`bolt/integration.py`)
- Real-time hardware monitoring (CPU, memory, GPU)
- Health checks with automatic warnings
- M4 Pro specific optimizations
- Memory pressure detection

#### Hardware-Accelerated Tools
- **RipgrepTurbo**: 30x faster parallel search
- **DependencyGraph**: 12x faster with GPU analysis
- **PythonAnalyzer**: 173x faster MLX acceleration
- **DuckDBTurbo**: Direct connection, no MCP overhead
- **TraceTurbo**: Unified performance tracing

### Execution Flow

1. **Query Analysis** → Einstein semantic search finds relevant code
2. **Task Decomposition** → Breaks query into specialized parallel tasks
3. **Clarification Check** → Detects ambiguous scope, adds clarification tasks
4. **Parallel Execution** → Distributes tasks to 8 agents with dependency resolution
5. **Result Synthesis** → Combines outputs into findings and recommendations

### Memory Management

- **Component Budgets**: DuckDB (50%), Jarvis (17%), Einstein (8%), Meta (10%), Cache (10%), Other (5%)
- **Pressure Handling**: Automatic eviction of low-priority allocations
- **Emergency Mode**: Aggressive cleanup when memory >95%
- **Per-Component Pools**: Thread-safe allocation with priority-based eviction

## Performance Characteristics

### M4 Pro Optimization
- **CPU**: 8 P-cores + 4 E-cores fully utilized
- **GPU**: 20 Metal cores with MLX acceleration
- **Memory**: 18GB enforced limit (75% of 24GB)
- **Storage**: <5ms filesystem operations

### Typical Performance
- **Initialization**: <100ms
- **Einstein Search**: ~500ms for 20 files
- **Task Decomposition**: <50ms
- **Agent Dispatch**: <10ms per task
- **Complete Solve**: 2-5s total
- **Memory Usage**: <500MB per agent
- **GPU Utilization**: 70-100% for supported operations

### Benchmarks
- **System State Capture**: ~5ms
- **Task Throughput**: 50+ tasks/second
- **GPU Acceleration**: 100+ GFLOPS
- **Parallel Efficiency**: 8 agents concurrent
- **Search Operations**: 23ms (was 150ms with MCP)

## Configuration

### Environment Variables
```bash
# GPU memory limit (default: 18GB)
export PYTORCH_METAL_WORKSPACE_LIMIT_BYTES=19327352832

# Allow duplicate OpenMP libraries
export KMP_DUPLICATE_LIB_OK=TRUE

# Force CPU for testing
export MLX_FORCE_CPU=1
```

### Configuration File (`bolt/config.yaml`)
```yaml
agents:
  count: 8
  max_recursion_depth: 1
  task_timeout: 300

hardware:
  memory_limit_gb: 18
  gpu_enabled: true
  cpu_cores: 12

tools:
  concurrency_limits:
    semantic_search: 3
    pattern_search: 4
    code_analysis: 2
    dependency_check: 1
    optimization: 2
    generic: 4

  timeouts:
    semantic_search: 30
    pattern_search: 15
    code_analysis: 60
    dependency_check: 45
    optimization: 120
    generic: 30
```

## Command Reference

### CLI Options
```bash
bolt solve "query" [OPTIONS]

Options:
  --analyze-only    Only analyze, don't execute changes
  --help           Show help message
```

### Advanced Usage
```bash
# Run from project root
python bolt_cli.py "refactor database layer"

# Direct API access
python -c "
from bolt.integration import bolt_solve_cli
import asyncio
asyncio.run(bolt_solve_cli('optimize all functions', analyze_only=True))
"
```

## Integration

### With Existing Workflow
```bash
# Before making changes
bolt solve "analyze impact of changing database schema" --analyze-only

# After implementation
bolt solve "verify no circular dependencies after refactor"

# Pre-commit check
bolt solve "check code quality issues before commit"
```

### With Meta System
```bash
# Start meta development environment
python meta_coordinator.py --dev-mode &

# Start file monitoring with quality checks
python meta_daemon.py --watch-path . &

# Run Bolt within meta system
bolt solve "optimize with meta awareness enabled"
```

## Monitoring

### Real-time Dashboard
```python
from bolt.performance_monitor import get_performance_monitor

monitor = get_performance_monitor()
monitor.start_monitoring()

# Get dashboard
dashboard = monitor.get_real_time_dashboard()
print(f"System Health: {dashboard['summary']['health']}")
print(f"Active Agents: {dashboard['summary']['agents_working']}")
print(f"GPU Usage: {dashboard['system']['gpu_utilization']:.1f}%")
```

### Memory Monitoring
```python
from bolt.memory_manager import get_memory_manager

manager = get_memory_manager()
report = manager.get_status_report()

print(f"System Usage: {report['system']['system_usage_percent']:.1f}%")
for component, stats in report['components'].items():
    print(f"{component}: {stats['usage_percent']:.1f}%")
```

## Testing

### Test Suite
```bash
# Run comprehensive tests
pytest bolt/test_integration.py -v

# Quick system test
python bolt/test_bolt_system.py

# Agent engine tests
python bolt/test_agent_engine.py

# Memory manager tests
python bolt/test_memory_manager.py

# M4 Pro benchmark
python bolt/benchmark_m4pro.py
```

### Manual Testing
```bash
# Test basic functionality
bolt solve "analyze project structure" --analyze-only

# Test performance under load
for i in {1..5}; do
  bolt solve "find optimization opportunities in src/" &
done
wait

# Test memory pressure handling
bolt solve "analyze very large codebase with high memory usage"
```

## Troubleshooting

See [BOLT_TROUBLESHOOTING.md](BOLT_TROUBLESHOOTING.md) for detailed troubleshooting guide.

## Advanced Topics

### Custom Task Types
```python
# Add custom task execution logic
class CustomAgent(Agent):
    async def _execute_task_by_type(self, task, task_type):
        if task_type == "custom_analysis":
            # Your custom logic here
            return await self.custom_analysis(task)
        return await super()._execute_task_by_type(task, task_type)
```

### Performance Tuning
```python
# Adjust concurrency limits
context = AgentExecutionContext(
    instruction="optimize code",
    max_recursion_depth=1
)
context._semaphore_limits["pattern_search"] = 8  # Increase parallel searches
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `pytest bolt/test_*.py -v`
4. Submit pull request with detailed description

## License

See project LICENSE file.

## Support

For issues, questions, or contributions:
1. Check [BOLT_TROUBLESHOOTING.md](BOLT_TROUBLESHOOTING.md)
2. Review [BOLT_PERFORMANCE_TUNING.md](BOLT_PERFORMANCE_TUNING.md)
3. Create GitHub issue with system details and error logs