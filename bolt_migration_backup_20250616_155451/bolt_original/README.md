# Bolt: 8-Agent System with M4 Pro Hardware Acceleration

A comprehensive integration layer that orchestrates 8 parallel agents with hardware-accelerated tools, optimized for Apple M4 Pro.

## Features

- **8 Parallel Agents**: Concurrent task execution with intelligent routing
- **M4 Pro Optimization**: MLX GPU acceleration, Metal monitoring, 18GB memory limit
- **Hardware-Accelerated Tools**: 10-30x faster than MCP servers
- **Real-Time Monitoring**: CPU, memory, and GPU usage tracking
- **Memory Safety**: Automatic throttling under pressure
- **Task Dependencies**: Complex dependency graph resolution
- **Einstein Integration**: Semantic code search and optimization

## Quick Start

```bash
# Basic usage
bolt solve "optimize all functions"

# Analyze only (no changes)
bolt solve "find memory leaks" --analyze-only

# Run from project root
python bolt_cli.py "refactor database layer"
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make CLI executable
chmod +x bolt_cli.py

# Optional: Add to PATH
ln -s $(pwd)/bolt_cli.py /usr/local/bin/bolt
```

## Architecture

### Core Components

1. **BoltIntegration** (`bolt/integration.py`)
   - Central orchestrator for all agents
   - System state monitoring
   - Task queue management
   - Query analysis and execution

2. **Agent** (`bolt/integration.py`)
   - Individual worker with tool access
   - Hardware-accelerated execution
   - Task state tracking

3. **SystemState** (`bolt/integration.py`)
   - Real-time hardware monitoring
   - Health checks and warnings
   - GPU memory tracking

4. **Accelerated Tools**
   - RipgrepTurbo: 30x faster search
   - DependencyGraph: 12x faster with GPU
   - PythonAnalyzer: 173x faster
   - DuckDBTurbo: No MCP overhead
   - TraceTurbo: Unified tracing

### Task Flow

1. User submits query → `bolt solve "query"`
2. Query analyzed → Task breakdown created
3. Tasks submitted → Dependencies resolved
4. Agents execute → Parallel with health checks
5. Results collected → Formatted output

## Testing

```bash
# Run comprehensive test suite
pytest bolt/test_integration.py -v

# Quick system test
python bolt/test_bolt_system.py

# M4 Pro benchmark
python bolt/benchmark_m4pro.py
```

## Performance

On M4 Pro (8P+4E cores, 20 GPU cores):

- System state capture: ~5ms
- Task throughput: 50+ tasks/second
- GPU acceleration: 100+ GFLOPS
- Memory limit: 18GB enforced
- Parallel efficiency: 8 agents concurrent

## Examples

### Optimization
```bash
bolt solve "optimize database queries for speed"
# Creates tasks:
# - Search for query patterns
# - Analyze performance bottlenecks
# - Profile memory usage
# - Identify optimization opportunities
```

### Debugging
```bash
bolt solve "debug memory leak in trading module"
# Creates tasks:
# - Trace execution paths
# - Analyze error patterns
# - Check dependency conflicts
```

### Refactoring
```bash
bolt solve "refactor wheel strategy for maintainability"
# Creates tasks:
# - Analyze code structure
# - Detect code smells
# - Check cyclic dependencies
```

## API Usage

```python
from bolt.integration import BoltIntegration

async def example():
    # Create system
    bolt = BoltIntegration(num_agents=8)
    await bolt.initialize()
    
    # Execute query
    result = await bolt.execute_query("optimize performance")
    
    # Process results
    for task_result in result['results']:
        print(f"{task_result['task']}: {task_result['status']}")
    
    await bolt.shutdown()
```

## Configuration

Environment variables:
- `PYTORCH_METAL_WORKSPACE_LIMIT_BYTES`: GPU memory limit (default: 18GB)
- `KMP_DUPLICATE_LIB_OK`: Allow duplicate OpenMP libraries
- `MLX_FORCE_CPU`: Force MLX to use CPU (testing only)

## Troubleshooting

### High Memory Usage
The system automatically throttles when memory exceeds 85%. Tasks are requeued until memory pressure reduces.

### GPU Not Detected
Check MLX/PyTorch installation:
```python
import mlx.core as mx
print(mx.metal.is_available())

import torch
print(torch.backends.mps.is_available())
```

### Task Failures
Check agent logs in output. Common issues:
- Missing dependencies
- File not found
- Tool initialization failure

## Development

### Adding New Tools
1. Create tool in `src/unity_wheel/accelerated_tools/`
2. Add to Agent._init_tools()
3. Implement task logic in Agent._execute_task_logic()

### Custom Task Types
1. Extend query analysis in BoltIntegration.analyze_query()
2. Add task execution logic
3. Update tests

## License

See project LICENSE file.