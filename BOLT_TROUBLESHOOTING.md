# Bolt Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Python Version Compatibility
```bash
# Error: Python 3.9+ required
❌ Python 3.8.x found
✅ Solution: Upgrade Python
brew install python@3.11
python3.11 -m pip install --upgrade pip
```

#### MLX Installation Failures
```bash
# Error: MLX not available
❌ Failed to install mlx
✅ Solutions:
# For Apple Silicon
pip install --upgrade mlx mlx-lm

# For Intel Mac (limited GPU support)
pip install mlx-cpu-only
export MLX_FORCE_CPU=1
```

#### Permission Errors
```bash
# Error: Permission denied creating symlink
❌ ln: /usr/local/bin/bolt: Permission denied
✅ Solutions:
# Use sudo for system install
sudo ln -s $(pwd)/bolt_cli.py /usr/local/bin/bolt

# Or add to local PATH
echo 'export PATH="$PATH:$(pwd)"' >> ~/.zshrc
source ~/.zshrc
```

### Runtime Issues

#### Memory Errors

##### High Memory Usage Warning
```bash
⚠️ System warning: High memory usage: 87%
```
**Cause**: System memory above 85% threshold  
**Solutions**:
```bash
# 1. Reduce agent count
bolt solve "query" --agents=4

# 2. Clear system caches
sudo purge

# 3. Close other applications
# 4. Check memory allocations
python -c "
from bolt.memory_manager import get_memory_manager
manager = get_memory_manager()
print(manager.get_status_report())
"
```

##### Memory Allocation Failed
```bash
❌ MemoryError: Could not allocate 1000MB for duckdb
```
**Solutions**:
```python
# Check component budgets
from bolt.memory_manager import get_memory_manager
manager = get_memory_manager()

# Get current allocations
for component in ['duckdb', 'jarvis', 'einstein']:
    stats = manager.get_component_stats(component)
    print(f"{component}: {stats['usage_percent']:.1f}% of {stats['max_mb']:.0f}MB")

# Force cleanup if needed
manager.enforce_limits(strict=True)
```

##### GPU Memory Overflow
```bash
❌ Metal memory allocation failed
```
**Solutions**:
```bash
# 1. Reduce GPU memory limit
export PYTORCH_METAL_WORKSPACE_LIMIT_BYTES=16106127360  # 15GB

# 2. Force CPU fallback
export MLX_FORCE_CPU=1

# 3. Check GPU memory usage
python -c "
import mlx.core as mx
print(f'MLX available: {mx.metal.is_available()}')
print(f'Metal memory info: {mx.metal.get_memory_info()}')
"
```

#### Performance Issues

##### Slow Performance
```bash
bolt solve "query" taking >30 seconds
```
**Diagnostic Steps**:
```python
# 1. Check system health
from bolt.integration import SystemState
state = SystemState.capture()
print(f"CPU: {state.cpu_percent}%, Memory: {state.memory_percent}%")
print(f"Warnings: {state.warnings}")

# 2. Monitor performance
from bolt.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
monitor.start_monitoring()
# Run your query
dashboard = monitor.get_real_time_dashboard()
print(f"Bottleneck: {dashboard['summary']['bottleneck']}")
```

**Solutions**:
```bash
# Increase concurrency limits
# Edit bolt/integration.py AgentExecutionContext._semaphore_limits

# Reduce scope
bolt solve "query with specific scope" --analyze-only

# Check hardware acceleration
python bolt/benchmark_m4pro.py
```

##### Agent Timeouts
```bash
❌ Task agent_0_task_123 timed out after 30s
```
**Solutions**:
```python
# Increase timeouts in context
context = AgentExecutionContext(instruction="query")
context.tool_timeouts["code_analysis"] = 120  # 2 minutes
context.tool_timeouts["optimization"] = 300   # 5 minutes
```

#### System Health Issues

##### High CPU Usage
```bash
⚠️ System warning: High CPU usage: 92%
```
**Solutions**:
```bash
# 1. Reduce agent count
bolt solve "query" --agents=4

# 2. Check for background processes
top -o cpu

# 3. Throttle agent execution
python -c "
import time
from bolt.integration import BoltIntegration
# Add delays between agent spawns
"
```

##### GPU Not Detected
```bash
⚠️ GPU backend: none
```
**Diagnostic**:
```python
# Check MLX installation
try:
    import mlx.core as mx
    print(f"MLX Metal available: {mx.metal.is_available()}")
    print(f"MLX devices: {mx.metal.device_info()}")
except ImportError:
    print("MLX not installed")

# Check PyTorch MPS
try:
    import torch
    print(f"PyTorch MPS available: {torch.backends.mps.is_available()}")
except ImportError:
    print("PyTorch not available")
```

**Solutions**:
```bash
# Reinstall MLX
pip uninstall mlx mlx-lm
pip install mlx mlx-lm

# Check macOS version (requires 12.3+)
sw_vers

# Verify Metal support
system_profiler SPDisplaysDataType | grep Metal
```

### Task Execution Issues

#### Einstein Search Failures
```bash
❌ Einstein search failed: Index not initialized
```
**Solutions**:
```python
# Manual Einstein initialization
from einstein.unified_index import UnifiedIndex
from pathlib import Path

index = UnifiedIndex(base_path=Path("."))
await index.initialize()
print("Einstein index initialized")
```

#### Tool Initialization Failures
```bash
❌ Agent agent_0 error: Tool initialization failed
```
**Diagnostic**:
```python
# Check individual tools
from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
from src.unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph

try:
    rg = get_ripgrep_turbo()
    print("Ripgrep: OK")
except Exception as e:
    print(f"Ripgrep error: {e}")

try:
    dg = get_dependency_graph()
    print("Dependency graph: OK")
except Exception as e:
    print(f"Dependency graph error: {e}")
```

#### Dependency Resolution Failures
```bash
❌ Task dependency cycle detected
```
**Solutions**:
```python
# Check task dependencies manually
from bolt.integration import BoltIntegration
bolt = BoltIntegration()
analysis = await bolt.analyze_query("your query")

# Print task dependencies
for i, task in enumerate(analysis['tasks']):
    print(f"Task {i}: {task['description']}")
    print(f"  Dependencies: {task.get('dependencies', 'None')}")
```

### Development Issues

#### Import Errors
```bash
❌ ImportError: No module named 'bolt.integration'
```
**Solutions**:
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Add project root to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute imports
python -c "
import sys
sys.path.insert(0, '/full/path/to/wheel-trading')
from bolt.integration import BoltIntegration
"
```

#### Testing Failures
```bash
❌ pytest bolt/test_integration.py failing
```
**Solutions**:
```bash
# Run with verbose output
pytest bolt/test_integration.py -v -s

# Run individual test
pytest bolt/test_integration.py::test_basic_query -v

# Check test dependencies
pip install pytest pytest-asyncio

# Clean test environment
rm -rf __pycache__ .pytest_cache
```

### System Monitoring

#### Real-time Diagnostics
```python
#!/usr/bin/env python3
"""Real-time Bolt system diagnostics"""

import asyncio
import time
from bolt.integration import BoltIntegration, SystemState
from bolt.memory_manager import get_memory_manager
from bolt.performance_monitor import get_performance_monitor

async def diagnose_system():
    print("🔍 Bolt System Diagnostics")
    print("=" * 50)
    
    # System state
    state = SystemState.capture()
    print(f"✅ System Health: {'OK' if state.is_healthy else 'WARNING'}")
    print(f"📊 CPU: {state.cpu_percent:.1f}%")
    print(f"💾 Memory: {state.memory_percent:.1f}%")
    print(f"🔥 GPU: {state.gpu_backend} ({state.gpu_memory_used_gb:.1f}GB)")
    
    if state.warnings:
        print("\n⚠️ Warnings:")
        for warning in state.warnings:
            print(f"  - {warning}")
    
    # Memory manager
    print(f"\n💾 Memory Manager Status:")
    manager = get_memory_manager()
    report = manager.get_status_report()
    
    print(f"📈 Total Allocated: {report['system']['total_allocated_mb']:.1f}MB")
    for component, stats in report['components'].items():
        usage = stats['usage_percent']
        status = "🔴" if usage > 90 else "🟡" if usage > 75 else "🟢"
        print(f"  {status} {component}: {usage:.1f}% ({stats['allocated_mb']:.1f}MB)")
    
    # Test basic functionality
    print(f"\n🧪 Testing Basic Functionality:")
    bolt = BoltIntegration(num_agents=2)  # Reduced for testing
    
    try:
        await bolt.initialize()
        print("✅ Initialization: OK")
        
        # Test query analysis
        result = await bolt.analyze_query("test query")
        print(f"✅ Query Analysis: OK ({len(result.get('tasks', []))} tasks)")
        
        await bolt.shutdown()
        print("✅ Shutdown: OK")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
    
    print(f"\n🏁 Diagnostics Complete")

if __name__ == "__main__":
    asyncio.run(diagnose_system())
```

### Performance Tuning

#### Optimization Checklist
```bash
# 1. Check hardware detection
python -c "
from bolt.hardware_state import get_hardware_state
hw = get_hardware_state()
print(f'P-cores: {hw.cpu.p_cores}, E-cores: {hw.cpu.e_cores}')
print(f'Memory: {hw.memory.total_gb:.1f}GB')
print(f'GPU cores: {hw.gpu.cores}')
"

# 2. Verify accelerated tools
python -c "
from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
rg = get_ripgrep_turbo()
print(f'Ripgrep cores: {rg.max_workers}')
"

# 3. Test GPU acceleration
python bolt/test_gpu_acceleration.py

# 4. Benchmark system
python bolt/benchmark_m4pro.py
```

### Getting Help

#### Log Collection
```bash
# Enable debug logging
export BOLT_LOG_LEVEL=DEBUG

# Run with full logging
bolt solve "query" --analyze-only 2>&1 | tee bolt_debug.log

# Collect system info
python -c "
import sys, platform, os
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.machine()}')
print(f'CPU cores: {os.cpu_count()}')

try:
    import psutil
    mem = psutil.virtual_memory()
    print(f'Memory: {mem.total // (1024**3)}GB total, {mem.available // (1024**3)}GB available')
except:
    pass
"
```

#### Bug Reports
When reporting issues, include:
1. Full error message and stack trace
2. System information (OS, Python version, hardware)
3. Command that triggered the issue
4. Debug logs (`BOLT_LOG_LEVEL=DEBUG`)
5. Memory/CPU usage during failure
6. Steps to reproduce

#### Emergency Recovery
```bash
# Stop all Bolt processes
pkill -f bolt
pkill -f "python.*bolt"

# Clear caches
python -c "
import gc
gc.collect()
"

# Reset system state
python -c "
from bolt.memory_manager import get_memory_manager
manager = get_memory_manager()
manager.enforce_limits(strict=True)
"

# Restart with minimal configuration
bolt solve "simple test query" --analyze-only --agents=1
```