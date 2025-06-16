# Unity Wheel Memory Management Framework

A comprehensive memory management system designed specifically for trading applications with intelligent allocation strategies, pressure monitoring, and resource scheduling.

## Features

### 🧠 Core Memory Management
- **Unified Memory Manager**: Central coordinator for all memory operations
- **Component-Specific Budgets**: Separate memory pools for trading data, ML models, database operations, and caching
- **Intelligent Allocation**: Smart allocation strategies optimized for each component type
- **Real-time Monitoring**: Continuous memory pressure detection and adaptive responses

### 📊 Memory Pools
- **Standard Memory Pool**: Virtual allocation tracking with fragmentation management
- **Shared Tensor Pool**: Optimized for ML operations with GPU-aware tensor management
- **Circular Pool**: For streaming data with automatic wrap-around
- **Object Pool**: Reusable object management with factory pattern

### 🔄 Pressure Management
- **Real-time Monitoring**: Continuous system memory tracking
- **Trend Analysis**: Predictive pressure detection with slope analysis
- **Adaptive Thresholds**: Dynamic pressure levels based on system state
- **Emergency Protocols**: Automatic cleanup when memory becomes critical

### 🧹 Cleanup System
- **Multi-level Cleanup**: Light, moderate, aggressive, and emergency cleanup strategies
- **Component-Specific**: Tailored cleanup for different data types
- **Automated Scheduling**: Background cleanup based on pressure and time intervals
- **Emergency Measures**: Comprehensive cleanup protocols for critical situations

### 📋 Resource Scheduling
- **Memory-Aware Scheduling**: Task scheduling based on memory availability
- **Priority Management**: Intelligent task prioritization with preemption support
- **Adaptive Strategies**: Dynamic scheduling based on system state
- **Resource Tracking**: Performance monitoring and optimization recommendations

## Architecture

```
UnifiedMemoryManager
├── PressureMonitor (Real-time monitoring)
├── CleanupSystem (Automated cleanup)
├── ComponentPools
│   ├── TradingDataPool (35% budget - 7GB)
│   ├── MLModelsPool (25% budget - 5GB)
│   ├── DatabasePool (25% budget - 5GB)
│   └── CachePool (10% budget - 2GB)
└── ResourceScheduler (Memory-aware task scheduling)
```

## Usage Examples

### Basic Memory Allocation

```python
from unity_wheel.memory import get_memory_manager

# Get the global memory manager
manager = get_memory_manager()

# Allocate memory for trading data
alloc_id = manager.allocate(
    component='trading_data',
    size_mb=100,
    description='SPY options chain',
    priority=8,
    tags=['options', 'real_time']
)

# Use the memory...

# Deallocate when done
manager.deallocate(alloc_id)
```

### Context Managers (Recommended)

```python
from unity_wheel.memory import (
    allocate_for_trading,
    allocate_for_ml,
    allocate_tensor_memory
)

# Trading data allocation
with allocate_for_trading(100, "Options processing", priority=8) as alloc_id:
    # Process options data
    pass

# ML model allocation
with allocate_for_ml(500, "BERT model loading", priority=9) as alloc_id:
    # Load and use ML model
    pass

# Tensor allocation with automatic size calculation
with allocate_tensor_memory((1000, 512), np.float32, "Embeddings") as alloc_id:
    # Use tensor memory
    pass
```

### Adaptive Memory Allocation

```python
from unity_wheel.memory import allocate_adaptive_memory, allocate_with_fallback

# Adaptive allocation - tries smaller sizes until successful
with allocate_adaptive_memory(200, "Large dataset processing") as (alloc_id, actual_size):
    print(f"Got {actual_size}MB (requested 200MB)")
    # Work with whatever size was allocated

# Fallback allocation - tries primary size, falls back to secondary
with allocate_with_fallback(100, 50, "Data processing") as (alloc_id, size):
    if size == 50:
        print("Using fallback allocation")
    # Process with available memory
```

### Resource Scheduling

```python
from unity_wheel.memory import (
    get_resource_scheduler,
    schedule_trading_task,
    schedule_ml_task,
    TaskPriority,
    ResourceRequirements
)

scheduler = get_resource_scheduler()

# Schedule a trading task
task_id = schedule_trading_task(
    name="Process options chain",
    func=process_options,
    memory_mb=150,
    priority=TaskPriority.HIGH,
    symbol="SPY"
)

# Schedule an ML task with custom requirements
requirements = ResourceRequirements(
    memory_mb=800,
    estimated_duration_seconds=300,
    component='ml_models',
    memory_priority=9
)

task_id = scheduler.submit_task(
    name="Train model",
    func=train_model,
    requirements=requirements,
    priority=TaskPriority.CRITICAL
)

# Check task status
status = scheduler.get_task_status(task_id)
print(f"Task state: {status.state}")
```

### Memory Pools

```python
from unity_wheel.memory import (
    create_ml_tensor_pool,
    create_trading_data_pool,
    SharedTensorPool
)
import numpy as np

# Create specialized pools
tensor_pool = create_ml_tensor_pool(1000)  # 1GB tensor pool
trading_pool = create_trading_data_pool(500)  # 500MB trading pool

# Allocate tensors in shared memory
alloc_id = tensor_pool.allocate_tensor(
    shape=(1000, 512),
    dtype=np.float32,
    description="Embedding vectors"
)

# Get tensor for computation
tensor = tensor_pool.get_tensor(alloc_id)
# Use tensor...

tensor_pool.deallocate(alloc_id)
```

### Monitoring and Statistics

```python
from unity_wheel.memory import memory_usage_report

# Get comprehensive memory report
report = memory_usage_report()
print(f"System usage: {report['system']['system_usage_percent']:.1f}%")
print(f"Total allocated: {report['system']['allocated_mb']:.1f}MB")

# Component-specific usage
for component, stats in report['components'].items():
    print(f"{component}: {stats['usage_percent']:.1f}% "
          f"({stats['allocated_mb']:.1f}/{stats['budget_mb']:.1f}MB)")

# Scheduler statistics
scheduler = get_resource_scheduler()
stats = scheduler.get_statistics()
print(f"Task completion rate: {stats['performance']['completion_rate']:.1%}")
print(f"Average wait time: {stats['performance']['average_wait_time']:.1f}s")
```

## Configuration

### System Configuration (M4 Pro - 24GB)
- **Total Memory**: 24GB
- **Usable Memory**: 20GB (leaving 4GB for system)
- **Component Budgets**:
  - Trading Data: 35% (7GB) - Price data, options chains, market data
  - ML Models: 25% (5GB) - Neural networks, embeddings, training
  - Database: 25% (5GB) - DuckDB, SQLite operations
  - Cache: 10% (2GB) - General caching
  - System Buffer: 5% (1GB) - Emergency buffer

### Pressure Thresholds
- **Normal**: < 70% memory usage
- **Low Pressure**: 70-80% memory usage
- **Medium Pressure**: 80-85% memory usage
- **High Pressure**: 85-90% memory usage
- **Critical**: 90-95% memory usage
- **Emergency**: > 95% memory usage

## Best Practices

### 1. Use Context Managers
Always use context managers for automatic cleanup:
```python
# Good
with allocate_for_trading(100, "Options data") as alloc_id:
    process_data()

# Avoid manual allocation/deallocation
alloc_id = manager.allocate(...)
# ... (might forget to deallocate)
manager.deallocate(alloc_id)
```

### 2. Set Appropriate Priorities
- **Critical (9-10)**: System-critical operations, active ML models
- **High (7-8)**: Real-time trading data, important calculations
- **Normal (5-6)**: Standard operations, query results
- **Low (3-4)**: Background tasks, cache operations
- **Deferred (1-2)**: Non-urgent operations, temporary data

### 3. Tag Your Allocations
Use descriptive tags for better monitoring:
```python
with allocate_for_trading(100, "SPY options", tags=['options', 'real_time', 'spy']) as alloc_id:
    pass
```

### 4. Monitor Memory Pressure
Register callbacks for memory pressure events:
```python
def handle_pressure(pressure_level):
    if pressure_level > 0.9:
        # Take action: reduce batch sizes, clear caches, etc.
        pass

manager = get_memory_manager()
manager.register_pressure_callback(handle_pressure)
```

### 5. Use Appropriate Pool Types
- **Trading Data**: Use standard pools with LRU eviction
- **ML Models**: Use tensor pools for efficient GPU operations
- **Streaming Data**: Use circular pools for continuous data
- **Temporary Objects**: Use object pools for reusable resources

## Error Handling

### Memory Allocation Failures
```python
from unity_wheel.memory import allocate_for_trading

try:
    with allocate_for_trading(1000, "Large operation") as alloc_id:
        # Process data
        pass
except MemoryError as e:
    # Fallback to smaller allocation or alternative strategy
    with allocate_for_trading(500, "Reduced operation") as alloc_id:
        # Process with less memory
        pass
```

### Pressure Monitoring
```python
manager = get_memory_manager()

def emergency_handler():
    logger.critical("Emergency memory situation")
    # Clear all non-critical caches
    # Reduce processing batch sizes
    # Defer non-urgent operations

manager.register_emergency_callback(emergency_handler)
```

## Integration with Trading Components

### Options Processing
```python
from unity_wheel.memory import allocate_for_trading

def process_options_chain(symbol: str, expiry: str):
    estimated_size = estimate_options_memory(symbol, expiry)
    
    with allocate_for_trading(
        estimated_size, 
        f"Options chain {symbol} {expiry}",
        priority=7,
        tags=['options', symbol.lower()]
    ) as alloc_id:
        # Load and process options data
        options_data = load_options_chain(symbol, expiry)
        return analyze_options(options_data)
```

### ML Model Training
```python
from unity_wheel.memory import schedule_ml_task, ResourceRequirements

def schedule_model_training(model_config):
    requirements = ResourceRequirements(
        memory_mb=2000,  # 2GB for training
        estimated_duration_seconds=1800,  # 30 minutes
        component='ml_models',
        memory_priority=8
    )
    
    return schedule_ml_task(
        name=f"Train {model_config['name']}",
        func=train_model,
        memory_mb=2000,
        priority=TaskPriority.HIGH,
        config=model_config
    )
```

### Database Queries
```python
from unity_wheel.memory import allocate_for_database

def execute_large_query(query: str):
    # Estimate result set size
    estimated_mb = estimate_query_result_size(query)
    
    with allocate_for_database(
        estimated_mb,
        f"Query: {query[:50]}...",
        priority=6,
        tags=['query_result']
    ) as alloc_id:
        return execute_query(query)
```

## Performance Monitoring

The memory system provides comprehensive monitoring and statistics:

- **Allocation Success Rates**: Track allocation failures by component
- **Memory Pressure Events**: Monitor frequency and severity of pressure situations
- **Cleanup Effectiveness**: Measure how much memory is freed by cleanup operations
- **Task Scheduling Performance**: Track wait times and completion rates
- **Component Usage Patterns**: Analyze memory usage by component over time

Use these metrics to optimize your application's memory usage and improve overall system performance.