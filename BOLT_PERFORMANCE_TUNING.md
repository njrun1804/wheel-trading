# Bolt Performance Tuning Guide

## Overview

This guide provides detailed instructions for optimizing Bolt's performance on M4 Pro hardware, including hardware-specific tuning, memory optimization, and advanced configuration options.

## Hardware Optimization

### M4 Pro Specific Settings

#### CPU Configuration
```bash
# Set CPU affinity for optimal P-core usage
export BOLT_CPU_AFFINITY="0,1,2,3,4,5,6,7"  # P-cores only
export BOLT_E_CORE_USAGE="background"        # Use E-cores for monitoring

# Disable CPU throttling during intensive operations
sudo pmset -c disablesleep 1
sudo pmset -c displaysleep 0
```

#### Memory Configuration
```python
# Optimal memory allocation for M4 Pro (24GB)
MEMORY_CONFIG = {
    "system_reserve_gb": 4,      # Reserve for macOS
    "bolt_allocation_gb": 18,    # 75% for Bolt
    "emergency_buffer_gb": 2,    # Emergency reserve
}

# Component-specific tuning
COMPONENT_TUNING = {
    "duckdb": {
        "memory_limit": "9GB",           # 50% of Bolt allocation
        "threads": 8,                    # P-cores only
        "temp_directory": "/tmp/bolt"    # Fast SSD storage
    },
    "jarvis": {
        "memory_limit": "3GB",           # 17% of allocation
        "index_cache_size": "1GB",       # Aggressive caching
        "parallel_workers": 4            # Limited for memory efficiency
    },
    "einstein": {
        "memory_limit": "1.5GB",         # 8% of allocation
        "embedding_cache": "512MB",      # Cache embeddings
        "batch_size": 128                # Optimal for M4 Pro
    }
}
```

#### GPU Optimization
```python
# MLX GPU configuration for M4 Pro
import os
os.environ.update({
    # Use 18GB of 20GB available (leave buffer)
    "PYTORCH_METAL_WORKSPACE_LIMIT_BYTES": "19327352832",
    
    # Enable Metal shader caching
    "MLX_METAL_CACHE_ENABLE": "1",
    "MLX_METAL_CACHE_PATH": "/tmp/mlx_cache",
    
    # Optimize for M4 Pro's 20 GPU cores
    "MLX_GPU_CORES": "20",
    "MLX_MEMORY_POOL": "16GB",
    
    # Enable optimizations
    "MLX_FUSE_OPERATIONS": "1",
    "MLX_STREAM_PARALLEL": "1"
})
```

### Tool-Specific Performance Tuning

#### Ripgrep Turbo Optimization
```python
# Configure ripgrep for maximum M4 Pro performance
RIPGREP_CONFIG = {
    "max_workers": 12,              # All CPU cores
    "chunk_size": 1024,             # Optimal chunk size
    "memory_map": True,             # Use memory mapping
    "parallel_glob": True,          # Parallel file discovery
    "ignore_case": False,           # Faster exact matching
    "buffer_size": "64KB",          # Optimal buffer size
}

# File-specific optimizations
RIPGREP_FILE_TYPES = {
    "python": {"extensions": [".py"], "priority": "high"},
    "javascript": {"extensions": [".js", ".ts"], "priority": "medium"},
    "config": {"extensions": [".json", ".yaml", ".toml"], "priority": "low"}
}
```

#### Dependency Graph Acceleration
```python
# GPU-accelerated dependency analysis
DEPENDENCY_CONFIG = {
    "use_gpu": True,
    "gpu_batch_size": 256,          # Optimal for M4 Pro
    "cache_graph": True,            # Persistent caching
    "parallel_ast": 8,              # P-core parallelization
    "memory_limit": "2GB",          # Controlled memory usage
    "cycle_detection": "gpu"        # GPU-based cycle detection
}
```

#### Python Analysis Turbo
```python
# MLX-accelerated Python analysis
PYTHON_ANALYSIS_CONFIG = {
    "mlx_enabled": True,
    "gpu_memory_limit": "4GB",      # Dedicated GPU memory
    "batch_size": 64,               # Optimal batch size
    "parallel_files": 8,            # P-core parallelization
    "ast_cache": True,              # Cache parsed ASTs
    "complexity_threshold": 10,     # Skip simple functions
}
```

## System-Level Optimizations

### Memory Pressure Management

#### Dynamic Memory Allocation
```python
class OptimizedMemoryManager:
    def __init__(self):
        self.adaptive_limits = {
            "low_pressure": {"duckdb": 0.60, "jarvis": 0.20, "einstein": 0.10},
            "medium_pressure": {"duckdb": 0.45, "jarvis": 0.15, "einstein": 0.08},
            "high_pressure": {"duckdb": 0.30, "jarvis": 0.10, "einstein": 0.05}
        }
    
    def adjust_for_pressure(self, pressure_level):
        """Dynamically adjust memory limits based on system pressure"""
        limits = self.adaptive_limits[pressure_level]
        for component, percentage in limits.items():
            self.update_component_limit(component, percentage)
```

#### Memory Pool Optimization
```python
# Pre-allocate memory pools for better performance
MEMORY_POOLS = {
    "small_objects": {"size": "100MB", "count": 1000},
    "medium_objects": {"size": "500MB", "count": 100},
    "large_objects": {"size": "1GB", "count": 10}
}

# Memory allocation strategies
ALLOCATION_STRATEGIES = {
    "duckdb_queries": "large_pool_first",
    "jarvis_indices": "medium_pool_preferred",
    "einstein_embeddings": "small_pool_batch"
}
```

### Concurrency Optimization

#### Agent Pool Tuning
```python
# Optimal agent configuration for M4 Pro
AGENT_CONFIG = {
    "total_agents": 8,              # One per P-core
    "agent_affinity": {
        "agent_0": {"cpu": 0, "priority": "high"},
        "agent_1": {"cpu": 1, "priority": "high"},
        "agent_2": {"cpu": 2, "priority": "high"},
        "agent_3": {"cpu": 3, "priority": "high"},
        "agent_4": {"cpu": 4, "priority": "medium"},
        "agent_5": {"cpu": 5, "priority": "medium"},
        "agent_6": {"cpu": 6, "priority": "medium"},
        "agent_7": {"cpu": 7, "priority": "medium"}
    },
    "load_balancing": "round_robin",
    "task_stealing": True           # Allow work stealing
}
```

#### Task Queue Optimization
```python
# Priority-based task scheduling
TASK_SCHEDULING = {
    "queue_type": "priority_fifo",
    "max_queue_size": 1000,
    "priority_levels": {
        "CRITICAL": {"weight": 1.0, "timeout": 300},
        "HIGH": {"weight": 0.8, "timeout": 180},  
        "NORMAL": {"weight": 0.6, "timeout": 120},
        "LOW": {"weight": 0.4, "timeout": 60}
    },
    "batching": {
        "enabled": True,
        "batch_size": 4,
        "timeout_ms": 100
    }
}
```

### GPU Acceleration Tuning

#### MLX Performance Optimization
```python
# Advanced MLX configuration
import mlx.core as mx

# Configure MLX for optimal M4 Pro performance
mx.set_default_device(mx.gpu)
mx.set_memory_pool_size(16 * 1024 * 1024 * 1024)  # 16GB pool

# Operation-specific tuning
MLX_OPERATIONS = {
    "matrix_multiply": {
        "tile_size": 1024,
        "use_fast_math": True,
        "optimize_for_memory": False
    },
    "similarity_search": {
        "batch_size": 512,
        "use_half_precision": True,
        "stream_parallel": True
    },
    "embeddings": {
        "chunk_size": 256,
        "cache_embeddings": True,
        "async_compute": True
    }
}
```

#### GPU Memory Management
```python
class GPUMemoryOptimizer:
    def __init__(self):
        self.memory_pools = {
            "small": mx.memory.Pool(size=1024*1024*1024),    # 1GB
            "medium": mx.memory.Pool(size=4*1024*1024*1024), # 4GB
            "large": mx.memory.Pool(size=8*1024*1024*1024)   # 8GB
        }
    
    def allocate_optimal(self, size_bytes):
        """Choose optimal memory pool based on size"""
        if size_bytes < 100*1024*1024:        # <100MB
            return self.memory_pools["small"]
        elif size_bytes < 1*1024*1024*1024:   # <1GB
            return self.memory_pools["medium"]
        else:
            return self.memory_pools["large"]
```

## Advanced Configuration

### Einstein Optimization

#### Semantic Search Tuning
```python
# Einstein performance configuration
EINSTEIN_CONFIG = {
    "index_type": "faiss_gpu",          # GPU-accelerated index
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 128,                  # Optimal for M4 Pro
    "cache_size": 10000,                # Cache 10k embeddings
    "gpu_memory": "4GB",                # Dedicated GPU memory
    "parallel_queries": 4,              # Concurrent search queries
    "rerank_top_k": 100,                # Rerank top 100 results
}

# Code understanding optimization
CODE_ANALYSIS_CONFIG = {
    "ast_cache_size": 5000,             # Cache 5k AST trees
    "symbol_index_size": 50000,         # 50k symbols
    "dependency_cache": True,           # Cache dependency graphs
    "incremental_updates": True,        # Only update changed files
}
```

### Tool Timeout Optimization

#### Dynamic Timeout Adjustment
```python
class AdaptiveTimeoutManager:
    def __init__(self):
        self.base_timeouts = {
            "semantic_search": 30,
            "pattern_search": 15,
            "code_analysis": 60,
            "dependency_check": 45,
            "optimization": 120
        }
        self.performance_history = {}
    
    def get_adaptive_timeout(self, tool_type, complexity_score):
        """Calculate timeout based on historical performance and complexity"""
        base = self.base_timeouts[tool_type]
        
        # Adjust based on complexity
        complexity_multiplier = 1.0 + (complexity_score / 10.0)
        
        # Adjust based on recent performance
        if tool_type in self.performance_history:
            avg_time = self.performance_history[tool_type]["avg_time"]
            if avg_time > base * 0.8:  # If often near timeout
                complexity_multiplier *= 1.5
        
        return int(base * complexity_multiplier)
```

## Monitoring and Profiling

### Performance Monitoring Setup
```python
# Comprehensive performance monitoring
MONITORING_CONFIG = {
    "metrics_interval": 1.0,            # 1 second intervals
    "history_size": 3600,               # 1 hour of history
    "alert_thresholds": {
        "cpu_warning": 85,
        "cpu_critical": 95,
        "memory_warning": 85,
        "memory_critical": 95,
        "gpu_warning": 90,
        "gpu_critical": 98
    },
    "performance_baselines": {
        "task_completion_ms": 5000,      # 5 second baseline
        "memory_allocation_ms": 100,     # 100ms allocation time
        "gpu_operation_ms": 1000         # 1 second GPU ops
    }
}
```

### Profiling Tools
```python
# Built-in profiling utilities
class BoltProfiler:
    def __init__(self):
        self.enable_cpu_profiling = True
        self.enable_memory_profiling = True
        self.enable_gpu_profiling = True
        
    def profile_solve_operation(self, query):
        """Profile a complete solve operation"""
        with self.profile_context("solve_operation"):
            # CPU profiling
            cpu_profile = self.profile_cpu_usage()
            
            # Memory profiling
            memory_profile = self.profile_memory_usage()
            
            # GPU profiling
            gpu_profile = self.profile_gpu_usage()
            
            return {
                "cpu": cpu_profile,
                "memory": memory_profile,
                "gpu": gpu_profile
            }
```

## Benchmarking and Testing

### Performance Benchmarks
```bash
# Run comprehensive benchmarks
python bolt/benchmark_m4pro.py --full-suite

# Test specific components
python bolt/benchmark_m4pro.py --component=ripgrep
python bolt/benchmark_m4pro.py --component=gpu_acceleration
python bolt/benchmark_m4pro.py --component=memory_management

# Stress testing
python bolt/benchmark_m4pro.py --stress-test --duration=300
```

### Load Testing
```python
# Concurrent load testing
async def load_test_bolt():
    """Test Bolt under concurrent load"""
    tasks = []
    
    # Create multiple concurrent solve operations
    for i in range(8):  # One per agent
        task = asyncio.create_task(
            run_bolt_solve(f"test query {i}")
        )
        tasks.append(task)
    
    # Monitor system during load
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    
    # Wait for completion
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    performance_report = monitor.get_performance_report()
    return {
        "task_results": results,
        "performance": performance_report
    }
```

## Configuration Files

### Production Configuration (`bolt_production.yaml`)
```yaml
system:
  hardware: m4_pro
  memory_limit_gb: 18
  cpu_cores: 12
  gpu_enabled: true

agents:
  count: 8
  max_recursion_depth: 1
  task_timeout: 300
  memory_per_agent_mb: 512

tools:
  ripgrep:
    max_workers: 12
    chunk_size: 1024
    memory_map: true
  
  dependency_graph:
    use_gpu: true
    batch_size: 256
    cache_enabled: true
  
  python_analyzer:
    mlx_enabled: true
    batch_size: 64
    parallel_files: 8

memory:
  component_budgets:
    duckdb: 0.50
    jarvis: 0.17
    einstein: 0.08
    meta_system: 0.10
    cache: 0.10
    other: 0.05
  
  pressure_thresholds:
    warning: 0.85
    critical: 0.95
  
  pools:
    small_objects: 100MB
    medium_objects: 500MB
    large_objects: 1GB

performance:
  monitoring_enabled: true
  metrics_interval: 1.0
  profiling_enabled: false
  
  optimization:
    adaptive_timeouts: true
    dynamic_concurrency: true
    memory_pooling: true
    gpu_acceleration: true
```

### Environment Setup Script (`setup_performance.sh`)
```bash
#!/bin/bash
# Performance optimization setup for M4 Pro

# System optimizations
echo "Setting up M4 Pro optimizations..."

# Disable CPU throttling
sudo pmset -c disablesleep 1
sudo pmset -c displaysleep 0

# Set memory limits
export BOLT_MEMORY_LIMIT=18GB
export PYTORCH_METAL_WORKSPACE_LIMIT_BYTES=19327352832

# Enable performance monitoring
export BOLT_MONITORING=true
export BOLT_PROFILING=development

# GPU optimizations
export MLX_METAL_CACHE_ENABLE=1
export MLX_METAL_CACHE_PATH=/tmp/mlx_cache
export MLX_GPU_CORES=20
export MLX_MEMORY_POOL=16GB

# Tool configurations
export RIPGREP_MAX_WORKERS=12
export DEPENDENCY_GRAPH_GPU=true
export PYTHON_ANALYZER_MLX=true

echo "Performance optimizations applied!"
echo "Run 'bolt solve \"test performance\" --analyze-only' to verify"
```

## Troubleshooting Performance Issues

### Common Performance Problems

#### Slow Initialization
```python
# Profile initialization
import time
from bolt.integration import BoltIntegration

start_time = time.time()
bolt = BoltIntegration()
init_time = time.time() - start_time
print(f"Initialization took: {init_time:.2f}s")

# If >5 seconds, check:
# 1. Einstein index building
# 2. Tool initialization
# 3. GPU driver setup
```

#### Memory Leaks
```python
# Monitor memory usage over time
import gc
import psutil
import time

def monitor_memory_usage(duration=300):
    """Monitor memory usage for specified duration"""
    start_time = time.time()
    memory_samples = []
    
    while time.time() - start_time < duration:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_samples.append({
            "timestamp": time.time(),
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024   # MB
        })
        
        time.sleep(5)
    
    # Check for memory growth
    if memory_samples:
        initial_memory = memory_samples[0]["rss"]
        final_memory = memory_samples[-1]["rss"]
        growth = final_memory - initial_memory
        
        if growth > 100:  # More than 100MB growth
            print(f"⚠️ Potential memory leak: {growth:.1f}MB growth")
            gc.collect()  # Force garbage collection
```

#### GPU Utilization Issues
```python
# Check GPU utilization
import mlx.core as mx

def check_gpu_utilization():
    """Check GPU utilization and performance"""
    if not mx.metal.is_available():
        print("❌ Metal GPU not available")
        return
    
    # Get GPU memory info
    memory_info = mx.metal.get_memory_info()
    print(f"GPU Memory: {memory_info}")
    
    # Test GPU performance
    import numpy as np
    test_data = np.random.randn(1000, 1000).astype(np.float32)
    
    start_time = time.time()
    gpu_result = mx.matmul(mx.array(test_data), mx.array(test_data.T))
    mx.eval(gpu_result)  # Force evaluation
    gpu_time = time.time() - start_time
    
    print(f"GPU Matrix Multiply (1000x1000): {gpu_time:.3f}s")
    
    if gpu_time > 0.1:  # Should be <100ms
        print("⚠️ GPU performance may be suboptimal")
```

This performance tuning guide provides comprehensive optimization strategies for getting the best performance from Bolt on M4 Pro hardware. Regular monitoring and profiling will help identify bottlenecks and optimize performance for your specific use cases.