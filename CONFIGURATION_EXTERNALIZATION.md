# Configuration Externalization Implementation

This document describes the successful externalization of hardcoded values in both Einstein and Bolt systems to centralized configuration files with environment variable support.

## Overview

Previously, both Einstein and Bolt systems contained numerous hardcoded values scattered throughout the codebase. This implementation centralizes all configuration into structured schemas with:

- ✅ Default values defined in configuration classes
- ✅ Environment variable overrides
- ✅ YAML file configuration support  
- ✅ Validation using Pydantic models
- ✅ Backward compatibility with existing code

## Einstein System Configuration

### Configuration Structure

The Einstein configuration is defined in `einstein/einstein_config.py` with these main sections:

- **HardwareConfig**: CPU cores, memory, GPU/ANE availability
- **PerformanceConfig**: Timeouts, search limits, concurrency settings
- **CacheConfig**: Cache sizes and policies
- **DatabaseConfig**: SQLite pragma settings, query limits
- **MLConfig**: Model parameters, embedding dimensions, ANE settings
- **MonitoringConfig**: Logging and monitoring settings
- **PathConfig**: File system paths

### Externalized Values

The following hardcoded values were moved to configuration:

| Original Hardcoded Value | Configuration Location | Environment Variable |
|--------------------------|------------------------|----------------------|
| `embedding_dim = 384` | `config.ml.embedding_dimension_minilm` | `EINSTEIN_EMBEDDING_DIM_MINILM` |
| `LIMIT 100` | `config.database.default_query_limit` | `EINSTEIN_DB_QUERY_LIMIT` |
| `timeout_ms: int = 50` | `config.performance.search_timeout_ms` | `EINSTEIN_MAX_SEARCH_MS` |
| `cache_size=2560` | `config.database.cache_size_pages` | `EINSTEIN_DB_CACHE_PAGES` |
| Various search timeouts | `config.performance.*_timeout_ms` | `EINSTEIN_*_TIMEOUT_MS` |

### Usage Examples

```python
# Load configuration
from einstein.einstein_config import get_einstein_config

config = get_einstein_config()

# Use configured values instead of hardcoded ones
embedding_dim = config.ml.embedding_dimension_minilm  # Was: 384
query_limit = config.database.default_query_limit     # Was: 100
search_timeout = config.performance.search_timeout_ms # Was: 50
```

### Environment Variable Overrides

```bash
# Override default embedding dimension
export EINSTEIN_EMBEDDING_DIM_MINILM=512

# Override search timeout
export EINSTEIN_MAX_SEARCH_MS=75

# Override database query limit
export EINSTEIN_DB_QUERY_LIMIT=200
```

## Bolt System Configuration

### Configuration Structure

The Bolt configuration is defined in `bolt/core/config.py` with these main sections:

- **HardwareConfig**: Core counts, memory allocation, buffer alignment
- **PerformanceConfig**: Timeouts, batch sizes, workload thresholds
- **MemoryConfig**: Component memory budgets, pool settings
- **CircuitBreakerConfig**: Failure thresholds, timeout settings
- **BoltConfig**: Main configuration with legacy compatibility

### Externalized Values

The following hardcoded values were moved to configuration:

| Original Hardcoded Value | Configuration Location | Environment Variable |
|--------------------------|------------------------|----------------------|
| `p_cores: int = 8` | `config.hardware.performance_cores` | `BOLT_PERFORMANCE_CORES` |
| `e_cores: int = 4` | `config.hardware.efficiency_cores` | `BOLT_EFFICIENCY_CORES` |
| `metal_cores: int = 20` | `config.hardware.metal_cores` | `BOLT_METAL_CORES` |
| `timeout: float = 60.0` | `config.circuit_breaker.timeout_s` | `BOLT_CB_TIMEOUT` |
| `batch_size = 32` | `config.performance.default_batch_size` | `BOLT_DEFAULT_BATCH_SIZE` |
| `buffer_alignment = 16` | `config.hardware.buffer_alignment_bytes` | `BOLT_BUFFER_ALIGNMENT` |
| Memory budget ratios | `config.memory.*_budget_ratio` | `BOLT_*_BUDGET` |

### Usage Examples

```python
# Load configuration
from bolt.core.config import get_default_config

config = get_default_config()

# Use configured values instead of hardcoded ones
p_cores = config.hardware.performance_cores           # Was: 8
batch_size = config.performance.default_batch_size    # Was: 32
duckdb_budget = config.memory.duckdb_budget_ratio     # Was: 0.50
```

### Environment Variable Overrides

```bash
# Override core counts
export BOLT_PERFORMANCE_CORES=12
export BOLT_EFFICIENCY_CORES=8

# Override batch sizes
export BOLT_DEFAULT_BATCH_SIZE=64
export BOLT_GPU_BATCH_SIZE=512

# Override memory budgets
export BOLT_DUCKDB_BUDGET=0.60
export BOLT_JARVIS_BUDGET=0.20
```

## YAML Configuration Files

Both systems support YAML configuration files:

### Bolt Configuration Example

```yaml
# ~/.bolt/config.yaml
hardware:
  performance_cores: 12
  efficiency_cores: 8
  total_memory_gb: 32.0
  max_allocation_gb: 24.0

performance:
  default_batch_size: 64
  gpu_batch_size: 512
  short_timeout_s: 0.2

memory:
  duckdb_budget_ratio: 0.6
  jarvis_budget_ratio: 0.2

max_agents: 12
use_gpu: true
```

## Configuration Loading Priority

Configuration values are loaded in this priority order (highest to lowest):

1. **Environment Variables** (highest priority)
2. **YAML Configuration File** 
3. **Default Values** (lowest priority)

## Validation

Both systems use Pydantic for configuration validation:

- Type checking (int, float, bool, str)
- Range validation (min/max values)
- Custom validators for business logic
- Clear error messages for invalid configurations

## Migration Benefits

### Before (Hardcoded Values)
```python
# Scattered hardcoded values
embedding_dim = 384                    # in rapid_startup.py
LIMIT 100                             # in SQL queries
timeout_ms: int = 50                  # in search requests
p_cores: int = 8                      # in adaptive_concurrency.py
batch_size = 32                       # in gpu_acceleration.py
```

### After (Centralized Configuration)
```python
# Centralized, configurable values
config = get_config()
embedding_dim = config.ml.embedding_dimension_minilm
query_limit = config.database.default_query_limit
timeout_ms = config.performance.search_timeout_ms
p_cores = config.hardware.performance_cores
batch_size = config.performance.default_batch_size
```

## Testing

The configuration externalization is validated by `test_configuration_externalization.py`:

- ✅ Default configuration loading
- ✅ Environment variable overrides
- ✅ YAML file configuration
- ✅ Configuration validation
- ✅ Hardcoded value replacement verification

## Environment Variables Reference

### Einstein Environment Variables

| Variable | Description | Default | Type |
|----------|-------------|---------|------|
| `EINSTEIN_EMBEDDING_DIM_MINILM` | MiniLM embedding dimension | 384 | int |
| `EINSTEIN_MAX_SEARCH_MS` | Maximum search time | 50 | float |
| `EINSTEIN_DB_QUERY_LIMIT` | Default SQL query limit | 100 | int |
| `EINSTEIN_DB_CACHE_PAGES` | SQLite cache size in pages | 10000 | int |
| `EINSTEIN_USE_GPU` | Enable GPU acceleration | true | bool |
| `EINSTEIN_LOG_LEVEL` | Logging level | INFO | str |

### Bolt Environment Variables

| Variable | Description | Default | Type |
|----------|-------------|---------|------|
| `BOLT_PERFORMANCE_CORES` | Number of P-cores | 8 | int |
| `BOLT_EFFICIENCY_CORES` | Number of E-cores | 4 | int |
| `BOLT_METAL_CORES` | Number of Metal GPU cores | 20 | int |
| `BOLT_DEFAULT_BATCH_SIZE` | Default batch size | 32 | int |
| `BOLT_GPU_BATCH_SIZE` | GPU batch size | 256 | int |
| `BOLT_DUCKDB_BUDGET` | DuckDB memory budget ratio | 0.50 | float |
| `BOLT_SHORT_TIMEOUT` | Short operation timeout | 0.1 | float |
| `BOLT_BUFFER_ALIGNMENT` | Buffer alignment in bytes | 16 | int |

## Migration Impact

This configuration externalization provides:

1. **🎯 Tunable Performance**: Users can optimize settings for their hardware
2. **🔧 Environment-Specific Config**: Different settings for dev/staging/prod
3. **📊 Better Monitoring**: Centralized configuration makes it easier to track what settings are being used
4. **🧪 Testing Flexibility**: Easy to test with different configurations
5. **💾 Hardware Adaptation**: Settings can be automatically adjusted based on detected hardware
6. **🔒 Validation**: Invalid configurations are caught early with clear error messages

The implementation maintains full backward compatibility while enabling much more flexible configuration management.