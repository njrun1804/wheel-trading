# Einstein Environment Variables Reference

This document provides a comprehensive reference for all environment variables that can be used to configure the Einstein system.

## Overview

Environment variables provide a way to override configuration settings without modifying configuration files. This is particularly useful for:

- Different deployment environments (development, staging, production)
- Containerized deployments
- CI/CD pipelines
- Sensitive configuration values

## Variable Naming Convention

Einstein environment variables follow this naming pattern:
```
EINSTEIN_<SECTION>__<KEY>
```

- All uppercase
- `EINSTEIN_` prefix
- Double underscore (`__`) separates sections and subsections
- Single underscore (`_`) separates words within a key

### Examples:
- `EINSTEIN_PERFORMANCE__MAX_STARTUP_TIME_MS`
- `EINSTEIN_CACHE__HOT_CACHE_SIZE`
- `EINSTEIN_ML__ENABLE_ANE`

## Hardware Configuration

Override auto-detected hardware settings:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EINSTEIN_CPU_CORES` | integer | auto-detected | Number of CPU cores to use |
| `EINSTEIN_MEMORY_GB` | float | auto-detected | Maximum memory allocation in GB |
| `EINSTEIN_USE_GPU` | boolean | auto-detected | Enable GPU acceleration |
| `EINSTEIN_HARDWARE__CPU_PERFORMANCE_CORES` | integer | auto-detected | Number of performance cores (Apple Silicon) |
| `EINSTEIN_HARDWARE__CPU_EFFICIENCY_CORES` | integer | auto-detected | Number of efficiency cores (Apple Silicon) |
| `EINSTEIN_HARDWARE__HAS_GPU` | boolean | auto-detected | Override GPU detection |
| `EINSTEIN_HARDWARE__GPU_CORES` | integer | auto-detected | Number of GPU cores |
| `EINSTEIN_HARDWARE__HAS_ANE` | boolean | auto-detected | Override Apple Neural Engine detection |
| `EINSTEIN_HARDWARE__ANE_CORES` | integer | auto-detected | Number of ANE cores |
| `EINSTEIN_HARDWARE__PLATFORM_TYPE` | string | auto-detected | Platform type: apple_silicon, intel_mac, intel, amd, unknown |
| `EINSTEIN_HARDWARE__ARCHITECTURE` | string | auto-detected | Architecture: arm64, x86_64 |

### Examples:
```bash
# Force Einstein to use 8 CPU cores instead of auto-detected value
export EINSTEIN_CPU_CORES=8

# Limit memory usage to 16GB
export EINSTEIN_MEMORY_GB=16.0

# Disable GPU acceleration
export EINSTEIN_USE_GPU=false

# Override platform detection
export EINSTEIN_HARDWARE__PLATFORM_TYPE=apple_silicon
```

## Performance Configuration

Control performance targets and limits:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EINSTEIN_MAX_STARTUP_MS` | float | 500.0 | Maximum startup time in milliseconds |
| `EINSTEIN_MAX_SEARCH_MS` | float | 50.0 | Maximum search time in milliseconds |
| `EINSTEIN_MAX_MEMORY_GB` | float | 2.0 | Maximum memory usage in GB |
| `EINSTEIN_PERFORMANCE__MAX_CRITICAL_PATH_MS` | float | 200.0 | Critical path initialization time |
| `EINSTEIN_PERFORMANCE__MAX_BACKGROUND_INIT_MS` | float | 2000.0 | Background initialization time |
| `EINSTEIN_PERFORMANCE__TARGET_TEXT_SEARCH_MS` | float | 5.0 | Text search target time |
| `EINSTEIN_PERFORMANCE__TARGET_SEMANTIC_SEARCH_MS` | float | 20.0 | Semantic search target time |
| `EINSTEIN_PERFORMANCE__CACHE_MEMORY_LIMIT_MB` | float | 512.0 | Cache memory limit in MB |
| `EINSTEIN_SEARCH_CONCURRENCY` | integer | 4 | Maximum concurrent search operations |
| `EINSTEIN_FILE_IO_CONCURRENCY` | integer | 12 | Maximum concurrent file I/O operations |
| `EINSTEIN_PERFORMANCE__MAX_EMBEDDING_CONCURRENCY` | integer | 8 | Maximum concurrent embedding operations |
| `EINSTEIN_PERFORMANCE__MAX_ANALYSIS_CONCURRENCY` | integer | 6 | Maximum concurrent analysis operations |

### Examples:
```bash
# Allow longer startup time for slower systems
export EINSTEIN_MAX_STARTUP_MS=1000

# Increase search timeout for complex operations
export EINSTEIN_MAX_SEARCH_MS=100

# Limit memory usage for resource-constrained environments
export EINSTEIN_MAX_MEMORY_GB=1.0

# Increase concurrency for high-performance systems
export EINSTEIN_SEARCH_CONCURRENCY=8
export EINSTEIN_FILE_IO_CONCURRENCY=16
```

## Cache Configuration

Control cache sizes and policies:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EINSTEIN_HOT_CACHE_SIZE` | integer | 1000 | Hot cache entries |
| `EINSTEIN_WARM_CACHE_SIZE` | integer | 5000 | Warm cache entries |
| `EINSTEIN_SEARCH_CACHE_SIZE` | integer | 500 | Search result cache entries |
| `EINSTEIN_INDEX_CACHE_MB` | integer | 256 | Index cache size in MB |
| `EINSTEIN_CACHE__FILE_CACHE_SIZE` | integer | 1000 | File content cache entries |
| `EINSTEIN_CACHE__COMPRESS_THRESHOLD_BYTES` | integer | 1024 | Compression threshold |
| `EINSTEIN_CACHE__CACHE_TTL_SECONDS` | integer | 3600 | Cache time-to-live |
| `EINSTEIN_CACHE__MAX_CACHE_ENTRIES` | integer | 10000 | Maximum total cache entries |
| `EINSTEIN_CACHE__PREFETCH_COMMON_PATTERNS` | boolean | true | Enable pattern prefetching |
| `EINSTEIN_CACHE__PREFETCH_CACHE_SIZE` | integer | 1000 | Prefetch cache size |

### Examples:
```bash
# Increase cache sizes for systems with more memory
export EINSTEIN_HOT_CACHE_SIZE=2000
export EINSTEIN_WARM_CACHE_SIZE=10000
export EINSTEIN_INDEX_CACHE_MB=512

# Reduce cache sizes for memory-constrained systems
export EINSTEIN_HOT_CACHE_SIZE=500
export EINSTEIN_WARM_CACHE_SIZE=2000
export EINSTEIN_INDEX_CACHE_MB=128

# Disable prefetching to save memory
export EINSTEIN_CACHE__PREFETCH_COMMON_PATTERNS=false
```

## Path Configuration

Override default paths:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EINSTEIN_CACHE_DIR` | string | ./.einstein | Cache directory path |
| `EINSTEIN_PATHS__BASE_DIR` | string | current dir | Base directory path |
| `EINSTEIN_PATHS__LOGS_DIR` | string | cache_dir/logs | Logs directory path |
| `EINSTEIN_PATHS__ANALYTICS_DB_PATH` | string | cache_dir/analytics.db | Analytics database path |
| `EINSTEIN_PATHS__EMBEDDINGS_DB_PATH` | string | cache_dir/embeddings.db | Embeddings database path |
| `EINSTEIN_PATHS__MODELS_DIR` | string | cache_dir/models | Models directory path |
| `EINSTEIN_PATHS__ENABLE_CONCURRENT_DB` | boolean | true | Enable concurrent database access |
| `EINSTEIN_PATHS__MAX_DB_CONNECTIONS` | integer | 4 | Maximum database connections |
| `EINSTEIN_PATHS__DB_CONNECTION_TIMEOUT` | float | 30.0 | Database connection timeout |
| `EINSTEIN_PATHS__DB_LOCK_TIMEOUT` | float | 30.0 | Database lock timeout |

### Examples:
```bash
# Use custom cache directory
export EINSTEIN_CACHE_DIR="/var/cache/einstein"

# Use separate log directory
export EINSTEIN_PATHS__LOGS_DIR="/var/log/einstein"

# Increase database connection limits
export EINSTEIN_PATHS__MAX_DB_CONNECTIONS=8
export EINSTEIN_PATHS__DB_CONNECTION_TIMEOUT=60.0
```

## Machine Learning Configuration

Control ML and AI parameters:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EINSTEIN_LEARNING_RATE` | float | 0.1 | Adaptive learning rate |
| `EINSTEIN_SIMILARITY_THRESHOLD` | float | 0.7 | Similarity threshold for matches |
| `EINSTEIN_CONFIDENCE_THRESHOLD` | float | 0.3 | Minimum confidence for results |
| `EINSTEIN_ML__BANDIT_EXPLORATION_RATE` | float | 0.1 | Multi-armed bandit exploration rate |
| `EINSTEIN_ML__EMBEDDING_DIMENSION` | integer | 1536 | Embedding vector dimension |
| `EINSTEIN_ML__MAX_SEQUENCE_LENGTH` | integer | 512 | Maximum sequence length |
| `EINSTEIN_ML__BATCH_SIZE` | integer | 32 | Training batch size |
| `EINSTEIN_ML__MAX_TRAINING_SAMPLES` | integer | 10000 | Maximum training samples |
| `EINSTEIN_ML__RELEVANCE_THRESHOLD` | float | 0.5 | Relevance threshold for ranking |
| `EINSTEIN_ENABLE_ANE` | boolean | true | Enable Apple Neural Engine |
| `EINSTEIN_ANE_BATCH_SIZE` | integer | 256 | ANE batch size |
| `EINSTEIN_ANE_CACHE_MB` | integer | 512 | ANE cache size in MB |
| `EINSTEIN_ANE_WARMUP` | boolean | true | Warm up ANE on startup |
| `EINSTEIN_ANE_FALLBACK` | boolean | true | Fallback to CPU on ANE errors |

### Examples:
```bash
# Increase similarity threshold for more precise matches
export EINSTEIN_SIMILARITY_THRESHOLD=0.8

# Reduce confidence threshold to include more results
export EINSTEIN_CONFIDENCE_THRESHOLD=0.2

# Disable ANE acceleration
export EINSTEIN_ENABLE_ANE=false

# Increase ANE batch size for better throughput
export EINSTEIN_ANE_BATCH_SIZE=512
```

## Monitoring Configuration

Control monitoring and observability:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EINSTEIN_LOG_LEVEL` | string | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
| `EINSTEIN_DEBOUNCE_MS` | float | 250.0 | File change debounce delay |
| `EINSTEIN_MEMORY_CHECK_INTERVAL` | integer | 30 | Memory check interval in seconds |
| `EINSTEIN_MONITORING__FILE_WATCH_INTERVAL_S` | integer | 60 | File watch check interval |
| `EINSTEIN_MONITORING__PERFORMANCE_HISTORY_SIZE` | integer | 50 | Performance metrics history size |
| `EINSTEIN_MONITORING__SYSTEM_LOAD_HISTORY_SIZE` | integer | 20 | System load history size |
| `EINSTEIN_MONITORING__COOLDOWN_PERIOD_S` | float | 5.0 | Cooldown between operations |
| `EINSTEIN_MONITORING__GC_INTERVAL_S` | float | 30.0 | Garbage collection interval |
| `EINSTEIN_MONITORING__ENABLE_PERFORMANCE_LOGS` | boolean | true | Enable performance logging |
| `EINSTEIN_MONITORING__ENABLE_CACHE_STATS` | boolean | true | Enable cache statistics logging |

### Examples:
```bash
# Enable debug logging
export EINSTEIN_LOG_LEVEL=DEBUG

# Reduce file watch frequency
export EINSTEIN_MONITORING__FILE_WATCH_INTERVAL_S=120

# Increase performance history for better analytics
export EINSTEIN_MONITORING__PERFORMANCE_HISTORY_SIZE=100

# Disable cache statistics to reduce overhead
export EINSTEIN_MONITORING__ENABLE_CACHE_STATS=false
```

## Feature Flags

Enable or disable system features:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EINSTEIN_ENABLE_GPU_ACCELERATION` | boolean | true | Enable GPU acceleration |
| `EINSTEIN_ADAPTIVE_CONCURRENCY` | boolean | true | Enable adaptive concurrency |
| `EINSTEIN_PREDICTIVE_PREFETCH` | boolean | true | Enable predictive prefetching |
| `EINSTEIN_MEMORY_OPTIMIZATION` | boolean | true | Enable memory optimization |
| `EINSTEIN_REALTIME_INDEXING` | boolean | true | Enable real-time indexing |
| `EINSTEIN_FEATURES__ENABLE_EXPERIMENTAL` | boolean | false | Enable experimental features |

### Examples:
```bash
# Disable GPU acceleration for CPU-only environments
export EINSTEIN_ENABLE_GPU_ACCELERATION=false

# Enable experimental features for testing
export EINSTEIN_FEATURES__ENABLE_EXPERIMENTAL=true

# Disable memory optimization for debugging
export EINSTEIN_MEMORY_OPTIMIZATION=false
```

## Boolean Value Formats

Boolean environment variables accept multiple formats:

| True Values | False Values |
|-------------|--------------|
| `true` | `false` |
| `1` | `0` |
| `yes` | `no` |
| `on` | `off` |
| `True` | `False` |
| `TRUE` | `FALSE` |

## Configuration File Integration

Environment variables take precedence over configuration files. The precedence order is:

1. Environment variables (highest precedence)
2. Configuration files (config.yaml)
3. Default values (lowest precedence)

## Common Deployment Scenarios

### High-Performance Server
```bash
export EINSTEIN_CPU_CORES=32
export EINSTEIN_MEMORY_GB=64.0
export EINSTEIN_SEARCH_CONCURRENCY=16
export EINSTEIN_FILE_IO_CONCURRENCY=32
export EINSTEIN_HOT_CACHE_SIZE=5000
export EINSTEIN_WARM_CACHE_SIZE=20000
export EINSTEIN_INDEX_CACHE_MB=2048
```

### Memory-Constrained Environment
```bash
export EINSTEIN_MEMORY_GB=1.0
export EINSTEIN_HOT_CACHE_SIZE=250
export EINSTEIN_WARM_CACHE_SIZE=1000
export EINSTEIN_INDEX_CACHE_MB=64
export EINSTEIN_SEARCH_CONCURRENCY=2
export EINSTEIN_FILE_IO_CONCURRENCY=4
export EINSTEIN_CACHE__PREFETCH_COMMON_PATTERNS=false
```

### Development Environment
```bash
export EINSTEIN_LOG_LEVEL=DEBUG
export EINSTEIN_MONITORING__ENABLE_PERFORMANCE_LOGS=true
export EINSTEIN_DEBOUNCE_MS=100
export EINSTEIN_MAX_STARTUP_MS=2000
export EINSTEIN_FEATURES__ENABLE_EXPERIMENTAL=true
```

### Production Environment
```bash
export EINSTEIN_LOG_LEVEL=INFO
export EINSTEIN_MONITORING__ENABLE_PERFORMANCE_LOGS=false
export EINSTEIN_MAX_STARTUP_MS=500
export EINSTEIN_FEATURES__ENABLE_EXPERIMENTAL=false
export EINSTEIN_CACHE_DIR="/var/cache/einstein"
export EINSTEIN_PATHS__LOGS_DIR="/var/log/einstein"
```

## Docker Environment

When running Einstein in Docker containers:

```dockerfile
# Dockerfile example
ENV EINSTEIN_CACHE_DIR=/app/cache
ENV EINSTEIN_PATHS__LOGS_DIR=/app/logs
ENV EINSTEIN_MEMORY_GB=4.0
ENV EINSTEIN_LOG_LEVEL=INFO
```

```bash
# Docker run example
docker run -e EINSTEIN_MEMORY_GB=8.0 \
           -e EINSTEIN_LOG_LEVEL=DEBUG \
           -e EINSTEIN_CACHE_DIR=/tmp/cache \
           einstein:latest
```

## Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: einstein-config
data:
  EINSTEIN_MEMORY_GB: "16.0"
  EINSTEIN_LOG_LEVEL: "INFO"
  EINSTEIN_CACHE_DIR: "/app/cache"
  EINSTEIN_SEARCH_CONCURRENCY: "8"
  EINSTEIN_ENABLE_GPU_ACCELERATION: "true"
```

## Validation and Troubleshooting

### Check Current Configuration
Use the configuration test script:
```bash
python -m einstein.einstein_config
```

### Common Issues

1. **Invalid boolean values**: Use `true`/`false`, not `True`/`False`
2. **Invalid paths**: Ensure directories exist and are writable
3. **Resource limits**: Don't exceed system capabilities
4. **Type mismatches**: Use correct types (integer, float, string, boolean)

### Debug Configuration Loading
```bash
export EINSTEIN_LOG_LEVEL=DEBUG
python -c "from einstein.einstein_config import get_einstein_config; config = get_einstein_config(); print(config)"
```

## Security Considerations

- Don't put sensitive values in environment variables in production
- Use container secrets or secure configuration management
- Avoid logging environment variables that may contain sensitive data
- Use least-privilege principles for file system access

## Related Documentation

- [Einstein Configuration Guide](EINSTEIN_CONFIGURATION_GUIDE.md)
- [Einstein Performance Tuning](EINSTEIN_PERFORMANCE_TUNING.md)
- [Einstein Troubleshooting](EINSTEIN_TROUBLESHOOTING.md)