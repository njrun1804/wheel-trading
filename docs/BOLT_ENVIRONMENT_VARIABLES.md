# Bolt Environment Variables Reference

This document provides a comprehensive reference for all environment variables that can be used to configure the Bolt system.

## Overview

Environment variables provide a flexible way to configure the Bolt system across different deployment environments without modifying configuration files. This is essential for:

- Multi-environment deployments (development, staging, production)
- Containerized and cloud deployments
- CI/CD pipeline integration
- Runtime configuration adjustment

## Variable Naming Convention

Bolt environment variables follow this naming pattern:
```
BOLT_<SECTION>__<KEY>
```

- All uppercase
- `BOLT_` prefix
- Double underscore (`__`) separates sections and subsections
- Single underscore (`_`) separates words within a key

### Examples:
- `BOLT_AGENT_POOL__NUM_AGENTS`
- `BOLT_PERFORMANCE__TARGET_CPU_UTILIZATION`
- `BOLT_ACCELERATED_TOOLS__RIPGREP_TURBO`

## System Configuration

Basic system settings:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_SYSTEM__NAME` | string | "Bolt System" | System name for identification |
| `BOLT_SYSTEM__VERSION` | string | "1.0.0" | Configuration version |
| `BOLT_SYSTEM__ENVIRONMENT` | string | "development" | Environment: development, staging, production |
| `BOLT_SYSTEM__ENABLE_DEBUG` | boolean | false | Enable debug mode |
| `BOLT_SYSTEM__ENABLE_METRICS` | boolean | true | Enable metrics collection |
| `BOLT_SYSTEM__ENABLE_TRACING` | boolean | true | Enable distributed tracing |

### Examples:
```bash
# Set production environment
export BOLT_SYSTEM__ENVIRONMENT=production

# Enable debug mode for troubleshooting
export BOLT_SYSTEM__ENABLE_DEBUG=true

# Disable metrics collection
export BOLT_SYSTEM__ENABLE_METRICS=false
```

## Agent Pool Configuration

Control the multi-agent orchestrator:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_MAX_AGENTS` | integer | 8 | Maximum number of agents (legacy) |
| `BOLT_DEFAULT_AGENTS` | integer | 8 | Default number of agents (legacy) |
| `BOLT_AGENT_POOL__NUM_AGENTS` | integer | 12 | Number of agents (recommended: CPU cores) |
| `BOLT_AGENT_POOL__MIN_AGENTS` | integer | 4 | Minimum number of agents |
| `BOLT_AGENT_POOL__MAX_AGENTS` | integer | 16 | Maximum number of agents |
| `BOLT_AGENT_POOL__ENABLE_WORK_STEALING` | boolean | true | Enable work stealing between agents |
| `BOLT_AGENT_POOL__STEAL_THRESHOLD` | float | 0.2 | Work stealing threshold (0.0-1.0) |
| `BOLT_AGENT_POOL__MAX_STEAL_ATTEMPTS` | integer | 4 | Maximum steal attempts per cycle |
| `BOLT_BATCH_SIZE` | integer | 32 | Default batch size for operations (legacy) |
| `BOLT_AGENT_POOL__BATCH_SIZE` | integer | 32 | Batch size for operations |
| `BOLT_AGENT_POOL__BATCH_TIMEOUT_MS` | integer | 5 | Timeout for batch collection |
| `BOLT_AGENT_POOL__MONITOR_INTERVAL_MS` | integer | 50 | Agent monitoring interval |
| `BOLT_AGENT_POOL__REBALANCE_INTERVAL_MS` | integer | 1000 | Load rebalancing interval |
| `BOLT_AGENT_POOL__PROACTIVE_STEALING_INTERVAL_MS` | integer | 20 | Proactive stealing check interval |

### Examples:
```bash
# Scale to 16 agents for high-performance systems
export BOLT_AGENT_POOL__NUM_AGENTS=16
export BOLT_AGENT_POOL__MAX_AGENTS=20

# Optimize for CPU-intensive workloads
export BOLT_AGENT_POOL__ENABLE_WORK_STEALING=true
export BOLT_AGENT_POOL__STEAL_THRESHOLD=0.15

# Increase batch size for throughput
export BOLT_AGENT_POOL__BATCH_SIZE=64
```

## Token Optimization Configuration

Control token allocation and management:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_TOKEN_OPTIMIZATION__ENABLE_DYNAMIC_TOKENS` | boolean | true | Enable dynamic token allocation |
| `BOLT_TOKEN_OPTIMIZATION__ENABLE_DRIFT_COMPENSATION` | boolean | true | Compensate for context drift |
| `BOLT_TOKEN_OPTIMIZATION__MAX_CONTEXT_SIZE` | integer | 64000 | Maximum context size (tokens) |

### Complexity Budget Variables:
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_TOKEN_OPTIMIZATION__SIMPLE_MIN_TOKENS` | integer | 1000 | Simple task minimum tokens |
| `BOLT_TOKEN_OPTIMIZATION__SIMPLE_TARGET_TOKENS` | integer | 2000 | Simple task target tokens |
| `BOLT_TOKEN_OPTIMIZATION__SIMPLE_MAX_TOKENS` | integer | 3000 | Simple task maximum tokens |
| `BOLT_TOKEN_OPTIMIZATION__MODERATE_MIN_TOKENS` | integer | 4000 | Moderate task minimum tokens |
| `BOLT_TOKEN_OPTIMIZATION__MODERATE_TARGET_TOKENS` | integer | 6000 | Moderate task target tokens |
| `BOLT_TOKEN_OPTIMIZATION__MODERATE_MAX_TOKENS` | integer | 8000 | Moderate task maximum tokens |
| `BOLT_TOKEN_OPTIMIZATION__DETAILED_MIN_TOKENS` | integer | 10000 | Detailed task minimum tokens |
| `BOLT_TOKEN_OPTIMIZATION__DETAILED_TARGET_TOKENS` | integer | 15000 | Detailed task target tokens |
| `BOLT_TOKEN_OPTIMIZATION__DETAILED_MAX_TOKENS` | integer | 20000 | Detailed task maximum tokens |
| `BOLT_TOKEN_OPTIMIZATION__EXHAUSTIVE_MIN_TOKENS` | integer | 25000 | Exhaustive task minimum tokens |
| `BOLT_TOKEN_OPTIMIZATION__EXHAUSTIVE_TARGET_TOKENS` | integer | 40000 | Exhaustive task target tokens |
| `BOLT_TOKEN_OPTIMIZATION__EXHAUSTIVE_MAX_TOKENS` | integer | 50000 | Exhaustive task maximum tokens |

### Examples:
```bash
# Increase context size for complex tasks
export BOLT_TOKEN_OPTIMIZATION__MAX_CONTEXT_SIZE=128000

# Adjust token budgets for resource-constrained environments
export BOLT_TOKEN_OPTIMIZATION__DETAILED_MAX_TOKENS=15000
export BOLT_TOKEN_OPTIMIZATION__EXHAUSTIVE_MAX_TOKENS=30000

# Disable dynamic tokens for predictable usage
export BOLT_TOKEN_OPTIMIZATION__ENABLE_DYNAMIC_TOKENS=false
```

## CPU Optimization Configuration

Control CPU usage and affinity (M4 Pro optimized):

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_CPU_OPTIMIZATION__ENABLE_CPU_OPTIMIZATION` | boolean | true | Enable CPU optimization |
| `BOLT_CPU_OPTIMIZATION__ENABLE_CORE_AFFINITY` | boolean | true | Enable CPU core affinity |
| `BOLT_CPU_OPTIMIZATION__TOTAL_CORES` | integer | 12 | Total CPU cores |
| `BOLT_CPU_OPTIMIZATION__LOAD_THRESHOLD` | float | 0.8 | Load threshold for rebalancing |
| `BOLT_CPU_OPTIMIZATION__REBALANCE_INTERVAL_SECONDS` | float | 1.0 | Rebalancing check interval |
| `BOLT_CPU_OPTIMIZATION__PROCESS_PRIORITY` | integer | -5 | Process priority (-20 to 19) |

### Examples:
```bash
# Optimize for 16-core system
export BOLT_CPU_OPTIMIZATION__TOTAL_CORES=16

# Increase load threshold for CPU-intensive workloads
export BOLT_CPU_OPTIMIZATION__LOAD_THRESHOLD=0.9

# Set higher process priority
export BOLT_CPU_OPTIMIZATION__PROCESS_PRIORITY=-10

# Disable core affinity for Docker environments
export BOLT_CPU_OPTIMIZATION__ENABLE_CORE_AFFINITY=false
```

## Memory Management

Control memory usage and garbage collection:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_MAX_MEMORY_GB` | float | 18.0 | Maximum memory usage in GB (legacy) |
| `BOLT_MEMORY__MAX_MEMORY_USAGE_GB` | float | 18.0 | Maximum memory usage in GB |
| `BOLT_MEMORY__MAX_MEMORY_USAGE_PERCENT` | float | 80.0 | Maximum memory usage (% of total) |
| `BOLT_MEMORY__ENABLE_MEMORY_MONITORING` | boolean | true | Enable memory monitoring |
| `BOLT_MEMORY__MEMORY_CHECK_INTERVAL_MS` | integer | 1000 | Memory check interval |
| `BOLT_MEMORY__GC_THRESHOLD` | float | 0.85 | Garbage collection threshold |
| `BOLT_MEMORY__BUFFER_ALLOCATION_SIZE_MB` | integer | 16 | Buffer allocation size (MB) |
| `BOLT_MEMORY__MAX_BUFFER_POOL_SIZE_MB` | integer | 256 | Maximum buffer pool size (MB) |

### Examples:
```bash
# Increase memory limit for high-memory systems
export BOLT_MEMORY__MAX_MEMORY_USAGE_GB=32.0

# Use percentage-based memory limiting
export BOLT_MEMORY__MAX_MEMORY_USAGE_PERCENT=75.0

# Adjust GC threshold for better performance
export BOLT_MEMORY__GC_THRESHOLD=0.9

# Increase buffer sizes for I/O intensive workloads
export BOLT_MEMORY__BUFFER_ALLOCATION_SIZE_MB=32
export BOLT_MEMORY__MAX_BUFFER_POOL_SIZE_MB=512
```

## Performance Configuration

Set performance targets and limits:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_ASYNC_TIMEOUT` | float | 300.0 | Async operation timeout in seconds (legacy) |
| `BOLT_PERFORMANCE__TARGET_CPU_UTILIZATION` | float | 0.80 | Target CPU utilization (0.0-1.0) |
| `BOLT_PERFORMANCE__TARGET_TASKS_PER_SECOND` | float | 150.0 | Target task throughput |
| `BOLT_PERFORMANCE__MAX_TASK_DURATION_SECONDS` | float | 30.0 | Maximum task duration |
| `BOLT_PERFORMANCE__INITIALIZATION_TIMEOUT_SECONDS` | integer | 30 | Initialization timeout |
| `BOLT_PERFORMANCE__SHUTDOWN_TIMEOUT_SECONDS` | integer | 10 | Shutdown timeout |
| `BOLT_PERFORMANCE__TASK_TIMEOUT_SECONDS` | integer | 300 | Default task timeout |
| `BOLT_PERFORMANCE__MAX_QUEUE_SIZE` | integer | 1000 | Maximum task queue size |
| `BOLT_PERFORMANCE__QUEUE_TIMEOUT_MS` | integer | 100 | Queue operation timeout |
| `BOLT_PERFORMANCE__MAX_RETRY_ATTEMPTS` | integer | 3 | Maximum retry attempts |
| `BOLT_PERFORMANCE__RETRY_DELAY_MS` | integer | 1000 | Base retry delay |
| `BOLT_PERFORMANCE__RETRY_BACKOFF_MULTIPLIER` | float | 2.0 | Exponential backoff multiplier |

### Examples:
```bash
# Increase performance targets for high-end systems
export BOLT_PERFORMANCE__TARGET_CPU_UTILIZATION=0.9
export BOLT_PERFORMANCE__TARGET_TASKS_PER_SECOND=300.0

# Extend timeouts for complex operations
export BOLT_PERFORMANCE__MAX_TASK_DURATION_SECONDS=60.0
export BOLT_PERFORMANCE__TASK_TIMEOUT_SECONDS=600

# Increase queue size for high-throughput scenarios
export BOLT_PERFORMANCE__MAX_QUEUE_SIZE=5000
```

## Hardware Acceleration

Control GPU and specialized hardware:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_USE_GPU` | boolean | true | Enable GPU acceleration (legacy) |
| `BOLT_PREFER_MLX` | boolean | true | Prefer MLX over other GPU backends (legacy) |
| `BOLT_HARDWARE__ENABLE_GPU` | boolean | true | Enable GPU acceleration |
| `BOLT_HARDWARE__GPU_MEMORY_LIMIT_GB` | float | 8.0 | GPU memory limit |
| `BOLT_HARDWARE__ENABLE_METAL` | boolean | true | Enable Metal acceleration (macOS) |
| `BOLT_HARDWARE__ENABLE_MLX` | boolean | true | Enable MLX acceleration (Apple Silicon) |
| `BOLT_HARDWARE__ACCELERATION_PREFERENCE` | string | "auto" | Acceleration preference: auto, cpu, gpu, metal, mlx |
| `BOLT_HARDWARE__FALLBACK_ON_ERROR` | boolean | true | Fallback to CPU on acceleration errors |

### Examples:
```bash
# Force CPU-only mode
export BOLT_HARDWARE__ENABLE_GPU=false
export BOLT_HARDWARE__ACCELERATION_PREFERENCE=cpu

# Optimize for Apple Silicon
export BOLT_HARDWARE__ENABLE_MLX=true
export BOLT_HARDWARE__ENABLE_METAL=true
export BOLT_HARDWARE__GPU_MEMORY_LIMIT_GB=16.0

# Force specific acceleration backend
export BOLT_HARDWARE__ACCELERATION_PREFERENCE=mlx
```

## Integration Configuration

Control system integrations:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_INTEGRATIONS__ENABLE_EINSTEIN_INTEGRATION` | boolean | true | Enable Einstein system integration |
| `BOLT_INTEGRATIONS__ENABLE_MCP_INTEGRATION` | boolean | true | Enable MCP server integration |
| `BOLT_INTEGRATIONS__ENABLE_TRADING_SYSTEM_INTEGRATION` | boolean | true | Enable trading system integration |
| `BOLT_INTEGRATIONS__INTEGRATION_TIMEOUT_SECONDS` | integer | 30 | Integration operation timeout |
| `BOLT_INTEGRATIONS__HEALTH_CHECK_INTERVAL_SECONDS` | integer | 60 | Health check interval |
| `BOLT_INTEGRATIONS__INTEGRATION_RETRY_ATTEMPTS` | integer | 3 | Retry attempts for failed integrations |
| `BOLT_INTEGRATIONS__INTEGRATION_RETRY_DELAY_MS` | integer | 5000 | Retry delay for integrations |

### Examples:
```bash
# Disable specific integrations for standalone mode
export BOLT_INTEGRATIONS__ENABLE_EINSTEIN_INTEGRATION=false
export BOLT_INTEGRATIONS__ENABLE_TRADING_SYSTEM_INTEGRATION=false

# Increase integration timeouts for slow networks
export BOLT_INTEGRATIONS__INTEGRATION_TIMEOUT_SECONDS=60
export BOLT_INTEGRATIONS__HEALTH_CHECK_INTERVAL_SECONDS=120
```

## Logging Configuration

Control logging behavior:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_LOG_LEVEL` | string | "INFO" | Logging level (legacy) |
| `BOLT_LOG_FILE` | string | null | Log file path (legacy) |
| `BOLT_LOGGING__LEVEL` | string | "INFO" | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `BOLT_LOGGING__ENABLE_FILE_LOGGING` | boolean | true | Enable logging to file |
| `BOLT_LOGGING__LOG_FILE` | string | "bolt.log" | Log file name |
| `BOLT_LOGGING__LOG_DIRECTORY` | string | "./logs" | Log directory |
| `BOLT_LOGGING__MAX_LOG_SIZE_MB` | integer | 100 | Maximum log file size (MB) |
| `BOLT_LOGGING__BACKUP_COUNT` | integer | 5 | Number of backup log files |
| `BOLT_LOGGING__ENABLE_CONSOLE_LOGGING` | boolean | true | Enable console logging |
| `BOLT_LOGGING__CONSOLE_FORMAT` | string | "detailed" | Console format: simple, detailed, json |
| `BOLT_LOGGING__ENABLE_PERFORMANCE_LOGGING` | boolean | true | Enable performance logging |
| `BOLT_LOGGING__ENABLE_ERROR_LOGGING` | boolean | true | Enable error logging |
| `BOLT_LOGGING__ENABLE_AUDIT_LOGGING` | boolean | false | Enable audit logging |
| `BOLT_LOGGING__ROTATION_POLICY` | string | "size" | Rotation policy: size, time, both |
| `BOLT_LOGGING__ROTATION_INTERVAL` | string | "daily" | Rotation interval: hourly, daily, weekly |

### Examples:
```bash
# Enable debug logging
export BOLT_LOG_LEVEL=DEBUG
export BOLT_LOGGING__LEVEL=DEBUG

# Use custom log directory
export BOLT_LOGGING__LOG_DIRECTORY="/var/log/bolt"

# Enable JSON logging for structured log processing
export BOLT_LOGGING__CONSOLE_FORMAT=json

# Increase log file size limits
export BOLT_LOGGING__MAX_LOG_SIZE_MB=500
export BOLT_LOGGING__BACKUP_COUNT=10
```

## Validation Configuration

Control validation behavior:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_VALIDATION__SUCCESS_RATE_THRESHOLD` | float | 0.90 | Minimum success rate threshold |
| `BOLT_VALIDATION__ENABLE_STARTUP_VALIDATION` | boolean | true | Enable startup validation |
| `BOLT_VALIDATION__ENABLE_RUNTIME_VALIDATION` | boolean | true | Enable runtime validation |
| `BOLT_VALIDATION__ENABLE_PERFORMANCE_VALIDATION` | boolean | true | Enable performance validation |
| `BOLT_VALIDATION__ENABLE_INTEGRATION_VALIDATION` | boolean | true | Enable integration validation |
| `BOLT_VALIDATION__VALIDATION_TIMEOUT_SECONDS` | integer | 60 | Validation timeout |
| `BOLT_VALIDATION__HEALTH_CHECK_TIMEOUT_SECONDS` | integer | 30 | Health check timeout |
| `BOLT_VALIDATION__FAIL_FAST_ON_VALIDATION_ERROR` | boolean | true | Fail fast on validation errors |
| `BOLT_VALIDATION__CONTINUE_ON_MINOR_ERRORS` | boolean | true | Continue on minor validation errors |

### Examples:
```bash
# Lower success rate threshold for development
export BOLT_VALIDATION__SUCCESS_RATE_THRESHOLD=0.75

# Disable performance validation for debugging
export BOLT_VALIDATION__ENABLE_PERFORMANCE_VALIDATION=false

# Increase validation timeouts
export BOLT_VALIDATION__VALIDATION_TIMEOUT_SECONDS=120
```

## Error Handling Configuration

Control error handling and recovery:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_ERROR_HANDLING__ENABLE_GRACEFUL_DEGRADATION` | boolean | true | Enable graceful degradation |
| `BOLT_ERROR_HANDLING__DEGRADATION_LEVELS` | integer | 3 | Number of degradation levels |
| `BOLT_ERROR_HANDLING__MAX_RETRY_ATTEMPTS` | integer | 3 | Maximum retry attempts |
| `BOLT_ERROR_HANDLING__RETRY_DELAY_SECONDS` | float | 1.0 | Base retry delay |
| `BOLT_ERROR_HANDLING__RETRY_BACKOFF_FACTOR` | float | 2.0 | Exponential backoff factor |
| `BOLT_ERROR_HANDLING__ENABLE_CIRCUIT_BREAKER` | boolean | true | Enable circuit breaker |
| `BOLT_ERROR_HANDLING__CIRCUIT_BREAKER_THRESHOLD` | integer | 5 | Circuit breaker failure threshold |
| `BOLT_ERROR_HANDLING__CIRCUIT_BREAKER_TIMEOUT_SECONDS` | integer | 30 | Circuit breaker timeout |
| `BOLT_ERROR_HANDLING__CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | integer | 60 | Circuit breaker recovery timeout |
| `BOLT_ERROR_HANDLING__ENABLE_ERROR_REPORTING` | boolean | true | Enable error reporting |

### Examples:
```bash
# Increase retry attempts for unstable environments
export BOLT_ERROR_HANDLING__MAX_RETRY_ATTEMPTS=5
export BOLT_ERROR_HANDLING__RETRY_DELAY_SECONDS=2.0

# Adjust circuit breaker for high-error scenarios
export BOLT_ERROR_HANDLING__CIRCUIT_BREAKER_THRESHOLD=10
export BOLT_ERROR_HANDLING__CIRCUIT_BREAKER_TIMEOUT_SECONDS=60
```

## Monitoring Configuration

Control monitoring and metrics:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_ENABLE_MONITORING` | boolean | true | Enable system monitoring (legacy) |
| `BOLT_MONITORING_INTERVAL` | float | 1.0 | Monitoring interval in seconds (legacy) |
| `BOLT_MONITORING__ENABLE_SYSTEM_MONITORING` | boolean | true | Enable system monitoring |
| `BOLT_MONITORING__SYSTEM_METRICS_INTERVAL_SECONDS` | integer | 10 | System metrics collection interval |
| `BOLT_MONITORING__ENABLE_PERFORMANCE_MONITORING` | boolean | true | Enable performance monitoring |
| `BOLT_MONITORING__PERFORMANCE_METRICS_INTERVAL_SECONDS` | integer | 5 | Performance metrics interval |
| `BOLT_MONITORING__ENABLE_RESOURCE_MONITORING` | boolean | true | Enable resource monitoring |
| `BOLT_MONITORING__RESOURCE_METRICS_INTERVAL_SECONDS` | integer | 30 | Resource metrics interval |
| `BOLT_MONITORING__ENABLE_HEALTH_CHECKS` | boolean | true | Enable health checks |
| `BOLT_MONITORING__HEALTH_CHECK_INTERVAL_SECONDS` | integer | 60 | Health check interval |
| `BOLT_MONITORING__METRICS_RETENTION_DAYS` | integer | 30 | Metrics retention period |
| `BOLT_MONITORING__ENABLE_METRICS_EXPORT` | boolean | false | Enable metrics export |

### Examples:
```bash
# Increase monitoring frequency for production
export BOLT_MONITORING__SYSTEM_METRICS_INTERVAL_SECONDS=5
export BOLT_MONITORING__PERFORMANCE_METRICS_INTERVAL_SECONDS=1

# Enable metrics export
export BOLT_MONITORING__ENABLE_METRICS_EXPORT=true
export BOLT_MONITORING__METRICS_RETENTION_DAYS=90

# Disable resource monitoring to reduce overhead
export BOLT_MONITORING__ENABLE_RESOURCE_MONITORING=false
```

## Accelerated Tools Configuration

Control accelerated tool features:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_ACCELERATED_TOOLS__RIPGREP_TURBO` | boolean | true | Enable ripgrep turbo |
| `BOLT_ACCELERATED_TOOLS__DEPENDENCY_GRAPH` | boolean | true | Enable dependency graph turbo |
| `BOLT_ACCELERATED_TOOLS__PYTHON_ANALYSIS` | boolean | true | Enable python analysis turbo |
| `BOLT_ACCELERATED_TOOLS__DUCKDB_TURBO` | boolean | true | Enable DuckDB turbo |
| `BOLT_ACCELERATED_TOOLS__TRACE_TURBO` | boolean | true | Enable tracing turbo |
| `BOLT_ACCELERATED_TOOLS__PYTHON_HELPERS` | boolean | true | Enable python helpers turbo |

### Tool-Specific Configuration:
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_ACCELERATED_TOOLS__RIPGREP__MAX_PARALLEL_SEARCHES` | integer | 12 | Maximum parallel ripgrep searches |
| `BOLT_ACCELERATED_TOOLS__RIPGREP__SEARCH_TIMEOUT_SECONDS` | integer | 30 | Search timeout |
| `BOLT_ACCELERATED_TOOLS__DEPENDENCY_GRAPH__MAX_PARALLEL_ANALYSIS` | integer | 8 | Maximum parallel analysis |
| `BOLT_ACCELERATED_TOOLS__DEPENDENCY_GRAPH__CACHE_SIZE` | integer | 1000 | Cache size for dependency graph |
| `BOLT_ACCELERATED_TOOLS__PYTHON_ANALYSIS__ENABLE_GPU_ACCELERATION` | boolean | true | Enable GPU acceleration for analysis |
| `BOLT_ACCELERATED_TOOLS__PYTHON_ANALYSIS__BATCH_SIZE` | integer | 64 | Analysis batch size |
| `BOLT_ACCELERATED_TOOLS__DUCKDB__MAX_CONNECTIONS` | integer | 24 | Maximum DuckDB connections |
| `BOLT_ACCELERATED_TOOLS__DUCKDB__CONNECTION_TIMEOUT_SECONDS` | integer | 30 | Connection timeout |
| `BOLT_ACCELERATED_TOOLS__TRACE__ENABLE_ALL_BACKENDS` | boolean | true | Enable all tracing backends |
| `BOLT_ACCELERATED_TOOLS__TRACE__BUFFER_SIZE` | integer | 10000 | Trace buffer size |

### Examples:
```bash
# Disable specific accelerated tools
export BOLT_ACCELERATED_TOOLS__RIPGREP_TURBO=false
export BOLT_ACCELERATED_TOOLS__PYTHON_ANALYSIS=false

# Optimize ripgrep for high-performance systems
export BOLT_ACCELERATED_TOOLS__RIPGREP__MAX_PARALLEL_SEARCHES=24
export BOLT_ACCELERATED_TOOLS__RIPGREP__SEARCH_TIMEOUT_SECONDS=60

# Increase DuckDB connection limits
export BOLT_ACCELERATED_TOOLS__DUCKDB__MAX_CONNECTIONS=48
```

## Cache Configuration

Control caching behavior:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BOLT_CACHE_SIZE` | integer | 1000 | Cache size for results (legacy) |

### Examples:
```bash
# Increase cache size for better performance
export BOLT_CACHE_SIZE=5000
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

## Configuration Precedence

Configuration values are applied in this order (highest to lowest precedence):

1. Environment variables
2. Configuration files (config.yaml)
3. Default values

## Common Deployment Scenarios

### High-Performance Development Server
```bash
export BOLT_SYSTEM__ENVIRONMENT=development
export BOLT_AGENT_POOL__NUM_AGENTS=16
export BOLT_MEMORY__MAX_MEMORY_USAGE_GB=32.0
export BOLT_PERFORMANCE__TARGET_CPU_UTILIZATION=0.9
export BOLT_LOGGING__LEVEL=DEBUG
export BOLT_HARDWARE__ENABLE_MLX=true
```

### Production Server
```bash
export BOLT_SYSTEM__ENVIRONMENT=production
export BOLT_AGENT_POOL__NUM_AGENTS=12
export BOLT_MEMORY__MAX_MEMORY_USAGE_GB=24.0
export BOLT_PERFORMANCE__TARGET_TASKS_PER_SECOND=200.0
export BOLT_LOGGING__LEVEL=INFO
export BOLT_MONITORING__ENABLE_METRICS_EXPORT=true
export BOLT_ERROR_HANDLING__ENABLE_ERROR_REPORTING=true
```

### Resource-Constrained Environment
```bash
export BOLT_AGENT_POOL__NUM_AGENTS=4
export BOLT_MEMORY__MAX_MEMORY_USAGE_GB=8.0
export BOLT_HARDWARE__ENABLE_GPU=false
export BOLT_PERFORMANCE__TARGET_CPU_UTILIZATION=0.7
export BOLT_MONITORING__SYSTEM_METRICS_INTERVAL_SECONDS=30
export BOLT_ACCELERATED_TOOLS__PYTHON_ANALYSIS=false
```

### Container/Docker Deployment
```bash
export BOLT_LOGGING__LOG_DIRECTORY=/app/logs
export BOLT_CPU_OPTIMIZATION__ENABLE_CORE_AFFINITY=false
export BOLT_MEMORY__MAX_MEMORY_USAGE_PERCENT=70.0
export BOLT_VALIDATION__VALIDATION_TIMEOUT_SECONDS=120
export BOLT_INTEGRATIONS__INTEGRATION_TIMEOUT_SECONDS=60
```

## Docker Environment

```dockerfile
# Dockerfile example
ENV BOLT_SYSTEM__ENVIRONMENT=production
ENV BOLT_AGENT_POOL__NUM_AGENTS=12
ENV BOLT_MEMORY__MAX_MEMORY_USAGE_GB=16.0
ENV BOLT_LOGGING__LOG_DIRECTORY=/app/logs
ENV BOLT_LOGGING__LEVEL=INFO
```

```bash
# Docker run example
docker run -e BOLT_AGENT_POOL__NUM_AGENTS=16 \
           -e BOLT_MEMORY__MAX_MEMORY_USAGE_GB=24.0 \
           -e BOLT_LOGGING__LEVEL=DEBUG \
           -e BOLT_HARDWARE__ENABLE_MLX=true \
           bolt:latest
```

## Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bolt-config
data:
  BOLT_SYSTEM__ENVIRONMENT: "production"
  BOLT_AGENT_POOL__NUM_AGENTS: "12"
  BOLT_MEMORY__MAX_MEMORY_USAGE_GB: "24.0"
  BOLT_PERFORMANCE__TARGET_TASKS_PER_SECOND: "200.0"
  BOLT_LOGGING__LEVEL: "INFO"
  BOLT_HARDWARE__ENABLE_MLX: "true"
  BOLT_MONITORING__ENABLE_METRICS_EXPORT: "true"
```

## Validation and Troubleshooting

### Check Current Configuration
```bash
python -c "from bolt.core.config import get_default_config; print(get_default_config().dict())"
```

### Common Issues

1. **Invalid agent count**: Ensure `num_agents` doesn't exceed system capabilities
2. **Memory limits**: Don't exceed available system memory
3. **Boolean format**: Use lowercase `true`/`false`
4. **Type mismatches**: Ensure correct types (int, float, string, bool)
5. **Path permissions**: Ensure log directories are writable

### Debug Configuration Loading
```bash
export BOLT_LOGGING__LEVEL=DEBUG
python -c "from bolt.core.config import get_default_config; config = get_default_config(); print('Agents:', config.agent_pool.num_agents)"
```

## Security Considerations

- Use container secrets for sensitive configuration in production
- Avoid logging environment variables containing sensitive data
- Use least-privilege file system permissions
- Consider using configuration management tools for production deployments

## Related Documentation

- [Bolt Configuration Guide](BOLT_CONFIGURATION_GUIDE.md)
- [Bolt Performance Tuning](BOLT_PERFORMANCE_TUNING.md)
- [Bolt Troubleshooting](BOLT_TROUBLESHOOTING.md)