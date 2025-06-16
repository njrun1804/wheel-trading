# Einstein Configuration System

The Einstein configuration system provides centralized, externalized configuration management with automatic hardware detection and environment variable support.

## Overview

All hardcoded configurations have been externalized to `einstein_config.py`, which provides:

- **Automatic hardware detection** - Detects CPU cores, memory, GPU, and platform type
- **Environment variable overrides** - All settings can be overridden via environment variables
- **Dynamic adjustments** - Configuration automatically adjusts based on detected hardware
- **Type-safe configuration** - Uses dataclasses for structured configuration

## Configuration Structure

### Hardware Configuration (Auto-detected)
- CPU cores (total, performance, efficiency)
- Memory (total and available)
- GPU availability and cores
- Platform type (Apple Silicon, Intel, AMD)
- Architecture (arm64, x86_64)

### Performance Configuration
- Startup time targets
- Search performance targets
- Memory usage limits
- Concurrency limits

### Cache Configuration
- LRU cache sizes
- Memory management settings
- Cache policies and TTLs

### Path Configuration
- Base directory
- Cache directories
- Database paths
- Model storage paths

### ML Configuration
- Learning rates
- Embedding dimensions
- Training parameters
- Similarity thresholds

### Monitoring Configuration
- File watching settings
- Performance monitoring
- Memory monitoring
- Logging levels

## Environment Variables

All configuration values can be overridden using environment variables:

### Performance Settings
- `EINSTEIN_MAX_STARTUP_MS` - Maximum startup time in milliseconds (default: 500)
- `EINSTEIN_MAX_SEARCH_MS` - Maximum search time in milliseconds (default: 50)
- `EINSTEIN_MAX_MEMORY_GB` - Maximum memory usage in GB (default: 2.0)
- `EINSTEIN_SEARCH_CONCURRENCY` - Max concurrent searches (default: 4)
- `EINSTEIN_FILE_IO_CONCURRENCY` - Max concurrent file operations (default: 12)

### Cache Settings
- `EINSTEIN_HOT_CACHE_SIZE` - Hot cache entry limit (default: 1000)
- `EINSTEIN_WARM_CACHE_SIZE` - Warm cache entry limit (default: 5000)
- `EINSTEIN_SEARCH_CACHE_SIZE` - Search cache entry limit (default: 500)
- `EINSTEIN_INDEX_CACHE_MB` - Index cache size in MB (default: 256)

### ML Settings
- `EINSTEIN_LEARNING_RATE` - Adaptive learning rate (default: 0.1)
- `EINSTEIN_SIMILARITY_THRESHOLD` - Similarity threshold (default: 0.7)
- `EINSTEIN_CONFIDENCE_THRESHOLD` - Confidence threshold (default: 0.3)

### Monitoring Settings
- `EINSTEIN_LOG_LEVEL` - Logging level (DEBUG, INFO, WARN, ERROR) (default: INFO)
- `EINSTEIN_DEBOUNCE_MS` - File change debounce delay (default: 250)
- `EINSTEIN_MEMORY_CHECK_INTERVAL` - Memory check interval in seconds (default: 30)

### Feature Flags
- `EINSTEIN_USE_GPU` - Enable GPU acceleration (true/false) (default: auto-detect)
- `EINSTEIN_ADAPTIVE_CONCURRENCY` - Enable adaptive concurrency (default: true)
- `EINSTEIN_PREDICTIVE_PREFETCH` - Enable predictive prefetching (default: true)
- `EINSTEIN_MEMORY_OPTIMIZATION` - Enable memory optimization (default: true)
- `EINSTEIN_REALTIME_INDEXING` - Enable real-time indexing (default: true)

### Path Settings
- `EINSTEIN_CACHE_DIR` - Override cache directory location

## Usage

### Basic Usage

```python
from einstein.einstein_config import get_einstein_config

# Get configuration (auto-detects hardware)
config = get_einstein_config()

# Access configuration values
max_workers = config.hardware.cpu_cores
cache_size = config.cache.hot_cache_size
search_timeout = config.performance.max_search_time_ms
```

### With Environment Overrides

```bash
# Set environment variables
export EINSTEIN_MAX_SEARCH_MS=25
export EINSTEIN_HOT_CACHE_SIZE=2000
export EINSTEIN_USE_GPU=false

# Run your application
python your_app.py
```

### Testing Configuration

```bash
# Run the configuration test suite
python test_einstein_config.py
```

## Hardware Auto-Detection

The system automatically detects:

1. **CPU Configuration**
   - Total logical cores
   - Performance cores (P-cores) on Apple Silicon
   - Efficiency cores (E-cores) on Apple Silicon

2. **Memory**
   - Total system memory
   - Available memory
   - Adjusts limits to use max 80% of available

3. **GPU**
   - Metal GPU on macOS
   - NVIDIA/AMD GPUs via GPUtil
   - Automatically disables GPU features if not available

4. **Platform**
   - Detects Apple Silicon, Intel, AMD
   - Architecture (ARM64, x86_64)

## Dynamic Adjustments

The configuration automatically adjusts based on hardware:

### Low Memory Systems (<16GB)
- Reduces cache sizes by 50%
- Lowers memory usage targets
- Adjusts concurrency limits

### CPU Core Scaling
- File I/O concurrency scales with CPU cores
- Search concurrency limited to CPU_CORES / 2
- Analysis concurrency adjusted for available cores

### GPU Availability
- Disables GPU acceleration if no GPU detected
- Adjusts ML batch sizes based on GPU memory

### Performance Targets
- Slower systems get relaxed timing constraints
- Search and startup targets scale with CPU speed

## Migration from Hardcoded Values

All hardcoded values have been replaced:

| Old Hardcoded Value | New Configuration Path |
|-------------------|----------------------|
| `12` cores | `config.hardware.cpu_cores` |
| `8` P-cores | `config.hardware.cpu_performance_cores` |
| `4` E-cores | `config.hardware.cpu_efficiency_cores` |
| `24GB` RAM | `config.hardware.memory_total_gb` |
| `19.2GB` allocated | `config.hardware.memory_available_gb * 0.8` |
| `20` GPU cores | `config.hardware.gpu_cores` |
| `1000` cache size | `config.cache.hot_cache_size` |
| `50ms` search target | `config.performance.max_search_time_ms` |
| `500ms` startup target | `config.performance.max_startup_time_ms` |
| `2GB` memory limit | `config.performance.max_memory_usage_gb` |
| `.einstein/` paths | `config.paths.*` |

## Best Practices

1. **Always use configuration** - Never hardcode values
2. **Document defaults** - Explain why default values were chosen
3. **Test with overrides** - Verify environment variables work
4. **Handle missing config** - Gracefully handle configuration errors
5. **Log configuration** - Log active configuration on startup

## Extending Configuration

To add new configuration options:

1. Add field to appropriate dataclass in `einstein_config.py`
2. Add environment variable support in `ConfigLoader`
3. Document the new setting in this README
4. Update components to use the new configuration

## Troubleshooting

### Configuration Not Loading
- Check file permissions on `.einstein` directory
- Verify environment variables are set correctly
- Check logs for hardware detection errors

### Performance Issues
- Review auto-adjusted values for your hardware
- Override with environment variables if needed
- Monitor actual vs. target performance

### Memory Issues  
- Check `EINSTEIN_MAX_MEMORY_GB` setting
- Verify cache sizes are appropriate
- Enable memory optimization features