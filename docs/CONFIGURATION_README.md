# Configuration System Documentation

This directory contains comprehensive documentation and examples for configuring both Einstein and Bolt systems. The configuration system provides flexible, environment-aware settings management with validation and migration tools.

## 📁 Documentation Structure

```
docs/
├── CONFIGURATION_README.md                    # This file - overview and quick start
├── EINSTEIN_CONFIGURATION_GUIDE.md            # Complete Einstein configuration guide
├── BOLT_CONFIGURATION_GUIDE.md               # Complete Bolt configuration guide
├── EINSTEIN_ENVIRONMENT_VARIABLES.md         # Einstein environment variable reference
├── BOLT_ENVIRONMENT_VARIABLES.md             # Bolt environment variable reference
├── CONFIGURATION_MIGRATION_GUIDE.md          # Migration from hardcoded values
└── examples/
    └── config_validation/
        ├── einstein_config_validation.py      # Einstein validation examples
        └── bolt_config_validation.py          # Bolt validation examples

templates/
├── einstein/
│   └── config.yaml.example                   # Einstein configuration template
└── bolt/
    └── config.yaml.example                   # Bolt configuration template
```

## 🚀 Quick Start

### Einstein Configuration

1. **Use Default Configuration** (auto-detects hardware):
   ```python
   from einstein.einstein_config import get_einstein_config
   config = get_einstein_config()
   ```

2. **Custom Configuration File**:
   ```bash
   cp einstein/config.yaml.example einstein/config.yaml
   # Edit config.yaml as needed
   ```

3. **Environment Variables**:
   ```bash
   export EINSTEIN_LOG_LEVEL=DEBUG
   export EINSTEIN_MAX_STARTUP_MS=1000
   export EINSTEIN_HOT_CACHE_SIZE=2000
   ```

### Bolt Configuration

1. **Use Default Configuration**:
   ```python
   from bolt.core.config import get_default_config
   config = get_default_config()
   ```

2. **Custom Configuration**:
   ```bash
   cp bolt/config.yaml.example bolt/config.yaml
   # Edit config.yaml as needed
   ```

3. **Environment Variables**:
   ```bash
   export BOLT_AGENT_POOL__NUM_AGENTS=16
   export BOLT_HARDWARE__ENABLE_MLX=true
   export BOLT_LOGGING__LEVEL=INFO
   ```

## 📖 Documentation Overview

### Configuration Guides

| Document | Purpose | Audience |
|----------|---------|----------|
| [Einstein Configuration Guide](EINSTEIN_CONFIGURATION_GUIDE.md) | Complete Einstein configuration reference | All Einstein users |
| [Bolt Configuration Guide](BOLT_CONFIGURATION_GUIDE.md) | Complete Bolt configuration reference | All Bolt users |
| [Migration Guide](CONFIGURATION_MIGRATION_GUIDE.md) | Moving from hardcoded to external config | Developers |

### Reference Documentation

| Document | Purpose | Use Case |
|----------|---------|----------|
| [Einstein Environment Variables](EINSTEIN_ENVIRONMENT_VARIABLES.md) | All Einstein env vars | DevOps, deployment |
| [Bolt Environment Variables](BOLT_ENVIRONMENT_VARIABLES.md) | All Bolt env vars | DevOps, deployment |

### Examples and Tools

| File | Purpose | Usage |
|------|---------|-------|
| `einstein_config_validation.py` | Configuration validation examples | Testing, validation |
| `bolt_config_validation.py` | Configuration validation examples | Testing, validation |
| `config.yaml.example` templates | Configuration templates | Initial setup |

## ⚙️ Configuration Features

### Auto-Detection

Both systems automatically detect hardware capabilities:

- **CPU**: Core count, performance vs efficiency cores (Apple Silicon)
- **Memory**: Total and available memory
- **GPU**: Apple Metal, NVIDIA/AMD GPUs
- **ANE**: Apple Neural Engine (Apple Silicon)

### Environment Support

Three-tier configuration hierarchy:

1. **Environment Variables** (highest precedence)
2. **Configuration Files** (medium precedence)  
3. **Default Values** (lowest precedence)

### Validation

Built-in validation ensures:

- Type correctness
- Value ranges
- Hardware compatibility
- Cross-section consistency

### Migration Support

Tools and guides for migrating from hardcoded values:

- Hardcoded value scanners
- Migration scripts
- Validation tools
- Rollback procedures

## 🛠️ Configuration Management

### Development Workflow

1. **Start with defaults** - Use auto-detection for initial setup
2. **Customize as needed** - Create configuration files for specific needs
3. **Use environment variables** - Override for different environments
4. **Validate configuration** - Use validation tools to ensure correctness
5. **Monitor performance** - Track impact of configuration changes

### Deployment Strategy

#### Development Environment
```bash
# Enable debug features
export EINSTEIN_LOG_LEVEL=DEBUG
export BOLT_SYSTEM__ENABLE_DEBUG=true

# Reduce resource usage
export EINSTEIN_MAX_MEMORY_GB=4.0
export BOLT_AGENT_POOL__NUM_AGENTS=4
```

#### Production Environment
```bash
# Optimize for performance
export EINSTEIN_LOG_LEVEL=INFO
export BOLT_SYSTEM__ENVIRONMENT=production

# Scale for production
export EINSTEIN_HOT_CACHE_SIZE=5000
export BOLT_AGENT_POOL__NUM_AGENTS=12
```

#### Container Environment
```bash
# Container-friendly settings
export EINSTEIN_CACHE_DIR=/app/cache
export BOLT_LOGGING__LOG_DIRECTORY=/app/logs

# Respect container limits
export BOLT_MEMORY__MAX_MEMORY_USAGE_PERCENT=70.0
```

### Best Practices

1. **Version Control**: Keep configuration files in version control
2. **Environment Separation**: Use different configs for different environments
3. **Secrets Management**: Don't put sensitive values in configuration files
4. **Validation**: Always validate configuration before deployment
5. **Documentation**: Document custom configuration choices
6. **Monitoring**: Monitor configuration changes and their impact

## 🔧 Common Configuration Patterns

### High-Performance System

```yaml
# Einstein
performance:
  max_memory_usage_gb: 16.0
  max_search_concurrency: 16
  max_file_io_concurrency: 24

cache:
  hot_cache_size: 10000
  warm_cache_size: 50000
  index_cache_size_mb: 2048

# Bolt
agent_pool:
  num_agents: 16
  batch_size: 128

performance:
  target_cpu_utilization: 0.9
  target_tasks_per_second: 500.0

memory:
  max_memory_usage_gb: 32.0
```

### Resource-Constrained System

```yaml
# Einstein
performance:
  max_memory_usage_gb: 2.0
  max_search_concurrency: 2

cache:
  hot_cache_size: 250
  warm_cache_size: 1000
  index_cache_size_mb: 64

# Bolt
agent_pool:
  num_agents: 2
  batch_size: 8

memory:
  max_memory_usage_gb: 4.0
  gc_threshold: 0.7

hardware:
  enable_gpu: false
```

### Apple Silicon Optimization

```yaml
# Einstein
ml:
  enable_ane: true
  ane_batch_size: 512
  ane_cache_size_mb: 1024

performance:
  max_embedding_concurrency: 8  # P-cores
  max_file_io_concurrency: 12   # All cores

# Bolt
cpu_optimization:
  enable_core_affinity: true
  p_cores: [0, 1, 2, 3, 4, 5, 6, 7]
  e_cores: [8, 9, 10, 11]

hardware:
  enable_mlx: true
  enable_metal: true
  acceleration_preference: "mlx"
```

## 🧪 Testing Configuration

### Validation Scripts

Run validation to ensure configuration is correct:

```bash
# Einstein validation
python examples/config_validation/einstein_config_validation.py

# Bolt validation  
python examples/config_validation/bolt_config_validation.py
```

### Configuration Testing

Test different configurations:

```python
# Test Einstein configuration
from einstein.einstein_config import get_einstein_config
from examples.config_validation.einstein_config_validation import EinsteinConfigValidator

config = get_einstein_config()
validator = EinsteinConfigValidator(config)
is_valid = validator.validate_all()

# Test Bolt configuration
from bolt.core.config import get_default_config
from examples.config_validation.bolt_config_validation import BoltConfigValidator

config = get_default_config()
validator = BoltConfigValidator(config)
is_valid = validator.validate_all()
```

### Performance Testing

Measure impact of configuration changes:

```python
import time

def test_performance_impact():
    # Test with different configurations
    configs = [
        {"cache_size": 1000, "agents": 8},
        {"cache_size": 5000, "agents": 12},
        {"cache_size": 10000, "agents": 16},
    ]
    
    for config in configs:
        start_time = time.time()
        # Run your workload
        duration = time.time() - start_time
        print(f"Config {config}: {duration:.2f}s")
```

## 🚨 Troubleshooting

### Common Issues

1. **Configuration not loading**
   - Check file permissions
   - Verify YAML syntax
   - Check file paths

2. **Environment variables not working**
   - Verify variable names (case-sensitive)
   - Check variable format (use double underscores)
   - Confirm variable scope

3. **Performance degradation**
   - Compare with baseline
   - Check resource limits
   - Validate cache sizes

4. **Hardware not detected**
   - Check hardware detection logs
   - Override with environment variables
   - Verify dependencies installed

### Debug Configuration Loading

Enable debug logging to troubleshoot:

```bash
# Einstein debug
export EINSTEIN_LOG_LEVEL=DEBUG
python -c "from einstein.einstein_config import get_einstein_config; get_einstein_config()"

# Bolt debug
export BOLT_LOGGING__LEVEL=DEBUG
python -c "from bolt.core.config import get_default_config; get_default_config()"
```

### Configuration Validation

Use validation tools to check configuration:

```bash
# Validate Einstein configuration
python -m examples.config_validation.einstein_config_validation

# Validate Bolt configuration
python -m examples.config_validation.bolt_config_validation
```

## 🔗 Related Documentation

- [Einstein System README](../EINSTEIN_README.md)
- [Bolt System README](../BOLT_README.md)  
- [Hardware Optimization Guide](../HARDWARE_OPTIMIZATION_GUIDE.md)
- [Performance Tuning Guide](../PERFORMANCE_TUNING_GUIDE.md)
- [Deployment Guide](../DEPLOYMENT_GUIDE.md)

## 📞 Support

For configuration questions or issues:

1. Check the relevant configuration guide
2. Run validation scripts to identify issues
3. Review environment variable documentation
4. Check troubleshooting sections
5. Create an issue with configuration details and error messages

## 🔄 Contributing

When adding new configuration options:

1. Update the appropriate configuration class
2. Add environment variable support
3. Update validation logic
4. Add documentation to the guide
5. Include examples in validation scripts
6. Update migration tools if needed

This configuration system provides flexible, validated, environment-aware settings management for both Einstein and Bolt systems. Use the documentation and tools provided to optimize your deployment for your specific needs.