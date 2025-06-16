# 🚀 Quick Reference Guide

Unity Wheel Trading Bot v2.2 - Common commands and operations.

## Core Commands

### Get Trading Recommendations

```bash
# Basic recommendation
python run.py --portfolio 100000

# With specific account
python run.py --portfolio 100000 --account-id "12345678"

# Conservative settings
python run.py --portfolio 100000 --config examples/core/conservative_config.yaml
```

### System Diagnostics

```bash
# Full system health check
python run.py --diagnose

# Performance metrics
python run.py --performance

# Quick health check
./scripts/health_check.sh

# Database integrity check
python validate_database_schema.py
```

## GPU Orchestrator

### Starting the Orchestrator

```bash
# Interactive mode (recommended)
./orchestrate

# Single command
./orchestrate "optimize Greek calculations"

# Test GPU components
python test_gpu_orchestrator_isolated.py
```

## Development Commands

### Testing

```bash
# Run all tests
pytest --cov=src/unity_wheel

# Quick tests only
pytest -v -m "not slow"

# Specific test file
pytest tests/test_wheel.py -v

# Integration tests
pytest tests/test_autonomous_flow.py -v
```

### Code Quality

```bash
# Auto-format code
black src/ tests/
ruff format .

# Fix linting issues
ruff check --fix .

# Type checking
mypy src/ --strict

# Pre-commit checks
pre-commit run --all-files
```

### Data Management

```bash
# Collect latest data
python tools/download_unity_options_comprehensive.py

# Clean old data
python scripts/cleanup_old_data.py

# Verify data integrity
python validate_database_schema.py
```

## Orchestrator Tasks

### Trading Analysis
- `analyze Unity position`
- `optimize portfolio risk`
- `evaluate SPY options`

### Code Development  
- `refactor risk module`
- `optimize Greek calculations`
- `find performance bottlenecks`

### System Optimization
- `benchmark GPU acceleration`
- `profile memory usage`
- `measure calculation speed`

## Interactive Commands (Orchestrator)

- `help` - Show commands
- `stats` - Session statistics  
- `strategy` - View evolving strategies
- `cache` - Cache performance
- `exit` - Quit

## Tips

1. **First Run**: GPU warmup takes ~200ms (one-time)
2. **Complex Tasks**: Automatically use MCTS (1000+ alternatives)
3. **Confidence**: 95% is usually enough (early stopping)
4. **Evolution**: Strategies improve with use

## Performance

- **GPU**: 826 GFLOPS on M4 Pro
- **Alternatives**: 1000-5000 explored
- **Time**: 15-60s adaptive
- **Parallelism**: 128x operations

## Troubleshooting

```bash
# Fix imports
python scripts/fix_all_imports.py

# Test isolated
python test_gpu_orchestrator_isolated.py

# Check GPU
python -c "import mlx.core as mx; print(mx.default_device())"

# Memory monitoring
python scripts/monitor_memory.py

# Database locks
python scripts/check_database_locks.py
```

## Configuration

### Environment Variables

```bash
# Strategy settings
export WHEEL_STRATEGY__DELTA_TARGET=0.30
export WHEEL_RISK__MAX_POSITION_SIZE=0.20

# Performance tuning
export WHEEL_DATABENTO__LOADER__MAX_WORKERS=12
export USE_PURE_PYTHON=false

# Debugging
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development
```

### Config Files

- `config.yaml` - Main configuration
- `examples/core/conservative_config.yaml` - Conservative settings
- `config/database.yaml` - Database settings

## File Locations

### Core Components
- Entry point: `run.py`
- Main package: `src/unity_wheel/`
- Trading logic: `src/unity_wheel/strategy/wheel.py`
- Risk management: `src/unity_wheel/risk/`
- Math calculations: `src/unity_wheel/math/options.py`

### Data Storage
- Primary database: `data/wheel_trading_master.duckdb`
- Cache directory: `data/cache/`
- Configuration: `config.yaml`

## Common Workflows

### 1. Get Recommendation
```bash
python run.py --portfolio 100000
```

### 2. Development Cycle
```bash
# Make changes
# Run tests
pytest -v -m "not slow"
# Check code quality
pre-commit run --all-files
# Commit
git add . && git commit -m "feat: description"
```

### 3. Performance Analysis
```bash
# Run benchmarks
python scripts/run_benchmarks.py
# Profile code
python -m cProfile run.py --portfolio 100000
```

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System architecture overview
- [Integration Guide](INTEGRATION_GUIDE.md) - External service setup
- [Development Guide](DEVELOPMENT_GUIDE.md) - Development workflow
- [API Guide](docs/API_GUIDE.md) - Python API reference
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

---

**Last Updated**: June 2025  
**Version**: 2.2
