# 🚀 GPU Orchestrator Quick Reference

## Starting the Orchestrator

```bash
# Interactive mode (recommended)
./orchestrate

# Single command
./orchestrate "optimize Greek calculations"

# Test GPU components
python test_gpu_orchestrator_isolated.py
```

## Common Tasks

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

## Interactive Commands

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
python fix_all_imports.py

# Test isolated
python test_gpu_orchestrator_isolated.py

# Check GPU
python -c "import mlx.core as mx; print(mx.default_device())"
```
