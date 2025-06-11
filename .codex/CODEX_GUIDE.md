# Codex Optimization Guide

## 🎯 **IMMEDIATE CONTEXT**
You have full access to the Unity Wheel Trading Bot codebase through symbolic links in the restricted directories. The real code resides in `src/unity_wheel/` (123 Python files). The older `unity_trading/` path is kept as a symlink for compatibility.

## 🔥 **CRITICAL SUCCESS PATTERNS**

### ✅ **What Works for You:**
1. **File Access**: Prefer `src/unity_wheel/` for imports. The `unity_trading/` symlink is provided only for legacy support.
2. **Testing**: Run `pytest tests/test_*.py -v` - all 106+ tests pass
3. **Type Checking**: `mypy --strict src/unity_wheel data_pipeline app --ignore-missing-imports`
4. **Quick Validation**: `./scripts/housekeeping.sh --unity-check` (2 seconds)
5. **Auto-Fix**: `./scripts/housekeeping.sh --fix` resolves file placement issues
6. Legacy packages `ml_engine`, `risk_engine`, and `strategy_engine` have been removed; use modules under `src/unity_wheel`.

### ❌ **What Blocks You:**
1. **Execution Code**: Never add `execute_trade()`, `place_order()`, `broker.execute()`
2. **Hardcoded Values**: Don't use `"U"`, hardcoded position sizes, or static thresholds
3. **Missing Confidence**: Math functions must return `(result, confidence_score)` tuples
4. **Wrong Directories**: `test_*.py` must be in `tests/`, not root

## 🛠 **OPTIMIZATION WORKFLOWS**

### **Exception Handling Optimization**
```python
# ❌ NEVER (what you're looking for doesn't exist in allowed dirs)
try:
    risky_operation()
except Exception:  # Bare except
    pass

# ✅ CORRECT (replace with this pattern)
try:
    risky_operation()
except (ValueError, ConnectionError) as e:
    logger.error(f"Operation failed: {e}")
    return fallback_value()
```

**Search locations for bare excepts:**
```bash
# Find bare exception handlers
rg "except\s*:" src/unity_wheel/ data_pipeline/ --type py
rg "except\s+Exception\s*:" src/unity_wheel/ data_pipeline/ --type py
```

### **Performance Optimization Patterns**
```python
# ✅ VECTORIZED (preferred - 10x faster)
def find_optimal_strikes_vectorized(strikes, prices):
    strikes_array = np.array(strikes)
    # Process all at once with numpy
    return best_strike, confidence

# ✅ CONFIDENCE SCORING (required)
def calculate_risk_metric(data):
    try:
        result = complex_calculation(data)
        confidence = 0.95 if len(data) > 100 else 0.7
        return result, confidence
    except Exception as e:
        return 0.0, 0.0
```

### **Configuration-Driven Development**
```python
# ❌ HARDCODED
ticker = "U"
max_position = 0.20

# ✅ CONFIG-DRIVEN
from data_pipeline.config.loader import get_config
config = get_config()
ticker = config.unity.ticker
max_position = config.risk.position_limits.max_position_size
```

## 📁 **DIRECTORY MAPPING**

| Codex Access | Real Location | Purpose |
|--------------|---------------|---------|
| `unity_trading/` | `src/unity_wheel/` (canonical) | Main codebase (123 files) |
| `data_pipeline/config/` | `src/config/` | Configuration system |
| `data_pipeline/patterns/` | `src/patterns/` | Reusable patterns |
| `tests/` | `tests/` | Test suite (106+ tests) |
| `app/` | `app/` | Application layer |

## 🧪 **TESTING STRATEGY**

```bash
# Quick test your changes
pytest tests/test_wheel.py tests/test_math.py -v

# Full test suite
pytest -v

# Test specific functionality
pytest tests/test_databento_unity.py tests/test_position_sizing.py -v

# Performance tests
pytest tests/test_performance_benchmarks.py -v
```

## 🔧 **COMMON OPTIMIZATION TASKS**

### 1. **Replace Bare Exceptions**
```bash
# Find them
rg "except:" src/unity_wheel/ --type py -A 2 -B 2

# Replace pattern
except ValueError as e:
    logger.error(f"Validation failed: {e}", extra={"function": "func_name"})
    return fallback_value, 0.0
```

### 2. **Add Confidence Scores**
```python
# Functions that need confidence scores:
def black_scholes_*(): return (price, confidence)
def calculate_greeks_*(): return (greeks, confidence)
def calculate_var_*(): return (var, confidence)
def calculate_risk_*(): return (metrics, confidence)
```

### 3. **Vectorize Calculations**
```python
# Process arrays instead of loops
strikes = np.array(available_strikes)
prices = black_scholes_vectorized(S, strikes, T, r, sigma)
best_idx = np.argmin(scores)
```

### 4. **Extract Hardcoded Values**
```python
# Add to config.yaml:
unity:
  ticker: "U"
  contracts_per_trade: 1

# Use in code:
ticker = config.unity.ticker
contracts = config.unity.contracts_per_trade
```

## 🚀 **QUICK VALIDATION**

```bash
# Before making changes
./scripts/housekeeping.sh --unity-check

# After making changes
./scripts/housekeeping.sh --fix --dry-run  # Preview fixes
./scripts/housekeeping.sh --fix            # Apply fixes
pytest tests/test_wheel.py -v              # Test core functionality
```

## 💡 **SUCCESS METRICS**

- **Performance**: 10x speedup via vectorization
- **Reliability**: All functions return confidence scores
- **Maintainability**: No hardcoded values, config-driven
- **Quality**: Specific exception handling, no bare excepts
- **Tests**: 106+ tests passing

## 🎯 **YOUR OPTIMIZATION FOCUS**

1. **Exception Handling**: Replace bare `except:` with specific exceptions
2. **Performance**: Vectorize loops using numpy arrays
3. **Confidence**: Add confidence scores to calculation functions
4. **Configuration**: Move hardcoded values to config.yaml
5. **Testing**: Ensure all changes have test coverage

## ⚡ **IMMEDIATE ACTION ITEMS**

```bash
# 1. Check current status
./scripts/housekeeping.sh --unity-check

# 2. Find optimization targets
rg "except:" src/unity_wheel/ --type py
rg "for.*in.*range" src/unity_wheel/ --type py | grep -v test

# 3. Make changes using patterns above

# 4. Validate
pytest tests/test_wheel.py -v
./scripts/housekeeping.sh --fix

# 5. Commit
./scripts/commit-workflow.sh -m "Optimize exception handling and performance" -y
```

Remember: You have full access to optimize the codebase. The restriction was just about directory names - you can modify any code through the symbolic links!
