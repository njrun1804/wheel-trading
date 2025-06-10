> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Required Changes for Full Alignment

## Summary of Adjustments Needed

Based on your refined requirements, here's what needs to be adjusted in the current project:

### ✅ Created Examples
1. **`src/diagnostics.py`** - Self-diagnosis system for autonomous operation
2. **`src/config/unity.py`** - Unity-specific configuration with risk parameters
3. **`run.py`** - Example of properly typed, deterministic CLI
4. **Updated `requirements.txt`** - Added mypy, hypothesis, pytz, click, rich

### 🔴 Critical Changes Still Needed

#### 1. **Type Safety Throughout**
```python
# Every function needs:
from __future__ import annotations
from typing import Final, Literal, TypedDict, Never

# Example update for src/utils/math.py:
def black_scholes_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: Literal["call", "put"] = "call"
) -> float | np.ndarray:
```

#### 2. **Implement Risk Analytics**
Create `src/utils/analytics.py`:
- `calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float`
- `calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float`
- `half_kelly_size(win_prob: float, win_size: float, loss_size: float) -> float`
- `calculate_cagr(returns: np.ndarray, periods_per_year: int = 252) -> float`

#### 3. **Objective Function Implementation**
Update `src/wheel.py` to use:
```python
def score_position(self, expected_return: float, cvar: float) -> float:
    """Score = CAGR - 0.20 × |CVaR₉₅|"""
    return expected_return - 0.20 * abs(cvar)
```

#### 4. **Property-Based Tests**
Add to each test file:
```python
from hypothesis import given, strategies as st
from hypothesis import assume

@given(
    current_price=st.floats(min_value=10.0, max_value=100.0),
    strikes=st.lists(st.floats(min_value=5.0, max_value=150.0), min_size=3, max_size=20)
)
def test_optimal_strike_properties(current_price: float, strikes: list[float]) -> None:
    """Property: optimal strike is always in the provided list."""
    assume(len(set(strikes)) == len(strikes))  # No duplicates
    # Test implementation
```

#### 5. **Structured Logging**
Replace all print/logger calls with:
```python
logger.info("Event occurred", extra={
    "event_type": "calculation",
    "ticker": "U",
    "values": {"price": 35.50, "delta": 0.30},
    "timestamp": datetime.now().isoformat()
})
```

### 🟡 Medium Priority Changes

1. **Update all examples** from SPY/AAPL to Unity (U)
2. **Add deterministic behavior**:
   - Set random seeds
   - Sort all collections before processing
   - Use Decimal for financial calculations
3. **Create integration tests** in `tests/test_integration.py`
4. **Add performance benchmarks** with pytest-benchmark

### 🟢 Good to Have (Later)

1. **API documentation** generation with Sphinx
2. **Monitoring dashboard** for autonomous operation
3. **Historical backtesting** framework
4. **ML model stubs** for future enhancement

## Migration Path

### Phase 1: Type Safety (Do First)
```bash
# Add to all Python files
from __future__ import annotations

# Run type checker
mypy src/ --strict

# Fix all type errors
```

### Phase 2: Unity Focus
```bash
# Update configs
cp src/config/unity.py src/config/__init__.py

# Update all tests to use U
find tests -name "*.py" -exec sed -i 's/SPY/U/g' {} \;
```

### Phase 3: Risk Implementation
```bash
# Create analytics module
touch src/utils/analytics.py

# Add CVaR and CAGR calculations
# Update wheel.py to use objective function
```

### Phase 4: Testing Enhancement
```bash
# Run property tests
pytest tests/ -v --hypothesis-show-statistics

# Check determinism
pytest tests/ --randomly-seed=1234
pytest tests/ --randomly-seed=1234  # Should be identical
```

## Validation Checklist

Before considering aligned:
- [ ] All functions have complete type hints
- [ ] Self-diagnostics pass (`python run.py --diagnose`)
- [ ] All examples use Unity (U)
- [ ] CVaR objective function implemented
- [ ] Property-based tests added
- [ ] Structured JSON logging available
- [ ] Performance <200ms verified
- [ ] Documentation updated for every change

## Note on Documentation

Every code change must include:
1. **Docstring** with type information
2. **Unit test** with property test
3. **Integration test** if affects multiple modules
4. **README update** if changes user interface
5. **CLAUDE.md update** if changes architecture

The current codebase is well-structured but needs these enhancements for full autonomous operation capability.
