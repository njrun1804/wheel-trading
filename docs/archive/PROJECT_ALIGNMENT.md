> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Project Alignment with Refined Requirements

## Current Gaps & Required Adjustments

### 1. **Type Safety & Machine Maintainability** 🔴
**Current**: Partial type hints, basic logging
**Required**:
- Exhaustive type hints on ALL functions, including return types
- Structured logging with JSON output for machine parsing
- Deterministic outputs (no random seeds, consistent ordering)
- Self-diagnosis capabilities

**Actions**:
```python
# Example of required type safety
from typing import TypedDict, Literal, Final, Never
from decimal import Decimal  # For deterministic float operations

class PositionRecommendation(TypedDict):
    action: Literal["HOLD", "ROLL", "CLOSE", "OPEN"]
    ticker: Final[Literal["U"]]  # Unity only
    rationale: str
    confidence: Decimal
    diagnostics: dict[str, Any]
```

### 2. **Unity (U) Focus** 🟡
**Current**: Examples use SPY/AAPL
**Required**: Unity-specific implementation

**Actions**:
- Update all examples to use ticker "U"
- Add Unity-specific configuration (typical IV ranges, liquidity constraints)
- Document Unity's options chain characteristics

### 3. **Objective Function** 🔴
**Current**: Simple delta targeting
**Required**: CAGR - 0.20 × |CVaR₉₅| with ½-Kelly sizing

**Actions**:
- Implement CVaR calculation in analytics module
- Add CAGR estimation from expected returns
- Implement half-Kelly position sizing
- Document the mathematical derivation

### 4. **Testing Philosophy** 🔴
**Current**: Unit tests only
**Required**: Property-based testing + self-diagnosis

**Actions**:
```python
# Add to requirements.txt
hypothesis>=6.0.0

# Example property test
from hypothesis import given, strategies as st

@given(
    S=st.floats(min_value=0.01, max_value=1000),
    K=st.floats(min_value=0.01, max_value=1000),
    T=st.floats(min_value=0.001, max_value=2.0),
    r=st.floats(min_value=-0.05, max_value=0.10),
    sigma=st.floats(min_value=0.01, max_value=3.0)
)
def test_black_scholes_put_call_parity(S, K, T, r, sigma):
    """Property: Put-Call parity always holds."""
    call = black_scholes_price(S, K, T, r, sigma, "call")
    put = black_scholes_price(S, K, T, r, sigma, "put")
    parity = call - put
    expected = S - K * np.exp(-r * T)
    assert abs(parity - expected) < 1e-10
```

### 5. **Autonomous Operation** 🔴
**Current**: Basic CLI output
**Required**: Machine-readable, self-diagnosing system

**Actions**:
- Structured JSON output mode
- Health check endpoint
- Automatic validation of all calculations
- Recovery strategies for common failures

### 6. **Documentation Integration** 🟡
**Current**: Separate documentation
**Required**: Every code change includes docs/tests/integration

**Actions**:
- Create `docs/` directory with:
  - API reference (auto-generated)
  - Integration guides
  - Mathematical foundations
- Add doctest examples to all functions
- Create integration test suite

## Immediate Priority Adjustments

### Phase 1: Type Safety & Logging (Do First)
1. Add `from __future__ import annotations` to all modules
2. Add exhaustive type hints
3. Replace print statements with structured logging
4. Add `--json` flag to run.py for machine output

### Phase 2: Unity Configuration
1. Create `src/config/unity.py` with U-specific parameters
2. Update all examples and tests
3. Add Unity market hours validation

### Phase 3: Risk Analytics Foundation
1. Implement CVaR in `src/utils/analytics.py`
2. Add CAGR calculation
3. Implement half-Kelly sizing
4. Document objective function mathematics

### Phase 4: Testing Enhancement
1. Add hypothesis to requirements
2. Create property-based test suite
3. Add determinism tests
4. Create self-diagnosis module

## Configuration Updates Needed

```python
# src/config/unity.py
from decimal import Decimal
from typing import Final

class UnityConfig:
    TICKER: Final[str] = "U"

    # Risk parameters
    CVAR_PERCENTILE: Final[Decimal] = Decimal("0.95")
    OBJECTIVE_RISK_WEIGHT: Final[Decimal] = Decimal("0.20")

    # Position sizing
    KELLY_FRACTION: Final[Decimal] = Decimal("0.5")  # Half-Kelly

    # Unity-specific
    TYPICAL_IV_RANGE: tuple[Decimal, Decimal] = (Decimal("0.30"), Decimal("0.80"))
    MIN_BID_ASK_SPREAD: Final[Decimal] = Decimal("0.05")

    # Market hours (ET)
    MARKET_OPEN: Final[str] = "09:30:00"
    MARKET_CLOSE: Final[str] = "16:00:00"
```

## Integration Requirements

Every PR/commit must include:
1. **Code changes** with exhaustive types
2. **Tests**: Unit + property + integration
3. **Documentation**: Docstrings + user guide updates
4. **Validation**: Self-diagnosis passing
5. **Performance**: <200ms decision time maintained

## Summary

The current codebase provides a good foundation but needs:
- 🔴 **Critical**: Type safety, structured logging, CVaR/CAGR objective
- 🟡 **Important**: Unity focus, property testing
- 🟢 **Good**: Current architecture supports these additions

No architectural changes needed, just enhancement of existing modules with stricter machine-maintainability requirements.
