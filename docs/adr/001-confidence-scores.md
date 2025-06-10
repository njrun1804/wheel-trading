# ADR-001: All Calculations Return Confidence Scores

## Status
Accepted

## Context
The Unity Wheel Trading Bot operates autonomously, making financial decisions without human intervention. We need a mechanism to assess the reliability of every calculation and decision to prevent acting on low-quality data or uncertain computations.

## Decision
All calculation functions must return a `CalculationResult` object or tuple containing:
1. The calculated value
2. A confidence score (0.0 to 1.0)
3. Optional warnings list

## Consequences

### Positive
- Autonomous system can make informed decisions about when to trade
- Clear degradation path when data quality is poor
- Easier debugging with confidence tracking
- Natural circuit breaker when confidence drops

### Negative
- Every function needs confidence scoring logic
- Slightly more complex return types
- Performance overhead of confidence calculation

### Neutral
- Requires consistent patterns across codebase
- Testing must verify confidence scoring

## Implementation
```python
@dataclass
class CalculationResult:
    value: float
    confidence: float
    warnings: List[str] = field(default_factory=list)

# Example usage
def black_scholes_price_validated(...) -> CalculationResult:
    # Validation
    if not all(x > 0 for x in [S, K, T, sigma]):
        return CalculationResult(np.nan, 0.0, ["Invalid inputs"])

    # Calculation
    price = black_scholes_formula(...)

    # Confidence scoring
    confidence = 0.99  # High confidence for valid inputs
    if T < 0.01:  # Near expiration
        confidence *= 0.8

    return CalculationResult(price, confidence, [])
```

## References
- Original design discussion: See ADAPTIVE_SYSTEM_DESIGN.md
- Implementation: src/unity_wheel/math/options.py
- Tests: tests/test_math.py
