# ADR-002: Unified Position Sizing via DynamicPositionSizer

## Status
Accepted (January 2025)

## Context
The codebase had multiple implementations of position sizing logic scattered across different modules, leading to inconsistencies and maintenance challenges. Unity has specific constraints (max 3 concurrent puts) that need consistent enforcement.

## Decision
All position sizing must go through the `DynamicPositionSizer` class in `src/unity_wheel/utils/position_sizing.py`. No other module should implement position sizing logic independently.

## Consequences

### Positive
- Single source of truth for position limits
- Consistent application of risk constraints
- Easier to modify sizing rules globally
- Better testing coverage

### Negative
- Additional abstraction layer
- All modules must import from utils
- Breaking change for existing code

### Neutral
- Requires refactoring existing implementations
- Must maintain backward compatibility during migration

## Implementation
```python
from src.unity_wheel.utils.position_sizing import DynamicPositionSizer

# Initialize with config
sizer = DynamicPositionSizer(config)

# Calculate position
result = sizer.calculate_position_size(
    portfolio_value=100000,
    option_price=1.50,
    strike_price=45.0,
    current_positions=existing_positions,
    market_regime="normal"
)

# Always check confidence
if result.confidence > 0.7:
    contracts = result.contracts
else:
    # Handle low confidence
    pass
```

## Constraints Enforced
1. Maximum 20% of portfolio per position
2. Maximum 3 concurrent Unity puts
3. Minimum position value of $1,000
4. Kelly criterion with 1/2 sizing for safety

## Migration Path
1. ✅ Create DynamicPositionSizer class
2. ✅ Migrate WheelStrategy.calculate_position_size
3. ✅ Update all callers to use new API
4. ✅ Remove old implementations
5. ✅ Add deprecation warnings

## References
- Implementation: src/unity_wheel/utils/position_sizing.py
- Tests: tests/test_position_sizing.py
- Migration PR: #[TBD]
