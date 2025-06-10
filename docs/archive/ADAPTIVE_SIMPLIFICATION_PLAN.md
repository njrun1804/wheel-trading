> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Adaptive System Simplification Plan

## What to Remove

### 1. Generic adaptive.py
- **Current**: Two adaptive systems (generic + Unity-specific)
- **Action**: Remove generic, keep only Unity-specific
- **Reason**: We only trade Unity, no need for abstract system

### 2. Overly Complex Market Conditions
- **Remove from MarketConditions**:
  - `volatility_percentile` (just use absolute vol)
  - `price_momentum` (not used effectively)
  - `volume_ratio` (Unity always liquid enough)

- **Keep**:
  - `realized_volatility` (critical)
  - `iv_rank` (edge indicator)
  - `days_to_earnings` (Unity-specific risk)
  - `current_drawdown` (risk management)

### 3. Unnecessary Config Sections
- **Remove**:
  - ML integration settings (not implemented)
  - Multiple scaling toggles (just always scale)
  - Abstract "confidence" scoring

- **Keep**:
  - Circuit breakers (hard limits)
  - Position sizing rules
  - Unity-specific thresholds

### 4. Complex Backtesting
- **Current**: Full market simulation
- **Better**: Simple historical win/loss rates
- **Focus**: Drawdown prevention, not alpha generation

## What to Add

### 1. Outcome Tracking
```python
@dataclass
class WheelOutcome:
    date: datetime
    recommendation: Dict
    actual_result: Optional[Dict] = None
    pnl: Optional[float] = None

# Store outcomes for continuous improvement
outcomes_db = DuckDB("wheel_outcomes.db")
```

### 2. Simple A/B Testing
```python
def get_recommendation_with_experiment():
    if random.random() < 0.1:  # 10% experimental
        params = get_experimental_params()
        track_experiment = True
    else:
        params = get_standard_params()
        track_experiment = False
```

### 3. Unity Earnings Calendar Integration
- Automated earnings date fetching
- Pre-earnings parameter adjustments
- Post-earnings vol crush detection

## Simplified Architecture

```
wheel-trading/
├── src/
│   ├── unity_wheel/
│   │   ├── adaptive/
│   │   │   ├── __init__.py
│   │   │   ├── unity_sizing.py    # Position sizing only
│   │   │   ├── unity_params.py    # Parameter selection
│   │   │   └── outcomes.py        # Track results
│   │   └── strategy/
│   │       └── wheel.py           # Uses adaptive directly
└── config.yaml                    # Simplified config
```

## Key Principles

1. **Unity Only**: Remove all multi-asset abstractions
2. **Practical Focus**: What actually affects P&L
3. **Measurable**: Track every recommendation outcome
4. **Simple Rules**: If/then, not complex math
5. **Fail Safe**: Conservative defaults always

## Metrics That Matter

### Keep Tracking:
- Win rate at different deltas
- Average loss when assigned
- Drawdown from peak
- Earnings skip effectiveness

### Stop Tracking:
- Complex Greeks beyond delta
- Correlation matrices
- Statistical confidence scores
- Multi-regime classifications

## Configuration Simplification

### Before (config.yaml):
```yaml
risk:
  limits:
    max_var_95: 0.05
    max_cvar_95: 0.075
    max_kelly_fraction: 0.25
  circuit_breakers:
    max_position_pct: 0.20
  adaptive_limits:
    enabled: true
    volatility_scaling: true
    confidence_scaling: true
```

### After (config.yaml):
```yaml
unity_wheel:
  max_position_pct: 0.20      # Of portfolio
  max_drawdown: 0.20          # Stop at -20%

  # Unity-specific
  normal_vol_range: [0.40, 0.60]
  skip_earnings_days: 7

  # Simple rules
  high_vol_reduction: 0.70    # Multiply size by this
  drawdown_reduction: 0.50    # Per 10% drawdown
```

## Testing Simplification

### Current: Complex backtesting
### Better: Simple validation
```python
def validate_rule(rule_name, historical_data):
    """Did this rule help or hurt?"""
    with_rule = apply_rule(historical_data)
    without_rule = baseline(historical_data)

    return {
        'improvement': with_rule['objective'] - without_rule['objective'],
        'drawdown_reduction': with_rule['max_dd'] - without_rule['max_dd'],
        'worth_keeping': improvement > 0.01  # 1% improvement threshold
    }
```

## Implementation Priority

1. **Week 1**: Remove generic adaptive system
2. **Week 2**: Simplify Unity adaptive to core rules
3. **Week 3**: Add outcome tracking
4. **Week 4**: Validate with historical data

## Success Metrics

- Fewer than 10 adaptive parameters
- All rules explainable in one sentence
- Drawdown reduced by >20% vs static
- Code reduced by >30%
- Zero ML dependencies
