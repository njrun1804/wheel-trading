# Smart Values Migration Guide

## Overview

This guide outlines the migration from static hardcoded values to smart, adaptive parameters that adjust based on market conditions, ML recommendations, and confidence scores.

## Identified Hardcoded Values

### 1. Risk Limits (src/unity_wheel/risk/limits.py)
```python
# Current hardcoded values:
max_position_pct: float = 0.20
max_contracts: int = 10
min_portfolio_value: float = 10000
max_volatility: float = 1.5
max_gap_percent: float = 0.10
min_volume_ratio: float = 0.5
max_daily_loss_pct: float = 0.02
max_weekly_loss_pct: float = 0.05
max_consecutive_losses: int = 3
min_confidence: float = 0.30
max_warnings: int = 3
```

### 2. Performance Thresholds
- Cache TTL values (currently in config.yaml)
- Retry counts and timeouts
- Confidence score minimums
- SLA targets

### 3. Strategy Parameters
- Delta bounds (currently static in optimizer)
- DTE ranges
- Roll triggers
- Position sizing limits

## Migration Strategy

### Phase 1: Centralize Static Values
1. Move all identified hardcoded values to `config.yaml`
2. Create new sections as needed:
   ```yaml
   risk:
     circuit_breakers:
       max_consecutive_losses: 3
       max_daily_loss_pct: 0.02
       max_weekly_loss_pct: 0.05
     
     adaptive_limits:
       base_max_position_pct: 0.20
       volatility_scaling_enabled: true
       confidence_scaling_enabled: true
   ```

### Phase 2: Implement Adaptive Functions
1. **Volatility-Based Scaling**
   ```python
   def get_adaptive_position_limit(base_limit: float, market_state: MarketState) -> float:
       """Scale position limits based on market volatility."""
       vol_percentile = market_state.volatility_percentile
       # Reduce position size in high volatility
       scaling_factor = 1.0 - (0.5 * vol_percentile)
       return base_limit * scaling_factor
   ```

2. **Confidence-Based Adjustment**
   ```python
   def get_confidence_adjusted_threshold(
       base_value: float,
       confidence: float,
       min_multiplier: float = 0.5,
       max_multiplier: float = 1.5
   ) -> float:
       """Adjust thresholds based on system confidence."""
       multiplier = min_multiplier + (max_multiplier - min_multiplier) * confidence
       return base_value * multiplier
   ```

3. **ML-Driven Parameters**
   ```python
   def get_ml_optimized_parameter(
       parameter_name: str,
       default_value: float,
       market_features: Dict[str, float],
       historical_outcomes: pd.DataFrame
   ) -> Tuple[float, float]:
       """Get ML-optimized parameter with confidence."""
       # Use decision engine to predict optimal value
       optimized_value, confidence = decision_engine.predict_parameter(
           parameter_name,
           market_features,
           historical_outcomes
       )
       return optimized_value if confidence > 0.6 else default_value, confidence
   ```

### Phase 3: Implement Smart Defaults
1. **Dynamic Risk Limits**
   - Position size adjusts with account volatility
   - Loss limits tighten after consecutive losses
   - Margin usage scales with market regime

2. **Adaptive Strategy Parameters**
   - Delta targets adjust with IV rank
   - DTE targets consider earnings calendar
   - Roll triggers learn from past outcomes

3. **Performance-Based Tuning**
   - Cache TTL shortens during high volatility
   - Retry counts increase during network issues
   - Confidence thresholds adapt to recent accuracy

## Implementation Checklist

### Immediate Actions
- [ ] Audit all source files for hardcoded values
- [ ] Create comprehensive config sections
- [ ] Implement config loader with validation
- [ ] Add environment variable overrides

### Smart Value Implementation
- [ ] Create AdaptiveConfig class
- [ ] Implement market state calculators
- [ ] Add confidence scoring to all limits
- [ ] Build historical performance tracker
- [ ] Implement ML parameter optimizer

### Testing & Validation
- [ ] Unit tests for adaptive functions
- [ ] Property tests for parameter bounds
- [ ] Integration tests with market scenarios
- [ ] Performance benchmarks for calculations

## Example Implementation

```python
class AdaptiveRiskLimits:
    """Smart risk limits that adapt to market conditions."""
    
    def __init__(self, config: Config, market_state: MarketState):
        self.config = config
        self.market_state = market_state
        self.performance_tracker = PerformanceTracker()
        
    def get_position_limit(self) -> float:
        """Get adaptive position size limit."""
        base_limit = self.config.risk.base_max_position_pct
        
        # Scale by volatility
        vol_scalar = self._get_volatility_scalar()
        
        # Scale by recent performance
        perf_scalar = self._get_performance_scalar()
        
        # Scale by confidence
        conf_scalar = self._get_confidence_scalar()
        
        # Combine scalars (multiplicative)
        final_limit = base_limit * vol_scalar * perf_scalar * conf_scalar
        
        # Apply bounds
        return max(0.05, min(final_limit, 1.0))  # 5% to 100%
```

## Benefits

1. **Improved Risk Management**
   - Automatic tightening during adverse conditions
   - Expansion during favorable regimes
   - Learning from historical outcomes

2. **Better Performance**
   - Optimized parameters for current conditions
   - Reduced manual tuning
   - Faster adaptation to regime changes

3. **Enhanced Reliability**
   - Graceful degradation under stress
   - Self-healing thresholds
   - Autonomous operation

## Monitoring

Track effectiveness of smart values:
- Parameter drift over time
- Improvement in outcomes
- Confidence score accuracy
- Override frequency

## Rollback Strategy

All smart values must have static fallbacks:
```python
value = get_smart_value() if confidence > min_confidence else static_default
```

This ensures system stability during the migration period.