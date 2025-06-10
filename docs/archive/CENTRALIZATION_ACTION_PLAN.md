> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Hardcoded Values Centralization Action Plan

## Executive Summary

The codebase contains ~150+ hardcoded values across risk management, strategy parameters, network configuration, and data processing. This action plan prioritizes centralizing the most critical values that directly affect trading decisions and system reliability.

## Priority 1: Critical Trading Parameters (Do First)

These values directly affect P&L and must be centralized immediately:

### 1.1 Risk Limits Configuration

**File**: Update `config.yaml` to add:
```yaml
risk:
  limits:
    max_position_pct: 0.20          # Max position as % of portfolio
    max_consecutive_losses: 3       # Stop after N losses
    min_confidence: 0.30            # Minimum confidence to trade
    max_daily_loss_pct: 0.02        # 2% daily loss limit
    max_weekly_loss_pct: 0.05       # 5% weekly loss limit
    max_contracts: 10               # Maximum contracts per position
    min_portfolio_value: 10000      # Minimum account size to trade

  market_conditions:
    max_volatility: 1.5             # 150% annual volatility limit
    max_gap_percent: 0.10           # 10% gap/shock limit
    min_volume_ratio: 0.5           # Minimum volume vs average

  var_cvar:
    max_var_95: 0.05                # 5% VaR limit
    max_cvar_95: 0.075              # 7.5% CVaR limit

  kelly:
    max_fraction: 0.25              # Maximum Kelly fraction
    default_fraction: 0.50          # Half-Kelly default

  greeks:
    max_delta_exposure: 100.0       # Maximum delta exposure
    max_gamma_exposure: 10.0        # Maximum gamma exposure
    max_vega_exposure: 1000.0       # Maximum vega exposure ($)
    max_theta_decay: 100.0          # Maximum daily theta decay

  margin:
    max_utilization: 0.50           # 50% margin utilization max
    safety_factor: 0.80             # 80% of available margin
```

**Code Changes Required**:
1. Update `src/unity_wheel/risk/limits.py` to read from config
2. Update `src/unity_wheel/risk/analytics.py` to use config values
3. Update `src/unity_wheel/strategy/wheel.py` position sizing logic

### 1.2 Adaptive System Parameters

**File**: Add to `config.yaml`:
```yaml
adaptive:
  # Regime detection
  regime_persistence_days: 3        # Days to confirm regime change

  # Per-regime parameters
  regime_params:
    normal:
      put_delta: 0.30               # Target delta for puts
      target_dte: 35                # Days to expiration
      roll_profit_target: 0.50      # Roll at 50% profit
      position_size_factor: 1.0     # 100% of base size

    volatile:
      put_delta: 0.25               # Lower delta in volatile markets
      target_dte: 28                # Shorter duration
      roll_profit_target: 0.25      # Take profits quickly
      position_size_factor: 0.7     # 70% of base size

    stressed:
      put_delta: 0.20               # Much lower delta
      target_dte: 21                # Shortest duration
      roll_profit_target: 0.25      # Quick profits
      position_size_factor: 0.5     # 50% of base size

    low_volatility:
      put_delta: 0.35               # Higher delta for more premium
      target_dte: 42                # Longer duration
      roll_profit_target: 0.50      # Standard profit target
      position_size_factor: 1.2     # 120% of base size (opportunity)

  # Stop conditions
  stop_conditions:
    max_volatility: 1.0             # 100% volatility = stop
    max_drawdown: 0.20              # 20% drawdown = stop
    min_days_to_earnings: 7         # Skip if earnings < 7 days
```

**Code Changes Required**:
1. Update `src/unity_wheel/strategy/adaptive_base.py` to use config
2. Remove hardcoded regime parameters
3. Make position sizing use config factors

## Priority 2: Operational Parameters (Do Second)

These affect system reliability and performance:

### 2.1 Network and API Configuration

**File**: Update `config.yaml`:
```yaml
data:
  api_timeouts:
    connect: 5.0                    # Connection timeout (seconds)
    read: 30.0                      # Read timeout
    write: 10.0                     # Write timeout
    total: 60.0                     # Total request timeout

  retry:
    max_attempts: 3                 # Maximum retry attempts
    delays: [1, 2, 5]               # Exponential backoff delays
    rate_limit_wait: 60             # Wait after rate limit

  cache_ttl:
    account_data: 30                # Account data cache (seconds)
    option_chains: 900              # 15 minutes for option chains
    greeks: 300                     # 5 minutes for Greeks
    market_data: 60                 # 1 minute for spot prices
    historical: 86400               # 1 day for historical data
```

### 2.2 Performance Monitoring

**File**: Add to `config.yaml`:
```yaml
performance:
  sla:
    black_scholes_ms: 50            # Black-Scholes calculation
    greeks_ms: 100                  # Greeks calculation
    risk_metrics_ms: 200            # Risk metrics calculation
    decision_ms: 500                # Total decision time
    api_call_ms: 5000               # External API calls

  monitoring:
    metrics_retention_days: 90      # Keep metrics for 90 days
    alert_threshold_pct: 150        # Alert if >150% of SLA
    sample_rate: 0.1                # Sample 10% of operations
```

## Priority 3: Data Quality Parameters (Do Third)

### 3.1 Market Data Validation

**File**: Add to `config.yaml`:
```yaml
data:
  quality:
    max_spread_pct: 10.0            # Maximum bid-ask spread %
    min_quote_size: 1               # Minimum quote size
    max_price_change_pct: 50.0      # Maximum price change %
    min_options_per_expiry: 10      # Minimum option contracts
    stale_data_minutes: 15          # Data considered stale after
    min_liquidity:
      volume: 10                    # Minimum daily volume
      open_interest: 100            # Minimum open interest
      bid_size: 1                   # Minimum bid size
```

### 3.2 Databento Integration

**File**: Add databento section:
```yaml
databento:
  filters:
    moneyness_range: 0.20           # 20% around spot price
    max_expirations: 3              # Keep 3 nearest expirations
    min_volume: 0                   # Include all volume levels

  storage:
    local_retention_days: 30        # Keep 30 days locally
    compression: true               # Enable compression
    partitioning: "daily"           # Partition by day
```

## Implementation Steps

### Step 1: Update Configuration Schema (1 hour)

1. Update `src/config/schema.py` to add new configuration classes:
```python
@dataclass
class AdaptiveConfig:
    """Adaptive system configuration."""
    regime_persistence_days: int = 3
    regime_params: Dict[str, RegimeParams] = field(default_factory=dict)
    stop_conditions: StopConditions = field(default_factory=StopConditions)
```

2. Add validation for new fields
3. Ensure backward compatibility

### Step 2: Update Risk Module (2 hours)

1. Modify `src/unity_wheel/risk/limits.py`:
```python
# Before
max_position_pct: float = 0.20

# After
max_position_pct: float = field(init=False)

def __post_init__(self):
    config = get_config()
    self.max_position_pct = config.risk.limits.max_position_pct
```

2. Update all risk calculations to use config values
3. Add config validation on startup

### Step 3: Update Adaptive System (2 hours)

1. Modify `src/unity_wheel/strategy/adaptive_base.py`:
```python
# Before
self.base_position_pct = 0.20
params = {
    'put_delta': 0.30,
    'target_dte': 35,
    'roll_profit_target': 0.50,
}

# After
self.base_position_pct = config.risk.limits.max_position_pct
regime_config = config.adaptive.regime_params[regime.value.lower()]
params = {
    'put_delta': regime_config.put_delta,
    'target_dte': regime_config.target_dte,
    'roll_profit_target': regime_config.roll_profit_target,
}
```

### Step 4: Update Network Configuration (1 hour)

1. Update all HTTP clients to use config timeouts
2. Centralize retry logic using config values
3. Update cache TTLs from config

### Step 5: Testing and Validation (2 hours)

1. Run all existing tests to ensure compatibility
2. Add tests for config validation
3. Test with different config values
4. Verify no hardcoded values remain

## Migration Strategy

### Phase 1: Add to Config (No Breaking Changes)
- Add all new config sections
- Keep hardcoded values as fallbacks
- Log warnings when using defaults

### Phase 2: Update Code (Gradual)
- Update one module at a time
- Run tests after each module
- Keep backward compatibility

### Phase 3: Remove Hardcoded Values
- Remove all hardcoded defaults
- Require config values
- Update documentation

## Validation Checklist

- [ ] All risk limits in config.yaml
- [ ] Adaptive parameters per regime
- [ ] Network timeouts centralized
- [ ] Cache TTLs configurable
- [ ] Performance SLAs defined
- [ ] Data quality thresholds set
- [ ] No credentials hardcoded
- [ ] All tests passing
- [ ] Config validation working
- [ ] Documentation updated

## Benefits

1. **Flexibility**: Adjust parameters without code changes
2. **Testing**: Easy to test different configurations
3. **Monitoring**: Track which parameters affect performance
4. **Safety**: Clear limits and boundaries
5. **Compliance**: All risk parameters auditable
6. **Performance**: Tune for different market conditions

## Next Steps

1. Review and approve this plan
2. Create feature branch for changes
3. Implement Priority 1 items first
4. Test thoroughly with production data
5. Deploy with current values first
6. Tune parameters based on performance
