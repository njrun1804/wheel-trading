# Migration Guide: v1.x to v2.0

## Overview

Unity Wheel Trading Bot v2.0 introduces significant architectural improvements focused on autonomous operation, self-monitoring, and resilience. This guide helps you migrate from v1.x to v2.0.

## Major Changes

### 1. **New Package Structure**
- **Old**: `src/` with flat module structure
- **New**: `src/unity_wheel/` with organized subpackages

```python
# Old imports
from src.wheel import find_optimal_put_strike
from src.models import Position

# New imports
from unity_wheel.strategy import WheelStrategy
from unity_wheel.models import Position
```

### 2. **Enhanced API with Self-Validation**

The new API includes comprehensive validation and confidence scoring:

```python
# Old approach
strike = find_optimal_put_strike(current_price, strikes, volatility)

# New approach
advisor = WheelAdvisor(wheel_params, risk_limits)
recommendation = advisor.advise_position(market_snapshot)
# Returns structured recommendation with confidence and risk metrics
```

### 3. **Structured Logging**

All modules now use structured machine-parseable logging:

```python
# Old
logging.info(f"Selected strike {strike}")

# New
structured_logger.log(
    level="INFO",
    message="Strike selected",
    context={
        "strike": strike,
        "confidence": confidence,
        "delta": delta,
    }
)
```

### 4. **Performance Monitoring**

Built-in performance tracking with SLA monitoring:

```python
from unity_wheel.monitoring import get_performance_monitor

# Automatic performance tracking via decorators
@timed_operation(threshold_ms=200.0)
def your_function():
    pass
```

### 5. **Feature Flags for Graceful Degradation**

Control feature availability dynamically:

```python
from unity_wheel.utils import get_feature_flags

flags = get_feature_flags()
if flags.is_enabled("advanced_greeks"):
    # Use advanced features
    pass
```

### 6. **Data Quality Validation**

Comprehensive market data validation:

```python
from unity_wheel.data import get_market_validator

validator = get_market_validator()
result = validator.validate(market_data)
if not result.is_valid:
    # Handle poor quality data
    pass
```

## Migration Steps

### Step 1: Update Imports

Replace old imports with new package structure:

```python
# Replace these old imports
from src.wheel import WheelStrategy
from src.config.base import Config
from src.utils.math import black_scholes_price

# With these new imports
from unity_wheel.strategy import WheelStrategy
from src.config import get_config
from unity_wheel.math import black_scholes_price_validated
```

### Step 2: Update Configuration

The configuration system now supports auto-tuning:

```python
# Old
config = Config.from_file("config.yaml")

# New
from src.config import get_config, get_config_loader
config = get_config()
loader = get_config_loader()

# Enable configuration tracking
loader.track_parameter_usage("strategy.delta_target")
```

### Step 3: Use New API

Replace direct function calls with the advisor API:

```python
# Old approach
strike = find_optimal_put_strike(...)
contracts = calculate_position_size(...)

# New approach
advisor = WheelAdvisor(
    wheel_params=WheelParameters(
        target_delta=0.30,
        target_dte=45,
        max_position_size=0.20,
    ),
    risk_limits=RiskLimits()
)

market_snapshot = MarketSnapshot(
    ticker="U",
    current_price=35.50,
    buying_power=100000,
    option_chain=option_data,
    # ... other fields
)

recommendation = advisor.advise_position(market_snapshot)
```

### Step 4: Add Error Recovery

Wrap external calls with recovery decorators:

```python
from unity_wheel.utils import with_recovery, RecoveryStrategy

@with_recovery(strategy=RecoveryStrategy.RETRY, max_attempts=3)
def fetch_market_data():
    # Your data fetching logic
    pass
```

### Step 5: Enable Monitoring

Add performance monitoring to critical paths:

```python
# Command line
python run_aligned.py --performance  # View performance metrics
python run_aligned.py --export-metrics  # Export for dashboards

# In code
from unity_wheel.monitoring import get_performance_monitor
monitor = get_performance_monitor()
stats = monitor.get_all_stats()
```

## Breaking Changes

1. **Function Signatures**: Many functions now return structured results with confidence scores
2. **Error Handling**: Functions may raise `ValidationError` for invalid inputs
3. **Async Support**: Some functions are now async (especially in broker module)
4. **Required Fields**: Market snapshots require additional fields (timestamp, implied_volatility)

## New Features to Leverage

1. **Self-Diagnostics**: Run health checks with `--diagnose`
2. **Observability Export**: Export metrics in multiple formats
3. **Auto-Tuning**: Enable configuration auto-tuning based on outcomes
4. **Circuit Breakers**: Automatic degradation for unreliable services
5. **Caching**: Automatic caching of expensive calculations

## Example: Complete Migration

### Old Code (v1.x)
```python
from src.wheel import find_optimal_put_strike
from src.config.base import Config

config = Config.from_file("config.yaml")
current_price = 35.50
strikes = [30, 32.5, 35, 37.5, 40]
volatility = 0.65

strike = find_optimal_put_strike(
    current_price, 
    strikes, 
    volatility,
    config.delta_target
)
print(f"Recommended strike: {strike}")
```

### New Code (v2.0)
```python
from unity_wheel import WheelAdvisor, MarketSnapshot
from unity_wheel.strategy import WheelParameters

# Initialize with parameters
advisor = WheelAdvisor(
    wheel_params=WheelParameters(target_delta=0.30)
)

# Create market snapshot
snapshot = MarketSnapshot(
    timestamp=datetime.now(timezone.utc),
    ticker="U",
    current_price=35.50,
    buying_power=100000,
    option_chain={
        "30.0": {"bid": 0.80, "ask": 0.90, ...},
        "32.5": {"bid": 1.20, "ask": 1.30, ...},
        # ... more strikes
    },
    implied_volatility=0.65,
)

# Get recommendation with confidence
rec = advisor.advise_position(snapshot)
if rec["action"] == "ADJUST":
    print(f"Recommended: {rec['rationale']}")
    print(f"Confidence: {rec['confidence']:.1%}")
    print(f"Risk metrics: {rec['risk']}")
```

## Support

For questions or issues during migration:
1. Check the diagnostic output: `python run_aligned.py --diagnose`
2. Review the logs for detailed error information
3. Consult the API documentation in the docstrings
4. File an issue with version information: `python run_aligned.py --version`