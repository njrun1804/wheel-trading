# Architecture Overview

Unity Wheel Trading Bot v2.2 - Single-user options wheel strategy recommendation system.

## High-Level Flow
```
Market Data → Validation → Risk Analysis → Strategy → Recommendation
     ↓             ↓            ↓             ↓            ↓
  Schwab      DataQuality   RiskLimits   WheelParams   Decision
 Databento     Anomalies     Analytics    Adaptive      Logging
```

## Entry Points
- **`run.py`** - Simple CLI wrapper for recommendations
- **`src/unity_wheel/cli/run.py:264`** - Main application logic
- **`examples/single_account_simple.py`** - Basic usage example

## Core Modules

### 🎯 **`src/unity_wheel/api/`** - External API Interface
- `advisor.py:106` - Main recommendation engine (`WheelAdvisor.advise_position()`)
- `advisor_simple.py` - Simplified interface for basic usage

### 📊 **`src/unity_wheel/strategy/`** - Trading Logic
- `wheel.py:626` - Core wheel strategy implementation
 - `wheel.py:153` - Vectorized put strike selection (10x performance boost)
 - `wheel.py:291` - Vectorized call strike selection

### ⚠️ **`src/unity_wheel/risk/`** - Risk Management
- `analytics.py:798` - Portfolio risk calculations with confidence scores
- `limits.py:89` - Circuit breakers and position limits
- `unity_margin.py` - Unity-specific margin requirements

### 🧮 **`src/unity_wheel/math/`** - Financial Mathematics
- `options.py:746` - Black-Scholes pricing, Greeks (all with validation)
- `options.py:123` - `black_scholes_price_validated()` - Core pricing function

### 📈 **`src/unity_wheel/data_providers/`** - Market Data
- `databento/client.py` - Real-time options data via Databento API
- `schwab/client.py` - Account data via Schwab API
- `fred/fred_client.py` - Risk-free rates from Federal Reserve

### ⚙️ **`src/config/`** - Configuration System
- `schema.py:924` - Complete configuration validation schemas
- `loader.py` - Smart config loading with environment overrides
- `config.yaml` - Main configuration file

### 🔧 **`src/unity_wheel/utils/`** - Utilities
- `position_sizing.py:71` - Unified position sizing logic
- `trading_calendar.py` - Market hours and option expiry detection
- `validate.py` - Environment and data validation

## Key Design Principles

1. **Confidence-Based**: Every calculation returns `(value, confidence)` tuples
2. **Never Crash**: All operations use `@with_recovery` decorators
3. **Performance SLAs**: Black-Scholes <0.2ms, full recommendation <200ms
4. **Type Safety**: 100% type hints with mypy strict mode
5. **Observable**: Structured logging with automatic performance tracking

## Data Flow

```python
# 1. Market Data Collection
schwab_data = SchwabClient().get_positions()
option_data = DatabentoClient().get_option_chain("U")

# 2. Risk Analysis
risk_metrics = RiskAnalytics().calculate_portfolio_risk(portfolio)

# 3. Strategy Execution
optimal_strike = WheelStrategy().find_optimal_put_strike_vectorized(strikes)

# 4. Recommendation Generation
recommendation = WheelAdvisor().advise_position(account, positions, chains)
```

## Safety Systems

- **Circuit Breakers**: Stop trading at 150% volatility or -20% drawdown
- **Position Limits**: Max 20% portfolio per position, 3 concurrent Unity puts
- **Rate Limiting**: 60 requests/minute with exponential backoff
- **Confidence Thresholds**: Minimum 30% confidence required for trades

## Performance Optimizations (v2.2)

 - **Vectorized Calculations**: Process put and call strikes simultaneously (10x speedup)
- **Lazy Loading**: Data loaded only when needed
- **Caching**: 5-minute TTL on expensive calculations
- **Memory Efficient**: <100MB typical usage

## Testing Strategy

- **Unit Tests**: Property-based testing with Hypothesis
- **Integration Tests**: `test_autonomous_flow.py` - Full recommendation pipeline
- **Benchmarks**: `test_performance_benchmarks.py` - SLA validation
- **CI/CD**: Tests in both Ubuntu (development) and macOS (runtime) environments
