# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 QUICK START (Most Common Tasks)

```bash
# Get trading recommendation
python run.py -p 100000

# If errors, run diagnostics
python run.py --diagnose

# Quick commit
./scripts/commit-workflow.sh -y

# Run tests for new modules
pytest tests/test_databento_unity.py tests/test_position_sizing.py -v
```

**Key Files:** `run.py` (entry), `src/unity_wheel/cli/run.py:112` (main), `api/advisor.py:106` (logic), `config.yaml` (settings)

**Common Errors:**
- "Invalid credentials" → `python scripts/setup-secrets.py`
- "Rate limit" → Wait 60s
- "No liquid strikes" → `export DATABENTO_SKIP_VALIDATION=true`

**Quick Navigation:** [Commands](#quick-reference) • [Files](#key-file-locations) • [Errors](#troubleshooting) • [Config](#configuration-system) • [Architecture](#architecture) • [Workflows](#common-workflows)

---

## Project Overview

Unity Wheel Trading Bot v2.2 - An autonomous options wheel strategy recommendation system with self-monitoring, auto-tuning, and enterprise-grade reliability. Designed for single-user local operation with NO BROKER INTEGRATION (recommendations only).

### Recent Optimizations (Jan 2025)
- **5x Performance**: Vectorized option calculations process all strikes at once
- **Configurable Ticker**: No more hardcoded "U" - uses `config.unity.ticker`
- **Enhanced Safety**: All calculations now return confidence scores
- **Unified Position Sizing**: Single implementation via `DynamicPositionSizer`
- **Better Error Handling**: No bare except clauses, specific exceptions only
- **Backtest Validated**: 27-30% annual returns with optimized params (delta=0.40, DTE=30)

### ⚠️ Known File System Issues (Clean these up!)
```bash
# 1. Remove duplicate files with " 2" suffix (iCloud sync artifacts)
find . -name "* 2.*" -type f -delete

# 2. Remove redundant engine directories
rm -rf ml_engine/ risk_engine/ strategy_engine/

# 3. Remove nested duplicate structure
rm -rf Documents/com~apple~CloudDocs/

# 4. Clean up __pycache__ and temp files
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

## Quick Reference

### One-Liner Cheat Sheet
```bash
# Get recommendation
python run.py -p 100000

# Quick health check
./scripts/housekeeping.sh -q

# Test single function
python -c "from src.unity_wheel.math.options import black_scholes_price_validated as bs; print(bs(100,100,1,0.05,0.2,'call'))"

# View config value
python -c "from src.config.loader import get_config as gc; print(gc().strategy.greeks.delta_target)"

# Check performance
python -c "from src.unity_wheel.metrics import metrics_collector as mc; print(mc.get_performance_stats())"

# Validate environment
python -m src.unity_wheel.utils.validate

# Quick commit
git add -A && git commit -m "msg" && git push

# Export metrics
python run.py --export-metrics > metrics.json
```

### Most Used Commands
```bash
# Primary operations
python run.py --portfolio 100000      # Get recommendation
python run.py --diagnose             # System health
pre-commit run --all-files                   # Run all checks
./scripts/commit-workflow.sh -y              # Auto-commit

# Testing
pytest tests/test_autonomous_flow.py -v      # Integration test
pytest tests/test_math.py::test_black_scholes_edge_cases -v  # Specific test

# Monitoring
python src/unity_wheel/monitoring/scripts/live_monitor.py      # Real-time dashboard
python src/unity_wheel/monitoring/scripts/daily_health_check.py  # Morning checks
```

### Key File Locations
```python
# Entry Points
run.py                   # Simple wrapper script
src/unity_wheel/cli/run.py:264       # main() function
src/unity_wheel/cli/run.py:112       # generate_recommendation()

# Core Components
src/unity_wheel/api/advisor.py:74           # WheelAdvisor class
src/unity_wheel/api/advisor.py:106          # advise_position() - Main logic
src/unity_wheel/strategy/wheel.py:153       # find_optimal_put_strike_vectorized() - NEW!
src/unity_wheel/strategy/wheel.py:626       # WheelStrategy implementation
src/unity_wheel/risk/analytics.py:798       # Risk calculations (now with confidence)
src/unity_wheel/math/options.py:746         # Black-Scholes, Greeks
src/unity_wheel/utils/position_sizing.py:71 # DynamicPositionSizer (unified)
src/unity_wheel/utils/trading_calendar.py   # Trading day detection & option expiries
src/unity_wheel/utils/trading_calendar_enhancements.py # Early closes & earnings

# Configuration
src/config/schema.py:924                    # All config schemas
src/config/loader.py                        # Config loading & tracking
config.yaml                                  # Main config file

# Data Integration
src/unity_wheel/schwab/client.py                       # Schwab API client
src/unity_wheel/data_providers/databento/client.py     # Databento client
src/unity_wheel/data_providers/fred/                   # FRED data
src/unity_wheel/data_providers/schwab/                 # Schwab data modules
```

### Critical Constants & Performance SLAs
```python
# Position Limits
MAX_POSITION_SIZE = 0.20          # 20% of portfolio per position
MAX_CONCURRENT_PUTS = 3           # Unity-specific limit
MIN_CONFIDENCE = 0.30             # 30% minimum confidence required
UNITY_TICKER = config.unity.ticker # Configurable, defaults to "U"

# Risk Thresholds
MAX_VOLATILITY = 1.50             # 150% - stop trading above this
MAX_DRAWDOWN = -0.20              # -20% - circuit breaker
CONSECUTIVE_LOSS_LIMIT = 3        # Stop after 3 losses
EARNINGS_BUFFER_DAYS = 7          # Skip trades 7 days before earnings

# Trading Parameters (Optimized from Backtests)
TARGET_DELTA = 0.40               # Optimal for Unity's high vol (was 0.30)
TARGET_DTE = 30                   # Optimal for Unity (was 45)
CONTRACTS_PER_TRADE = 100         # Unity shares per contract

# Unity Volatility Environment (June 2025)
CURRENT_VOLATILITY = 0.87         # 87% - extreme high
VOL_6M_RANGE = (0.44, 1.31)      # 44-131% over 6 months
EXPECTED_ANNUAL_RETURN = 0.27     # 27-30% based on backtests
GAP_EVENTS_PER_YEAR = 38         # >10% moves

# Performance SLAs (milliseconds)
BLACK_SCHOLES_SLA = 0.2           # Options pricing
GREEKS_SLA = 0.3                  # All Greeks calculation
VAR_SLA = 10.0                    # VaR calculation (1000 points)
STRIKE_SELECTION_SLA = 100.0      # Find optimal strike
DECISION_SLA = 200.0              # Full recommendation
API_CALL_SLA = 1000.0             # External API calls

# Circuit Breaker Settings
FAILURE_THRESHOLD = 5             # Consecutive failures before open
RESET_TIMEOUT = 60                # Seconds before circuit reset
RATE_LIMIT = 60                   # Requests per minute
```

### Trading Calendar Usage
```python
# Check if market is open
from src.unity_wheel.utils import is_trading_day
if not is_trading_day(datetime.now()):
    print("Market is closed")

# Find next option expiration
from src.unity_wheel.utils import SimpleTradingCalendar
calendar = SimpleTradingCalendar()
next_expiry = calendar.get_next_expiry_friday(datetime.now())
trading_days = calendar.days_to_next_expiry(datetime.now())

# Check for early close or earnings
from src.unity_wheel.utils.trading_calendar_enhancements import EnhancedTradingCalendar
enhanced = EnhancedTradingCalendar()
is_early_close = enhanced.is_early_close(datetime.now())
near_earnings = enhanced.is_near_unity_earnings(datetime.now(), days_buffer=7)
```

### Common Code Patterns
```python
# Validated calculation pattern
from src.unity_wheel.math.options import black_scholes_price_validated
result = black_scholes_price_validated(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
if result.confidence > 0.95:
    price = result.value

# Decorator usage
@timed_operation(threshold_ms=10.0)
@with_recovery(strategy=RecoveryStrategy.FALLBACK)
@cached(ttl_seconds=300)
def expensive_calculation():
    pass

# Config access
from src.config.loader import get_config
config = get_config()
delta = config.strategy.greeks.delta_target

# Structured logging
logger.info("Operation completed", extra={
    "function": "calculate_risk",
    "execution_time_ms": 5.2,
    "confidence": 0.98
})
```

### Quick Debugging Actions
```python
# When recommendation fails
python run.py --diagnose  # Check system health
python -c "from src.unity_wheel.data_providers.databento import validate_connection; validate_connection()"  # Check data
python -c "from src.unity_wheel.schwab import test_auth; test_auth()"  # Check auth

# When performance is slow
python -c "from src.unity_wheel.utils.cache import get_cache; print(get_cache().stats())"  # Cache stats
python -c "from src.unity_wheel.metrics import metrics_collector as mc; print(mc.get_slow_operations(threshold_ms=50))"  # Slow ops

# When getting unexpected results
python -c "from src.config.loader import get_config_loader as gcl; print(gcl().get_unused_parameters())"  # Unused config
python -c "from src.unity_wheel.monitoring.diagnostics import run_diagnostics; run_diagnostics(verbose=True)"  # Full diagnostics
```

### Common Gotchas
```python
# ❌ WRONG: Using relative imports
from math import black_scholes  # Will import system math module!

# ✅ RIGHT: Use absolute imports
from src.unity_wheel.math.options import black_scholes_price_validated

# ❌ WRONG: Not checking confidence
price = calculate_option_price(...).value  # May be NaN!

# ✅ RIGHT: Always check confidence
result = calculate_option_price(...)
if result.confidence > 0.95:
    price = result.value

# ❌ WRONG: Catching all exceptions
try:
    risky_operation()
except:  # Never do this!
    pass

# ✅ RIGHT: Specific exception handling
try:
    risky_operation()
except (ValueError, KeyError) as e:
    logger.error(f"Operation failed: {e}")
    return fallback_value()
```

## Troubleshooting

### Common Error Messages & Solutions

#### Authentication Errors
```bash
# "Invalid credentials" or "Failed to retrieve credentials"
python scripts/setup-secrets.py  # Re-run setup
export SCHWAB_CLIENT_ID=xxx
export SCHWAB_CLIENT_SECRET=xxx

# "Access token expired"
# Auto-refreshes, but if persistent:
python tools/verification/schwab_oauth_fixed.py

# "Rate limit exceeded"
# Wait 60s for circuit breaker reset
# Check: python -c "from src.unity_wheel.auth.rate_limiter import get_rate_limiter; print(get_rate_limiter().get_status())"
```

#### Data Quality Issues
```bash
# "Insufficient data for reliable VaR calculation"
# Need 20+ data points, check:
python -c "from src.unity_wheel.data_providers.databento import get_historical_data; print(len(get_historical_data('U', days=30)))"

# "Data quality issues: X errors found"
python run.py --diagnose  # See specific issues

# "No liquid strikes available"
# Check liquidity thresholds:
# MIN_VOLUME = 100, MIN_OPEN_INTEREST = 100
```

#### Performance Violations
```bash
# "SLA violation: operation took Xms (threshold: Yms)"
# Enable profiling to identify bottleneck:
python -c "from src.unity_wheel.utils import enable_profiling; enable_profiling()"
python run.py --portfolio 100000  # Run with profiling

# View performance stats:
python -c "from src.unity_wheel.metrics import metrics_collector; print(metrics_collector.get_sla_report())"
```

### Common Issues
```bash
# Databento connection issues
export DATABENTO_SKIP_VALIDATION=true       # Temporary skip
python tools/debug/debug_databento.py       # Debug connection

# Schwab auth problems
python tools/verification/schwab_oauth_fixed.py  # Fix OAuth
python tools/verification/schwab_status_check.py # Check status

# Secret/credential issues
python scripts/test-secrets.py              # Validate all secrets
python tools/verification/verify_secret.py  # Check specific secret

# Config validation errors
python -c "from src.config.loader import validate_config; validate_config()"
```

### Performance Debugging
```python
# Profile slow operations
from src.unity_wheel.utils import enable_profiling
enable_profiling()  # Logs all operations >10ms

# Check cache effectiveness
from src.unity_wheel.metrics import metrics_collector
print(metrics_collector.get_cache_stats())

# View SLA violations
print(metrics_collector.get_sla_report())

# Check circuit breaker status
from src.unity_wheel.auth.rate_limiter import get_rate_limiter
print(get_rate_limiter().get_circuit_status())
```

### Log Analysis & Metrics
```bash
# View recent errors
grep -E "ERROR|CRITICAL" logs/wheel.log | tail -20

# Count warnings by type
grep "WARNING" logs/wheel.log | grep -oE '"message":"[^"]+"' | sort | uniq -c | sort -nr

# Performance violations
grep "SLA violation" logs/wheel.log | tail -10

# View structured logs (JSON)
jq '.level == "ERROR"' logs/wheel.json | jq -s '.[0:5]'

# Export metrics database
sqlite3 exports/metrics.db "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 10;"

# View recommendation history
sqlite3 exports/metrics.db "SELECT decision_id, action, confidence, expected_return FROM decisions WHERE confidence > 0.5;"
```

### Database Queries
```sql
-- Recent high-confidence decisions
SELECT decision_id, action, confidence, expected_return, execution_time_ms
FROM decisions
WHERE confidence > 0.7
ORDER BY timestamp DESC
LIMIT 10;

-- Performance trends
SELECT
  DATE(timestamp) as date,
  AVG(execution_time_ms) as avg_time,
  MAX(execution_time_ms) as max_time,
  COUNT(*) as count
FROM decisions
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Feature usage analysis
SELECT features_used, COUNT(*) as usage_count
FROM decisions
GROUP BY features_used
ORDER BY usage_count DESC;
```

## Architecture

### Key Classes & Methods
```python
# Main Recommendation Flow
WheelAdvisor.advise_position()              # advisor.py:106
├── validate_market_data()                   # advisor.py:138
├── WheelStrategy.find_optimal_put_strike()  # wheel.py:187
├── calculate_position_size()                # wheel.py:325
├── RiskAnalyzer.calculate_metrics()         # analytics.py:156
└── create_recommendation()                  # advisor.py:273

# Risk Calculations
RiskAnalytics.calculate_portfolio_risk()     # analytics.py:432
├── calculate_var()                          # analytics.py:234
├── calculate_cvar()                         # analytics.py:289
├── calculate_kelly_fraction()               # analytics.py:367
└── check_risk_limits()                      # limits.py:89

# Options Math (all with confidence scores)
black_scholes_price_validated()              # options.py:123
calculate_greeks_validated()                 # options.py:287
implied_volatility_validated()               # options.py:412
probability_itm_validated()                  # options.py:189
```

### Directory Structure (Clean Version)
```
src/
├── config/          # Config validation (schema.py:924 lines)
├── unity_wheel/
│   ├── api/         # External API (advisor.py)
│   ├── math/        # Options calculations
│   ├── risk/        # Risk management
│   ├── strategy/    # Trading strategies
│   ├── adaptive/    # Adaptive system logic
│   ├── data_providers/
│   │   ├── databento/   # Options data
│   │   ├── schwab/      # Account data
│   │   └── fred/        # Economic data
│   ├── monitoring/  # Health checks & diagnostics
│   └── analytics/   # Performance tracking
└── tests/           # Property-based tests
```

### Architectural Improvements Needed
1. **Centralize Configuration**: Create ConfigurationService singleton
2. **Define Async Boundaries**: Clear async/sync module separation
3. **Consolidate Adaptive Logic**: All adaptive code → adaptive/ module
4. **Replace Lazy Imports**: Use proper dependency injection
5. **Standardize Data Providers**: Align provider structures

### Design Principles & Guidelines

**🚨 CRITICAL: NO SYNTHETIC DATA POLICY**
- **NEVER** use synthetic, mock, dummy, or generated market data
- **ALWAYS** use real data from authorized providers (Databento, Schwab, FRED)
- **PROHIBITED**: Black-Scholes generated option prices, synthetic Greeks, mock volume/OI
- **REQUIRED**: All market data MUST come from live APIs with proper credentials
- **VERIFICATION**: Any data collection MUST be validated against known real market data

1. **Every calculation returns confidence** - `(value, confidence)` tuples
2. **Never crash** - Use `@with_recovery`, return NaN with explanation
3. **Log everything** - Function name, inputs, outputs, timing
4. **Type everything** - 100% type hints, mypy strict mode
5. **Track performance** - Alert on calculations >10ms
6. **Validate inputs** - Use Pydantic models everywhere
7. **Test edge cases** - Property-based testing with Hypothesis
8. **Self-monitor** - Automatic SLA tracking and alerting
9. **Real data only** - NO synthetic/mock/dummy data allowed under any circumstance

## Quick Test Commands

```bash
# Fast tests only (30 seconds)
pytest -v -m "not slow"

# Test specific module
pytest tests/test_math.py -v

# Test with coverage
pytest --cov=src --cov-report=html

# Quick validation
python -c "from unity_wheel.math import black_scholes_price_validated as bs; print(bs(100,100,1,0.05,0.2,'call'))"

# Performance benchmark
pytest tests/test_performance_benchmarks.py::test_black_scholes_performance -v
```

## Configuration System

### Quick Config Access
```python
# Get config value
from src.config.loader import get_config
config = get_config()
delta = config.strategy.greeks.delta_target      # 0.30
max_pos = config.risk.position_limits.max_position_size  # 0.20

# Check config health
from src.config.loader import get_config_loader
print(get_config_loader().generate_health_report())

# Override via environment
export WHEEL_STRATEGY__GREEKS__DELTA_TARGET=0.25
```

### Most Important Settings
```yaml
strategy:
  greeks:
    delta_target: 0.30          # Target delta for puts
  expiration:
    days_to_expiry_target: 45   # Target DTE

risk:
  position_limits:
    max_position_size: 0.20     # 20% of portfolio max
  circuit_breakers:
    max_volatility: 1.50        # Stop at 150% vol
    max_drawdown: -0.20         # Stop at -20% drawdown

operations:
  api:
    max_concurrent_puts: 3      # Unity-specific
    min_confidence: 0.30        # 30% minimum
```

### Unity Adaptive System (Key Feature)

The project includes a Unity-specific adaptive system at `src/unity_wheel/strategy/adaptive_base.py:221` that adjusts position sizing based on market conditions.

#### Quick Usage:
```python
from src.unity_wheel.strategy.adaptive_wheel import create_adaptive_wheel_strategy

# Simple usage
strategy = create_adaptive_wheel_strategy(portfolio_value=200000)
rec = strategy.get_recommendation(
    unity_price=35.00,
    available_strikes=[30, 32.5, 35, 37.5, 40],
    available_expirations=[7, 14, 21, 28, 35, 42],
    portfolio_drawdown=-0.05
)

if rec['should_trade']:
    print(f"Trade ${rec['position_size']:,.0f} at ${rec['recommended_strike']}")
```

#### Adaptive Rules Summary:
- **<40% vol**: 120% position (opportunity)
- **40-60% vol**: 100% position (normal)
- **60-80% vol**: 70% position (caution)
- **>80% vol**: 50% position (defensive)
- **>100% vol**: STOP TRADING
- **Earnings <7 days**: SKIP TRADE
- **Drawdown >20%**: STOP TRADING

#### Quick Volatility Check:
```python
# Check current Unity volatility tier
from src.unity_wheel.adaptive import get_volatility_tier
tier = get_volatility_tier(current_vol=0.65)  # Returns: 'caution'

# Check if should trade
from src.unity_wheel.adaptive import should_trade_unity
can_trade = should_trade_unity(vol=0.85, drawdown=-0.15, days_to_earnings=10)  # Returns: (False, "High volatility")
```

#### Usage Examples:

```python
# Create adaptive wheel strategy
from src.unity_wheel.strategy.adaptive_wheel import create_adaptive_wheel_strategy

# Initialize with portfolio value
strategy = create_adaptive_wheel_strategy(portfolio_value=200000)

# Get recommendation
recommendation = strategy.get_recommendation(
    unity_price=35.00,
    available_strikes=[30, 32.5, 35, 37.5, 40],
    available_expirations=[7, 14, 21, 28, 35, 42],
    portfolio_drawdown=-0.05  # Currently down 5%
)

# Check result
if recommendation['should_trade']:
    print(f"Trade: ${recommendation['position_size']:,.0f} position")
    print(f"Strike: ${recommendation['recommended_strike']}")
    print(f"DTE: {recommendation['target_dte']} days")
else:
    print(f"Skip: {recommendation['skip_reason']}")

# Record outcome later
strategy.record_outcome(
    recommendation['recommendation_id'],
    actual_pnl=1500,  # Made $1,500
    was_assigned=False
)
```

#### Adaptive Rules:

1. **Volatility-Based Position Sizing**:
   - <40% vol: 120% of base (opportunity)
   - 40-60% vol: 100% of base (normal)
   - 60-80% vol: 70% of base (caution)
   - >80% vol: 50% of base (defensive)

2. **Drawdown Management**:
   - Linear reduction from 0% to -20%
   - Complete stop at -20% drawdown
   - Preserves capital during losses

3. **Earnings Awareness**:
   - Skip trades <7 days to earnings
   - Adjust DTE to expire before earnings
   - Avoids Unity's ±15-25% earnings moves

4. **Parameter Adaptation**:
   - High vol: Lower delta (20-25 vs 30)
   - High vol: Shorter DTE (28 vs 35)
   - High vol: Quick profits (25% vs 50%)

5. **Stop Conditions**:
   - Volatility >100%: Stop trading
   - Drawdown >20%: Stop trading
   - Earnings <7 days: Skip trade

#### Configuration:

```yaml
# No complex configuration needed!
# Adaptive rules are built into the system
# Just set your risk tolerance:

risk:
  max_position_size: 1.00  # For aggressive traders
  circuit_breakers:
    max_position_pct: 0.20   # Base position size
    min_portfolio_value: 10000
```

## Performance Targets

- Black-Scholes: <0.2ms per calculation
- Greeks: <0.3ms for all Greeks
- Risk metrics: <10ms for 1000 data points
- IV solver: <5ms with fallback to bisection
- Memory: <100MB for typical portfolio

## Objective Function

Maximize: **CAGR - 0.20 × |CVaR₉₅|** with **½-Kelly** position sizing

## Backtest Performance (Real Data)

### 1-Year Results (June 2024-2025)
- **Total Return**: 27.0%
- **Sharpe Ratio**: 3.72 (exceptional)
- **Win Rate**: 100% (8 trades, all profitable)
- **Max Drawdown**: 0% (risk management worked)
- **Avg Trade P&L**: $1,825

### Key Success Factors
1. **Avoided all 40 earnings periods** (Unity moves ±15-25%)
2. **Managed through 38 gap events** (>10% moves)
3. **Optimized parameters**: Delta 0.40, DTE 30
4. **Conservative sizing**: 20% max position

## Common Workflows

### 1. Add New Risk Check
```python
# Quick template for adding a new risk check
# 1. Add to risk/limits.py:
class RiskLimits:
    def check_new_limit(self, value: float) -> Tuple[bool, str]:
        threshold = self.config.risk.new_limit_threshold
        if value > threshold:
            return False, f"New limit exceeded: {value:.2f} > {threshold:.2f}"
        return True, ""

# 2. Add to schema.py:
class RiskConfig(BaseModel):
    new_limit_threshold: float = Field(0.5, ge=0, le=1)

# 3. Update analytics.py:
def check_risk_limits(self, ...):
    checks.append(self.limits.check_new_limit(calculated_value))

# 4. Test: pytest tests/test_risk.py::test_new_limit -v
```

### 2. Add New Options Calculation
```python
# Template for new validated calculation
# In src/unity_wheel/math/options.py:
@timed_operation(threshold_ms=1.0)
@validate_inputs
def new_calculation_validated(
    S: float, K: float, T: float, r: float, sigma: float
) -> CalculationResult:
    \"\"\"New calculation with validation.\"\"\"
    try:
        # Validate inputs
        if not all(x > 0 for x in [S, K, T, sigma]):
            return CalculationResult(value=float('nan'), confidence=0.0)

        # Calculate
        result = your_math_here(S, K, T, r, sigma)

        # Validate output
        if not (0 <= result <= S):
            return CalculationResult(value=result, confidence=0.5)

        return CalculationResult(value=result, confidence=0.99)

    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        return CalculationResult(value=float('nan'), confidence=0.0)
```

### 3. Debug Production Issue
```bash
# 1. Check logs for errors
grep -E "ERROR|CRITICAL" logs/wheel.log | tail -50 | grep -E "function|message"

# 2. Run diagnostics
python run.py --diagnose > diagnostics.txt

# 3. Check specific subsystem
python -c "
from src.unity_wheel.schwab import SchwabClient
from src.unity_wheel.auth import get_auth_client
auth = get_auth_client()
print('Auth valid:', auth.is_authenticated())
print('Token expires:', auth.token_expires_at)
"

# 4. Enable debug logging
export LOG_LEVEL=DEBUG
python run.py --portfolio 100000 2>&1 | tee debug.log

# 5. Check metrics database
sqlite3 exports/metrics.db "
SELECT * FROM decisions
WHERE confidence < 0.3 OR execution_time_ms > 500
ORDER BY timestamp DESC LIMIT 10;"
```

### 4. Performance Optimization
```bash
# 1. Profile current performance
pytest tests/test_performance_benchmarks.py -v

# 2. Make changes with @timed_operation decorator

# 3. Verify improvement
python -c "from src.unity_wheel.metrics import metrics_collector; print(metrics_collector.get_performance_stats())"
```

## Data Sources

### Databento (Options Data)
```python
# Quick setup (uses Google Cloud Secrets for API key)
from src.unity_wheel.data_providers.databento import DatabentoClient
client = DatabentoClient()  # Auto-retrieves key from Google Secrets

# Get Unity options
from src.unity_wheel.data_providers.databento.integration import get_wheel_candidates
candidates = await get_wheel_candidates("U", target_delta=0.30)

# Debug connection
python tools/debug/debug_databento.py

# Full documentation: docs/DATABENTO_UNITY_GUIDE.md
```

### Schwab (Account Data)
```python
# OAuth setup required - see tools/verification/schwab_oauth_fixed.py
from src.unity_wheel.schwab import SchwabClient
client = SchwabClient()  # Uses env vars
positions = await client.get_positions()
```

## Critical Implementation Notes

### Data Flow
```
Market Data → Validation → Risk Analysis → Strategy → Recommendation
     ↓             ↓            ↓             ↓            ↓
  Schwab      DataQuality   RiskLimits   WheelParams   Decision
 Databento     Anomalies     Analytics    Adaptive      Logging
```

### Key Decorators
- `@timed_operation(threshold_ms=10.0)` - Performance tracking
- `@with_recovery(strategy=RecoveryStrategy.FALLBACK)` - Error handling
- `@cached(ttl_seconds=300)` - Result caching
- `@validate_inputs` - Pydantic validation
- `@track_confidence` - Confidence scoring

### Testing Strategy
- Unit tests: Fast, isolated, property-based
- Integration tests: `test_autonomous_flow.py`
- Benchmarks: `test_performance_benchmarks.py`
- Always validate with known test cases
- NO BROKER INTEGRATION - recommendations only

## v2.2 Enhancements (January 2025)

### Performance Optimizations
- **Vectorized Strike Selection**: Process all strikes at once with numpy (10x faster)
- **Lazy Import Removal**: Moved imports out of hot paths
- **Confidence-Based Filtering**: Skip low-confidence calculations early

### Code Quality Improvements
- **No Bare Excepts**: All exceptions are specific with proper logging
- **Configurable Ticker**: Unity ticker from config, not hardcoded
- **Unified Position Sizing**: Single source of truth for position calculations
- **Enhanced Tests**: New tests for databento_unity and position_sizing modules

### New Methods
- `WheelStrategy.find_optimal_put_strike_vectorized()` - 10x faster strike selection
- `WheelStrategy.find_optimal_call_strike_vectorized()` - vectorized call selection
- `RiskAnalytics.aggregate_portfolio_greeks()` - Now returns confidence score
- `PositionSizeResult.confidence` - All position sizing includes confidence

## v2.0 Enhancements

### New Utilities
- **Structured Logging**: Use `StructuredLogger` for JSON logs
- **Performance Monitoring**: Apply `@timed_operation` decorator
- **Caching**: Use `@cached` for expensive calculations
- **Recovery**: Apply `@with_recovery` for external calls
- **Feature Flags**: Check with `get_feature_flags().is_enabled()`

### Autonomous Features
1. **Data Validation**: All market data validated automatically
2. **Anomaly Detection**: Unusual market conditions flagged
3. **Circuit Breakers**: External calls protected from failures
4. **Auto-Tuning**: Configuration optimizes based on outcomes
5. **Observability**: Metrics exported in multiple formats

### Monitoring & Alerts
- SLA violations tracked automatically
- Feature degradation monitored
- Performance trends analyzed
- System health scored continuously

### Git Hooks
Pre-commit hooks run automatically:
- Code formatting
- Type checking
- Security scanning
- System diagnostics
- Configuration validation

### Shell Scripts
- `scripts/autonomous-checks.sh` - Full system validation
- `scripts/monitor.sh` - Continuous monitoring daemon
- `scripts/dev.sh` - Development environment setup
- `scripts/maintenance.sh` - Periodic maintenance tasks

## Safety Features

### Performance Tracking
- **Location**: `src/unity_wheel/analytics/performance_tracker.py`
- **Purpose**: Learn from actual results vs predictions
- **Usage**: Automatically records recommendations and outcomes
- **Benefits**: Identifies when confidence is miscalibrated

### Risk Limits
- **Location**: `src/unity_wheel/risk/limits.py`
- **Purpose**: Circuit breakers for autonomous safety
- **Key Limits**:
  - Max 20% portfolio per position
  - Stop after 3 consecutive losses
  - No trading above 150% volatility
  - Minimum 30% confidence required

### Monitoring & Health Checks

#### Daily Health Check
```bash
python src/unity_wheel/monitoring/scripts/daily_health_check.py  # Run every morning
# Checks:
# - Data freshness (<5min stale)
# - Config validation
# - Credentials active
# - Performance within SLAs
# - Risk limits not breached
```

#### Live Monitor
```bash
python src/unity_wheel/monitoring/scripts/live_monitor.py  # Real-time dashboard
# Updates every 10 seconds:
# - Current positions & P&L
# - Risk metrics (VaR, Greeks)
# - Market data status
# - System health score
# - Alert notifications
```

#### Quick Validation
```bash
./scripts/housekeeping.sh --quick  # <30 second check
python -m src.unity_wheel.utils.validate     # Full environment validation
```

## Environment Variables

### Required
```bash
SCHWAB_CLIENT_ID=xxx
SCHWAB_CLIENT_SECRET=xxx
DATABENTO_API_KEY=xxx        # Optional if DATABENTO_SKIP_VALIDATION=true
```

### Optional Overrides
```bash
WHEEL_STRATEGY__DELTA_TARGET=0.25
WHEEL_ML__ENABLED=true
WHEEL_TRADING__MODE=paper
WHEEL_RISK__MAX_POSITION_SIZE=0.30
DATABENTO_SKIP_VALIDATION=true
```

## Memory & Resource Optimization

### Memory Usage Tips
```python
# Check current memory usage
import psutil
import os
process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Clear caches if needed
from src.unity_wheel.utils.cache import get_cache
get_cache().clear()  # Clears all cached data

# Garbage collection for large operations
import gc
gc.collect()  # Force garbage collection
```

### Resource Limits
- **Target Memory**: <100MB for typical portfolio
- **Cache Size**: 1000 entries max
- **Log Rotation**: 100MB per file, 5 files max
- **Database Size**: 50MB for metrics.db

## Integration Testing Shortcuts

```bash
# Test full recommendation flow
pytest tests/test_autonomous_flow.py::test_full_recommendation_flow -v -s

# Test with real market data (requires credentials)
pytest tests/test_e2e_recommendation_flow.py -v -m "integration"

# Benchmark performance
pytest tests/test_performance_benchmarks.py -v --benchmark-only

# Test specific strategy
pytest tests/test_adaptive_system.py::test_volatility_based_sizing -v
```

## Monitoring Specific Subsystems

```python
# Monitor Schwab connection
from src.unity_wheel.schwab import monitor_connection
monitor_connection(interval=60)  # Check every 60s

# Monitor Databento data quality
from src.unity_wheel.data_providers.databento import monitor_data_quality
monitor_data_quality(symbol="U", interval=300)  # Every 5 min

# Monitor risk limits
from src.unity_wheel.risk import monitor_risk_limits
monitor_risk_limits(portfolio_value=100000)  # Real-time risk monitoring
```

## Less Common Topics (See below for details)

- [Memory & Resource Optimization](#memory--resource-optimization)
- [Integration Testing Shortcuts](#integration-testing-shortcuts)
- [Monitoring Specific Subsystems](#monitoring-specific-subsystems)
- [v2.0 Enhancements](#v20-enhancements)
- [Git Hooks & Shell Scripts](#git-hooks)

---

# DETAILED SECTIONS BELOW

The sections below contain detailed information for less common tasks. The quick reference above covers 90% of typical usage.
