# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unity Wheel Trading Bot v2.0 - An autonomous options wheel strategy recommendation system with self-monitoring, auto-tuning, and enterprise-grade reliability. Designed for single-user local operation with NO BROKER INTEGRATION (recommendations only).

## Commands

### Primary Commands (v2.0):

- `python run_aligned.py --portfolio 100000` - Get trading recommendation
- `python run_aligned.py --diagnose` - Run system diagnostics
- `python run_aligned.py --performance` - View performance metrics
- `python run_aligned.py --export-metrics` - Export dashboard metrics
- `python run_aligned.py --version` - Show version info
- `./scripts/autonomous-checks.sh` - Run all autonomous checks
- `./scripts/monitor.sh` - Start continuous monitoring

### Development Commands:

- `pre-commit run --all-files` - Run all quality checks
- `pytest tests/test_autonomous_flow.py -v` - Run integration tests
- `black src/ tests/` - Auto-format code
- `mypy src/` - Type checking

### Quick Development Flow:

```bash
# After making changes:
pre-commit run --all-files

# Check system health:
python run_aligned.py --diagnose

# Monitor performance:
python run_aligned.py --performance

# View configuration health:
python -c "from src.config.loader import get_config_loader; print(get_config_loader().generate_health_report())"
```

## Architecture

### Current Structure:

```
wheel-trading/
├── config.yaml         # Comprehensive configuration file
├── pyproject.toml      # Poetry configuration with exact pins
├── run.py              # CLI entry point for decisions
├── src/
│   ├── config/         # Intelligent configuration system
│   │   ├── schema.py   # Pydantic validation schemas
│   │   ├── loader.py   # Config loader with tracking
│   │   └── integration.py # Legacy compatibility
│   ├── unity_wheel/
│   │   ├── models/     # Immutable data models with validation
│   │   │   ├── position.py # Position with OCC symbol parsing
│   │   │   ├── greeks.py   # Greeks with range validation
│   │   │   └── account.py  # Account state tracking
│   │   ├── math/       # Self-validating options mathematics
│   │   │   └── options.py  # BS with confidence scores
│   │   ├── risk/       # Risk analytics with self-monitoring
│   │   │   └── analytics.py # VaR, CVaR, Kelly, limits
│   │   ├── schwab/     # Reliable Schwab client with validation
│   │   │   ├── client.py    # Main client with retry logic
│   │   │   ├── types.py     # Position/account data models
│   │   │   └── exceptions.py # Error hierarchy
│   │   └── validate.py # Environment validation script
│   ├── wheel.py        # Core wheel strategy implementation
│   └── utils/          # Utilities
└── tests/              # Property-based test suite
```

### Key Design Principles:

1. **Autonomous Operation** - Self-monitoring, self-healing, self-optimizing
2. **Self-Validation** - Every calculation includes confidence score
3. **Type Safety** - 100% type hints, mypy strict mode
4. **Immutable Models** - All data models are frozen dataclasses
5. **Property Testing** - Hypothesis for edge case discovery
6. **Structured Logging** - Machine-parseable JSON logs
7. **Performance Monitoring** - Automatic SLA tracking
8. **Graceful Degradation** - Feature flags for resilience

## Development Guidelines

1. **Logging Everything** - Include function name, inputs, outputs
2. **Confidence Scores** - All calculations return (result, confidence)
3. **100% Coverage** - Tests must maintain complete coverage
4. **Error Recovery** - Never crash, return NaN with explanation
5. **Performance Tracking** - Log any calculation >10ms

## Testing Requirements

Before any changes:
```bash
# Validate environment
poetry run python -m unity_wheel.validate

# Run full test suite
poetry run pytest -v

# Check specific functionality
poetry run python -c "
from unity_wheel.math import black_scholes_price_validated
result = black_scholes_price_validated(100, 100, 1, 0.05, 0.2, 'call')
print(f'Price: {result.value:.2f}, Confidence: {result.confidence:.0%}')
"
```

## Configuration System

### Overview

The project uses an intelligent YAML-based configuration system with:
- **Comprehensive validation** using Pydantic schemas
- **Environment variable overrides** (WHEEL_SECTION__PARAM format)
- **Parameter usage tracking** to identify unused settings
- **Impact tracking** to suggest parameter tuning
- **Health reporting** with warnings and recommendations
- **Self-tuning capabilities** based on outcome tracking

### Key Configuration Sections:

1. **Strategy** (`strategy.*`)
   - `delta_target`: Target delta for short puts (default: 0.30)
   - `days_to_expiry_target`: Target DTE (default: 45)
   - `roll_triggers`: Profit targets, delta breaches, DTE thresholds

2. **Risk** (`risk.*`)
   - `max_position_size`: Max position as % of portfolio (default: 0.20)
   - `kelly_fraction`: Kelly criterion fraction (default: 0.50 = Half-Kelly)
   - `limits`: VaR, CVaR, Greek exposures, margin utilization

3. **Data** (`data.*`)
   - `cache_ttl`: Cache expiration times
   - `api_timeouts`: Connection and request timeouts
   - `quality`: Data staleness and minimum liquidity thresholds

4. **ML** (`ml.*`)
   - `enabled`: Toggle ML enhancement
   - `features`: IV rank, skew, realized vol, macro factors
   - `models`: Probability and volatility model configurations

### Environment Variable Overrides:

```bash
# Override delta target
export WHEEL_STRATEGY__DELTA_TARGET=0.25

# Enable ML features
export WHEEL_ML__ENABLED=true

# Set trading mode
export WHEEL_TRADING__MODE=paper
```

### Configuration Health Monitoring:

```python
# Check configuration health
from src.config.loader import get_config_loader
loader = get_config_loader()
print(loader.generate_health_report())

# Track parameter usage
loader.track_parameter_usage("strategy.delta_target")

# Report decision impact
loader.track_parameter_impact("strategy.delta_target", 0.75)
```

## Performance Targets

- Black-Scholes: <0.2ms per calculation
- Greeks: <0.3ms for all Greeks
- Risk metrics: <10ms for 1000 data points
- IV solver: <5ms with fallback to bisection
- Memory: <100MB for typical portfolio

## Objective Function

Maximize: **CAGR - 0.20 × |CVaR₉₅|** with **½-Kelly** position sizing

## Schwab Integration

The project now includes a reliable Schwab client with:
- **Automatic retry logic** for network failures
- **Position validation** with OCC symbol parsing
- **Corporate action detection** from position anomalies
- **Fallback to cached data** during outages
- **Self-validation** of all data consistency

### Schwab Client Usage:

```python
from src.unity_wheel.schwab import SchwabClient

async with SchwabClient(client_id, client_secret) as client:
    positions = await client.get_positions()  # Never cached
    account = await client.get_account()      # Cached briefly (30s)

    # Detect corporate actions
    actions = client.detect_corporate_actions(positions)
```

### Required Environment Variables:
- `SCHWAB_CLIENT_ID` - OAuth client ID
- `SCHWAB_CLIENT_SECRET` - OAuth client secret

## Future Features Roadmap

1. **Schwab OAuth Flow** - Complete OAuth implementation
2. **Decision Engine** - Multi-criteria scoring with explanations
3. **ML Enhancement** - Probability adjustments, pattern recognition
4. **Backtesting** - Historical validation with transaction costs

## Databento Integration

The project now includes comprehensive Databento integration for options data:

### Key Features:
- **Rate-limited client** with automatic retry logic
- **Smart data filtering** to reduce storage by 80%
- **Hybrid storage** (local for recent, cloud for historical)
- **Comprehensive validation** for data quality
- **Integration with wheel strategy** for candidate selection

### Usage:
```python
# Find wheel candidates
from src.unity_wheel.databento.integration import DatentoIntegration
integration = DatentoIntegration(client, storage)
candidates = await integration.get_wheel_candidates(
    underlying="U",
    target_delta=0.30,
    dte_range=(30, 60)
)
```

### Storage:
- Local: 30 days of data (~5GB for Unity)
- Cloud: Optional GCS/BigQuery for historical
- Monthly cost: <$1 for typical usage

See `DATABENTO_INTEGRATION.md` for full details.

## Notes

- Always validate calculations with known test cases
- Use property-based testing for new functions
- Log confidence degradation for monitoring
- Prefer self-diagnostic approaches over manual debugging
- NO BROKER INTEGRATION - recommendations only

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

### Daily Health Check
- **Script**: `./daily_health_check.py`
- **Run**: Every morning before trading
- **Checks**: Data freshness, config health, credentials, performance

### Live Monitor
- **Script**: `./monitor_live.py`
- **Purpose**: Real-time dashboard with alerts
- **Updates**: Every 10 seconds
- **Features**: Risk status, performance, market data, alerts
