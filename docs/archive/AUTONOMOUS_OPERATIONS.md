> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Autonomous Operations Guide

## Overview

Unity Wheel Trading Bot v2.0 operates autonomously with minimal human intervention. This guide explains how the autonomous features work together to ensure reliable operation.

## Autonomous Monitoring

### Continuous Health Checks

The system continuously monitors its own health through multiple mechanisms:

1. **System Diagnostics** (`run_aligned.py --diagnose`)
   - Math validation with known test cases
   - Model integrity checks
   - Risk calculation verification
   - Type consistency validation
   - Performance benchmarks

2. **Performance Monitoring** (`run_aligned.py --performance`)
   - Operation latency tracking
   - SLA violation detection
   - Success rate monitoring
   - Slow operation identification

3. **Data Quality Validation**
   - Market data freshness checks
   - Price reasonableness validation
   - Option chain consistency
   - Spread validity checks
   - Liquidity requirements

### Automated Shell Scripts

#### `scripts/monitor.sh`
Runs continuous monitoring loop (default: every 5 minutes):
- Health checks
- Performance monitoring
- Alert detection
- Metric exports
- Automatic cleanup of old data

```bash
# Start monitoring
./scripts/monitor.sh

# With custom interval (seconds)
MONITOR_INTERVAL=600 ./scripts/monitor.sh

# Disable metric exports
EXPORT_METRICS=false ./scripts/monitor.sh
```

#### `scripts/autonomous-checks.sh`
Comprehensive system validation:
- Runs diagnostics
- Checks performance
- Validates configuration
- Monitors feature flags
- Exports metrics
- Cleans cache

```bash
# Run all checks
./scripts/autonomous-checks.sh

# Enable dev mode (includes tests)
DEV_MODE=true ./scripts/autonomous-checks.sh
```

## Git Pre-commit Hooks

Quality checks run automatically before commits:

```yaml
# .pre-commit-config.yaml includes:
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)
- Security scanning (bandit)
- System diagnostics
- Configuration validation
- Feature flag health checks
```

Install hooks:
```bash
pre-commit install
```

## Self-Healing Mechanisms

### 1. Circuit Breakers
Protect against cascading failures:
```python
@circuit_breaker("api_breaker")
def external_api_call():
    # Automatically opens circuit after failures
    # Falls back to mock data when open
```

### 2. Feature Flags
Dynamic feature control:
```python
flags = get_feature_flags()
if flags.is_enabled("advanced_feature"):
    # Use feature
else:
    # Use fallback
```

Features auto-disable after repeated failures.

### 3. Error Recovery
Automatic retry and fallback:
```python
@with_recovery(strategy=RecoveryStrategy.RETRY, max_attempts=3)
def risky_operation():
    # Automatically retries on failure
```

### 4. Data Quality Fallbacks
- Invalid data rejected
- Missing data handled gracefully
- Anomalies flagged but processed
- Stale data triggers warnings

## Self-Optimization

### Configuration Auto-Tuning

The system learns from outcomes:

```python
# Automatic parameter tracking
tuner = get_auto_tuner()
tuner.record_outcome(decision_id, parameters, outcome)

# Get tuning recommendations
recommendations = tuner.get_recommendations(current_config)
```

Parameters automatically adjust based on:
- Success rates
- Average returns
- Risk-adjusted performance
- Confidence levels

### Performance Optimization

- Automatic caching of expensive calculations
- Performance monitoring identifies bottlenecks
- SLA violations trigger alerts
- Resource usage tracked

## Observability

### Metric Exports

Export metrics for external dashboards:

```bash
python run.py --export-metrics
```

Formats supported:
- **JSON**: Universal format
- **InfluxDB**: Time-series database
- **Prometheus**: Monitoring system
- **CSV**: Spreadsheet analysis

### Structured Logging

All logs are machine-parseable JSON:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "module": "unity_wheel.strategy",
  "message": "Strike selected",
  "context": {
    "strike": 32.5,
    "delta": -0.30,
    "confidence": 0.95
  }
}
```

### Decision Audit Trail

Every decision is logged with:
- Unique decision ID
- Features used
- Confidence score
- Risk metrics
- Execution time

## Alerts and Notifications

The system alerts on:
- Critical diagnostic failures
- Performance SLA violations
- Feature degradation
- Data quality issues
- Configuration conflicts

Alerts appear in:
- Console output
- Log files
- Monitor script output
- Pre-commit hook failures

## Maintenance Tasks

### Automatic Cleanup
- Cache files older than 7 days
- Export files older than 30 days
- Log rotation when > 10MB

### Manual Maintenance
```bash
# Clean all temporary data
find . -name "*.cache" -delete
find exports/ -name "*.json" -mtime +7 -delete

# Reset feature flags
rm feature_flags.json

# Clear performance history
rm metrics.db
```

## Troubleshooting

### System Not Healthy
1. Run diagnostics: `python run.py --diagnose`
2. Check logs in `logs/monitor.log`
3. Review feature flag status
4. Validate configuration

### Performance Issues
1. Check metrics: `python run.py --performance`
2. Look for SLA violations
3. Review cache hit rates
4. Check for degraded features

### Data Quality Problems
1. Review validation results
2. Check for anomalies
3. Verify data sources
4. Look at circuit breaker states

## Best Practices

1. **Let it run**: The system self-monitors and self-heals
2. **Review alerts**: Don't ignore repeated warnings
3. **Trust auto-tuning**: Let parameters optimize over time
4. **Monitor trends**: Use exported metrics for analysis
5. **Keep hooks enabled**: Pre-commit catches issues early

## Summary

The autonomous operation features work together to create a self-managing system:

- **Monitoring** detects issues
- **Validation** ensures quality
- **Recovery** handles failures
- **Optimization** improves performance
- **Observability** provides insights

The result is a trading recommendation system that requires minimal manual intervention while maintaining high reliability and performance.
