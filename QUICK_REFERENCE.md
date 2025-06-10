# Unity Wheel Bot v2.0 - Quick Reference

> **⚠️ Note:** The default `config.yaml` is tuned for high risk/high return strategies.
> For a more conservative approach, see `examples/core/conservative_config.yaml`

## 🚀 Daily Commands

```bash
# Morning health check (run first!)
./daily_health_check.py

# Get trading recommendation
python run.py --portfolio 100000

# Quick system check
python run.py --diagnose

# View performance
python run.py --performance

# Live monitoring dashboard
./monitor_live.py
```

## 🤖 Autonomous Operations

```bash
# Start monitoring (runs continuously)
./scripts/monitor.sh

# One-time full check
./scripts/autonomous-checks.sh

# Weekly maintenance
./scripts/maintenance.sh
```

## 🔍 Troubleshooting

```bash
# Check version
python run.py --version

# Export metrics for analysis
python run.py --export-metrics

# View configuration health
python -c "from src.config.loader import get_config_loader; print(get_config_loader().generate_health_report())"

# Check feature flags
python -c "from unity_wheel.utils import get_feature_flags; f = get_feature_flags(); print(f.get_status_report()['summary'])"
```

## ⚙️ Configuration Overrides

```bash
# Change parameters via environment
export WHEEL_STRATEGY__DELTA_TARGET=0.25
export WHEEL_STRATEGY__DAYS_TO_EXPIRY_TARGET=30

# Control monitoring
export MONITOR_INTERVAL=300  # 5 minutes
export EXPORT_METRICS=true
```

## 📊 Key Files

- `logs/monitor.log` - Continuous monitoring output
- `exports/` - Metric export files
- `feature_flags.json` - Feature states
- `metrics.db` - Performance history
- `config.yaml` - Main configuration

## 🚨 Common Issues

### "System diagnostics failed"
```bash
# Check specific failures
python run.py --diagnose --verbose

# Reset feature flags
rm feature_flags.json
```

### "Performance degraded"
```bash
# View slow operations
python run.py --performance

# Clear cache
find . -name "*.cache" -delete
```

### "Configuration issues"
```bash
# Validate config
python -c "from src.config import get_config"

# Check for env overrides
env | grep WHEEL_
```

## 📈 Monitoring Dashboard

```bash
# Export latest metrics
python run.py --export-metrics

# Files created:
# - exports/dashboard_YYYYMMDD_HHMMSS.json
# - exports/influx_YYYYMMDD_HHMMSS.txt
# - exports/prometheus_YYYYMMDD_HHMMSS.txt
# - exports/metrics_YYYYMMDD_HHMMSS.csv
```

## 🛡️ Git Hooks

```bash
# Install (one-time)
pre-commit install

# Run manually
pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

## 📞 Support

1. Run diagnostics first: `python run.py --diagnose`
2. Check logs: `tail -f logs/monitor.log`
3. Review alerts in monitoring output
4. Check version: `python run.py --version`

---
*Remember: This is a recommendation system only - no broker integration!*
