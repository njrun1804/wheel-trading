# Database Validation Tools

This directory contains comprehensive validation tools for the wheel trading database to ensure data integrity and safety before making financial decisions.

## Quick Start

```bash
# Daily health check (run every morning before trading)
python validate.py --daily

# Comprehensive validation (run weekly or after data updates)
python validate.py --comprehensive

# Investigate specific Greek calculation issues
python validate.py --greeks
```

## Validation Scripts

### 1. `validate.py` - Unified Validation Interface
**Main entry point for all validation tasks**

```bash
python validate.py --daily          # Quick health check
python validate.py --comprehensive  # Full validation
python validate.py --greeks         # Greek investigation
```

### 2. `daily_health_check.py` - Quick Daily Checks
**5 essential checks that should pass before trading**

- ✅ Data freshness (< 3 days old)
- ✅ Option Greeks integrity
- ✅ Price data sanity
- ✅ Unity stock data availability
- ✅ Unity option chain coverage

**Output:** Health score + trading readiness status

### 3. `validate_database_comprehensive.py` - Full Validation
**Comprehensive 11-point validation covering:**

#### Financial Data Integrity
- Negative prices/volumes detection
- Option bid-ask spread validation
- Arbitrage violation checks

#### Mathematical Consistency
- Greek calculation accuracy
- Implied volatility ranges
- Black-Scholes compliance

#### Data Completeness
- Price history gap analysis
- Option chain completeness
- Data staleness assessment

#### Statistical Anomalies
- Price outlier detection (Z-score > 5)
- Discontinuity identification (>50% jumps)
- Return distribution validation

**Output:** JSON report + health score + trading safety assessment

### 4. `investigate_greek_issues.py` - Greek Analysis
**Deep dive into option Greek calculation issues**

- Identifies specific problematic options
- Compares stored vs. theoretical Greeks
- Analyzes systematic patterns
- Provides fix recommendations

## Output Files

### `database_validation_report.json`
Complete validation results in machine-readable format
```json
{
  "financial_integrity": {...},
  "mathematical_consistency": {...},
  "data_completeness": {...},
  "statistical_anomalies": {...},
  "summary": {
    "health_score": 81.8,
    "safe_for_trading": false
  }
}
```

### `database_health_summary.md`
Human-readable executive summary with:
- Health score and trading readiness
- Detailed findings by category
- Risk assessment and recommendations
- Action plan with priorities

## Current Database Status

**As of 2025-06-11:**
- **Health Score:** 81.8% (Good)
- **Safe for Trading:** ⚠️ NO (theta issues need fixing)
- **Main Issue:** 23,302 options (17.2%) have positive theta
- **Data Quality:** Excellent (no pricing errors, complete chains)

## Daily Workflow

### Before Market Open
```bash
python validate.py --daily
```
**Expected:** 100% health score, "TRADING STATUS: READY"

### Weekly Deep Check
```bash
python validate.py --comprehensive
```
**Review:** `database_health_summary.md` for detailed analysis

### When Issues Found
```bash
python validate.py --greeks
```
**Action:** Follow recommendations in output

## Health Score Interpretation

| Score | Status | Action |
|-------|--------|---------|
| 90-100% | 🟢 Ready | Trade normally |
| 70-89% | 🟡 Caution | Extra monitoring, reduced position sizes |
| <70% | 🔴 Not Ready | Fix issues before trading |

## Key Metrics Monitored

### Critical (Must Pass)
- ✅ No negative prices/volumes
- ✅ Delta signs correct (calls: 0-1, puts: -1-0)
- ✅ Data freshness (<3 days)
- ✅ Unity data availability

### Important (Monitor)
- ⚠️ Theta calculations (17.2% currently positive)
- ⚠️ Option chain completeness
- ⚠️ Statistical outliers

### Informational
- 📊 Implied volatility ranges
- 📊 Return distributions
- 📊 Data coverage statistics

## Automation Setup

### Add to Crontab
```bash
# Daily check at 8:30 AM ET (before market open)
30 8 * * 1-5 cd /path/to/wheel-trading && python validate.py --daily

# Weekly comprehensive check Sunday at 6 PM
0 18 * * 0 cd /path/to/wheel-trading && python validate.py --comprehensive
```

### GitHub Actions (Optional)
```yaml
# Add to .github/workflows/data-validation.yml
- name: Validate Database
  run: python validate.py --comprehensive
```

## Troubleshooting

### Common Issues

**"Database not found"**
```bash
# Check if database exists
ls -la data/wheel_trading_master.duckdb
```

**"Import errors"**
```bash
# Install dependencies
pip install -r requirements.txt
```

**"Positive theta warnings"**
```bash
# This is known issue - investigate specifics
python validate.py --greeks
```

### False Positives

**Weekend data staleness:** Normal on Monday mornings
**High positive theta count:** Expected for deep OTM puts
**Small option chain gaps:** Acceptable for illiquid strikes

## Integration with Trading System

### Before Each Trade
```python
from daily_health_check import daily_health_check

if not daily_health_check():
    print("🚨 Database health check failed - aborting trades")
    sys.exit(1)
```

### Position Sizing Adjustment
```python
# Reduce position size for options with data quality issues
if option_has_positive_theta(option_symbol):
    position_size *= 0.75  # Reduce by 25%
```

## Support

For issues with validation tools:
1. Check `database_health_summary.md` for detailed explanations
2. Run `python validate.py --greeks` for Greek-specific issues
3. Review the JSON report for machine-readable details
4. Check database structure with `duckdb data/wheel_trading_master.duckdb -c "SHOW TABLES"`

---

**Remember:** These tools are designed to catch data quality issues before they impact your trading. When in doubt, err on the side of caution and investigate further before making financial decisions.
