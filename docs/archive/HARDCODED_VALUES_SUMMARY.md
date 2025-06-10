> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Hardcoded Values Summary

## Overview
This document summarizes all hardcoded values and dummy data found in the Unity Wheel Trading Bot codebase, organized by category with recommendations for centralization.

## 1. Risk Management Constants

### Critical Risk Limits (MUST CENTRALIZE)
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| `max_position_pct` | risk/limits.py:32 | 0.20 | Max position size | Move to config.yaml → risk.limits |
| `max_consecutive_losses` | risk/limits.py:44 | 3 | Stop after 3 losses | Move to config.yaml → risk.circuit_breakers |
| `max_volatility` | risk/limits.py:35 | 1.5 | 150% vol limit | Make adaptive based on market regime |
| `min_confidence` | risk/limits.py:47 | 0.30 | Min confidence to trade | Move to config.yaml → risk.limits |
| `max_daily_loss_pct` | risk/limits.py:42 | 0.02 | 2% daily loss limit | Move to config.yaml → risk.limits |
| `max_weekly_loss_pct` | risk/limits.py:43 | 0.05 | 5% weekly loss limit | Move to config.yaml → risk.limits |

### Risk Analytics Limits
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| `max_var_95` | risk/analytics.py:84 | 0.05 | 5% VaR limit | Move to config.yaml → risk.var |
| `max_cvar_95` | risk/analytics.py:85 | 0.075 | 7.5% CVaR limit | Move to config.yaml → risk.cvar |
| `max_kelly_fraction` | risk/analytics.py:86 | 0.25 | Max Kelly sizing | Move to config.yaml → risk.kelly |
| `max_delta_exposure` | risk/analytics.py:87 | 100.0 | Delta limit | Move to config.yaml → risk.greeks |
| `max_margin_utilization` | risk/analytics.py:90 | 0.5 | 50% margin max | Move to config.yaml → risk.margin |

## 2. Strategy Parameters

### Wheel Strategy Constants
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| `min_premium_yield` | wheel.py:51 | 0.01 | 1% min premium | Move to config.yaml → strategy.filters |
| `roll_dte_threshold` | wheel.py:52 | 7 | Roll at 7 DTE | Move to config.yaml → strategy.roll_triggers |
| `roll_delta_threshold` | wheel.py:53 | 0.70 | Roll at 70 delta | Move to config.yaml → strategy.roll_triggers |
| Strike filters | wheel.py:140,311 | 0.8, 1.1 | 80-110% of spot | Move to config.yaml → strategy.strike_range |

### Adaptive System Parameters
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| `base_position_pct` | adaptive.py:99 | 0.20 | Base position size | Use config risk.limits.max_position_pct |
| `_regime_persistence_days` | adaptive.py:103 | 3 | Regime confirmation | Move to config.yaml → adaptive.regime_persistence |
| Delta targets | adaptive.py:244-253 | 0.20-0.30 | Per-regime deltas | Move to config.yaml → adaptive.regime_params |
| DTE targets | adaptive.py:245-261 | 21-42 | Per-regime DTEs | Move to config.yaml → adaptive.regime_params |

## 3. Network & API Configuration

### API URLs (ACCEPTABLE AS-IS)
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| Schwab OAuth URLs | oauth.py:30-32 | Official URLs | OAuth endpoints | Keep hardcoded (official API) |
| Redirect URI | oauth.py:30 | 127.0.0.1:8182 | OAuth callback | Keep for local-only operation |

### Timeouts & Retries (SHOULD CENTRALIZE)
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| Connection timeout | schema.py:125 | 5.0s | Connect timeout | Move to config.yaml → data.api_timeouts |
| Read timeout | schema.py:126 | 30.0s | Read timeout | Move to config.yaml → data.api_timeouts |
| Total timeout | client.py:93 | 30s | Request timeout | Use config values |
| Retry after | client.py:160 | 60s | Rate limit retry | Move to config.yaml → data.retry |

## 4. Data Processing Constants

### Databento Limits (PROVIDER-SPECIFIC)
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| `MONEYNESS_RANGE` | storage_adapter.py:24 | 0.20 | 20% of spot | Move to config.yaml → databento.filters |
| `MAX_EXPIRATIONS` | storage_adapter.py:25 | 3 | Keep 3 expiries | Move to config.yaml → databento.filters |
| `MAX_SPREAD_PCT` | validation.py:33 | 10.0 | Max bid-ask spread | Move to config.yaml → data.quality |
| Rate limits | client.py:41-43 | Various | API limits | Keep hardcoded (provider limits) |

### Cache TTLs (SHOULD CENTRALIZE)
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| `INTRADAY_TTL_MINUTES` | storage_adapter.py:26 | 15 | Intraday cache | Move to config.yaml → data.cache_ttl |
| `GREEKS_TTL_MINUTES` | storage_adapter.py:27 | 5 | Greeks cache | Move to config.yaml → data.cache_ttl |
| Default cache TTL | data_ingestion.py:387 | 5 min | General cache | Use config values |

## 5. Performance Thresholds

### Operation Timing Thresholds
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| Black-Scholes threshold | Various | 50ms | Calc timing | Move to config.yaml → performance.sla |
| Greeks threshold | Various | 100ms | Calc timing | Move to config.yaml → performance.sla |
| Decision threshold | advisor.py:39 | 200ms | Decision time | Move to config.yaml → performance.sla |

## 6. Database & Storage

### Database Files (ACCEPTABLE AS-IS)
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| `wheel_outcomes.db` | adaptive.py:252 | Outcomes DB | Track results | Keep, but make path configurable |
| `performance.db` | performance_tracker.py:40 | Performance DB | Track trades | Keep, but make path configurable |
| `schwab_data.db` | data_ingestion.py:87 | Account data | Store positions | Keep, but make path configurable |
| `wheel_cache.duckdb` | duckdb_cache.py:38 | Market cache | Cache data | Keep, but make path configurable |

### Table Names (KEEP HARDCODED)
- All table names should remain hardcoded as they are part of the schema definition
- Consider creating a schema versioning system if tables need to change

## 7. Mock/Dummy Data

### Placeholder Code (MUST REMOVE)
| Location | Type | Current | Action Required |
|----------|------|---------|----------------|
| anomaly_detector.py:400 | Placeholder spread | 0.02 | Implement proper calculation |
| integration.py:336 | Placeholder comment | N/A | Complete implementation |
| market_snapshot.py:144 | Disabled cache | False | Enable cache check |
| client.py:116 | OAuth placeholder | Comment | Implement OAuth flow |

## 8. Calculation Constants

### Mathematical Constants (KEEP LOCAL)
| Value | Location | Current | Purpose | Recommendation |
|-------|----------|---------|---------|----------------|
| Annual days | Various | 365.0 | Time conversion | Keep hardcoded |
| Trading days | Various | 252 | Market days | Keep hardcoded |
| Contract multiplier | Various | 100 | Options size | Keep hardcoded |
| Z-scores | Various | 1.65, 2.33 | Statistics | Keep hardcoded |

## Priority Actions

### High Priority (Affects Trading)
1. **Risk Limits**: Move all risk management constants to config.yaml
2. **Strategy Parameters**: Centralize wheel strategy parameters
3. **Adaptive Parameters**: Move regime-specific values to config
4. **Remove Placeholders**: Complete or remove all placeholder implementations

### Medium Priority (Operational)
1. **Network Config**: Centralize timeouts and retry settings
2. **Cache TTLs**: Move all cache durations to config
3. **Performance SLAs**: Define operation timing limits in config
4. **Database Paths**: Make storage locations configurable

### Low Priority (Nice to Have)
1. **Data Quality**: Centralize validation thresholds
2. **Provider Limits**: Document but keep provider-specific limits hardcoded
3. **Schema Version**: Add database schema versioning

## Implementation Guide

### Step 1: Update config.yaml
```yaml
risk:
  limits:
    max_position_pct: 0.20
    max_consecutive_losses: 3
    min_confidence: 0.30
    max_daily_loss_pct: 0.02
    max_weekly_loss_pct: 0.05
  var:
    max_var_95: 0.05
  cvar:
    max_cvar_95: 0.075
  kelly:
    max_fraction: 0.25
  greeks:
    max_delta_exposure: 100.0
    max_gamma_exposure: 10.0
    max_vega_exposure: 1000.0

adaptive:
  regime_persistence_days: 3
  regime_params:
    normal:
      put_delta: 0.30
      target_dte: 35
      roll_profit_target: 0.50
    volatile:
      put_delta: 0.25
      target_dte: 28
      roll_profit_target: 0.25
    stressed:
      put_delta: 0.20
      target_dte: 21
      roll_profit_target: 0.25
```

### Step 2: Update Code References
Replace hardcoded values with config references:
```python
# Before
max_position_pct: float = 0.20

# After
max_position_pct: float = config.risk.limits.max_position_pct
```

### Step 3: Add Smart Value Support
For adaptive values, implement dynamic adjustment:
```python
# Static threshold
if volatility > 1.5:

# Adaptive threshold
volatility_limit = self._get_adaptive_limit('volatility', market_regime)
if volatility > volatility_limit:
```

## Validation Checklist

- [ ] All critical risk limits moved to config
- [ ] Strategy parameters centralized
- [ ] Network timeouts configurable
- [ ] Cache TTLs in config
- [ ] Placeholder code removed or completed
- [ ] Database paths configurable
- [ ] Adaptive parameters support market regimes
- [ ] No credentials or secrets hardcoded
- [ ] All changes tested with existing tests
