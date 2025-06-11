# Sequential Implementation Guide - Unity Wheel Trading Bot

## Overview
This guide provides the exact sequence of tasks and prompts needed to complete the Unity Wheel Trading Bot. Each task is designed to be completed in a single Claude Code session.

## Task Flow & Dependencies

```
Task 1: Schwab OAuth ──────────┐
                               ├──> Task 2: Capital Allocation ─┐
                               │                                 ├──> Task 5: Enhanced Decision Engine
Task 3: Cost Analysis ─────────┘                                 │
                                                                 │
Task 4: Data Pipeline ───────────────────────────────────────────┘
                                                                 │
                                                                 v
                                              Task 6: Backtesting System
                                                                 │
                                                                 v
                                              Task 7: ML Enhancement (Optional)
                                                                 │
                                                                 v
                                              Task 8: Monitoring & Alerts
                                                                 │
                                                                 v
                                              Task 9: Integration & Docs
```

## Prerequisites
- MacOS 15.5 MBP M4 Silicon
- VS Code with Claude Code CLI
- Schwab Developer Account with API credentials
- Databento API access
- Python 3.11+ with Poetry installed

## Quick Reference

### Key Commands During Development
```bash
# Check system health
python run.py --diagnose

# Run unit tests for current work
pytest tests/test_<module>.py -v

# Check code quality
pre-commit run --all-files

# View configuration health
python -c "from src.config.loader import get_config_loader; print(get_config_loader().generate_health_report())"

# Run with mock data for testing
# (obsolete flag removed)
```

## Data Requirements & Costs

### Databento
- **Unity options**: ~$20-30/month for daily updates
- **Storage**: ~5GB for 30 days of Unity options
- **Initial backfill**: ~$100 for 2 years history
- **Rate limits**: 100 requests/minute

### Schwab API
- **Free** with brokerage account
- **Rate limits**: 120 requests/minute
- **OAuth tokens**: 30-minute access, 7-day refresh

## Common Pitfalls to Avoid

1. **Don't hardcode credentials** - Always use SecretManager
2. **Don't ignore rate limits** - Use built-in retry logic
3. **Don't skip validation** - Every calculation needs confidence scores
4. **Don't trust external data** - Always validate and sanitize
5. **Don't forget logging** - Log inputs, outputs, and decisions
6. **Don't optimize prematurely** - Get it working first
7. **Don't ignore edge cases** - Use property-based testing

---

## Task 1: Implement Schwab OAuth Authentication
**Dependencies**: None
**Estimated Time**: 2-3 hours

### Prompt:
```
Implement complete Schwab OAuth2 authentication flow in src/unity_wheel/schwab/client.py. Currently the authenticate() method just raises NotImplementedError.

Requirements:
1. Implement full OAuth2 PKCE flow following Schwab's documentation
2. Add automatic token refresh when tokens expire
3. Store tokens securely using the existing SecretManager
4. Create a setup script at scripts/setup_schwab_auth.py that walks through initial authentication
5. Add retry logic for auth failures
6. Test by successfully fetching real account data
7. Update tests/test_schwab.py with integration tests (use mock for CI)

The implementation must be fully autonomous - handle all error cases, log everything, and provide clear user guidance. Ensure all token handling is secure and follows OAuth2 best practices.

Success criteria: Can run `python scripts/setup_schwab_auth.py` to authenticate, then use SchwabClient to fetch real positions.
```

### Validation Steps:
1. Run `python scripts/setup_schwab_auth.py` - should guide through OAuth
2. Test token refresh: `python -c "from src.unity_wheel.schwab import SchwabClient; client = SchwabClient(); print(client.test_connection())"`
3. Verify real data: `python -c "from src.unity_wheel.schwab import SchwabClient; client = SchwabClient(); positions = client.get_positions(); print(f'Found {len(positions)} positions')"`
4. Check token storage: Verify tokens are in SecretManager, not in files
5. Run integration tests: `pytest tests/test_schwab.py -v`

---

## Task 2: Create Multi-Asset Capital Allocation Framework
**Dependencies**: Task 1 (need real account data)
**Estimated Time**: 3-4 hours

### Prompt:
```
Create a comprehensive capital allocation framework that tracks ALL capital sources and optimizes across them. The current system only considers Unity positions, but we need to optimize across:
- AMEX loan: $45k at 7% interest
- Schwab margin: available credit line with variable rate
- Cash: opportunity cost
- Stock positions: current holdings
- Options positions: existing contracts

Create:
1. src/unity_wheel/models/capital.py with:
   - LoanPosition dataclass (principal, rate, payments)
   - MarginStatus dataclass (available, used, rate)
   - CapitalAllocation class that tracks all sources
   - Methods to calculate total cost of capital

2. src/unity_wheel/analytics/portfolio_optimizer.py with:
   - optimize_allocation() that maximizes CAGR - 0.20×|CVaR₉₅| across ALL capital
   - Should recommend: pay down loan vs invest, use margin vs cash, position sizes
   - Factor in tax efficiency (even though environment is tax-free)
   - Include transaction costs in optimization

3. Update config.yaml to add:
   - capital.loans section for tracking debts
   - capital.margin section for margin parameters
   - capital.optimization section for solver settings

4. Add comprehensive tests covering all allocation scenarios

The optimizer must explain every decision clearly and provide confidence scores.
```

### Validation Steps:
1. Test capital tracking: `python -c "from src.unity_wheel.models.capital import CapitalAllocation; ca = CapitalAllocation(); print(ca.get_total_capital())"`
2. Verify loan calculations: Check 7% APR on $45k = $3,150/year interest
3. Test optimizer logic: Run scenarios where paying loan beats investing
4. Validate decisions: `python -c "from src.unity_wheel.analytics.portfolio_optimizer import optimize_allocation; result = optimize_allocation(...); print(result.explanation)"`
5. Run property tests: `pytest tests/test_capital_allocation.py -v`

---

## Task 3: Implement Cost-of-Change Analysis
**Dependencies**: Tasks 1-2 (need capital framework)
**Estimated Time**: 2 hours

### Prompt:
```
Add transaction cost modeling to properly evaluate the benefit of position changes. Currently the system ignores execution costs.

Implement in src/unity_wheel/analytics/transaction_costs.py:
1. CostAnalyzer class that models:
   - Bid-ask spreads (from real market data)
   - Commission costs (if any)
   - Assignment/exercise fees
   - Early close penalties for options
   - Tax implications of trades (even in tax-free account, model it)

2. Add to WheelStrategy.evaluate_position_change():
   - Calculate total cost of changing position
   - Compare benefit vs cost
   - Only recommend if net positive after costs
   - Provide break-even analysis

3. Update decision engine to use cost analysis:
   - Score all actions including "do nothing"
   - Rank by net benefit after costs
   - Show cost breakdown in recommendations

4. Add configuration for cost parameters in config.yaml

Include property-based tests to ensure cost calculations are always positive and transaction recommendations are profitable net of costs.
```

### Validation Steps:
1. Test cost calculations: `python -c "from src.unity_wheel.analytics.transaction_costs import CostAnalyzer; ca = CostAnalyzer(); print(ca.calculate_total_cost(...))"`
2. Verify spreads are realistic (typically 0.01-0.05 for liquid options)
3. Test edge cases: zero liquidity, wide spreads, early exercise
4. Validate net benefit: All recommendations must be +EV after costs
5. Run property tests: `pytest tests/test_transaction_costs.py::test_costs_always_positive -v`

---

## Task 4: Connect Real-Time Data Pipeline
**Dependencies**: Tasks 1-3 (need auth and cost model)
**Estimated Time**: 2-3 hours

### Prompt:
```
Wire together all data sources to feed real information into the decision engine. Currently using mock data.

1. Update src/unity_wheel/api/advisor.py:
   - Remove all mock data
   - Connect to real Schwab client for positions/account
   - Use Databento for real-time option chains
   - Add circuit breakers for data quality

2. Create src/unity_wheel/data_providers/base/pipeline.py:
   - DataPipeline class that orchestrates all sources
   - Automatic fallback to cached data if sources fail
   - Data quality validation
   - Staleness detection and warnings

3. Update run.py:
   - Use real data pipeline
   - Show data source status in output

4. Add monitoring in src/unity_wheel/monitoring/data_quality.py:
   - Track data freshness
   - Monitor source availability
   - Alert on quality issues
   - Log all data anomalies

Test with real market data during market hours. Ensure graceful degradation when sources are unavailable.
```

### Validation Steps:
1. Test during market hours: `python run.py --portfolio 100000`
2. Verify data sources: Check logs show "Schwab: CONNECTED", "Databento: CONNECTED"
3. Test failover: Disconnect network, should use cached data with warning
4. Monitor latency: `python run.py --portfolio 100000 --timing`
5. Validate data quality: No stale prices (>15 min old during market hours)

---

## Task 5: Enhance Decision Engine with Multi-Asset Logic
**Dependencies**: Tasks 1-4 (need all components)
**Estimated Time**: 3 hours

### Prompt:
```
Integrate the capital allocation framework into the decision engine to provide holistic recommendations.

1. Update src/unity_wheel/api/advisor.py WheelAdvisor class:
   - Load CapitalAllocation with all debts/margin
   - Call portfolio optimizer before position recommendations
   - Consider opportunity cost of capital in all decisions

2. Enhance recommendation output to include:
   - Loan paydown recommendations (if better than investing)
   - Margin usage decisions (when beneficial)
   - Cash allocation (how much to keep liquid)
   - Position sizing considering all capital sources

3. Add decision logic for:
   - When to pay down AMEX loan vs invest
   - When to use Schwab margin vs cash
   - How to optimize across all capital sources
   - Clear explanations for each recommendation

4. Update src/unity_wheel/recommendations/models.py:
   - Add LoanAction, MarginAction recommendation types
   - Include capital efficiency metrics
   - Show impact on total portfolio

Every recommendation must show its effect on the overall objective function (CAGR - 0.20×|CVaR₉₅|).
```

### Validation Steps:
1. Test full recommendation: `python run.py --portfolio 200000 --verbose`
2. Verify output includes: loan decisions, margin usage, position sizing
3. Check objective function: Each action shows impact on CAGR - 0.20×|CVaR₉₅|
4. Test edge case: When loan paydown beats investing (high market vol)
5. Validate explanations: Every decision has clear reasoning

---

## Task 6: Implement Comprehensive Backtesting
**Dependencies**: Tasks 1-5 (need complete system)
**Estimated Time**: 4 hours

### Prompt:
```
Complete the backtesting system to validate strategies before live use. The framework exists but isn't functional.

1. Implement src/unity_wheel/analytics/backtest_engine.py:
   - Event-driven backtester with realistic execution
   - Use actual bid-ask spreads from historical data
   - Model assignments based on pin risk
   - Include all transaction costs
   - Track loan interest and margin costs

2. Create src/unity_wheel/analytics/historical_data_loader.py:
   - Load options data from Databento
   - Cache preprocessed data locally
   - Provide point-in-time data access
   - Handle corporate actions

3. Add walk-forward optimization:
   - Rolling window parameter selection
   - Out-of-sample validation
   - Regime change detection
   - Parameter stability analysis

4. Generate comprehensive reports:
   - Performance attribution
   - Risk model validation
   - Transaction cost analysis
   - Parameter sensitivity

5. Create scripts/run_backtest.py CLI:
   - Backtest specific date ranges
   - Compare strategies
   - Export results

Validate backtester with known scenarios. Results must match forward testing within 5%.
```

### Validation Steps:
1. Run standard backtest: `python scripts/run_backtest.py --symbol U --start 2023-01-01 --end 2023-12-31`
2. Verify transaction costs: Spread costs should be 1-3% of profits
3. Test assignment modeling: ~30% of ITM options at expiry assigned
4. Compare with paper trading: Results within 5% of forward test
5. Check report completeness: Sharpe, max drawdown, win rate, profit factor all present

---

## Task 7: Add ML Enhancement Layer (Optional)
**Dependencies**: Tasks 1-6 (need historical data)
**Estimated Time**: 3 hours

### Prompt:
```
Implement ML models to enhance decision-making, but only if they demonstrate clear value over analytical approaches.

1. Create src/unity_wheel/ml/feature_engineering.py:
   - Extract microstructure features
   - Calculate regime indicators
   - Build volatility features
   - Generate training datasets

2. Implement src/unity_wheel/ml/models.py:
   - Probability of profit (PoP) adjustment model
   - Volatility regime classifier
   - Assignment probability predictor
   - Use simple, interpretable models (GBM, linear)

3. Add ML integration to decision engine:
   - Blend ML predictions with analytical
   - Always provide confidence intervals
   - Explain feature importance
   - A/B test framework

4. Create training pipeline:
   - Automatic retraining triggers
   - Performance monitoring
   - Drift detection
   - Model versioning

Only deploy ML if backtesting shows >5% improvement in Sharpe ratio. All predictions must be explainable.
```

### Validation Steps:
1. Train models: `python scripts/train_ml_models.py --symbol U`
2. Compare performance: A/B test ML vs analytical approach
3. Verify improvement: Must show >5% Sharpe improvement to deploy
4. Test explainability: `python -c "from src.unity_wheel.ml.models import explain_prediction; print(explain_prediction(...))"`
5. Monitor drift: Check model performance weekly, retrain if degraded

---

## Task 8: Production Monitoring and Alerts
**Dependencies**: Tasks 1-6 minimum
**Estimated Time**: 2 hours

### Prompt:
```
Build comprehensive monitoring to ensure the system runs reliably without intervention.

1. Enhance src/unity_wheel/monitoring/health_monitor.py:
   - Real-time system health scoring
   - Component availability tracking
   - Performance metrics collection
   - Automatic issue detection

2. Create src/unity_wheel/monitoring/alerts.py:
   - Define alert conditions (data staleness, errors, anomalies)
   - Implement notification system (log files, optional email)
   - Smart alert grouping to avoid spam
   - Self-resolving issue tracking

3. Add performance tracking:
   - Compare predictions to outcomes
   - Track strategy effectiveness
   - Monitor execution quality
   - Generate daily/weekly reports

4. Implement scripts/monitor_daemon.py:
   - Continuous monitoring service
   - Auto-restart on failures
   - Resource usage tracking
   - Performance optimization

5. Create operational dashboard:
   - System status overview
   - Recent decisions and outcomes
   - Risk metrics trends
   - Data quality indicators

The monitoring must detect and report issues before they impact recommendations.
```

### Validation Steps:
1. Start monitor: `python scripts/monitor_daemon.py --daemon`
2. Test alerts: Disconnect data source, should alert within 60 seconds
3. Check dashboard: `http://localhost:8080` shows system status
4. Verify metrics: Performance tracking updates after each recommendation
5. Test recovery: Kill and restart, should resume without data loss

---

## Task 9: Final Integration Testing and Documentation
**Dependencies**: All previous tasks
**Estimated Time**: 2 hours

### Prompt:
```
Perform comprehensive integration testing and create operational documentation.

1. Create tests/test_end_to_end_production.py:
   - Test complete flow with real data
   - Verify all components integrate properly
   - Test failure scenarios
   - Validate recommendations make sense

2. Update documentation:
   - Create OPERATIONS_GUIDE.md with daily usage
   - Document all configuration options
   - Add troubleshooting guide
   - Include performance tuning tips

3. Add system validation:
   - Pre-flight checks before recommendations
   - Data consistency validation
   - Configuration sanity checks
   - Resource availability confirmation

4. Create scripts/daily_operations.py:
   - One command to get daily recommendations
   - Include all safety checks
   - Show clear, actionable output
   - Log everything for audit trail

5. Final optimization:
   - Profile and optimize hot paths
   - Minimize startup time
   - Reduce memory usage
   - Cache optimization

Success criteria: System provides clear, actionable recommendations considering all capital sources with full confidence in the results.
```

### Validation Steps:
1. Run full system test: `python scripts/daily_operations.py`
2. Verify all components: Health check shows 100% operational
3. Test with real portfolio: Recommendations match manual analysis
4. Check documentation: Operations guide is complete and accurate
5. Final sign-off: Run for 1 week without manual intervention

---

## Completion Checklist

After all tasks:
- [ ] Schwab OAuth works reliably
- [ ] Real positions and account data flow
- [ ] Multi-asset optimization considers loans/margin/cash
- [ ] Transaction costs factored into decisions
- [ ] Backtesting validates strategy
- [ ] Monitoring ensures reliability
- [ ] System runs autonomously
- [ ] All decisions are explainable
- [ ] Performance meets objectives

## Time Estimate

- **Core System** (Tasks 1-5): 1.5-2 weeks
- **Validation** (Task 6): 3-4 days
- **Enhancement** (Task 7): 2-3 days (optional)
- **Production** (Tasks 8-9): 3-4 days

**Total**: 3-4 weeks for complete system

## Testing Strategy

### Unit Testing Approach
```bash
# After each task, run focused tests
pytest tests/test_<module>.py -v -s

# Run with coverage
pytest tests/test_<module>.py --cov=src/unity_wheel/<module> --cov-report=html

# Property-based testing for math-heavy modules
pytest tests/test_<module>.py -k "test_property" -v
```

### Integration Testing
```bash
# After Tasks 1-5 complete
pytest tests/test_integrated_system.py -v

# After Task 6 (backtesting)
pytest tests/test_backtest_validation.py -v

# Final end-to-end
pytest tests/test_end_to_end_production.py -v
```

### Manual Testing Checklist
- [ ] Schwab auth works with real credentials
- [ ] Can fetch actual positions
- [ ] Recommendations consider all capital sources
- [ ] Transaction costs properly calculated
- [ ] Risk limits enforced correctly
- [ ] Explanations are clear and actionable

## Troubleshooting Guide

### Common Issues & Solutions

1. **Schwab OAuth Fails**
   - Check redirect URI matches exactly
   - Ensure PKCE is implemented correctly
   - Verify client_id/secret are correct
   - Check token expiration handling

2. **Databento Rate Limits**
   - Implement exponential backoff
   - Cache aggressively
   - Batch requests when possible
   - Monitor usage via their dashboard

3. **Memory Usage High**
   - Profile with `memory_profiler`
   - Check for DataFrame copies
   - Use generators for large datasets
   - Clear caches periodically

4. **Calculations Don't Match**
   - Verify market data timestamps
   - Check for timezone issues
   - Validate Greek calculations
   - Compare with known good values

5. **Performance Issues**
   - Profile with `cProfile`
   - Check for N+1 queries
   - Verify caching is working
   - Look for unnecessary loops

### Debug Commands
```bash
# Check system health
python run.py --diagnose

# Verbose logging
export LOG_LEVEL=DEBUG
python run.py --portfolio 100000

# Profile performance
python -m cProfile -o profile.stats run.py --portfolio 100000
python -m pstats profile.stats

# Check configuration
python -c "from src.config.loader import get_config_loader; loader = get_config_loader(); print(loader.get_unused_parameters())"
```

## Final Validation

Before considering the system complete:

1. **Accuracy Validation**
   - Backtest results match forward testing ±5%
   - Greeks match market values ±1%
   - Risk metrics are consistent

2. **Performance Validation**
   - Response time <200ms for recommendations
   - Memory usage <100MB typical
   - Can handle 1000 positions

3. **Reliability Validation**
   - 24 hours continuous operation
   - Handles all market conditions
   - Recovers from all failures

4. **User Experience**
   - Clear, actionable recommendations
   - Explanations make sense
   - Confidence in results
