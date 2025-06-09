# Unity Wheel Trading Bot - Remaining Work Assessment

> **Note**: For detailed implementation steps and exact prompts for each task, see [SEQUENTIAL_IMPLEMENTATION_GUIDE.md](./SEQUENTIAL_IMPLEMENTATION_GUIDE.md)

## Executive Summary

The Unity Wheel Trading Bot has solid foundations but lacks critical components for production use. The core wheel strategy logic, risk analytics, and adaptive configuration are well-implemented. However, the system cannot currently provide real recommendations due to missing broker authentication, reliance on mock data, and absence of multi-asset optimization logic.

Most critically, the current implementation focuses solely on Unity wheel strategy without considering the broader capital allocation decisions mentioned in your goals (AMEX loan, Schwab margin, cash allocation, tax considerations).

## Gap Analysis: Intended vs Actual

### ✅ What's Been Built Well
1. **Core Wheel Strategy** - Robust implementation with validation
2. **Risk Analytics** - Comprehensive VaR, CVaR, Kelly sizing
3. **Options Math** - Black-Scholes, Greeks with confidence scores
4. **Adaptive Configuration** - Market-aware parameter adjustment
5. **Databento Integration** - Ready for options data
6. **Framework Architecture** - Clean, extensible design

### ❌ Critical Missing Pieces
1. **Schwab OAuth Flow** - Cannot get real positions/account data
2. **Multi-Asset Optimization** - No consideration of loans, cash, margin
3. **Real Data Pipeline** - System uses mock data
4. **Tax Considerations** - Not implemented
5. **Cost of Position Changes** - Not factored into decisions
6. **ML Implementation** - Config exists but no actual models
7. **Backtesting Engine** - Framework only, not functional

### ⚠️ Partially Complete
1. **Decision Engine** - Basic version works, advanced version disconnected
2. **CLI Interface** - Works but with mock data
3. **Performance Tracking** - Framework exists, needs real data

## Remaining Work - Sequential Order

### Phase 0: Critical Infrastructure [1-2 days]
**Purpose**: Enable real data flow

1. **Implement Schwab OAuth Flow**
   - Complete OAuth2 authentication in `schwab/client.py`
   - Add token refresh logic
   - Create setup script for initial auth
   - Test with real account data retrieval

2. **Connect Real Data Sources**
   - Wire Databento to decision engine
   - Implement real-time option chain fetching
   - Add market data caching layer
   - Validate data quality checks work

### Phase 1: Multi-Asset Optimization [3-4 days]
**Purpose**: Address your actual goals - optimal capital allocation

3. **Create Capital Allocation Framework**
   ```python
   class CapitalAllocation:
       - Track AMEX loan ($45k @ 7%)
       - Track Schwab margin availability/cost
       - Model cash opportunity cost
       - Consider tax implications (even if tax-free)
   ```

4. **Implement Portfolio-Level Optimizer**
   - Objective: Maximize CAGR - 0.20×|CVaR₉₅| across ALL assets
   - Decisions:
     * Pay down AMEX loan vs invest
     * Use Schwab margin vs cash
     * Allocate between Unity positions and cash
     * Consider transaction costs

5. **Add Cost-of-Change Analysis**
   - Model bid-ask spreads for position changes
   - Factor assignment/exercise costs
   - Include early close penalties
   - Optimize for net benefit after costs

### Phase 2: Complete Decision Engine [2-3 days]
**Purpose**: Generate actionable recommendations

6. **Integrate Multi-Asset Logic**
   - Enhance `WheelAdvisor` with capital allocation
   - Add loan paydown recommendations
   - Include margin usage decisions
   - Factor in opportunity costs

7. **Connect All Data Sources**
   - Real positions from Schwab
   - Live options chains from Databento
   - Current margin rates
   - Market conditions for adaptation

8. **Implement Recommendation Ranking**
   - Score all possible actions (including "do nothing")
   - Consider risk budget across entire portfolio
   - Provide clear explanations
   - Show confidence levels

### Phase 3: Backtesting & Validation [2-3 days]
**Purpose**: Validate strategy before live use

9. **Complete Backtesting Engine**
   - Load historical data from Databento
   - Simulate wheel strategy with real spreads
   - Model assignment probabilities correctly
   - Track all costs and slippage

10. **Add Walk-Forward Optimization**
    - Test parameter stability
    - Validate out-of-sample performance
    - Detect regime changes
    - Optimize for your objective function

11. **Performance Attribution**
    - Separate alpha from market beta
    - Measure strategy effectiveness
    - Validate risk models
    - Track prediction accuracy

### Phase 4: ML Enhancement [Optional - 2-3 days]
**Purpose**: Improve edge if analytical approach insufficient

12. **Feature Engineering**
    - Extract microstructure features
    - Calculate regime indicators
    - Build volatility predictors
    - Create assignment probability model

13. **Train Enhancement Models**
    - PoP adjustment model
    - Volatility regime classifier
    - Edge prediction model
    - Only deploy if backtested improvement

### Phase 5: Production Hardening [1-2 days]
**Purpose**: Make system reliable for daily use

14. **Complete Monitoring**
    - Real-time health checks
    - Performance tracking
    - Anomaly detection
    - Alert on degradation

15. **Add Safety Features**
    - Circuit breakers for market stress
    - Sanity checks on all recommendations
    - Automatic fallback modes
    - Clear error reporting

## Recommended Implementation Order

Given your specific goals, I recommend this priority order:

1. **Week 1**: Schwab OAuth + Multi-Asset Framework
   - Without real data, nothing else matters
   - Multi-asset optimization is core to your goals

2. **Week 2**: Complete Decision Engine + Backtesting
   - Validate strategy before risking capital
   - Ensure recommendations are sound

3. **Week 3**: Production Hardening + Monitoring
   - Make system reliable for daily use
   - Add safety nets

4. **Optional**: ML Enhancement
   - Only if backtesting shows need
   - Keep it simple initially

## Key Design Changes Needed

1. **Expand Beyond Unity-Only**
   - Current system assumes only Unity positions
   - Need portfolio-wide view
   - Consider all capital sources

2. **Add Loan/Margin Logic**
   - Model cost of capital properly
   - Compare investment returns vs debt service
   - Optimize total portfolio return

3. **Include Transaction Costs**
   - Current system ignores execution costs
   - Need realistic cost modeling
   - Optimize for net benefit

## Success Metrics

System is complete when it can:
1. Connect to real Schwab account
2. Fetch current positions and balances
3. Consider AMEX loan in decisions
4. Evaluate Schwab margin usage
5. Recommend optimal actions across all capital
6. Factor in all costs of changes
7. Provide clear explanations
8. Track performance vs predictions
9. Run autonomously with monitoring

## Estimated Timeline

- **Minimum Viable**: 2 weeks (Phases 0-2)
- **Production Ready**: 3 weeks (add Phase 3 & 5)
- **Full Featured**: 4 weeks (include ML)

## Next Immediate Steps

1. Implement Schwab OAuth to get real data
2. Create `CapitalAllocation` class for multi-asset tracking
3. Extend decision engine beyond Unity-only focus
4. Add cost-of-change calculations
5. Wire everything together with real data

The foundation is solid, but the system needs to expand beyond pure Unity wheel strategy to address your actual capital allocation goals.

## Implementation Guide

I've created a detailed [SEQUENTIAL_IMPLEMENTATION_GUIDE.md](./SEQUENTIAL_IMPLEMENTATION_GUIDE.md) with:
- 9 sequential tasks for full completion
- Exact prompts to give Claude Code for each task
- Dependencies clearly marked
- Time estimates for planning

Each prompt is self-contained and designed to be given to a fresh Claude Code instance after your standard preamble.
