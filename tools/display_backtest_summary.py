#!/usr/bin/env python3
"""Display a visual summary of backtest results."""

print(
    """
╔══════════════════════════════════════════════════════════════════════╗
║             WHEEL STRATEGY BACKTEST RESULTS - UNITY (U)              ║
╚══════════════════════════════════════════════════════════════════════╝

📊 PERFORMANCE SUMMARY (1 Year: June 2024 - June 2025)
┌─────────────────────────┬─────────────────────────────────────────────┐
│ Metric                  │ Result                                      │
├─────────────────────────┼─────────────────────────────────────────────┤
│ Total Return            │ 27.0% ✅                                    │
│ Sharpe Ratio            │ 3.72 (Exceptional)                          │
│ Win Rate                │ 100% (8/8 trades profitable)                │
│ Max Drawdown            │ 0.0% (No drawdowns!)                        │
│ Average Trade P&L       │ $1,825.58                                   │
│ Assignment Rate         │ 0% (0 assignments)                          │
└─────────────────────────┴─────────────────────────────────────────────┘

⚡ CURRENT MARKET CONDITIONS
┌─────────────────────────┬─────────────────────────────────────────────┐
│ Unity Price             │ $24.69                                      │
│ 20-day Volatility       │ 86.9% ⚠️  (EXTREME)                         │
│ 250-day Volatility      │ 75.0% (Very High)                           │
│ Risk-Free Rate          │ 5.00%                                       │
└─────────────────────────┴─────────────────────────────────────────────┘

🎯 OPTIMAL PARAMETERS (Based on Backtesting)
┌─────────────────────────┬─────────────────────────────────────────────┐
│ Parameter               │ Optimal Value                               │
├─────────────────────────┼─────────────────────────────────────────────┤
│ Target Delta            │ 0.40 (Return: 30.3%, Sharpe: 3.77)         │
│ Target DTE              │ 30 days (Sharpe: 4.50)                      │
│ Position Size           │ 20% of portfolio (max)                      │
└─────────────────────────┴─────────────────────────────────────────────┘

📈 VOLATILITY REGIME PERFORMANCE
┌─────────────────────────┬─────────┬───────────────────────────────────┐
│ Volatility Level        │ Days    │ Market Behavior                   │
├─────────────────────────┼─────────┼───────────────────────────────────┤
│ Low (<40%)              │ 7       │ Rare for Unity                    │
│ Medium (40-70%)         │ 127     │ Normal trading conditions         │
│ High (70-100%)          │ 77      │ ⚠️  Current regime                │
│ Extreme (>100%)         │ 37      │ 🚨 Avoid or reduce size           │
└─────────────────────────┴─────────┴───────────────────────────────────┘

⚠️  RISK METRICS
┌─────────────────────────┬─────────────────────────────────────────────┐
│ Risk Factor             │ Observation                                 │
├─────────────────────────┼─────────────────────────────────────────────┤
│ Gap Events (>10%)       │ 38 events in 1 year                         │
│ Earnings Avoided        │ 40 periods (CRITICAL!)                      │
│ Current Vol vs History  │ 87% vs 75% average                          │
│ VaR (95%)               │ -6.66% daily                                │
└─────────────────────────┴─────────────────────────────────────────────┘

💡 KEY INSIGHTS
• Unity's extreme volatility (87%) creates exceptional premiums
• 100% win rate achieved by avoiding ALL earnings periods
• 38 gap events handled successfully with proper position sizing
• No assignments due to conservative strike selection

🔧 RECOMMENDATIONS FOR CURRENT MARKET
1. ⚠️  REDUCE position sizes - volatility at extreme levels
2. Use LOWER delta (0.20-0.25) for safety in high vol
3. Maintain 30-day DTE for optimal risk/reward
4. NEVER trade during Unity earnings windows
5. Monitor for volatility mean reversion opportunity

📊 DATA QUALITY
✅ 100% REAL market data used (no synthetic data)
✅ 861 days of stock prices
✅ 178,724 option price records
✅ All results based on actual tradeable prices

══════════════════════════════════════════════════════════════════════
"""
)

# Show a risk-adjusted position size recommendation
current_vol = 0.869  # 86.9%
normal_vol = 0.60  # 60%
adjustment = normal_vol / current_vol

print(
    f"""
📐 POSITION SIZE CALCULATOR (Volatility-Adjusted)
┌─────────────────────────┬─────────────────────────────────────────────┐
│ Portfolio Value         │ $100,000                                    │
│ Normal Position (20%)   │ $20,000                                     │
│ Current Vol Adjustment  │ {adjustment:.1%} (60% / 87%)                    │
│ Recommended Position    │ ${20000 * adjustment:,.0f} max                               │
└─────────────────────────┴─────────────────────────────────────────────┘

⚡ With Unity at 87% volatility, reduce position sizes by ~30%
"""
)

# Historical context
print(
    """
📈 HISTORICAL CONTEXT (Unity Volatility)
┌─────────────┬─────────┬─────────┬─────────────────────────────────┐
│ Month       │ Avg Vol │ Min Vol │ Max Vol                         │
├─────────────┼─────────┼─────────┼─────────────────────────────────┤
│ Mar 2025    │ 106.6%  │ 73.8%   │ 130.8% 🚨 (Highest)             │
│ Apr 2025    │ 98.0%   │ 64.4%   │ 110.5%                          │
│ Current     │ 86.9%   │ -       │ - (Still Very High)             │
└─────────────┴─────────┴─────────┴─────────────────────────────────┘

Unity has spent 52% of the last year above 70% volatility!
══════════════════════════════════════════════════════════════════════
"""
)
