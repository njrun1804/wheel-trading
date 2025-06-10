> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Unity Wheel Adaptive System Design

## Overview

The adaptive system is specifically designed for Unity (U) wheel strategy recommendations, focusing on factors that actually matter for this single underlying tech stock.

## Key Adaptive Factors

### 1. Unity-Specific Volatility
- **Normal Range**: 40-60% annualized
- **High**: 60-80%
- **Extreme**: >80% (earnings, major announcements)
- **Position Sizing**: Inverse relationship with volatility

### 2. Drawdown-Based Position Sizing
- **0-5% drawdown**: Normal sizing
- **5-10% drawdown**: 75% of normal
- **10-15% drawdown**: 50% of normal
- **15-20% drawdown**: 25% of normal
- **>20% drawdown**: Stop trading

### 3. Earnings Awareness
- **>45 days**: Normal trading
- **8-45 days**: Adjust DTE to expire before earnings
- **<7 days**: Skip cycle or 50% position size
- **Earnings week**: Pause trading

### 4. IV Rank Adjustments
- **>80**: Increase size by 20% (great premiums)
- **50-80**: Normal sizing
- **<50**: Reduce size by 20% (poor premiums)

### 5. Market Regime
- **Normal**: 100% of calculated size
- **Volatile**: 80%
- **Stressed**: 60%
- **Crisis**: 40%

## Unity Wheel Parameters

### Put Delta Selection
```
Base: 0.30 (30 delta)
Adjustments:
- High IV skew (>5%): +0.05 (sell higher delta)
- Low IV skew (<-5%): -0.05 (sell lower delta)
- High assignment risk: -0.05 (more conservative)
Range: 0.15 to 0.40
```

### DTE Selection
```
Base: 35 days
Adjustments:
- High vol (>60%): -7 days (capture faster decay)
- Low vol (<30%): +7 days (need more time)
- Earnings adjustment: Expire 7+ days before
Range: 21 to 49 days
```

### Roll Triggers
- **Normal Volatility**:
  - Profit target: 50%
  - Loss trigger: -3x credit received

- **High Volatility**:
  - Profit target: 25% (take profits quickly)
  - Loss trigger: -2x credit (defensive rolls)

## Implementation Simplicity

The system avoids complex features that add little value:
- ❌ Multi-asset correlations (just Unity vs QQQ)
- ❌ Microstructure analysis
- ❌ Complex ML models
- ❌ Excessive statistical measures

Instead focuses on:
- ✅ Unity's actual volatility patterns
- ✅ Simple drawdown management
- ✅ Earnings cycle awareness
- ✅ Clear position sizing rules

## Kelly Sizing for Unity Wheel

Simplified Kelly calculation based on Unity wheel empirics:
- Win rate: ~70% at 30 delta
- Average win: 3% of position
- Average loss: 10% when assigned
- Half-Kelly applied for safety
- Maximum: 25% of portfolio

## Trading Pause Conditions

Automatic trading pause when:
1. Earnings week (<7 days)
2. Extreme volatility (>100%)
3. Deep drawdown (>20%)
4. Crisis regime detected
5. System confidence <30%

## Example Scenarios

### Scenario 1: Normal Unity Conditions
- Unity vol: 50%
- IV rank: 60
- Drawdown: -5%
- Days to earnings: 45
- **Result**: ~18% position size, 30 delta, 35 DTE

### Scenario 2: High Vol Tech Selloff
- Unity vol: 85%
- IV rank: 88
- Drawdown: -15%
- Correlation to QQQ: 0.90
- **Result**: ~8% position size, 25 delta, 28 DTE

### Scenario 3: Post-Earnings Vol Crush
- Unity vol: 40%
- IV rank: 20
- Drawdown: -8%
- Days to earnings: 85
- **Result**: ~14% position size, 28 delta, 42 DTE

## Benefits

1. **Focused**: Only adapts what matters for Unity
2. **Simple**: Clear rules, easy to validate
3. **Robust**: Drawdown protection built-in
4. **Practical**: Based on Unity's actual behavior
5. **Testable**: All logic is deterministic

## Integration Points

1. **Market Data**: Unity price, volatility, IV rank
2. **Portfolio**: Current drawdown, P&L history
3. **Calendar**: Earnings dates
4. **Risk System**: Position limits, regime detection

This focused approach ensures the adaptive system adds value without unnecessary complexity, perfectly aligned with the goal of maximizing CAGR - 0.20 × |CVaR₉₅| for Unity wheel strategies.
