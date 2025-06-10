> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Natural Risk-Based Position Sizing

## Philosophy: Let the Market Decide

Instead of artificial limits like "never exceed 33% per position," the system now uses natural risk-based sizing that responds to actual market conditions.

## How It Works

### Base Position: 25%
This is the starting point for "normal" Unity volatility (50-75% IV).

### Volatility Scaling

| Unity Volatility | Multiplier | Position Size | Logic |
|-----------------|------------|---------------|--------|
| <50% IV (rare) | 3.0x | 75% | Huge opportunity - Unity rarely this calm |
| 50-75% IV | 1.0x | 25% | Standard Unity behavior |
| 75-100% IV | 0.4x | 10% | Getting risky - common for Unity |
| >100% IV | 0.1x | 2.5% | Survival mode - Unity in crisis |

### Natural Limits in Action

**Scenario 1: Unity at 45% IV (unusual calm)**
- System allows up to 75% position
- But Kelly criterion might limit to 50%
- And margin requirements might limit to 60%
- Result: Natural limit emerges from risk, not arbitrary rule

**Scenario 2: Unity at 90% IV (typical volatility)**
- System scales down to 10% positions
- No need for artificial cap - risk itself limits size
- Can still trade, but safely

**Scenario 3: Unity at 120% IV (earnings/crisis)**
- System goes to 2.5% positions
- Nearly defensive, but still participating
- Natural protection without hard stop

## Benefits of This Approach

1. **Opportunistic**: Can take huge positions when genuinely safe
2. **Self-Protecting**: Automatically scales down with risk
3. **No Arbitrary Limits**: Position size matches actual risk
4. **Always Trading**: Even at 150% IV, still in the game at 2.5%

## Configuration Changes Made

```yaml
# Old (Artificial Limits)
max_position_pct: 0.33  # Never exceed 33%
volatility_factors:
  low: 1.50      # Only 50% boost
  extreme: 0.70  # Still 70% in crisis

# New (Natural Risk-Based)
max_position_pct: 1.00  # No artificial cap
volatility_factors:
  low: 3.00      # 300% when safe
  extreme: 0.10  # 10% when dangerous
```

## The Result

Your configuration now truly reflects your philosophy:
- **"No artificial limiters"** ✓
- **"100% based on risk modeling"** ✓
- **"Market situation determines size"** ✓

In practice, you'll rarely see positions above 50% due to natural risk constraints, but the system won't artificially prevent it if conditions truly warrant it.
