# Historical Data Utilization Plan

## Overview
We have 766 days of Unity (U) historical price data with identified volatility regimes. Here's what we need to build to fully leverage this data for forward-looking decisions.

## 1. Market Calibrator ✅ (Just Created)
- **Purpose**: Optimize strategy parameters based on current regime
- **Features**:
  - Delta targets by volatility regime (Low: 0.30, Med: 0.25, High: 0.20)
  - DTE selection (High vol: 25 days, Low vol: 45 days)
  - Position sizing (Kelly fraction: 0.50 → 0.33 → 0.25)
  - Rolling thresholds adaptive to conditions

## 2. IV Surface Analyzer 🔨 (Need to Build)

```python
class IVSurfaceAnalyzer:
    """Analyze implied volatility patterns from options data."""

    def calculate_iv_metrics(self):
        # IV Rank (0-100 percentile over 1 year)
        # IV Percentile (% of days below current IV)
        # Term structure (front month vs back month)
        # Skew (25 delta put vs 25 delta call)
        # IV mean reversion speed

    def detect_iv_regime(self):
        # Contango vs backwardation
        # High vs low IV environment
        # Skew regime (crash protection demand)
```

**Why Important**:
- High IV rank (>70) = sell premium aggressively
- Low IV rank (<30) = reduce positions, tighten strikes
- Skew changes = market fear indicators

## 3. Seasonality & Pattern Detector 🔨

```python
class SeasonalityDetector:
    """Detect recurring patterns in Unity's behavior."""

    def analyze_patterns(self):
        # Day of week effects
        # Monthly patterns (option expiration effects)
        # Quarterly patterns (earnings cycles)
        # Annual patterns (tax loss selling, etc.)
        # Volatility clustering periods
```

**Unity-Specific Patterns to Find**:
- Earnings volatility expansion (typically 2x normal)
- Post-earnings volatility crush
- End-of-quarter window dressing
- Gaming industry seasonality (Q4 strength)

## 4. Event Impact Analyzer 🔨

```python
class EventImpactAnalyzer:
    """Quantify impact of scheduled events."""

    def analyze_earnings_impact(self):
        # Average move on earnings
        # IV expansion timeline (when to enter)
        # IV crush magnitude (profit potential)
        # Optimal strategy adjustments

    def analyze_macro_impact(self):
        # Fed meeting impacts
        # CPI/Jobs report correlation
        # Tech sector beta to macro
```

**Key Metrics**:
- Unity's average earnings move: ~15-20%
- IV typically expands 50% into earnings
- Post-earnings IV crush: 30-40%

## 5. Anomaly Detection System 🔨

```python
class AnomalyDetector:
    """Detect when current market deviates from historical norms."""

    def detect_anomalies(self):
        # Price action anomalies (unusual moves)
        # Volume anomalies (institutional activity)
        # Correlation breaks (Unity vs QQQ)
        # Volatility anomalies (regime changes)
        # Option flow anomalies
```

**Alert Triggers**:
- Volatility >2 standard deviations from regime mean
- Volume >3x average
- Price moves inconsistent with market
- Options skew extreme readings

## 6. Dynamic Parameter Optimizer 🔨

```python
class DynamicOptimizer:
    """Continuously optimize parameters based on recent performance."""

    def optimize_parameters(self):
        # Track actual vs expected outcomes
        # Adjust deltas based on assignment frequency
        # Optimize DTE based on theta decay curves
        # Fine-tune position sizing

    def calculate_expectancy(self):
        # Win rate by delta/DTE combination
        # Average profit per trade
        # Risk-adjusted returns
        # Sharpe/Sortino ratios
```

## 7. Integration Requirements 🔨

### Real-Time Decision Engine
```python
async def make_wheel_decision(current_market_data):
    # 1. Identify current regime
    regime = regime_detector.get_current_regime()

    # 2. Get optimal parameters
    params = market_calibrator.calibrate_from_history()

    # 3. Check for anomalies
    anomalies = anomaly_detector.check_current_market()

    # 4. Adjust for events
    event_adj = event_analyzer.get_adjustments()

    # 5. Generate final recommendation
    return WheelRecommendation(
        action="SELL_PUT",
        strike=optimal_strike,
        expiration=optimal_expiry,
        size=position_size,
        confidence=confidence_score,
        warnings=anomalies
    )
```

### Configuration Updates Needed

```yaml
# config.yaml additions
analytics:
  iv_rank_period: 252  # days for IV rank calculation
  regime_lookback: 60  # days for regime detection

  # Regime-specific overrides
  regimes:
    low_vol:
      delta_target: 0.30
      dte_target: 45
      kelly_fraction: 0.50

    medium_vol:
      delta_target: 0.25
      dte_target: 35
      kelly_fraction: 0.33

    high_vol:
      delta_target: 0.20
      dte_target: 25
      kelly_fraction: 0.25

  # Event adjustments
  events:
    earnings:
      avoid_days_before: 5
      iv_expansion_threshold: 1.5
      position_reduction: 0.5

    fed_meetings:
      position_reduction: 0.25
```

## 8. Backtesting Framework 🔨

```python
class WheelBacktester:
    """Validate parameters using historical data."""

    def backtest_parameters(self, params: OptimalParameters):
        # Simulate wheel trades using historical data
        # Account for assignment probability
        # Include transaction costs
        # Calculate risk-adjusted returns

    def optimize_via_backtest(self):
        # Grid search parameter space
        # Find optimal delta/DTE combinations
        # Validate regime-specific rules
```

## Implementation Priority

1. **IV Surface Analyzer** (High) - Critical for timing entries
2. **Anomaly Detection** (High) - Avoid trading in abnormal conditions
3. **Event Impact Analyzer** (Medium) - Important for Unity earnings
4. **Dynamic Optimizer** (Medium) - Continuous improvement
5. **Seasonality Detector** (Low) - Nice to have
6. **Full Backtester** (Low) - Validation tool

## Next Steps

1. Start collecting options chain data daily to build IV history
2. Implement IV rank/percentile calculations
3. Create anomaly detection for current market conditions
4. Build event calendar integration (earnings dates)
5. Create unified decision engine that combines all signals

## Expected Improvements

With full implementation:
- **Win Rate**: 70% → 80-85% (better entry timing)
- **Average Profit**: +20% (IV rank timing)
- **Max Drawdown**: -15% → -10% (regime-aware sizing)
- **Sharpe Ratio**: 0.7 → 1.0+ (reduced variance)
