# Databento Historical Data Requirements

## Executive Summary

**The wheel trading system needs minimal historical data** - it's designed for real-time recommendations, not backtesting. Based on the codebase analysis:

- **Stock prices**: 250 days (daily close) for risk calculations
- **Options data**: Current chains only (no historical options needed)
- **Granularity**: Daily data sufficient, no intraday required

## Detailed Requirements

### 1. Stock Price History (Primary Need)

**What**: Daily closing prices for underlying stocks
**How Far Back**: 250 trading days (~1 year)
**Why**: 
- VaR/CVaR risk calculations (min 20, prefer 250 data points)
- Volatility calculations for risk metrics
- Sharpe ratio and performance tracking

**Storage**: ~2KB per symbol per year (minimal)

### 2. Implied Volatility History (Optional)

**What**: Daily IV for ATM options
**How Far Back**: 90 days
**Why**:
- IV rank/percentile calculations
- Volatility regime detection
- Entry timing optimization

**Storage**: ~1KB per symbol per quarter

### 3. Option Chains (Current Only)

**What**: Real-time option chains
**How Far Back**: None - current data only
**Why**:
- The system makes recommendations based on current market conditions
- No historical option prices needed for the core strategy

**Storage**: Covered by 15-minute cache TTL

## What We DON'T Need

❌ **Historical option prices** - The system doesn't backtest
❌ **Intraday data** - Daily granularity is sufficient  
❌ **Greeks history** - Calculated on demand
❌ **Trade/tick data** - Not used by the strategy
❌ **Years of history** - Maximum 1 year needed

## Implementation Plan

### For Initial Launch
```python
# Minimal viable data:
1. Current option chains (via Databento REST API)
2. 20 days of stock prices (for basic VaR)
```

### For Full Features
```python
# Complete data set:
1. 250 days of daily stock prices
2. 90 days of daily ATM IV (optional)
3. Current option chains with 15-min cache
```

## Data Fetching Strategy

### One-Time Historical Load
```python
async def load_historical_prices(symbol: str):
    """One-time load of historical prices for risk calculations."""
    
    # Fetch 250 days of daily bars
    end = datetime.now()
    start = end - timedelta(days=365)  # Get extra for holidays
    
    bars = await databento_client.get_daily_bars(
        symbol=symbol,
        start=start,
        end=end,
        dataset="XNAS.BASIC"  # Or appropriate dataset
    )
    
    # Store in DuckDB for risk calculations
    await storage.store_price_history(symbol, bars)
```

### Ongoing Updates
```python
# Daily cron job to append latest price
async def update_daily_price(symbol: str):
    """Add today's closing price to history."""
    
    latest = await databento_client.get_daily_bar(symbol)
    await storage.append_price(symbol, latest)
    
    # Trim to keep only 250 days
    await storage.trim_old_prices(symbol, days=250)
```

## Storage Estimates

For 10 symbols with full historical data:

| Data Type | Period | Size per Symbol | Total (10 symbols) |
|-----------|--------|-----------------|-------------------|
| Stock prices | 250 days | 2 KB | 20 KB |
| IV history | 90 days | 1 KB | 10 KB |
| Option chains | Current | 50 KB | 500 KB (cached) |
| **Total** | | | **~530 KB** |

This is negligible compared to the 5GB DuckDB limit.

## Code References

From the codebase:

1. **Risk Analytics** (`src/unity_wheel/risk/analytics.py`):
```python
# Line 164: Minimum data requirement
if len(returns_data) < 20:
    logger.warning("insufficient_data_for_var", 
                  data_points=len(returns_data))

# Line 196: Preferred data amount  
min_data_points = max(250, int(window * 1.5))
```

2. **Config** (`config.yaml`):
```yaml
# ML model lookback (optional feature)
volatility_model:
  lookback_days: 252  # One trading year
  
# Performance tracking
history_days: 90  # For decision tracking
```

3. **Strategy** (`src/unity_wheel/strategy/wheel.py`):
```python
# No historical data requirements
# All decisions based on current market snapshot
```

## Recommended Approach

1. **Start Simple**: Load 20 days of stock prices for basic risk metrics
2. **Add as Needed**: Expand to 250 days for full risk analytics
3. **Skip Options History**: Not needed for recommendations
4. **Use Daily Bars**: Sufficient for all calculations

## API Calls Required

### Initial Setup (per symbol)
- 1 call for historical daily bars (250 days)
- Total: 10 calls for 10 symbols

### Ongoing (per symbol per day)
- 1 call for current option chain (cached 15 min)
- 1 call for latest stock price (end of day)
- Total: ~20 calls/day for 10 symbols

This fits well within Databento's rate limits and keeps costs minimal.