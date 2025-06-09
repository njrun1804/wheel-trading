# Unity (U) Data Requirements - Final Summary

## Critical Clarification: NO BACKTESTING

**This system does NOT backtest strategies.** It makes real-time recommendations only.

The 250 days of historical data is ONLY for risk calculations:
- **VaR**: "What's my worst expected daily loss?"
- **Position Sizing**: "How much should I risk?"
- **Volatility**: "How wild is Unity's price movement?"

## For Unity (U) ONLY

### Historical Data Required

**Stock Prices**
- Symbol: U (Unity Software)
- Period: 250 trading days
- Type: Daily OHLC bars
- API Calls: 1 (one-time setup)
- Storage: 2KB
- Purpose: Calculate daily returns for risk metrics

**Options Data**
- Historical: NONE
- Current: Yes (15-minute cache)
- Purpose: Find 30-delta puts to sell

### Exact API Usage

**One-Time Setup**
```python
# 1 API call total
GET /timeseries.get_range
  dataset: "XASE.BASIC"  # Unity trades on NYSE American
  schema: OHLC_1D
  symbols: ["U"]
  start: 2024-06-09  # 250 trading days ago
  end: 2025-06-08
```

**Daily Operations**
```python
# When user requests recommendation:
GET /timeseries.get_range
  dataset: "OPRA.PILLAR"
  schema: DEFINITION + MBP_1
  symbols: ["U.OPT"]
  -> Cached for 15 minutes

# End of day (cron job):
GET /timeseries.get_range
  dataset: "XASE.BASIC"
  schema: OHLC_1D
  symbols: ["U"]
  -> Just yesterday's bar
```

### Storage Breakdown

```
~/.wheel_trading/cache/wheel_cache.duckdb

Tables:
- price_history: 250 rows × 8 columns = 2KB
- option_chains: ~50 rows (current only) = 50KB
- wheel_candidates: ~10 rows = 1KB

Total: ~53KB for Unity (0.001% of 5GB limit)
```

### Why 250 Days is Sufficient

The math shows:
- **VaR**: Need 12+ observations at 5th percentile ✓
- **Volatility**: ±8.8% estimation error (good enough) ✓
- **Kelly**: ~12 monthly periods for win rate stats ✓

With less data:
- 20 days: ±31% volatility error ❌
- 50 days: ±20% volatility error ⚠️
- 100 days: ±14% volatility error ⚠️
- 250 days: ±8.8% volatility error ✓

### Monthly Costs

For Unity only:
- Historical data load: $0.01 (one-time)
- Daily updates: ~$0.10/month
- Option chains: ~$0.20/month (with caching)
- **Total: <$0.50/month**

### Why Not 10 Symbols?

You're right - we're only trading Unity:
- No need for SPY, QQQ, etc.
- No diversification in wheel strategy
- Focused approach on one underlying
- Keeps it simple and cheap

## The Bottom Line

**For Unity wheel trading:**
1. Load 250 days of daily prices (2KB) - ONE TIME
2. Fetch current option chain when needed (cached 15 min)
3. Append yesterday's price each day

**NOT needed:**
- Historical option prices
- Intraday data
- Other symbols
- Backtesting infrastructure

This gives us everything needed for safe position sizing while keeping costs under $1/month.
