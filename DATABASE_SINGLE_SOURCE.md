# Database Single Source of Truth 🎯

**Last Updated**: June 14, 2025  
**Status**: ✅ CONSOLIDATED

## TL;DR

**USE THIS DATABASE FOR EVERYTHING:**
```
data/wheel_trading_optimized.duckdb
```

That's it. No other databases. No cache. No archives. Just this one.

## What's In This Database?

### 📈 Unity Stock Data (3.4 years)
- **Table**: `market.price_data`
- **Records**: 874 trading days
- **Date Range**: January 3, 2022 → June 13, 2025
- **Fields**: open, high, low, close, volume, returns, volatility

### 📊 Unity Options Data
- **Table**: `options.contracts`
- **Records**: 242+ contracts (growing daily)
- **Coverage**: All strikes within ±35% of spot price
- **Fields**: bid, ask, strike, expiration, implied_volatility

### 🤖 ML Features
- **Table**: `analytics.ml_features`
- **Records**: 30+ days
- **Includes**: VIX levels, market regime, realized volatility
- **Note**: FRED data (VIX, rates) now stored here

### 🎯 Trading Data
- **Tables**: `trading.positions`, `trading.decisions`
- **Purpose**: Track active positions and decision history

### 🎡 Wheel Opportunities
- **View**: `analytics.wheel_opportunities_mv`
- **Purpose**: Pre-filtered put options ready for wheel strategy

## What We Archived

All old databases have been moved to `data/archive/consolidated_20250614_060231/`:
- `~/.wheel_trading/cache/wheel_cache.duckdb` (contained historical data)
- `data/wheel_trading_master.duckdb` (old main database)
- `data/unified_trading.duckdb` (legacy unified attempt)

## Code Changes Made

✅ **21 files updated** to use the single database:
- All cache references removed
- All archive references updated
- All connection logic simplified

Key updates:
- `scripts/collect_eod_production.py` - Now uses main DB only
- `scripts/monitor_collection.py` - Simplified to single DB
- `src/config/schema.py` - All paths point to main DB
- `src/unity_wheel/config/unified_config.py` - Single DB path

## How to Use

### For Reading Data
```python
import duckdb
conn = duckdb.connect("data/wheel_trading_optimized.duckdb", read_only=True)

# Get Unity stock prices
df = conn.execute("""
    SELECT * FROM market.price_data 
    WHERE symbol = 'U' 
    ORDER BY date DESC
""").df()

# Get options
options = conn.execute("""
    SELECT * FROM options.contracts 
    WHERE symbol = 'U' AND expiration > CURRENT_DATE
""").df()
```

### For Writing Data
```python
# Only data collection scripts should write
conn = duckdb.connect("data/wheel_trading_optimized.duckdb")
```

## FRED Data Note

FRED economic indicators (VIX, interest rates, etc.) are now stored in:
- **Table**: `analytics.ml_features`
- **VIX Column**: `vix_level`
- **No separate FRED table needed**

## Performance

This database is optimized for M4 Pro hardware:
- Uses all 12 CPU cores
- Memory-mapped for speed
- Indexes on all critical queries
- 1.6-1.7x faster than old 3-DB setup

## Testing

Verify everything works:
```bash
# Test data integrity
python scripts/monitor_collection.py

# Run system diagnostics
python run.py --diagnose

# Collect new data
python scripts/collect_eod_production.py
```

## Important Rules

1. **NEVER** create new databases
2. **NEVER** use `~/.wheel_trading/cache/`
3. **ALWAYS** use `data/wheel_trading_optimized.duckdb`
4. **ONLY** this database has the 3.4 years of Unity data

## If You See Old References

If you find code referencing old databases:
1. Run: `python scripts/consolidate_database_references.py`
2. It will automatically fix all references
3. Old databases are safely archived

---

Remember: **One Database to Rule Them All** 💍