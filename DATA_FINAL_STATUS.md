# Unity Data Final Status

## What You Have

### ✅ Stock Data (COMPLETE)
- **861 days** of Unity daily stock data (2022-2025)
- Table: `price_history`
- Ready to use for backtesting

### ❌ Options Data (ISSUE)
- Databento API is not returning Unity options data
- Tried multiple symbol formats: `U.OPT`, `U`, specific contracts
- Getting "No data found" or "Invalid symbol" errors

## Possible Reasons

1. **Subscription Issue**: Your Databento subscription might not include Unity options
2. **Symbol Format**: Unity might use a non-standard symbol format in OPRA
3. **Data Availability**: Unity options might not be available in the OPRA.PILLAR dataset

## What You Can Do

### Option 1: Contact Databento Support
Ask them:
- "How do I access Unity Software (U) options data?"
- "What symbol format should I use for Unity options?"
- "Is Unity included in my subscription?"

### Option 2: Use Stock Data Only
You have complete Unity stock data for strategy development:
```python
# Access Unity stock data
import duckdb
conn = duckdb.connect("~/.wheel_trading/cache/wheel_cache.duckdb")
df = conn.execute("SELECT * FROM price_history WHERE symbol='U'").df()
```

### Option 3: Try Alternative Data Sources
- Interactive Brokers API
- Yahoo Finance (limited options data)
- Other options data providers

## Summary
- ✅ **Stock data**: 861 days ready to use
- ❌ **Options data**: Unable to download from Databento
- 🔒 **No synthetic data**: All existing data is real
