# Unity Options Import Complete ✅

## Summary

Successfully imported **1,040,215 Unity option records** spanning from March 28, 2023 to June 6, 2025 into the unified DuckDB storage.

## What Was Done

1. **Fixed decompression issue** - The zstandard files required streaming decompression (not direct decompression)
2. **Imported all Unity options** - Processed 28 monthly files containing 3 years of daily OHLCV data
3. **Created unified storage** - Data now available in `data/cache/wheel_cache.duckdb`
4. **Integrated with existing system** - Can be queried alongside FRED data and other market data

## Database Details

- **Table**: `unity_options_ohlcv`
- **Records**: 1,040,215
- **Unique symbols**: 9,421
- **Date range**: 2023-03-28 to 2025-06-06
- **File size**: 8.9 MB (parquet), ~50MB in DuckDB

## Sample Queries

### Find recent liquid Unity puts for wheel strategy
```sql
SELECT
    symbol,
    ts_event as date,
    CAST(SUBSTRING(symbol, 14, 8) AS FLOAT) / 1000 as strike,
    close as premium,
    volume
FROM unity_options_ohlcv
WHERE SUBSTRING(symbol, 13, 1) = 'P'  -- Puts only
    AND ts_event = (SELECT MAX(ts_event) FROM unity_options_ohlcv)
    AND CAST(SUBSTRING(symbol, 14, 8) AS FLOAT) / 1000 BETWEEN 30 AND 40
    AND volume > 0
ORDER BY volume DESC;
```

### Calculate annualized returns for puts
```sql
WITH parsed_options AS (
    SELECT
        symbol,
        ts_event,
        CAST(SUBSTRING(symbol, 2, 6) AS DATE FORMAT 'YYMMDD') as expiration,
        SUBSTRING(symbol, 13, 1) as option_type,
        CAST(SUBSTRING(symbol, 14, 8) AS FLOAT) / 1000 as strike,
        close as premium,
        volume
    FROM unity_options_ohlcv
)
SELECT
    symbol,
    strike,
    premium,
    ROUND(premium / strike * 365 / (expiration - ts_event), 3) as annualized_return,
    volume
FROM parsed_options
WHERE option_type = 'P'
    AND ts_event = (SELECT MAX(ts_event) FROM unity_options_ohlcv)
    AND strike BETWEEN 30 AND 40
    AND volume > 100
ORDER BY annualized_return DESC;
```

### Historical volatility analysis
```sql
-- Calculate historical put skew
WITH daily_skew AS (
    SELECT
        ts_event,
        MAX(CASE WHEN CAST(SUBSTRING(symbol, 14, 8) AS FLOAT) / 1000 = 30 THEN close END) /
        MAX(CASE WHEN CAST(SUBSTRING(symbol, 14, 8) AS FLOAT) / 1000 = 35 THEN close END) as skew_30_35
    FROM unity_options_ohlcv
    WHERE SUBSTRING(symbol, 13, 1) = 'P'
    GROUP BY ts_event
    HAVING skew_30_35 IS NOT NULL
)
SELECT
    ts_event,
    ROUND(skew_30_35, 3) as put_skew
FROM daily_skew
WHERE ts_event >= CURRENT_DATE - INTERVAL 30 DAY
ORDER BY ts_event DESC;
```

## Integration with Wheel Trading System

The data is now available for:
1. **Backtesting** - Historical option prices for strategy validation
2. **Risk Analysis** - Calculate historical volatility and correlations
3. **Liquidity Analysis** - Identify most liquid strikes and expirations
4. **Performance Tracking** - Compare actual vs predicted option prices

## Next Steps

1. **Run backtests** using historical data
2. **Analyze liquidity patterns** to refine strike selection
3. **Calculate implied volatility** surfaces
4. **Validate wheel strategy** parameters against historical performance

## Files Created
- `unity_options_etl.py` - ETL script (can be rerun anytime)
- `load_unity_to_duckdb.py` - DuckDB loader
- `data/unity-options/processed/unity_ohlcv_3y.parquet` - Processed data
- `data/cache/wheel_cache.duckdb` - Unified database with all data

The Unity options data is now fully integrated and ready for use! 🎉
