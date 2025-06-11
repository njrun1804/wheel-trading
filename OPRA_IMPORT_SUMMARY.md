# OPRA Data Import Summary

## What Was Created

1. **`import_opra_to_unified_db.py`** - Main import script that:
   - Imports OPRA options data into unified DuckDB storage
   - Decompresses `.zst` files automatically
   - Parses OPRA symbol format to extract strike, expiration, type
   - Filters to only import Unity (U) options
   - Creates unified `options_data` table for all options

2. **`verify_opra_import.py`** - Verification script that:
   - Checks if import was successful
   - Shows summary statistics
   - Provides useful SQL queries for analysis
   - Lists most liquid strikes

3. **`docs/OPRA_IMPORT_GUIDE.md`** - Complete documentation covering:
   - Import process and prerequisites
   - Database schema
   - Integration with existing system
   - Query examples
   - Troubleshooting

## Database Structure

The import creates/uses these tables in the unified DuckDB:
- `options_data` - Unified table for all options OHLCV data
- `opra_symbology` - Maps instrument IDs to option symbols

This integrates with existing tables:
- `databento_option_chains` - Real-time options chains
- `fred_observations` - Economic data
- `position_snapshots` - Account positions

## Quick Start

```bash
# Run the import
python import_opra_to_unified_db.py

# Verify it worked
python verify_opra_import.py

# Query the data
duckdb data/cache/wheel_cache.duckdb
```

## Key Features

- **Unified Storage**: All data (stocks, options, FRED) in one DuckDB instance
- **Automatic Compression**: Uses zstandard decompression
- **Smart Filtering**: Only imports Unity options to save space
- **Performance Indexes**: Created on common query patterns
- **Integration Ready**: Works with existing WheelStrategy and analytics

## Next Steps

1. Run the import: `python import_opra_to_unified_db.py`
2. The import will process all files and show progress
3. Use `verify_opra_import.py` to check results
4. Query historical options data for backtesting and analysis

The imported data is now available to all components of the wheel trading system through the unified Storage class.
