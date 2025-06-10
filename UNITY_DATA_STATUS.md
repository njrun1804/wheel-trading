# Unity Data Status

## Current Status (as of 2025-06-10)

### Stock Data
- **Real data**
- 861 trading days from January 3, 2022 to June 9, 2025
- Table: `price_history`

### Options Data
- **Real but incomplete**
- 206,236 tick records from March 28, 2023
- Symbol field missing for all records
- Table: `unity_options_ticks`
- Previous synthetic dataset (`databento_option_chains`) removed

### FRED Economic Data
- 9 series integrated
- 26,401 records in `fred_features`

### Database Path
`~/.wheel_trading/cache/wheel_cache.duckdb`

### Next Steps
1. Re-download options data with proper symbol mapping
2. Continue running `tools/download_unity_options_only.py`

**Bottom Line**: Stock data is complete and real. Options data is real but incomplete due to missing symbol information. No synthetic options data remains in the database.
