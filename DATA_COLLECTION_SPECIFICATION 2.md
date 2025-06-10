# Unity Data Collection Specification

## Overview
This document specifies exactly what data we are collecting for Unity (U) and where it is being stored.

## Stock Data Collection

### Table: `price_history`
- **Status**: Existing table with 513 days of data (May 2023 - June 2025)
- **New Data Needed**: January 1, 2022 - May 2023 (gap filling)

### Data Fields:
| Field | Type | Description |
|-------|------|-------------|
| symbol | VARCHAR | Always "U" for Unity |
| date | DATE | Trading date |
| open | DECIMAL(10,2) | Opening price |
| high | DECIMAL(10,2) | Daily high |
| low | DECIMAL(10,2) | Daily low |
| close | DECIMAL(10,2) | Closing price |
| volume | BIGINT | Daily volume |
| created_at | TIMESTAMP | When record was inserted |

### Source:
- **Dataset**: XNYS.PILLAR (Unity trades on NYSE)
- **Schema**: ohlcv-1d (daily bars)
- **Date Range**: January 1, 2022 to June 10, 2025
- **Expected Records**: ~861 trading days total

## Options Data Collection

### Table: `databento_option_chains`
- **Status**: Table will be created if not exists
- **Data Needed**: ALL historical options data

### Data Fields:
| Field | Type | Description |
|-------|------|-------------|
| symbol | VARCHAR | Always "U" for Unity |
| expiration | DATE | Option expiration date |
| strike | DECIMAL(10,2) | Strike price |
| option_type | VARCHAR(4) | "PUT" or "CALL" |
| bid | DECIMAL(10,4) | Bid price |
| ask | DECIMAL(10,4) | Ask price |
| mid | DECIMAL(10,4) | Mid price (calculated) |
| volume | INTEGER | Daily volume |
| open_interest | INTEGER | Open interest |
| implied_volatility | DECIMAL(6,4) | IV if available |
| delta | DECIMAL(5,4) | Delta Greek |
| gamma | DECIMAL(5,4) | Gamma Greek |
| theta | DECIMAL(5,4) | Theta Greek |
| vega | DECIMAL(5,4) | Vega Greek |
| rho | DECIMAL(5,4) | Rho Greek |
| timestamp | TIMESTAMP | When quote was captured |
| spot_price | DECIMAL(10,2) | Unity price at timestamp |
| moneyness | DECIMAL(5,4) | Strike/Spot ratio |
| created_at | TIMESTAMP | When record was inserted |

### Strike Selection Logic:
For each trading day:
1. Get Unity's closing price
2. Calculate strike range: 70% to 130% of closing price
3. Round to $2.50 intervals (Unity's standard)
4. Store ALL strikes in this range

#### Examples:
- Unity at $25: Strikes from $17.50 to $32.50 (7 strikes)
- Unity at $35: Strikes from $25.00 to $45.00 (9 strikes)
- Unity at $50: Strikes from $35.00 to $65.00 (13 strikes)

### Expiration Selection:
- **Type**: Monthly expirations only (3rd Friday of each month)
- **DTE Filter**: 21-49 days to expiration from each trading day
- **Result**: Typically 2-3 active expirations per day

### Source:
- **Dataset**: OPRA.PILLAR
- **Schema**: definition (for metadata) + bbo-1s or trades (for quotes)
- **Date Range**: January 1, 2023 to June 10, 2025
- **Symbols**: U.OPT (parent format) or specific option symbols
- **Expected Records**: ~10,000+ option quotes

## Data Collection Process

### Phase 1: Stock Data
1. Query existing `price_history` table to find gaps
2. Fetch missing data from Databento (2022-2023)
3. Insert only new records (no duplicates)

### Phase 2: Options Data
1. For each trading day in stock data:
   - Calculate dynamic strike range (70-130% of close)
   - Find valid expirations (21-49 DTE, monthly only)
   - Fetch option chains for each expiration
   - Filter to relevant strikes
   - Store in `databento_option_chains`

### Storage Optimization:
- **Moneyness Filter**: Storage adapter filters to 65-135% range
- **Our Collection**: 70-130% range (fits within storage range)
- **Result**: All collected data will be stored

## Data Volume Estimates

### Stock Data:
- 3.5 years × 252 trading days/year = ~861 records
- Size: ~36 KB

### Options Data:
- 2.5 years × 252 days × ~7 strikes × 3 expirations = ~13,230 records
- Size: ~1-2 MB

### Total Storage: <2 MB for complete dataset

## Why This Range?

The 70-130% strike range covers:
- **70-100%**: PUT strikes for wheel strategy entry (selling cash-secured puts)
- **100-130%**: CALL strikes for exit after assignment (selling covered calls)

This ensures complete coverage for the entire wheel strategy cycle.

## Current Status

As of June 10, 2025:
- ✅ Stock data: 513 days exist (May 2023 - June 2025)
- ❌ Stock data: ~348 days missing (Jan 2022 - May 2023)
- ❌ Options data: No historical options data exists
- 📊 Collection script: `tools/collect_all_unity_data.py` ready to run
