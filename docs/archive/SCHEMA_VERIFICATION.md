# Schema Verification Report

## Overview

This document verifies that our DuckDB cache schema correctly matches the API inputs from Schwab and Databento.

## Schwab API → DuckDB Schema Mapping

### Positions Data

**API Response Structure:**
```json
{
  "positions": [{
    "symbol": "U",
    "quantity": 100,
    "assetType": "EQUITY",
    "marketValue": 5000.00,
    "averagePrice": 48.50,
    "unrealizedPnL": 150.00,
    "realizedPnL": 0.00
  }]
}
```

**DuckDB Storage:**
- Table: `position_snapshots`
- Storage format: JSON blob in `positions` column
- All position fields are preserved in JSON format

**Verification:** ✅ Complete match - all API fields are stored

### Account Data

**API Response Structure:**
```json
{
  "securitiesAccount": {
    "accountNumber": "123456789",
    "type": "MARGIN",
    "currentBalances": {
      "liquidationValue": 100000.00,
      "cashBalance": 20000.00,
      "buyingPower": 40000.00,
      "marginBalance": 0.00,
      "maintenanceRequirement": 25000.00,
      "maintenanceCall": 0.00
    }
  }
}
```

**DuckDB Storage:**
- Table: `position_snapshots`
- Storage format: JSON blob in `account_data` column
- All account fields are preserved in JSON format

**Verification:** ✅ Complete match - all API fields are stored

## Databento API → DuckDB Schema Mapping

### Option Chain Data

**API Response Structure:**
```json
{
  "instrument_id": 12345,
  "ts_event": 1234567890000000000,
  "levels": [{
    "bid_px": 2500000000,  // $2.50 in 1e-9 dollars
    "ask_px": 2600000000,  // $2.60 in 1e-9 dollars
    "bid_sz": 100,
    "ask_sz": 150
  }]
}
```

**DuckDB Storage:**
- Table: `option_chains`
- Fields:
  - `symbol`: VARCHAR
  - `expiration`: DATE
  - `timestamp`: TIMESTAMP
  - `spot_price`: DECIMAL(10,2)
  - `data`: JSON (contains full chain data)

**Data Transformation:**
- Databento prices (1e-9 dollars) → Decimal dollars
- Nanosecond timestamps → Python datetime
- Instrument IDs mapped to option symbols

**Verification:** ✅ Complete match with proper transformations

### Greeks Cache

**Calculated Data Structure:**
```python
{
    "delta": 0.3012,
    "gamma": 0.0234,
    "theta": -0.0512,
    "vega": 0.1234,
    "rho": 0.0456,
    "iv": 0.2345
}
```

**DuckDB Storage:**
- Table: `greeks_cache`
- All Greeks stored as DECIMAL fields with appropriate precision

**Verification:** ✅ Proper precision for all Greeks values

## Schema Improvements Made

1. **JSON Storage for Flexibility**: Using JSON columns allows us to store complete API responses without schema changes when APIs add fields.

2. **Proper Decimal Precision**:
   - Prices: DECIMAL(10,2) for dollars
   - Greeks: DECIMAL(6,4) for delta, DECIMAL(8,6) for gamma, etc.

3. **Timestamp Handling**: All timestamps stored in UTC for consistency.

4. **Primary Keys**: Composite keys ensure no duplicate data.

## Missing Fields Analysis

### Currently Not Captured:
1. **Option Volume**: Databento provides volume data but we don't store it separately
2. **Exchange Information**: Which exchange the quote came from
3. **Quote Conditions**: Special trading conditions

### Recommendation:
These fields are available in the JSON blobs if needed. No schema changes required unless we need to query on these fields directly.

## Data Integrity Checks

### Schwab Data:
- ✅ All position types (STOCK, OPTION, CASH) handled
- ✅ Option symbols parsed correctly (OCC format)
- ✅ Corporate action detection from anomalies
- ✅ Account balance reconciliation

### Databento Data:
- ✅ Price normalization (1e-9 → decimal)
- ✅ Timestamp conversion (nanoseconds → datetime)
- ✅ Option type mapping (C/P → CALL/PUT)
- ✅ Moneyness filtering applied

## Conclusion

The DuckDB schema correctly captures all required fields from both Schwab and Databento APIs. The use of JSON columns provides flexibility for API changes while indexed fields enable efficient queries for the wheel strategy.