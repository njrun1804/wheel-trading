#!/usr/bin/env python3
"""Check actual data patterns to verify API functionality."""

import duckdb

conn = duckdb.connect('data/wheel_trading_optimized.duckdb', read_only=True)

print('=== ACTUAL DATA CHECK ===\n')

# Check latest stock data with issues
print('1. STOCK DATA ISSUES:')
result = conn.execute('''
    SELECT date, open, high, low, close, volume
    FROM market.price_data 
    WHERE symbol = 'U'
    AND date >= '2025-06-01'
    ORDER BY date DESC
''').fetchall()

print('   Recent data:')
for row in result[:10]:
    print(f'     {row[0]}: O={row[1]} H={row[2]} L={row[3]} C={row[4]} V={row[5]}')

# Check if any stock data has proper prices
print('\n2. NON-ZERO STOCK PRICES:')
result = conn.execute('''
    SELECT COUNT(*) as total,
           COUNT(CASE WHEN close > 0 THEN 1 END) as non_zero,
           MIN(date) as first_date,
           MAX(date) as last_date
    FROM market.price_data 
    WHERE symbol = 'U'
''').fetchone()
print(f'   Total records: {result[0]}')
print(f'   Non-zero closes: {result[1]} ({result[1]/result[0]*100:.1f}%)')
print(f'   Date range: {result[2]} to {result[3]}')

# Check sample of non-zero prices
result = conn.execute('''
    SELECT date, close, volume
    FROM market.price_data 
    WHERE symbol = 'U' AND close > 0
    ORDER BY date DESC
    LIMIT 10
''').fetchall()
if result:
    print('\n   Sample non-zero prices:')
    for row in result:
        vol_str = f'{row[2]:,}' if row[2] is not None else 'NULL'
        print(f'     {row[0]}: ${row[1]:.2f} (Vol: {vol_str})')

# Check options data
print('\n3. OPTIONS DATA:')
result = conn.execute('''
    SELECT timestamp, expiration, strike, option_type, bid, ask
    FROM options.contracts 
    WHERE symbol = 'U'
    ORDER BY timestamp DESC
    LIMIT 5
''').fetchall()

if result:
    print('   Latest 5 option records:')
    for row in result:
        print(f'     {row[0]}: Exp={row[1]} Strike=${row[2]} Type={row[3]} Bid=${row[4]} Ask=${row[5]}')
else:
    print('   NO OPTIONS DATA')

# Check for realistic vs dummy patterns
print('\n4. DATA PATTERNS:')
result = conn.execute('''
    SELECT 
        COUNT(DISTINCT bid) as unique_bids,
        COUNT(DISTINCT ask) as unique_asks,
        MIN(bid) as min_bid,
        MAX(bid) as max_bid,
        AVG(ask - bid) as avg_spread
    FROM options.contracts 
    WHERE symbol = 'U'
    AND bid > 0
''').fetchone()

if result[0]:
    print(f'   Unique bid prices: {result[0]}')
    print(f'   Unique ask prices: {result[1]}')
    print(f'   Bid range: ${result[2]} - ${result[3]}')
    print(f'   Average spread: ${result[4]:.4f}')

# Check FRED data
print('\n5. FRED DATA:')
result = conn.execute('''
    SELECT feature_date, vix_level, risk_free_rate, ted_spread
    FROM analytics.ml_features 
    WHERE symbol = 'U' 
    AND vix_level IS NOT NULL
    ORDER BY feature_date DESC
    LIMIT 5
''').fetchall()

if result:
    print('   Latest FRED data:')
    for row in result:
        print(f'     {row[0]}: VIX={row[1]:.2f} RF={row[2]:.4f} TED={row[3] if row[3] else "NULL"}')
else:
    print('   NO FRED DATA WITH VIX')

# Final verdict
print('\n=== VERDICT ===')
print('Based on the data patterns:')

# Check stock data
stock_result = conn.execute('''
    SELECT COUNT(CASE WHEN close > 0 THEN 1 END) * 100.0 / COUNT(*) as pct_valid
    FROM market.price_data WHERE symbol = 'U'
''').fetchone()

if stock_result[0] > 90:
    print('✅ Stock data appears to be from real API')
elif stock_result[0] > 50:
    print('⚠️  Stock data is partially valid')
else:
    print('❌ Stock data has issues - mostly zeros')

# Check options data
options_result = conn.execute('''
    SELECT COUNT(DISTINCT bid) as unique_bids
    FROM options.contracts 
    WHERE symbol = 'U' AND bid > 0
''').fetchone()

if options_result[0] > 100:
    print('✅ Options data shows realistic price variation')
elif options_result[0] > 10:
    print('⚠️  Options data has limited variation')
else:
    print('❌ Options data appears to be dummy/test data')

conn.close()