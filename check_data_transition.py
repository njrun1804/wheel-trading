#!/usr/bin/env python3
"""Check the data transition issues."""

import duckdb

conn = duckdb.connect('data/wheel_trading_optimized.duckdb', read_only=True)

print('=== CHECKING DATA ISSUES ===\n')

# 1. Check Unity stock prices over time
print('1. UNITY STOCK PRICE PROGRESSION:')
result = conn.execute('''
    SELECT 
        DATE_TRUNC('month', date) as month,
        AVG(close) as avg_close,
        MIN(close) as min_close,
        MAX(close) as max_close,
        COUNT(*) as days
    FROM market.price_data 
    WHERE symbol = 'U'
    GROUP BY DATE_TRUNC('month', date)
    ORDER BY month DESC
    LIMIT 12
''').fetchall()

for row in result:
    print(f'  {row[0].strftime("%Y-%m")}: Avg=${row[1]:.2f}, Range=${row[2]:.2f}-${row[3]:.2f} ({row[4]} days)')

# 2. Check the transition point
print('\n2. PRICE TRANSITION POINT:')
result = conn.execute('''
    SELECT date, close 
    FROM market.price_data 
    WHERE symbol = 'U' 
    AND date BETWEEN '2025-06-05' AND '2025-06-15'
    ORDER BY date
''').fetchall()

for row in result:
    print(f'  {row[0]}: ${row[1]:.2f}')

# 3. Check if we have multiple symbols
print('\n3. SYMBOLS IN DATABASE:')
result = conn.execute('''
    SELECT symbol, COUNT(*) as records, MIN(date) as first, MAX(date) as last
    FROM market.price_data
    GROUP BY symbol
''').fetchall()

for row in result:
    print(f'  {row[0]}: {row[1]} records from {row[2]} to {row[3]}')

# 4. Check actual Unity price range
print('\n4. ACTUAL UNITY (U) PRICE HISTORY:')
result = conn.execute('''
    SELECT 
        CASE 
            WHEN close < 50 THEN 'Under $50'
            WHEN close < 100 THEN '$50-100'
            WHEN close < 150 THEN '$100-150'
            ELSE 'Over $150'
        END as price_range,
        COUNT(*) as days,
        MIN(date) as first_date,
        MAX(date) as last_date
    FROM market.price_data 
    WHERE symbol = 'U'
    GROUP BY price_range
    ORDER BY MIN(close)
''').fetchall()

for row in result:
    print(f'  {row[0]}: {row[1]} days ({row[2]} to {row[3]})')

# 5. Check for data source changes
print('\n5. CHECK RECENT API CALLS:')
result = conn.execute('''
    SELECT 
        date,
        close,
        volume,
        CASE 
            WHEN volume IS NULL THEN 'Missing volume'
            WHEN close = 0 THEN 'Zero price'
            WHEN close < 50 THEN 'Low price'
            ELSE 'Normal'
        END as data_quality
    FROM market.price_data 
    WHERE symbol = 'U' 
    AND date >= '2025-06-01'
    ORDER BY date DESC
''').fetchall()

print('  Recent data quality:')
for row in result[:14]:
    print(f'    {row[0]}: ${row[1]:.2f} (Vol: {row[2] if row[2] else "NULL"}) - {row[3]}')

conn.close()