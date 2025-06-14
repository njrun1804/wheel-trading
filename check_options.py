#!/usr/bin/env python3
"""Check available options in database."""

import duckdb

conn = duckdb.connect('data/wheel_trading_optimized.duckdb')

print('Available PUT options:')
result = conn.execute('''
    SELECT strike, expiration, bid, ask, 
           CAST((expiration - CURRENT_DATE) AS INTEGER) as days_to_exp
    FROM options.contracts
    WHERE symbol = 'U' 
    AND option_type = 'PUT'
    AND bid > 0 AND ask > 0
    ORDER BY expiration, strike
    LIMIT 20
''').fetchall()

for row in result:
    print(f'  Strike: ${row[0]:.2f}, Exp: {row[1]}, Bid: ${row[2]:.2f}, Ask: ${row[3]:.2f}, Days: {row[4]}')

total = conn.execute('''
    SELECT COUNT(*) FROM options.contracts
    WHERE symbol = 'U' AND option_type = 'PUT'
    AND bid > 0 AND ask > 0
''').fetchone()[0]

print(f'\nTotal PUT options with valid quotes: {total}')

# Check date range
dates = conn.execute('''
    SELECT MIN(expiration), MAX(expiration), COUNT(DISTINCT expiration)
    FROM options.contracts
    WHERE symbol = 'U' AND bid > 0 AND ask > 0
''').fetchone()

print(f'\nExpiration range: {dates[0]} to {dates[1]} ({dates[2]} unique dates)')