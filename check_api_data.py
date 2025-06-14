#!/usr/bin/env python3
"""Check if API data is actually flowing correctly to the database."""

import duckdb
from datetime import datetime, timedelta

def main():
    conn = duckdb.connect('data/wheel_trading_optimized.duckdb', read_only=True)
    
    print('=== CHECKING API DATA FLOW ===\n')
    
    # 1. Check recent stock data
    print('1. STOCK DATA (Unity):')
    result = conn.execute('''
        SELECT date, close, volume 
        FROM market.price_data 
        WHERE symbol = 'U' 
        ORDER BY date DESC 
        LIMIT 5
    ''').fetchall()
    
    if result:
        print('   Latest 5 days:')
        for row in result:
            volume = f'{row[2]:,}' if row[2] is not None else 'NULL'
            print(f'     {row[0]}: Close=${row[1]:.2f}, Volume={volume}')
        
        # Check if we have recent data
        latest_date = result[0][0]
        days_old = (datetime.now().date() - latest_date).days
        if days_old > 3:
            print(f'   ⚠️  WARNING: Latest data is {days_old} days old!')
        else:
            print(f'   ✅ Data is current ({days_old} days old)')
    else:
        print('   ❌ NO STOCK DATA FOUND')
    
    # 2. Check options data
    print('\n2. OPTIONS DATA (Unity):')
    result = conn.execute('''
        SELECT DATE(timestamp) as date, 
               COUNT(*) as contracts,
               COUNT(DISTINCT strike_price) as strikes,
               COUNT(DISTINCT expiration) as expirations
        FROM options.contracts 
        WHERE symbol = 'U'
        GROUP BY DATE(timestamp)
        ORDER BY date DESC 
        LIMIT 5
    ''').fetchall()
    
    if result:
        print('   Latest 5 days with options:')
        for row in result:
            print(f'     {row[0]}: {row[1]} contracts, {row[2]} strikes, {row[3]} expirations')
    else:
        print('   ❌ NO OPTIONS DATA FOUND')
    
    # 3. Check FRED data
    print('\n3. FRED DATA (ML Features):')
    result = conn.execute('''
        SELECT feature_date,
               vix_level,
               risk_free_rate,
               ted_spread
        FROM analytics.ml_features 
        WHERE symbol = 'U'
        ORDER BY feature_date DESC 
        LIMIT 5
    ''').fetchall()
    
    if result:
        print('   Latest 5 days:')
        for row in result:
            vix = f'{row[1]:.2f}' if row[1] else 'NULL'
            rf = f'{row[2]:.4f}' if row[2] else 'NULL' 
            ted = f'{row[3]:.4f}' if row[3] else 'NULL'
            print(f'     {row[0]}: VIX={vix}, RF={rf}, TED={ted}')
    else:
        print('   ❌ NO FRED DATA FOUND')
    
    # 4. Check for dummy data patterns
    print('\n4. DATA QUALITY CHECKS:')
    
    # Check for suspicious patterns in stock data
    result = conn.execute('''
        SELECT COUNT(*) as total,
               COUNT(DISTINCT close) as unique_closes,
               COUNT(DISTINCT volume) as unique_volumes,
               MIN(close) as min_close,
               MAX(close) as max_close
        FROM market.price_data 
        WHERE symbol = 'U'
    ''').fetchone()
    
    if result[0] > 0:
        uniqueness_ratio = result[1] / result[0]
        print(f'   Stock data uniqueness: {uniqueness_ratio:.2%} unique prices')
        if uniqueness_ratio < 0.5:
            print('   ⚠️  WARNING: Low price variation - possible dummy data!')
        print(f'   Price range: ${result[3]:.2f} - ${result[4]:.2f}')
    
    # Check for test patterns in options
    result = conn.execute('''
        SELECT COUNT(*) as total,
               COUNT(DISTINCT bid_price) as unique_bids,
               COUNT(DISTINCT ask_price) as unique_asks
        FROM options.contracts 
        WHERE symbol = 'U'
        AND bid_price > 0
    ''').fetchone()
    
    if result[0] > 0:
        bid_uniqueness = result[1] / result[0]
        ask_uniqueness = result[2] / result[0]
        print(f'   Options bid uniqueness: {bid_uniqueness:.2%}')
        print(f'   Options ask uniqueness: {ask_uniqueness:.2%}')
        if bid_uniqueness < 0.1 or ask_uniqueness < 0.1:
            print('   ⚠️  WARNING: Low price variation - possible dummy data!')
    
    # 5. Check column mappings
    print('\n5. COLUMN MAPPING VERIFICATION:')
    
    # Check options columns
    result = conn.execute('''
        SELECT * FROM options.contracts 
        WHERE symbol = 'U'
        LIMIT 1
    ''').fetchone()
    
    if result:
        cols = conn.execute("DESCRIBE SELECT * FROM options.contracts").fetchall()
        print('   Options table columns:')
        important_cols = ['symbol', 'timestamp', 'expiration', 'strike_price', 
                          'option_type', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
        for col in cols:
            if col[0] in important_cols:
                print(f'     ✓ {col[0]}: {col[1]}')
    
    # Check specific data patterns that indicate real vs dummy data
    print('\n6. REAL DATA INDICATORS:')
    
    # Check if we have varying timestamps (not all at midnight)
    result = conn.execute('''
        SELECT COUNT(DISTINCT TIME(timestamp)) as unique_times
        FROM options.contracts 
        WHERE symbol = 'U'
        AND DATE(timestamp) = (SELECT MAX(DATE(timestamp)) FROM options.contracts WHERE symbol = 'U')
    ''').fetchone()
    
    if result[0] > 1:
        print(f'   ✅ Options have {result[0]} different timestamps (not dummy data)')
    else:
        print('   ⚠️  All options have same timestamp (possible dummy data)')
    
    # Check strike price patterns
    result = conn.execute('''
        SELECT strike_price, COUNT(*) as count
        FROM options.contracts 
        WHERE symbol = 'U'
        AND DATE(timestamp) = (SELECT MAX(DATE(timestamp)) FROM options.contracts WHERE symbol = 'U')
        GROUP BY strike_price
        ORDER BY strike_price
        LIMIT 10
    ''').fetchall()
    
    if result:
        print('   Strike prices distribution:')
        for strike, count in result[:5]:
            print(f'     ${strike:.2f}: {count} contracts')
    
    conn.close()

if __name__ == "__main__":
    main()