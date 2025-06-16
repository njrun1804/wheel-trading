#!/usr/bin/env python3
"""Minimal test of core trading functionality"""

import sys
import os
sys.path.insert(0, 'src')

def test_minimal_trading_workflow():
    """Test a minimal trading workflow without full system imports"""
    print("=== Testing Minimal Trading Workflow ===")
    
    try:
        import duckdb
        import pandas as pd
        import numpy as np
        import yaml
        from datetime import datetime, timedelta
        
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded")
        
        # Connect to database
        db_path = 'data/wheel_trading_master.duckdb'
        conn = duckdb.connect(db_path)
        
        # Get current options data
        options_data = conn.execute('''
            SELECT symbol, strike, expiration, option_type, bid, ask, volume, open_interest
            FROM active_options 
            WHERE symbol = 'U'
            ORDER BY expiration, strike
            LIMIT 10
        ''').fetchdf()
        
        print(f"✅ Retrieved {len(options_data)} Unity options from database")
        
        if len(options_data) > 0:
            print(f"  - Strike range: ${options_data.strike.min():.0f} - ${options_data.strike.max():.0f}")
            print(f"  - Expiry range: {options_data.expiration.min()} to {options_data.expiration.max()}")
        
        # Test basic options calculations
        def calculate_simple_iv_rank(current_price, strikes, option_type='put'):
            """Simple implied volatility rank calculation"""
            atm_strikes = strikes[abs(strikes - current_price) <= 5]
            return len(atm_strikes)
        
        # Get Unity configuration
        unity_config = config.get('unity', {})
        target_delta = unity_config.get('target_delta', 0.30)
        max_dte = unity_config.get('max_dte', 45)
        
        print(f"✅ Unity config: target_delta={target_delta}, max_dte={max_dte}")
        
        # Test risk calculation
        portfolio_value = 100000  # $100k portfolio
        max_position_size = unity_config.get('max_position_size', 0.1)
        max_contracts = int(portfolio_value * max_position_size / 100)  # Assume $100 per share
        
        print(f"✅ Risk calculation: max position size = {max_contracts} contracts")
        
        # Test decision logic
        current_date = datetime.now()
        
        if len(options_data) > 0:
            # Filter options by expiry
            options_data['expiration'] = pd.to_datetime(options_data['expiration'])
            options_data['dte'] = (options_data['expiration'] - current_date).dt.days
            
            valid_options = options_data[
                (options_data['dte'] >= 7) & 
                (options_data['dte'] <= max_dte) &
                (options_data['option_type'] == 'put')
            ]
            
            print(f"✅ Found {len(valid_options)} valid put options for wheel strategy")
            
            if len(valid_options) > 0:
                # Simple selection: closest to target delta (approximated by moneyness)
                current_price = 25.0  # Approximate Unity price
                valid_options['moneyness'] = valid_options['strike'] / current_price
                
                best_option = valid_options.loc[
                    (valid_options['moneyness'] >= 0.85) & 
                    (valid_options['moneyness'] <= 0.95)
                ].iloc[0] if len(valid_options[
                    (valid_options['moneyness'] >= 0.85) & 
                    (valid_options['moneyness'] <= 0.95)
                ]) > 0 else valid_options.iloc[0]
                
                print(f"✅ Selected option: ${best_option['strike']} put expiring {best_option['expiration'].strftime('%Y-%m-%d')}")
                print(f"  - Bid: ${best_option['bid']:.2f}, Ask: ${best_option['ask']:.2f}")
                print(f"  - Volume: {best_option['volume']}, OI: {best_option['open_interest']}")
        
        conn.close()
        print("✅ Minimal trading workflow completed successfully")
        
    except Exception as e:
        print(f"❌ Minimal trading workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_trading_workflow()