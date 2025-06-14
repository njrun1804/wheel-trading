#!/bin/bash
# Test data collection scripts

echo "Testing EOD collection..."
cd "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
"/opt/homebrew/Caskroom/miniconda/base/bin/python3" scripts/collect_eod_production.py

echo -e "\nTesting intraday collection..."
"/opt/homebrew/Caskroom/miniconda/base/bin/python3" scripts/collect_intraday.py --once

echo -e "\nChecking database..."
echo "SELECT COUNT(*) as options_count FROM options.contracts WHERE symbol='U';" | \
    duckdb "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/data/wheel_trading_optimized.duckdb"
