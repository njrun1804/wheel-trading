#!/bin/bash
# Wrapper script for collect_intraday.py
# Ensures proper environment and error handling

# Load environment variables if .env exists
if [ -f "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/.env" ]; then
    set -a
    source "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/.env"
    set +a
fi

# Change to project directory
cd "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"

# Run the script with proper Python path
exec "/opt/homebrew/Caskroom/miniconda/base/bin/python3" "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/collect_intraday.py" "$@"
