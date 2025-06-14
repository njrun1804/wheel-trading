#!/bin/bash
# Run tests with proper environment setup

# Set working directory
cd "$(dirname "$0")"

# Ensure test database exists
if [ ! -f "data/test_wheel_trading.duckdb" ]; then
    echo "Setting up test database..."
    python scripts/test_cleanup/01_setup_test_db.py
fi

# Run tests with proper PYTHONPATH
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH}"
export DATABASE_PATH="data/test_wheel_trading.duckdb"
export TESTING="true"

echo "Running tests with M4 Pro optimizations..."
python -m pytest tests/ -v --tb=short "$@"