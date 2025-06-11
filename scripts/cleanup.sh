#!/bin/bash
# Quick cleanup script for wheel trading bot

echo "🧹 Cleaning up wheel trading repository..."

# Remove cache directories
echo "Removing cache directories..."
rm -rf .mypy_cache/ .pytest_cache/ htmlcov/ .DS_Store

# Remove temporary files
echo "Removing temporary files..."
rm -f SESSION_*.txt SESSION_*.md
rm -f *.csv
rm -f test_*.py
rm -f *_report.json
rm -f diagnostics_history.json

# Remove performance results
echo "Removing performance results..."
rm -rf performance_results/

# Clean Python cache
echo "Cleaning Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Remove editor swap files
echo "Removing editor swap files..."
find . -type f -name "*.swp" -delete 2>/dev/null || true
find . -type f -name "*.swo" -delete 2>/dev/null || true
find . -type f -name "*~" -delete 2>/dev/null || true

echo "✅ Cleanup complete!"

# Show disk usage
echo ""
echo "📊 Current disk usage:"
du -sh data/ 2>/dev/null || echo "  data/: directory not found"
du -sh logs/ 2>/dev/null || echo "  logs/: directory not found"
du -sh . | grep -v "\.git"
