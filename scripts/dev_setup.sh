#!/bin/bash
# Quick development setup for Unity Wheel Bot
# Optimized for single-user macOS environment

set -e

echo "🚀 Unity Wheel Bot - Development Setup"
echo "====================================="

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" < "3.12" ]]; then
    echo "❌ Python 3.12+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION"

# Install Poetry if needed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✓ Poetry installed"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
poetry install

# Set up pre-commit hooks (simplified for Claude Code)
echo ""
echo "🔧 Setting up pre-commit hooks..."
poetry run pre-commit install

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p exports logs data/cache

# Quick validation
echo ""
echo "🧪 Running quick validation..."
poetry run python -c "
from src.unity_wheel.math.options import black_scholes_price_validated
result = black_scholes_price_validated(100, 100, 0.25, 0.05, 0.25, 'call')
print(f'✓ Math functions working: ${result.value:.2f}')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Configure credentials: poetry run python scripts/setup-secrets.py"
echo "2. Run health check: make quick"
echo "3. Get recommendation: make recommend"
echo ""
