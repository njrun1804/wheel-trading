#!/usr/bin/env bash
set -euo pipefail
# Quick health check for single-user recommendation system
# Focuses on what matters: math accuracy and recommendation capability


echo "🔍 Unity Wheel Bot - Quick Check"
echo "================================"

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Run from project root directory"
    exit 1
fi

# 1. Environment check
echo "📦 Environment:"
echo "   Python: $(python --version 2>&1)"
echo "   Location: $(pwd)"

# 2. Core functionality test
echo ""
echo "🧮 Testing core calculations..."
python -c "
from src.unity_wheel.math.options import black_scholes_price_validated
from decimal import Decimal

# Test known value
result = black_scholes_price_validated(100, 100, 0.25, 0.05, 0.25, 'call')
expected = 6.04  # Approximate known value

if abs(result.value - expected) < 0.1:
    print('   ✓ Black-Scholes accurate')
else:
    print(f'   ✗ Black-Scholes off: got {result.value:.2f}, expected ~{expected:.2f}')
    exit(1)

# Test edge case
edge = black_scholes_price_validated(100, 50, 0.01, 0.05, 0.50, 'put')
if edge.value > 0 and edge.confidence > 0:
    print('   ✓ Edge cases handled')
else:
    print('   ✗ Edge case failed')
    exit(1)
"

# 3. Configuration check
echo ""
echo "⚙️  Configuration:"
python -c "
from src.config.loader import get_config_loader

try:
    loader = get_config_loader()
    config = loader.config

    # Key settings for aggressive strategy
    print(f'   ✓ Max position size: {config[\"risk\"][\"max_position_size\"]:.0%}')
    print(f'   ✓ Max margin usage: {config[\"risk\"][\"max_margin_percent\"]:.0%}')
    print(f'   ✓ Target delta: {config[\"strategy\"][\"delta_target\"]}')
except Exception as e:
    print(f'   ✗ Config error: {e}')
    exit(1)
"

# 4. Quick recommendation test
echo ""
echo "💡 Testing recommendations..."
python -c "
from src.unity_wheel.api.advisor import WheelAdvisor
from src.unity_wheel.models.account import Account
from decimal import Decimal

advisor = WheelAdvisor()

# Test small account
small = Account(
    total_value=Decimal('10000'),
    cash_balance=Decimal('10000'),
    buying_power=Decimal('10000')
)
r1 = advisor.advise_position(small, [], {})
print(f'   ✓ Small account: {r1.primary_action.action_type} ({r1.confidence:.0%} confidence)')

# Test margin account
margin = Account(
    total_value=Decimal('100000'),
    cash_balance=Decimal('20000'),
    buying_power=Decimal('200000')
)
r2 = advisor.advise_position(margin, [], {})
print(f'   ✓ Margin account: {r2.primary_action.action_type} ({r2.confidence:.0%} confidence)')
"

echo ""
echo "✅ System ready for recommendations!"
echo ""
echo "📊 Get a recommendation:"
echo "   python run.py --portfolio 100000"
echo ""
