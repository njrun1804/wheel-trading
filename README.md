# Wheel Trading Strategy - Local Development

A sophisticated options wheel trading system designed for single-user workstation use.

## Quick Start

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
make -f Makefile.simple install

# 3. Configure
cp .env.local .env
# Edit .env with your settings

# 4. Run decision engine
python run.py

# Or with more detail:
python run.py --verbose --ticker AAPL --portfolio 50000
```

## Simple Workflow

```bash
# Run tests after changes
make -f Makefile.simple test

# Format code
make -f Makefile.simple format

# Quick check before committing
make -f Makefile.simple check
```

## Architecture

```
wheel-trading/
├── run.py              # Simple CLI entry point
├── src/
│   ├── config.py       # Configuration management
│   ├── main.py         # Main application
│   ├── models.py       # Data structures
│   ├── wheel.py        # Core strategy logic
│   └── utils/
│       └── math.py     # Options mathematics
└── tests/              # Test suite
```

## Current Features

- **Options Math**: Black-Scholes pricing, Greeks, implied volatility
- **Strike Selection**: Delta-based optimal strike finder
- **Position Sizing**: Risk-based position calculator
- **Decision Logic**: Roll triggers and timing

## Planned Features

1. **Risk Analytics** (Next)
   - Value at Risk (VaR/CVaR)
   - Kelly criterion sizing
   - Margin calculations

2. **Schwab Integration**
   - OAuth authentication
   - Real-time positions
   - Options chain data

3. **Decision Engine**
   - Multi-criteria scoring
   - Portfolio-aware recommendations
   - Clear explanations

4. **ML Enhancement** (Optional)
   - Probability adjustments
   - Dynamic parameters
   - Pattern recognition

## Development Principles

- **Single User**: Optimized for one trader on a private workstation
- **Self-Diagnostic**: Extensive logging and decision explanations
- **Maintainable**: Clear code structure, comprehensive tests
- **Pragmatic**: Start simple, add complexity only when needed

## Testing

```bash
# Run all tests
make -f Makefile.simple test

# Run specific test
python -m pytest tests/test_math.py -v

# Check coverage
python -m pytest --cov=src --cov-report=html
```

## Decision Output Example

```
🎯 Wheel Trading Decision Engine
📅 2024-01-15 09:30:00
==================================================

📊 Analyzing SPY
💰 Portfolio: $100,000
🎯 Target Delta: 0.3
📆 Target DTE: 45 days

💹 Current Price: $455

✅ RECOMMENDATION: Sell 4 SPY $445P
   Expiry: 45 days
   Max Risk: $178,000

==================================================
```

## Notes

- No cloud deployment needed - runs entirely local
- Configuration via .env file for security
- Designed to handle future ML/analytics modules
- Performance target: <200ms decision time