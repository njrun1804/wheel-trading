# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A sophisticated options wheel trading system designed for single-user local development with planned ML and risk analytics features.

## Commands

### Primary Development Commands:

- `python run.py` - Run decision engine with current data
- `make -f Makefile.simple test` - Run all tests
- `make -f Makefile.simple format` - Auto-format code
- `make -f Makefile.simple check` - Lint and check code quality

### Quick Development Flow:

```bash
# After making changes:
make -f Makefile.simple test
make -f Makefile.simple format

# Run the application:
python run.py --verbose
```

## Architecture

### Current Structure:

```
wheel-trading/
├── run.py              # CLI entry point for decisions
├── src/
│   ├── config.py       # Configuration management
│   ├── main.py         # Main application entry
│   ├── models.py       # Data models (Position, WheelPosition)
│   ├── wheel.py        # Core wheel strategy implementation
│   └── utils/
│       └── math.py     # Options mathematics (Black-Scholes, Greeks)
└── tests/              # Comprehensive test suite
```

### Planned Modules:

1. **src/utils/analytics.py** - Risk metrics (VaR, CVaR, Kelly sizing)
2. **src/auth/** - Schwab OAuth authentication
3. **src/data/** - Market data fetching and caching
4. **src/engine/** - Decision engine with scoring
5. **src/ml/** - Optional ML enhancement layer

## Development Guidelines

1. **Keep it simple** - This is for single-user local use
2. **Test everything** - Aim for >80% coverage
3. **Document decisions** - Log WHY each decision was made
4. **Performance matters** - Target <200ms for decisions
5. **Incremental features** - Add complexity only when needed

## Key Principles

- **Single User**: No multi-tenant complexity needed
- **Local Only**: No cloud deployment or CI/CD
- **Self-Diagnostic**: Extensive logging and explanations
- **Future-Ready**: Structure supports ML and advanced analytics

## Testing Requirements

Before any changes:
```bash
# Run tests
make -f Makefile.simple test

# Check specific functionality
python -c "from src.utils.math import black_scholes_price; print(black_scholes_price(100, 100, 1, 0.05, 0.2, 'call'))"
```

## Configuration

Environment variables in `.env`:
- `TRADING_MODE`: paper, backtest, or live
- `WHEEL_DELTA_TARGET`: Target delta (e.g., 0.30)
- `DAYS_TO_EXPIRY_TARGET`: Target DTE (e.g., 45)
- `MAX_POSITION_SIZE`: Max portfolio % per position

## Future Features Roadmap

1. **Risk Analytics** - VaR, CVaR, Kelly criterion
2. **Schwab Integration** - Real positions and options chains
3. **Decision Engine** - Multi-criteria scoring with explanations
4. **ML Enhancement** - Probability adjustments and pattern recognition

## Notes

- Use `python run.py --verbose` for detailed output
- All times are in local timezone
- Options math uses annualized parameters
- Position sizing respects portfolio constraints