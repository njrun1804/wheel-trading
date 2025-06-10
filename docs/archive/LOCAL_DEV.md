> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Local Development Guide

## Philosophy

This project is optimized for single-user local development with plans for sophisticated features (ML, risk analytics) to be added incrementally.

## Simplified Setup

1. **No Cloud Deployment** - Everything runs on your machine
2. **No CI/CD** - Test locally before committing
3. **No Pre-commit Hooks** - Format/lint when you want
4. **Simple Commands** - Basic make commands for common tasks

## Daily Workflow

```bash
# Morning: Check current recommendations
python run.py

# Development: Make changes and test
make -f Makefile.simple test

# Before commit: Format and check
make -f Makefile.simple format
make -f Makefile.simple check

# Commit directly
git add -A
git commit -m "Add feature X"
git push
```

## Project Structure

```
wheel-trading/
├── run.py                 # Quick CLI for decisions
├── Makefile.simple        # Local dev commands
├── requirements-local.txt # Minimal dependencies
├── .env.local            # Config template
│
├── src/                  # Source code
│   ├── config.py         # Settings management
│   ├── main.py           # Full application
│   ├── models.py         # Data structures
│   ├── wheel.py          # Strategy implementation
│   └── utils/
│       └── math.py       # Options mathematics
│
└── tests/                # Test files
    ├── test_main.py
    └── test_math.py
```

## Adding Features

Follow the incremental build plan:

1. **Risk Analytics** → Add to `src/utils/analytics.py`
2. **Schwab API** → Create `src/auth/` and `src/data/`
3. **Decision Engine** → Create `src/engine/`
4. **ML Layer** → Create `src/ml/` (optional)

## Testing Strategy

```bash
# Quick test during development
python -m pytest tests/test_math.py::TestBlackScholesPrice::test_call_option_atm -v

# Full test suite
make -f Makefile.simple test

# Coverage report
python -m pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Performance Considerations

- Target: <200ms for decision generation
- Use caching for market data (30s TTL)
- Vectorize calculations where possible
- Profile before optimizing

## Future Migration

When ready for production:
1. Add proper broker authentication
2. Implement data persistence
3. Add monitoring/alerting
4. Consider systemd service for automation

## Debugging Tips

```bash
# Verbose output
python run.py --verbose

# Python debugger
python -m pdb run.py

# Check specific calculation
python -c "from src.utils.math import black_scholes_price; print(black_scholes_price(100, 100, 1, 0.05, 0.2, 'call'))"
```

Remember: Keep it simple until complexity is needed!
