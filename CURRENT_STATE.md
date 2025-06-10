# Current State - June 2025

The Unity Wheel Trading Bot implements an autonomous wheel strategy focused on recommendation generation. Major components include:

- **Configuration**: YAML-driven with environment overrides and health reporting (`src/config/loader.py`).
- **Data Pipeline**: Async loaders for Schwab positions, Databento option chains, and FRED economic indicators.
- **Strategy Engine**: Wheel strategy logic in `strategy_engine/` with confidence-scored analytics.
- **Risk Engine**: VaR/CVaR calculations and position sizing utilities.
- **ML Engine**: Optional models for volatility forecasting and regime detection.
- **App**: CLI entry points in `app/` and `run.py` for user-facing commands.

Unit test coverage exceeds 90% and integration tests skip automatically without API keys. For contributing instructions see [CONTRIBUTING.md](CONTRIBUTING.md).
