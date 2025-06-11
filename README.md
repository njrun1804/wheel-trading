# Unity Wheel Trading Bot v2.0 - Autonomous Edition

> **⚠️ Note:** The default `config.yaml` is tuned for high risk/high return strategies.
> For a more conservative approach, see `examples/core/conservative_config.yaml`

A sophisticated options wheel strategy recommendation system for Unity Software Inc. (U). Features autonomous operation with self-monitoring, auto-tuning, and enterprise-grade reliability. Designed for single-user local operation with recommendations only (no broker trading integration).

## 🎯 Core Philosophy

**Autonomous Operation**: Self-monitoring, self-healing, and self-optimizing system that provides recommendations on demand. No trading execution - recommendations only.

## 🚀 Quick Start

```bash
# Get a wheel strategy recommendation
python run.py --portfolio 100000

# Run system diagnostics
python run.py --diagnose

# View performance metrics
python run.py --performance

# Continuous monitoring
./scripts/monitor.sh
```

## 💰 Cost Efficiency

- **< $50/month** total operational cost
- No streaming subscriptions
- Intelligent caching reduces API calls by 90%+
- Uses Google Cloud Secret Manager for credentials (only GCP dependency)

## 🏗️ Architecture

### Pull-When-Asked Flow

```
User requests recommendation
    ↓
Check local DuckDB cache (15-30 min TTL)
    ↓
If stale → Fetch from APIs:
    • Databento: Option chains (REST only)
    • FRED: Macro indicators
    ↓
Store in cache → Generate recommendation
```

### Storage Layers

1. **Local DuckDB** (~/.wheel_trading/cache/)
   - Primary storage for all data
   - 30-day automatic cleanup
   - < 5GB typical usage


## 📁 Project Structure

```
wheel-trading/
├── README.md                    # This file
├── CLAUDE.md                    # Claude Code instructions
├── INTEGRATION_GUIDE.md         # All external integrations
├── DEVELOPMENT_GUIDE.md         # Setup and development
├── config.yaml                  # Main configuration
├── run.py               # PRIMARY entry point (v2.0)
├── src/unity_wheel/             # Core implementation
├── tests/                       # All tests
├── examples/                    # Organized examples
│   ├── core/                    # Config, risk, validation
│   ├── data/                    # Databento and FRED
│   └── auth/                    # Authentication, secrets
├── tools/                       # Development utilities
│   ├── debug/                   # Debugging tools
│   ├── analysis/                # Data analysis scripts
│   └── verification/            # System verification
├── deployment/                  # Deployment configs
└── scripts/                     # Shell scripts
```

## 📦 Installation

### Quick Setup

```bash
# Clone repository
git clone <repository>
cd wheel-trading

# Install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up credentials
python scripts/setup-secrets.py
# This stores your API keys in Google Cloud Secret Manager or
# local encrypted storage depending on the environment.

# Verify installation
python -m unity_wheel.validate
```

For detailed setup instructions, see [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).

## 🔧 Configuration

### Required Credentials

1. **Databento**: API key for options data
2. **FRED** (optional): Free API key for economic data

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for detailed setup.

### Configuration System

The project uses an intelligent YAML-based configuration with:
- Comprehensive validation using Pydantic schemas
- Environment variable overrides (WHEEL_SECTION__PARAM format)
- Parameter usage tracking and health reporting
- Self-tuning based on outcome tracking

```bash
# Example overrides
export WHEEL_STRATEGY__DELTA_TARGET=0.30
export WHEEL_RISK__MAX_POSITION_SIZE=0.20
export WHEEL_ML__ENABLED=true
```

## 🎮 Usage

### Basic Recommendation

```bash
# Get recommendation with current market data
python run.py --portfolio 100000

# Output:
# 🎯 WHEEL STRATEGY RECOMMENDATION
# =====================================
# Action: SELL_PUT
# Symbol: U
# Strike: $32.50
# Expiration: 2024-03-15
# Contracts: 5
# Max Risk: $16,250
# Confidence: 85%
# Reason: Optimal delta match, high confidence
```

### Autonomous Features

```bash
# Run system diagnostics
python run.py --diagnose

# View performance metrics
python run.py --performance

# Export metrics dashboard
python run.py --export-metrics

# Continuous monitoring
./scripts/monitor.sh

# Run all autonomous checks
./scripts/autonomous-checks.sh
```

### Cloud Run Deployment

```bash
# Deploy as serverless job
gcloud run jobs deploy wheel-recommendation \
    --source . \
    --memory 2Gi \
    --task-timeout 5m

# Execute on demand
gcloud run jobs execute wheel-recommendation \
    --env-vars PORTFOLIO_VALUE=100000
```

## 📊 Data Management

### Cache TTLs

| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| Positions | 30 min | Changes slowly |
| Account | 30 sec | Critical for margin |
| Options | 15 min | Price sensitive |
| FRED | 24 hours | Daily updates |

### Storage Maintenance

```python
# Check storage usage
from unity_wheel.storage import Storage

storage = Storage()
await storage.initialize()
stats = await storage.get_storage_stats()
print(f"Cache size: {stats['db_size_mb']} MB")

# Clean old data (automatic after 30 days)
await storage.cleanup_old_data()
```

## 🧮 Core Features

### Self-Validating Mathematics

```python
from unity_wheel.math import black_scholes_price_validated

result = black_scholes_price_validated(
    S=35.50,    # Unity current price
    K=32.50,    # Strike price
    T=0.123,    # 45 days to expiration
    r=0.05,     # Risk-free rate
    sigma=0.65, # Implied volatility
    option_type="put"
)

print(f"Put price: ${result.value:.2f}")
print(f"Confidence: {result.confidence:.0%}")
```

### Risk Analytics

```python
from unity_wheel.risk import RiskAnalyzer

analyzer = RiskAnalyzer()

# Position sizing with Kelly criterion
kelly, confidence = analyzer.calculate_kelly_criterion(
    win_rate=0.70,      # 70% of puts expire worthless
    avg_win=1.0,        # Keep full premium
    avg_loss=3.0,       # Average loss if assigned
    apply_half_kelly=True  # Conservative sizing
)

print(f"Recommended size: {kelly:.1%} of portfolio")
```

## 🛠️ Development

### Running Tests

```bash
# All tests with coverage
poetry run pytest --cov=src/unity_wheel

# Specific module
poetry run pytest tests/test_storage.py -v

# Integration tests
poetry run pytest tests/test_autonomous_flow.py -v
```

### Code Quality

```bash
# Auto-format
poetry run black src/ tests/

# Type checking
poetry run mypy src/ --strict

# Pre-commit hooks (auto-installed)
poetry run pre-commit run --all-files
```

## 📈 Performance

### Calculation Benchmarks

- Black-Scholes: < 0.2ms per calculation
- Greeks (all): < 0.3ms
- Risk metrics: < 10ms for 1000 data points
- Cache lookup: < 1ms

### API Efficiency

- Cache hit rate: > 90% typical
- API calls per recommendation: 0-2 (with cache)
- Total latency: < 5 seconds (including API calls)

## 🔐 Security

- Credentials encrypted at rest
- OAuth tokens auto-refresh
- No credentials in code or config
- Machine-specific encryption keys
- Google Cloud Secret Manager stores credentials securely
  (the project's only Google Cloud dependency)

## 📝 Documentation

### Core Documentation
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - All external service integrations
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Setup, workflow, deployment
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common operations and commands
- [CLAUDE.md](CLAUDE.md) - Claude Code instructions

### Archived Documentation
Historical documentation has been archived in `docs/archive/` for reference.

## 🎯 Design Principles

1. **Autonomous Operation** - Self-monitoring, self-healing, self-optimizing
2. **Self-Validation** - Every calculation includes confidence score
3. **Type Safety** - 100% type hints, mypy strict mode
4. **Immutable Models** - All data models are frozen dataclasses
5. **Property Testing** - Hypothesis for edge case discovery
6. **Structured Logging** - Machine-parseable JSON logs
7. **Performance Monitoring** - Automatic SLA tracking
8. **Graceful Degradation** - Feature flags for resilience

## 🚫 What This Is NOT

- ❌ No automated trading execution (recommendations only)
- ❌ No broker integration for trading
- ❌ No real-time streaming data
- ❌ No multi-user support
- ❌ No complex infrastructure

## 🎯 Objective Function

**Maximize: CAGR - 0.20 × |CVaR₉₅|** with **½-Kelly** position sizing

## 📞 Support

- Issues: [GitHub Issues](https://github.com/yourusername/wheel-trading/issues)
- Documentation: [Wiki](https://github.com/yourusername/wheel-trading/wiki)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

**Remember**: This is a recommendation system only. Always verify recommendations before placing trades.

### Disclaimer

All outputs are for informational purposes only and do not constitute financial advice. Consult a licensed financial professional before making any investment decisions.
