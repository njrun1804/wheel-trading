# Development Guide

This guide covers setup, development workflow, and deployment for the Unity Wheel Trading Bot v2.0.

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (recommended) or pip
- Git with pre-commit hooks
- 2GB free disk space

### Initial Setup

1. **Clone and setup environment:**
```bash
git clone <repository>
cd wheel-trading
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Setup pre-commit hooks:**
```bash
pre-commit install
pre-commit run --all-files  # Verify setup
```

3. **Configure secrets:**
```bash
# Generate encryption key
python scripts/setup-secrets.py

# Set environment variable
export WHEEL_ENCRYPTION_KEY="<generated-key>"
```

4. **Add service credentials:**
```python
python -c "
from src.unity_wheel.secrets import SecretManager
import asyncio

async def setup():
    manager = SecretManager()

    # Add your API keys
    await manager.store_secret('databento', {'api_key': 'YOUR_KEY'})
    await manager.store_secret('schwab', {
        'client_id': 'YOUR_ID',
        'client_secret': 'YOUR_SECRET'
    })
    await manager.store_secret('fred', {'api_key': 'YOUR_KEY'})

asyncio.run(setup())
"
```

5. **Verify installation:**
```bash
# Run system validation
python -m unity_wheel.validate

# Run tests
pytest tests/ -v

# Get first recommendation
python run_aligned.py --portfolio 100000
```

## Development Workflow

### Code Quality Standards

All code must pass these checks before commit:

```bash
# Automatic via pre-commit
black src/ tests/           # Code formatting
isort src/ tests/          # Import sorting
mypy src/                  # Type checking
flake8 src/ tests/         # Linting
bandit -r src/             # Security scanning
```

### Testing Requirements

1. **Unit tests** for all new functions:
```python
# tests/test_feature.py
import pytest
from hypothesis import given, strategies as st

def test_calculation():
    """Test with known values."""
    result = calculate_something(100, 0.05)
    assert result == pytest.approx(105.0, rel=1e-4)

@given(
    value=st.floats(min_value=0, max_value=1e6),
    rate=st.floats(min_value=0, max_value=1)
)
def test_calculation_properties(value, rate):
    """Property-based testing for edge cases."""
    result = calculate_something(value, rate)
    assert result >= value  # Should never decrease
```

2. **Integration tests** for workflows:
```python
# tests/test_integration.py
async def test_recommendation_flow():
    """Test complete recommendation generation."""
    config = load_test_config()
    recommendation = await generate_recommendation(config)

    assert recommendation.confidence >= 0.8
    assert recommendation.position_size > 0
```

3. **Coverage requirements:**
```bash
# Run with coverage
pytest --cov=src --cov-report=html

# View report
open htmlcov/index.html

# Minimum 90% coverage required
```

### Adding New Features

1. **Create feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Implement with confidence scores:**
```python
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class CalculationResult:
    value: float
    confidence: float
    metadata: dict

def calculate_with_validation(
    input_value: float,
    **kwargs
) -> CalculationResult:
    """
    Calculate with built-in validation.

    Returns:
        CalculationResult with value and confidence score
    """
    # Validate inputs
    if input_value <= 0:
        return CalculationResult(
            value=float('nan'),
            confidence=0.0,
            metadata={'error': 'Invalid input'}
        )

    # Perform calculation
    result = complex_calculation(input_value)

    # Assess confidence
    confidence = assess_confidence(result, input_value)

    return CalculationResult(
        value=result,
        confidence=confidence,
        metadata={'method': 'complex_calculation'}
    )
```

3. **Add structured logging:**
```python
from src.unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)

@timed_operation("feature_calculation")
def your_feature(data: dict) -> dict:
    logger.info(
        "Starting calculation",
        extra={
            "function": "your_feature",
            "input_size": len(data),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    # Your implementation

    logger.info(
        "Calculation complete",
        extra={
            "function": "your_feature",
            "duration_ms": duration,
            "output_size": len(result)
        }
    )

    return result
```

4. **Update configuration schema:**
```python
# src/config/schema.py
class YourFeatureConfig(BaseModel):
    """Configuration for your feature."""

    enabled: bool = Field(default=True, description="Enable feature")
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold"
    )

    class Config:
        frozen = True
```

### Debugging

1. **Enable debug logging:**
```bash
export WHEEL_LOG_LEVEL=DEBUG
python run_aligned.py --diagnose
```

2. **Performance profiling:**
```python
# Add to any slow function
from src.unity_wheel.monitoring import profile_performance

@profile_performance
def slow_function():
    # Function will log if >10ms
    pass
```

3. **View metrics:**
```bash
# Real-time monitoring
./scripts/monitor.sh

# Export metrics
python run_aligned.py --export-metrics
```

## Environment Setup

### Directory Structure

```
~/.wheel/
├── secrets/          # Encrypted credentials
├── cache/           # API response cache
│   ├── databento/   # Options data
│   ├── schwab/      # Account data
│   └── fred/        # Economic data
├── logs/            # Application logs
└── metrics/         # Performance data
```

### Configuration

1. **Main config** (`config.yaml`):
```yaml
# Core settings
mode: "development"  # or "production"
portfolio_value: 100000
log_level: "INFO"

# Feature flags
features:
  ml_enhanced: false
  risk_parity: true
  auto_tune: true

# Integration settings
databento:
  enabled: true
  cache_ttl: 3600

schwab:
  enabled: true
  rate_limit: 120

fred:
  enabled: true
  fallback_rate: 0.05
```

2. **Environment overrides:**
```bash
# Override any config value
export WHEEL_MODE=production
export WHEEL_FEATURES__ML_ENHANCED=true
export WHEEL_DATABENTO__CACHE_TTL=7200
```

### Monitoring

1. **Structured logs** (JSON format):
```json
{
  "timestamp": "2024-01-20T10:30:00Z",
  "level": "INFO",
  "function": "generate_recommendation",
  "duration_ms": 45,
  "confidence": 0.92,
  "position_size": 10000,
  "message": "Recommendation generated"
}
```

2. **Metrics tracking:**
- SLA violations (>1s response time)
- Confidence degradation
- Cache hit rates
- API call counts

3. **Health checks:**
```bash
# Continuous monitoring
./scripts/autonomous-checks.sh

# One-time check
python run_aligned.py --diagnose
```

## Common Tasks

### Update Dependencies

```bash
# Update all dependencies
poetry update

# Update specific package
poetry add package@latest

# Regenerate requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

### Run Maintenance

```bash
# Clean old cache/logs
./scripts/maintenance.sh

# Optimize local database
python -c "
from src.unity_wheel.storage import optimize_storage
optimize_storage()
"
```

### Profile Performance

```python
# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code
    pass

# CPU profiling
import cProfile
cProfile.run('your_function()', 'profile_stats')
```

## Deployment Options

### Local Development

Default setup for single-user workstation:

```bash
# Development mode
python run_aligned.py --portfolio 100000

# Production mode
export WHEEL_MODE=production
python run_aligned.py --portfolio 100000
```

### Docker Container

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV WHEEL_MODE=production
CMD ["python", "run_aligned.py"]
```

Build and run:
```bash
docker build -t wheel-bot .
docker run -v ~/.wheel:/root/.wheel wheel-bot
```

### Cloud Run (GCP)

Deploy as scheduled job:

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT/wheel-bot

# Deploy job
gcloud run jobs replace cloud_run_job.yaml

# Run manually
gcloud run jobs execute wheel-bot
```

## Troubleshooting

### Common Issues

1. **Import errors:**
```bash
# Ensure in project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

2. **Permission denied:**
```bash
# Fix secret permissions
chmod 600 ~/.wheel/secrets/*
```

3. **Memory issues:**
```bash
# Increase limits
export WHEEL_CACHE__MAX_MEMORY_MB=500
```

4. **Slow performance:**
```python
# Enable profiling
python run_aligned.py --performance
```

### Debug Commands

```bash
# Validate environment
python -m unity_wheel.validate

# Check configuration
python -c "
from src.config.loader import get_config_loader
loader = get_config_loader()
print(loader.generate_health_report())
"

# Test integrations
python -c "
from src.unity_wheel.schwab import test_connection
test_connection()
"
```

## Best Practices

1. **Always include confidence scores** in calculations
2. **Log all external API calls** with timing
3. **Use property-based testing** for mathematical functions
4. **Implement graceful degradation** for all features
5. **Monitor resource usage** in production
6. **Document assumptions** in code comments
7. **Version all configuration changes**
8. **Profile before optimizing**

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests (maintain >90% coverage)
4. Ensure all checks pass
5. Submit pull request

For questions or issues, check the logs first:
```bash
tail -f ~/.wheel/logs/wheel-bot.log
```
