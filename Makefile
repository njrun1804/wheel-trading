# Simplified Makefile for Unity Wheel Bot (Single-User Recommendation System)

.PHONY: help quick recommend test clean install diagnose

# Default target
help:
	@echo "Unity Wheel Bot - Recommendation System"
	@echo "======================================"
	@echo ""
	@echo "Common tasks:"
	@echo "  make quick      - Quick system health check"
	@echo "  make recommend  - Get trading recommendation ($100k portfolio)"
	@echo "  make test       - Run critical tests only"
	@echo "  make clean      - Clean temporary files"
	@echo ""
	@echo "Setup & maintenance:"
	@echo "  make install    - Install/update dependencies"
	@echo "  make diagnose   - Full system diagnostics"
	@echo ""
	@echo "Examples:"
	@echo "  make recommend PORTFOLIO=50000   - Recommend for $50k portfolio"

# Quick health check - run this first
quick:
	@bash scripts/quick_check.sh

# Get recommendation (default $100k portfolio)
PORTFOLIO ?= 100000
recommend:
	poetry run python run_aligned.py --portfolio $(PORTFOLIO)

# Run only critical tests (math, risk, recommendations)
test:
	@echo "🧪 Running critical tests..."
	@poetry run pytest tests/test_math.py tests/test_options_properties.py -v --tb=short
	@poetry run pytest tests/test_e2e_recommendation_flow.py -v --tb=short
	@echo "✅ Core functionality verified"

# Install dependencies using Poetry
install:
	@echo "📦 Installing dependencies..."
	@poetry install --no-interaction
	@echo "✅ Dependencies installed"

# Full system diagnostics
diagnose:
	@echo "🏥 Running full diagnostics..."
	@poetry run python run_aligned.py --diagnose

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov .mypy_cache 2>/dev/null || true
	@rm -f .DS_Store 2>/dev/null || true
	@echo "✅ Cleaned"

# Hidden targets for Claude Code use
.PHONY: _format _typecheck _full-test

# Auto-format code (Claude Code handles this)
_format:
	@poetry run black src tests --quiet
	@poetry run isort src tests --quiet

# Type checking for critical modules
_typecheck:
	@poetry run mypy src/unity_wheel/math/ --strict
	@poetry run mypy src/unity_wheel/risk/ --strict

# Full test suite (for CI)
_full-test:
	@poetry run pytest -v --cov=src/unity_wheel
