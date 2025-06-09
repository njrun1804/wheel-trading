# Simple Makefile for local development
.PHONY: help run test dev clean install housekeeping-check

help:
	@echo "Wheel Trading - Local Development"
	@echo "  make install           - Install dependencies"
	@echo "  make run               - Run the application"
	@echo "  make test              - Run tests"
	@echo "  make dev               - Run in development mode (auto-reload)"
	@echo "  make clean             - Clean cache files"
	@echo "  make housekeeping-check - Check for file organization issues"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

run:
	python -m src.main

dev:
	python -m src.main --verbose

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

housekeeping-check:
	@echo "🔍 Checking for housekeeping violations..."
	@echo ""
	@echo "Checking test files..."
	@find . -name "test_*.py" -not -path "./tests/*" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./build/*" | grep . && echo "❌ Test files found outside tests/" && exit 1 || echo "✅ All tests properly organized"
	@echo ""
	@echo "Checking example files..."
	@find . -name "example_*.py" -not -path "./examples/*" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./build/*" | grep . && echo "❌ Example files found outside examples/" && exit 1 || echo "✅ All examples properly organized"
	@echo ""
	@echo "Checking for status docs in root..."
	@ls -1 *.md 2>/dev/null | grep -E "(SUMMARY|STATUS|COMPLETE|REPORT)\.md$$" | grep . && echo "❌ Status/summary docs found in root" && exit 1 || echo "✅ No status docs in root"
	@echo ""
	@echo "Checking for duplicate/old scripts..."
	@find . -name "pull_*_data.py" -not -path "./tools/*" -not -path "./venv/*" -not -path "./.venv/*" | grep . && echo "❌ Data pulling scripts found outside tools/" && exit 1 || echo "✅ Data scripts properly organized"
	@echo ""
	@echo "Checking for empty directories..."
	@find ./src -type d -empty | grep . && echo "⚠️  Empty directories found (consider removing)" || echo "✅ No empty directories"
	@echo ""
	@echo "📊 Summary Statistics:"
	@echo "  Root files: $$(ls -1 | grep -v -E '^(src|tests|examples|tools|scripts|deployment|docs|data|exports|venv|htmlcov|\.github)$$' | wc -l)"
	@echo "  Test files: $$(find tests -name "test_*.py" -type f 2>/dev/null | wc -l)"
	@echo "  Example files: $$(find examples -name "*.py" -type f 2>/dev/null | wc -l)"
	@echo "  Tool scripts: $$(find tools -name "*.py" -type f 2>/dev/null | wc -l)"
	@echo ""
	@echo "✨ Housekeeping check complete!"

# Quick commands for development
check: test
	python -m flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503,C901

format:
	python -m black src/ tests/
	python -m isort src/ tests/