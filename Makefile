# Simple Makefile for local development
.PHONY: help run test dev clean install

help:
	@echo "Wheel Trading - Local Development"
	@echo "  make install  - Install dependencies"
	@echo "  make run      - Run the application"
	@echo "  make test     - Run tests"
	@echo "  make dev      - Run in development mode (auto-reload)"
	@echo "  make clean    - Clean cache files"

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

# Quick commands for development
check: test
	python -m flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503,C901

format:
	python -m black src/ tests/
	python -m isort src/ tests/