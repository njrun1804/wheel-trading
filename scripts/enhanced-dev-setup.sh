#!/bin/bash

echo "Enhanced Development Setup for Wheel Trading Bot"
echo "=============================================="
echo ""

# 1. Python Static Analysis Tools
echo "1. Installing Python analysis tools..."
pip install --user pylsp-mypy pylsp-rope python-lsp-ruff

# 2. Git Hooks for Quality
echo "2. Setting up pre-commit hooks..."
pip install --user pre-commit
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF

# 3. Enhanced Git Integration
echo "3. Configuring Git for better Claude integration..."
git config core.commentChar ";"
git config merge.tool "vimdiff"
git config diff.algorithm "histogram"

# 4. Create Claude-friendly aliases
echo "4. Adding development aliases..."
cat >> ~/.zshrc << 'EOF'

# Wheel Trading Development Aliases
alias wt-test="pytest -xvs --tb=short"
alias wt-coverage="pytest --cov=src --cov-report=html"
alias wt-profile="python -m cProfile -o profile.stats"
alias wt-memory="python -m memory_profiler"
alias wt-security="bandit -r src/"
alias wt-deps="pipdeptree --warn silence | grep -E '^[^ ]'"
alias wt-clean="find . -type f -name '*.pyc' -delete && find . -type d -name '__pycache__' -delete"
alias wt-format="black . && ruff check . --fix"
alias wt-types="mypy src/ --install-types --non-interactive"

# Quick data checks
alias wt-data="duckdb data/market_data.db '.tables'"
alias wt-options="python -c 'from src.unity_wheel.analytics.iv_surface import IVSurface; IVSurface().plot()'"
alias wt-positions="python parse_my_positions.py"

# Claude-specific helpers
alias claude-summary="find . -name '*.py' -exec wc -l {} + | sort -n | tail -20"
alias claude-recent="git log --oneline --graph --all --since='1 week ago'"
alias claude-changes="git diff --stat HEAD~5..HEAD"
EOF

echo ""
echo "✓ Enhanced setup complete!"
