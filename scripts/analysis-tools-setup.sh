#!/bin/bash

echo "Setting up Analysis & Data Quality Tools for Wheel Trading"
echo "========================================================="
echo ""

# 1. Mathematical Analysis Tools
echo "1. Installing mathematical analysis tools..."
pip install --user \
    statsmodels \
    scipy \
    scikit-learn \
    sympy \
    numba \
    bottleneck

# 2. Data Quality & Validation
echo "2. Installing data validation tools..."
pip install --user \
    pandera \
    pydantic \
    jsonschema \
    cerberus

# 3. Performance Optimization for Analysis
echo "3. Installing performance tools..."
pip install --user \
    line_profiler \
    memory_profiler \
    snakeviz

# 4. Visualization for Analysis
echo "4. Installing visualization tools..."
pip install --user \
    seaborn \
    plotly \
    kaleido \
    dash

# 5. Create analysis helper scripts
cat > analysis_helpers.py << 'EOF'
"""Helper functions for mathematical analysis and validation."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any

def validate_option_params(delta: float, iv: float, dte: int) -> Dict[str, Any]:
    """Validate option parameters are within reasonable bounds."""
    issues = []

    if not 0 < delta < 1:
        issues.append(f"Delta {delta} outside valid range (0,1)")
    if not 0 < iv < 5:  # 500% IV is extreme but possible
        issues.append(f"IV {iv} outside reasonable range (0,5)")
    if dte < 0:
        issues.append(f"DTE {dte} cannot be negative")

    return {"valid": len(issues) == 0, "issues": issues}

def calculate_kelly_criterion(win_prob: float, win_amt: float, loss_amt: float) -> float:
    """Calculate optimal position size using Kelly Criterion."""
    if loss_amt == 0:
        return 0

    b = win_amt / loss_amt
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b
    return max(0, min(kelly, 0.25))  # Cap at 25% for safety

def analyze_returns_distribution(returns: pd.Series) -> Dict[str, float]:
    """Analyze return distribution for risk metrics."""
    return {
        "mean": returns.mean(),
        "std": returns.std(),
        "skew": stats.skew(returns),
        "kurtosis": stats.kurtosis(returns),
        "sharpe": returns.mean() / returns.std() * np.sqrt(252),
        "var_95": returns.quantile(0.05),
        "cvar_95": returns[returns <= returns.quantile(0.05)].mean()
    }
EOF

echo ""
echo "Creating Makefile for common analysis tasks..."
cat > Makefile << 'EOF'
.PHONY: analyze test profile validate clean

# Run position analysis
analyze:
	python run.py -p 100000 --verbose

# Run tests with coverage
test:
	pytest tests/ -v --cov=src --cov-report=html

# Profile performance
profile:
	python -m cProfile -o profile.stats run.py -p 100000
	snakeviz profile.stats

# Memory profiling
memory:
	python -m memory_profiler run.py -p 100000

# Validate data quality
validate:
	python quick_data_assessment.py
	python verify_database_integrity.py

# Check mathematical consistency
math-check:
	python -m pytest tests/test_math.py tests/test_options.py -v

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f profile.stats
	rm -f *.log
EOF

echo ""
echo "✓ Analysis tools setup complete!"
echo ""
echo "New capabilities:"
echo "- Enhanced mathematical analysis (statsmodels, scipy)"
echo "- Data validation (pandera, pydantic)"
echo "- Performance profiling (line_profiler, memory_profiler)"
echo "- Advanced visualizations (plotly, dash)"
echo ""
echo "Use 'make analyze' to run analysis with profiling"
echo "Use 'make validate' to check data quality"
echo "Use 'make math-check' to verify mathematical correctness"
