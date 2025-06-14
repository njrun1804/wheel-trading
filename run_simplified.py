#!/usr/bin/env python3
"""
Simplified run.py that bypasses complex risk analytics.
"""

import click
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from test_simple_advisor import get_simple_recommendation


@click.command()
@click.option('--portfolio', default=100000, help='Portfolio value')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--diagnose', is_flag=True, help='Run diagnostics')
def main(portfolio, format, diagnose):
    """Unity Wheel Trading System - Simplified Version."""
    
    if diagnose:
        # Run diagnostics
        from src.unity_wheel.cli.run import run_diagnostics
        run_diagnostics()
        return
    
    # Get recommendation using simplified logic
    get_simple_recommendation(portfolio)


if __name__ == "__main__":
    main()