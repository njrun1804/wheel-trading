#!/usr/bin/env python3
"""Unity wheel trading decision engine - aligned with autonomous operation requirements."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, Final, Literal, Never

import click
from rich.console import Console
from rich.table import Table

from src.config import get_settings
from src.config.unity import TICKER, COMPANY_NAME
from src.diagnostics import run_diagnostics
from src.wheel import WheelStrategy

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)

console = Console()

OutputFormat = Literal["text", "json"]


def setup_deterministic_environment() -> None:
    """Ensure deterministic behavior for all operations."""
    import numpy as np
    import random
    
    # Set seeds for reproducibility
    SEED: Final[int] = 42
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Set numpy error handling
    np.seterr(all='raise')  # Raise exceptions on numerical errors
    
    logger.info("Deterministic environment configured", extra={"seed": SEED})


def generate_recommendation(
    portfolio_value: float,
    current_price: float | None = None,
    output_format: OutputFormat = "text"
) -> dict[str, Any]:
    """Generate wheel strategy recommendation for Unity."""
    settings = get_settings()
    wheel = WheelStrategy()
    
    # Mock data - will be replaced with Schwab API
    if current_price is None:
        current_price = 35.50  # Unity typical price range
    
    strikes = [30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0]
    
    recommendation = {
        "timestamp": datetime.now().isoformat(),
        "ticker": TICKER,
        "company": COMPANY_NAME,
        "current_price": current_price,
        "portfolio_value": portfolio_value,
        "recommendation": None,
        "diagnostics": {
            "delta_target": settings.wheel_delta_target,
            "dte_target": settings.days_to_expiry_target,
            "max_position_size": settings.max_position_size
        }
    }
    
    try:
        # Find optimal put strike
        optimal_put = wheel.find_optimal_put_strike(
            current_price=current_price,
            available_strikes=strikes,
            volatility=0.65,  # Unity typical IV
            days_to_expiry=settings.days_to_expiry_target,
        )
        
        if optimal_put:
            contracts = wheel.calculate_position_size(TICKER, current_price, portfolio_value)
            
            recommendation["recommendation"] = {
                "action": "SELL_PUT",
                "strike": optimal_put,
                "contracts": contracts,
                "expiry_days": settings.days_to_expiry_target,
                "max_risk": optimal_put * 100 * contracts,
                "confidence": 0.85  # Placeholder - will be calculated
            }
        else:
            recommendation["recommendation"] = {
                "action": "HOLD",
                "reason": "No suitable strikes found",
                "confidence": 0.0
            }
            
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        recommendation["error"] = str(e)
        recommendation["recommendation"] = {
            "action": "ERROR",
            "reason": "Calculation failed",
            "confidence": 0.0
        }
    
    return recommendation


def display_recommendation_text(rec: dict[str, Any]) -> None:
    """Display recommendation in human-readable format."""
    console.print(f"\n[bold blue]🎯 Unity ({TICKER}) Wheel Strategy Recommendation[/bold blue]")
    console.print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    console.print("=" * 60)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Current Price", f"${rec['current_price']:.2f}")
    table.add_row("Portfolio Value", f"${rec['portfolio_value']:,.0f}")
    
    console.print(table)
    
    if rec.get("error"):
        console.print(f"\n[red]❌ ERROR: {rec['error']}[/red]")
    elif rec["recommendation"]["action"] == "SELL_PUT":
        r = rec["recommendation"]
        console.print(f"\n[green]✅ RECOMMENDATION: Sell {r['contracts']} {TICKER} ${r['strike']}P[/green]")
        console.print(f"   Expiry: {r['expiry_days']} days")
        console.print(f"   Max Risk: ${r['max_risk']:,.0f}")
        console.print(f"   Confidence: {r['confidence']:.1%}")
    else:
        console.print(f"\n[yellow]⏸️  {rec['recommendation']['action']}: {rec['recommendation']['reason']}[/yellow]")


@click.command()
@click.option('--portfolio', type=float, default=100000, help='Portfolio value')
@click.option('--price', type=float, help='Current Unity price (uses live data if not specified)')
@click.option('--format', 'output_format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--diagnose', is_flag=True, help='Run self-diagnostics')
@click.option('--verbose', is_flag=True, help='Verbose logging')
def main(
    portfolio: float,
    price: float | None,
    output_format: OutputFormat,
    diagnose: bool,
    verbose: bool
) -> int:
    """Unity wheel trading decision engine - autonomous operation mode."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run diagnostics if requested
    if diagnose:
        return run_diagnostics(output_format)
    
    # Set up deterministic environment
    setup_deterministic_environment()
    
    # Generate recommendation
    try:
        rec = generate_recommendation(portfolio, price, output_format)
        
        if output_format == "json":
            print(json.dumps(rec, indent=2))
        else:
            display_recommendation_text(rec)
        
        return 0
    except Exception as e:
        error_response = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "type": "FATAL"
        }
        
        if output_format == "json":
            print(json.dumps(error_response, indent=2))
        else:
            console.print(f"[red]FATAL ERROR: {e}[/red]")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())