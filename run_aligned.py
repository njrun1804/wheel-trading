#!/usr/bin/env python3
"""Unity wheel trading decision engine - aligned with autonomous operation requirements."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Final, Literal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from rich.console import Console
from rich.table import Table

import src.unity_wheel as unity_wheel
from src.config import get_settings
from src.config.unity import COMPANY_NAME, TICKER
from src.unity_wheel import __version__, get_version_string
from src.unity_wheel.__version__ import API_VERSION
from src.unity_wheel.api import MarketSnapshot, OptionData, WheelAdvisor
from src.unity_wheel.diagnostics import SelfDiagnostics
from src.unity_wheel.monitoring import get_performance_monitor
from src.unity_wheel.observability import get_observability_exporter
from src.unity_wheel.risk import RiskLimits
from src.unity_wheel.strategy import WheelParameters

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

console = Console()

OutputFormat = Literal["text", "json"]


def setup_deterministic_environment() -> None:
    """Ensure deterministic behavior for all operations."""
    import random

    import numpy as np

    # Set seeds for reproducibility
    SEED: Final[int] = 42
    np.random.seed(SEED)
    random.seed(SEED)

    # Set numpy error handling
    np.seterr(all="raise")  # Raise exceptions on numerical errors

    logger.info(
        "Deterministic environment configured",
        extra={
            "seed": SEED,
            "version": __version__,
            "api_version": API_VERSION,
        },
    )


def create_mock_market_snapshot(
    current_price: float,
    portfolio_value: float,
    ticker: str = TICKER,
) -> MarketSnapshot:
    """Create mock market snapshot for demo purposes."""
    strikes = [30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0]

    # Create mock option chain
    option_chain = {}
    for strike in strikes:
        # Calculate mock option prices
        distance = abs(current_price - strike) / current_price
        base_premium = 2.0 * (1 - distance)  # Simple approximation

        option_chain[str(strike)] = OptionData(
            strike=strike,
            expiration="2024-01-15",  # Mock date
            bid=max(0.10, base_premium - 0.05),
            ask=base_premium + 0.05,
            mid=base_premium,
            volume=100 + int(50 / (1 + distance)),
            open_interest=500 + int(200 / (1 + distance)),
            delta=-0.3 * (1 - distance),  # Approximate delta
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            implied_volatility=0.65,  # Unity typical IV
        )

    return MarketSnapshot(
        timestamp=datetime.now(),
        ticker=ticker,
        current_price=current_price,
        buying_power=portfolio_value,
        margin_used=0.0,
        positions=[],
        option_chain=option_chain,
        implied_volatility=0.65,
        risk_free_rate=0.05,
    )


def generate_recommendation(
    portfolio_value: float, current_price: float | None = None, output_format: OutputFormat = "text"
) -> dict[str, Any]:
    """Generate wheel strategy recommendation for Unity using new API."""
    settings = get_settings()

    # Create wheel parameters from settings
    wheel_params = WheelParameters(
        target_delta=settings.wheel_delta_target,
        target_dte=settings.days_to_expiry_target,
        max_position_size=settings.max_position_size,
    )

    # Create risk limits (could be from config)
    risk_limits = RiskLimits(
        max_var_95=0.05,
        max_cvar_95=0.075,
        max_margin_utilization=0.5,
    )

    # Initialize advisor
    advisor = WheelAdvisor(wheel_params, risk_limits)

    # Use mock data for demo
    if current_price is None:
        current_price = 35.50  # Unity typical price range

    # Create market snapshot
    market_snapshot = create_mock_market_snapshot(current_price, portfolio_value)

    try:
        # Get recommendation from advisor
        rec = advisor.advise_position(market_snapshot)

        # Convert to legacy format for compatibility
        recommendation = {
            "timestamp": datetime.now().isoformat(),
            "ticker": TICKER,
            "company": COMPANY_NAME,
            "current_price": current_price,
            "portfolio_value": portfolio_value,
            "recommendation": {
                "action": rec["action"],
                "rationale": rec["rationale"],
                "confidence": rec["confidence"],
                "risk_metrics": rec["risk"],
            },
            "details": rec.get("details", {}),
            "diagnostics": {
                "delta_target": wheel_params.target_delta,
                "dte_target": wheel_params.target_dte,
                "max_position_size": wheel_params.max_position_size,
            },
        }

        # Parse specific recommendation details
        if rec["action"] == "ADJUST" and "strike" in rec.get("details", {}):
            details = rec["details"]
            recommendation["recommendation"].update(
                {
                    "strike": details["strike"],
                    "contracts": details["contracts"],
                    "expiry_days": wheel_params.target_dte,
                    "max_risk": rec["risk"]["max_loss"],
                }
            )

    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}", exc_info=True)
        recommendation = {
            "timestamp": datetime.now().isoformat(),
            "ticker": TICKER,
            "company": COMPANY_NAME,
            "current_price": current_price,
            "portfolio_value": portfolio_value,
            "error": str(e),
            "recommendation": {
                "action": "ERROR",
                "rationale": f"Calculation failed: {str(e)}",
                "confidence": 0.0,
                "risk_metrics": {
                    "max_loss": 0.0,
                    "probability_assignment": 0.0,
                    "expected_return": 0.0,
                    "edge_ratio": 0.0,
                    "var_95": 0.0,
                    "margin_required": 0.0,
                },
            },
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
    table.add_row("Target Delta", f"{rec['diagnostics']['delta_target']:.2f}")
    table.add_row("Target DTE", f"{rec['diagnostics']['dte_target']} days")

    console.print(table)

    if rec.get("error"):
        console.print(f"\n[red]❌ ERROR: {rec['error']}[/red]")
    elif rec["recommendation"]["action"] == "ADJUST":
        r = rec["recommendation"]
        console.print(f"\n[green]✅ RECOMMENDATION: {r['rationale']}[/green]")
        if "strike" in r:
            console.print(f"   Strike: ${r['strike']:.2f}")
            console.print(f"   Contracts: {r['contracts']}")
            console.print(f"   Expiry: {r['expiry_days']} days")
            console.print(f"   Max Risk: ${r['max_risk']:,.0f}")
        console.print(f"   Confidence: {r['confidence']:.1%}")

        # Risk metrics
        rm = r["risk_metrics"]
        console.print(f"\n[yellow]Risk Metrics:[/yellow]")
        console.print(f"   Probability of Assignment: {rm['probability_assignment']:.1%}")
        console.print(f"   Expected Return: {rm['expected_return']:.2%}")
        console.print(f"   Edge Ratio: {rm['edge_ratio']:.3f}")
        console.print(f"   95% VaR: ${rm['var_95']:,.0f}")
    else:
        console.print(
            f"\n[yellow]⏸️  {rec['recommendation']['action']}: {rec['recommendation']['rationale']}[/yellow]"
        )
        console.print(f"   Confidence: {rec['recommendation']['confidence']:.1%}")


@click.command()
@click.option("--portfolio", type=float, default=100000, help="Portfolio value")
@click.option("--price", type=float, help="Current Unity price (uses mock data if not specified)")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option("--diagnose", is_flag=True, help="Run self-diagnostics")
@click.option("--performance", is_flag=True, help="Show performance metrics")
@click.option("--export-metrics", is_flag=True, help="Export metrics for dashboards")
@click.option("--version", is_flag=True, help="Show version information")
@click.option("--verbose", is_flag=True, help="Verbose logging")
def main(
    portfolio: float,
    price: float | None,
    output_format: OutputFormat,
    diagnose: bool,
    performance: bool,
    export_metrics: bool,
    version: bool,
    verbose: bool,
) -> int:
    """Unity wheel trading decision engine - autonomous operation mode."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show version if requested
    if version:
        if output_format == "json":
            print(
                json.dumps(
                    {
                        "version": __version__,
                        "components": {
                            k: v for k, v in unity_wheel.__version__.COMPONENT_VERSIONS.items()
                        },
                        "api_version": API_VERSION,
                    },
                    indent=2,
                )
            )
        else:
            print(get_version_string())
        return 0

    # Run diagnostics if requested
    if diagnose:
        diag = SelfDiagnostics()
        success = diag.run_all_checks()
        print(diag.report(format=output_format))
        return 0 if success else 1

    # Show performance metrics if requested
    if performance:
        monitor = get_performance_monitor()
        print(monitor.generate_report(format=output_format))
        return 0

    # Export metrics for dashboards if requested
    if export_metrics:
        exporter = get_observability_exporter()
        data = exporter.collect_current_metrics()

        # Export in multiple formats
        json_path = exporter.export_json(data)
        influx_path = exporter.export_influxdb(data)
        prom_path = exporter.export_prometheus(data)
        csv_path = exporter.export_csv(data)

        # Store in database
        exporter.store_metrics(data)

        # Generate summary
        summary = exporter.generate_summary_report(hours=24)

        if output_format == "json":
            print(
                json.dumps(
                    {
                        "exports": {
                            "json": str(json_path),
                            "influxdb": str(influx_path),
                            "prometheus": str(prom_path),
                            "csv": str(csv_path),
                        },
                        "summary": summary,
                    },
                    indent=2,
                )
            )
        else:
            console.print("[green]✅ Metrics exported successfully[/green]")
            console.print(f"   JSON: {json_path}")
            console.print(f"   InfluxDB: {influx_path}")
            console.print(f"   Prometheus: {prom_path}")
            console.print(f"   CSV: {csv_path}")
            console.print(f"\n[blue]24-hour Summary:[/blue]")
            console.print(f"   Total metrics: {summary['metrics']['total_count']}")
            console.print(f"   Unique metrics: {len(summary['metrics']['unique_metrics'])}")
            console.print(f"   Events: {len(summary['events'])}")

        return 0

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
        error_response = {"timestamp": datetime.now().isoformat(), "error": str(e), "type": "FATAL"}

        if output_format == "json":
            print(json.dumps(error_response, indent=2))
        else:
            console.print(f"[red]FATAL ERROR: {e}[/red]")

        return 1


if __name__ == "__main__":
    sys.exit(main())
