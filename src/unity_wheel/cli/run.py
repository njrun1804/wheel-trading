#!/usr/bin/env python3
"""Unity wheel trading decision engine - aligned with autonomous operation requirements."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
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
from src.unity_wheel.data_providers.base import FREDDataManager
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from src.unity_wheel.data_providers.databento.integration import DatabentoIntegration
from src.unity_wheel.data_providers.validation import validate_market_data
from src.unity_wheel.metrics import metrics_collector
from src.unity_wheel.monitoring import get_performance_monitor
from src.unity_wheel.monitoring.diagnostics import SelfDiagnostics
from src.unity_wheel.observability import get_observability_exporter
from src.unity_wheel.risk import RiskLimits
from src.unity_wheel.secrets.integration import SecretInjector
from src.unity_wheel.storage.storage import Storage
from src.unity_wheel.strategy import WheelParameters

# Configure simple logging to avoid conflicts with StructuredLogger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Just the message, since StructuredLogger already formats as JSON
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


# Removed create_mock_market_snapshot - now using real Databento data only


def generate_recommendation(
    portfolio_value: float, output_format: OutputFormat = "text"
) -> dict[str, Any]:
    """Generate wheel strategy recommendation for Unity using real market data."""
    settings = get_settings()

    # Initialize FRED data manager and retrieve risk metrics
    storage = Storage()
    asyncio.run(storage.initialize())
    fred_manager = FREDDataManager(storage=storage)
    rf_rate, _ = asyncio.run(fred_manager.get_or_fetch_risk_free_rate(3))
    regime, vix = asyncio.run(fred_manager.get_volatility_regime())
    logger.info(
        "Fetched FRED metrics",
        extra={"risk_free_rate": rf_rate, "volatility_regime": regime, "vix": vix},
    )

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
    # Scale limits based on volatility regime
    risk_limits.scale_by_volatility(vix / 100 if vix else 0.2)

    # Initialize advisor
    advisor = WheelAdvisor(wheel_params, risk_limits)

    # Import the real data function
    from .databento_integration import get_market_data_sync

    # Get real market data - fail if not available
    try:
        market_snapshot, confidence = get_market_data_sync(
            portfolio_value, TICKER, risk_free_rate=rf_rate
        )
        logger.info("Successfully fetched real Unity market data", extra={"confidence": confidence})
        current_price = market_snapshot.current_price

        # CRITICAL: Validate this is real market data, not mock/dummy data
        validate_market_data(market_snapshot)
        logger.info("Market data validation passed - using real data")

    except Exception as e:
        error_msg = f"Unable to fetch real Unity market data: {e}"
        logger.error(error_msg)
        # Re-raise to fail the program as requested
        raise

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
        # Re-raise to fail the program
        raise

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
@click.option(
    "--price",
    type=float,
    help="[DEPRECATED] Current Unity price - now fetched from real market data",
)
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
        perf_report = monitor.generate_report(format=output_format)
        metrics_report = metrics_collector.generate_report()
        if output_format == "json":
            print(
                json.dumps(
                    {
                        "performance": json.loads(perf_report),
                        "decision_metrics": metrics_report,
                    },
                    indent=2,
                )
            )
        else:
            print(perf_report)
            print("")
            print(metrics_report)
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
        rec = generate_recommendation(portfolio, output_format)

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
