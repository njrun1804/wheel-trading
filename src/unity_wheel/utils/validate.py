"""Environment validation script for unity-wheel-bot."""

from __future__ import annotations

import importlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from src.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class EnvironmentValidator:
    """Validate entire unity-wheel-bot environment."""

    def __init__(self):
        """Initialize validator."""
        self.results: Dict[str, Tuple[bool, str]] = {}
        self.critical_failures = 0
        self.warnings = 0

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        console.print("\n[bold blue]Unity Wheel Bot - Environment Validation[/bold blue]\n")

        # Check Python version
        self._check_python_version()

        # Check imports
        self._check_imports()

        # Check models
        self._check_models()

        # Check math functions
        self._check_math_functions()

        # Check risk analytics
        self._check_risk_analytics()

        # Check logging
        self._check_logging()

        # Display results
        self._display_results()

        return self.critical_failures == 0

    def _check_python_version(self) -> None:
        """Check Python version meets requirements."""
        required_major = 3
        required_minor = 9

        current_major = sys.version_info.major
        current_minor = sys.version_info.minor

        if current_major == required_major and current_minor >= required_minor:
            self.results["Python Version"] = (
                True,
                f"✓ {current_major}.{current_minor} (required: {required_major}.{required_minor}+)",
            )
        else:
            self.results["Python Version"] = (
                False,
                f"✗ {current_major}.{current_minor} (required: {required_major}.{required_minor}+)",
            )
            self.critical_failures += 1

    def _check_imports(self) -> None:
        """Check all required packages can be imported."""
        required_packages = [
            "numpy",
            "scipy",
            "pandas",
            "pydantic",
            "hypothesis",
            "pytest",
            "click",
            "rich",
        ]

        for package in required_packages:
            try:
                importlib.import_module(package)
                self.results[f"Import {package}"] = (True, "✓ Imported successfully")
            except ImportError as e:
                self.results[f"Import {package}"] = (False, f"✗ {str(e)}")
                self.critical_failures += 1

    def _check_models(self) -> None:
        """Check model functionality."""
        try:
            from src.unity_wheel.models import Account, Greeks, Position

            # Get Unity ticker from config
            config = get_config()
            ticker = config.unity.ticker

            # Test Position
            pos = Position(ticker, 100)
            assert pos.symbol == ticker
            assert pos.quantity == 100
            assert pos.position_type.value == "stock"

            # Test option position
            opt_symbol = f"{ticker}241220C00080000"
            opt = Position(opt_symbol, -1)
            assert opt.strike == 80.0
            assert opt.underlying == ticker

            self.results["Position Model"] = (True, "✓ All tests passed")

            # Test Greeks
            greeks = Greeks(delta=0.5, gamma=0.02, theta=-0.05)
            assert greeks.delta == 0.5

            self.results["Greeks Model"] = (True, "✓ All tests passed")

            # Test Account
            account = Account(
                cash_balance=50000.0,
                buying_power=100000.0,
                margin_used=10000.0,
            )
            assert account.margin_available == 40000.0

            self.results["Account Model"] = (True, "✓ All tests passed")

        except Exception as e:
            self.results["Models"] = (False, f"✗ {str(e)}")
            self.critical_failures += 1

    def _check_math_functions(self) -> None:
        """Check options math functionality."""
        try:
            from src.unity_wheel.math.options import (
                black_scholes_price_validated,
                calculate_all_greeks,
                implied_volatility_validated,
                probability_itm_validated,
            )

            # Test Black-Scholes
            result = black_scholes_price_validated(100, 100, 1, 0.05, 0.2, "call")
            assert result.confidence > 0.9
            assert 10 < result.value < 11  # Approximate expected value

            self.results["Black-Scholes"] = (
                True,
                f"✓ Price={result.value:.2f}, Confidence={result.confidence:.2f}",
            )

            # Test Greeks calculation
            greeks, confidence = calculate_all_greeks(100, 100, 1, 0.05, 0.2, "call")
            assert confidence > 0.9
            assert 0.5 < greeks["delta"] < 0.7

            self.results["Greeks Calculation"] = (True, "✓ All Greeks calculated")

            # Test IV calculation
            iv_result = implied_volatility_validated(10.45, 100, 100, 1, 0.05, "call")
            assert iv_result.confidence > 0.8
            assert 0.18 < iv_result.value < 0.22  # Should be close to 0.2

            self.results["Implied Volatility"] = (
                True,
                f"✓ IV={iv_result.value:.3f}, Confidence={iv_result.confidence:.2f}",
            )

            # Test probability ITM
            prob_result = probability_itm_validated(100, 100, 1, 0.05, 0.2, "call")
            assert prob_result.confidence > 0.9
            assert 0.5 < prob_result.value < 0.6

            self.results["Probability ITM"] = (True, "✓ Calculation successful")

        except Exception as e:
            self.results["Math Functions"] = (False, f"✗ {str(e)}")
            self.critical_failures += 1

    def _check_risk_analytics(self) -> None:
        """Check risk analytics functionality."""
        try:
            from src.unity_wheel.risk.analytics import RiskAnalyzer, RiskLimits

            # Create analyzer
            limits = RiskLimits(max_var_95=0.05, max_cvar_95=0.075)
            analyzer = RiskAnalyzer(limits)

            # Test VaR calculation
            returns = np.random.normal(0.001, 0.02, 1000)
            var, confidence = analyzer.calculate_var(returns, 0.95)
            assert 0 < var < 0.1
            assert confidence > 0.5

            self.results["VaR Calculation"] = (
                True,
                f"✓ VaR={var:.3f}, Confidence={confidence:.2f}",
            )

            # Test CVaR calculation
            cvar, confidence = analyzer.calculate_cvar(returns, 0.95)
            assert cvar >= var  # CVaR should be >= VaR
            assert confidence > 0.5

            self.results["CVaR Calculation"] = (True, "✓ CVaR calculated correctly")

            # Test Kelly criterion
            kelly, confidence = analyzer.calculate_kelly_criterion(0.6, 1.5, 1.0)
            assert 0 < kelly < 0.25  # Should be reasonable
            assert confidence > 0.8

            self.results["Kelly Criterion"] = (
                True,
                f"✓ Kelly={kelly:.3f}, Confidence={confidence:.2f}",
            )

        except Exception as e:
            self.results["Risk Analytics"] = (False, f"✗ {str(e)}")
            self.critical_failures += 1

    def _check_logging(self) -> None:
        """Check logging configuration."""
        try:
            # Test logging
            test_logger = logging.getLogger("unity_wheel.test")
            test_logger.info("Test log message")

            # Check if we can write to a log file
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            test_file = log_dir / "test.log"
            with open(test_file, "w") as f:
                f.write("Test log entry\n")

            # Clean up
            test_file.unlink()

            self.results["Logging"] = (True, "✓ Logging configured correctly")

        except Exception as e:
            self.results["Logging"] = (False, f"✗ {str(e)}")
            self.warnings += 1

    def _display_results(self) -> None:
        """Display validation results in a table."""
        table = Table(title="Validation Results")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        for component, (passed, details) in self.results.items():
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            table.add_row(component, status, details)

        console.print(table)

        # Summary
        total_checks = len(self.results)
        passed_checks = sum(1 for passed, _ in self.results.values() if passed)

        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"Total checks: {total_checks}")
        console.print(f"Passed: [green]{passed_checks}[/green]")
        console.print(f"Failed: [red]{total_checks - passed_checks}[/red]")
        console.print(f"Critical failures: [red]{self.critical_failures}[/red]")
        console.print(f"Warnings: [yellow]{self.warnings}[/yellow]")

        if self.critical_failures > 0:
            console.print("\n[bold red]❌ Environment validation FAILED[/bold red]")
        else:
            console.print("\n[bold green]✅ Environment validation PASSED[/bold green]")


def self_diagnostic_test() -> None:
    """Run self-diagnostic test of calculations."""
    console.print("\n[bold blue]Self-Diagnostic Test[/bold blue]\n")

    from src.unity_wheel.math.options import black_scholes_price_validated

    # Known test cases
    test_cases = [
        # (S, K, T, r, sigma, type, expected_min, expected_max)
        (100, 100, 1, 0.05, 0.2, "call", 10.0, 11.0),
        (100, 110, 1, 0.05, 0.2, "put", 9.0, 10.0),
        (100, 90, 0.25, 0.05, 0.3, "call", 11.5, 12.5),
    ]

    table = Table(title="Calculation Accuracy Test")
    table.add_column("Test Case")
    table.add_column("Result")
    table.add_column("Confidence")
    table.add_column("Status")

    for S, K, T, r, sigma, opt_type, exp_min, exp_max in test_cases:
        result = black_scholes_price_validated(S, K, T, r, sigma, opt_type)

        case_str = f"{opt_type.upper()} S={S} K={K} T={T} σ={sigma}"

        if exp_min <= result.value <= exp_max and result.confidence > 0.8:
            status = "[green]PASS[/green]"
        else:
            status = "[red]FAIL[/red]"

        table.add_row(case_str, f"{result.value:.3f}", f"{result.confidence:.2f}", status)

    console.print(table)


def main() -> None:
    """Main entry point for validation."""
    validator = EnvironmentValidator()
    success = validator.run_all_checks()

    # Run self-diagnostic
    self_diagnostic_test()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
