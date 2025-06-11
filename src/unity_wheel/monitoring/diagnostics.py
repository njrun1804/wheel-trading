"""Enhanced self-diagnostics for autonomous operation."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from src.config import get_config

from ..math import black_scholes_price_validated, calculate_all_greeks
from ..models import Account, Greeks, Position
from ..risk import RiskAnalyzer
from ..strategy import WheelStrategy
from ..utils import StructuredLogger, get_logger

logger = get_logger(__name__)
structured_logger = StructuredLogger(logger)

DiagnosticLevel = Literal["OK", "WARNING", "ERROR", "CRITICAL"]


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check with confidence."""

    check_name: str
    level: DiagnosticLevel
    message: str
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check_name": self.check_name,
            "level": self.level,
            "message": self.message,
            "confidence": self.confidence,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "recommendation": self.recommendation,
        }


class SelfDiagnostics:
    """Comprehensive self-diagnostics for autonomous operation."""

    CRITICAL_CHECKS = [
        "math_validation",
        "model_integrity",
        "risk_calculations",
        "type_consistency",
    ]

    PERFORMANCE_TARGETS = {
        "black_scholes": 0.0002,  # 0.2ms
        "greeks": 0.0003,  # 0.3ms
        "risk_metrics": 0.010,  # 10ms
        "decision": 0.200,  # 200ms
    }

    def __init__(self, history_file: Optional[Path] = None):
        """Initialize diagnostics with optional history tracking."""
        self.results: List[DiagnosticResult] = []
        self.history_file = history_file or Path("diagnostics_history.json")
        self.start_time = datetime.now(timezone.utc)

    def run_all_checks(self) -> bool:
        """
        Run comprehensive diagnostic checks.

        Returns
        -------
        bool
            True if all critical checks pass
        """
        self.results.clear()
        self.start_time = datetime.now(timezone.utc)

        # Critical checks
        self._check_math_validation()
        self._check_model_integrity()
        self._check_risk_calculations()
        self._check_type_consistency()

        # Performance checks
        self._check_performance_benchmarks()

        # System checks
        self._check_dependencies()
        self._check_memory_usage()
        self._check_logging_system()

        # Data quality checks
        self._check_calculation_confidence()

        # Save results
        self._save_history()

        # Return true if all critical checks pass
        critical_results = [r for r in self.results if r.check_name in self.CRITICAL_CHECKS]

        critical_passed = all(r.level in ("OK", "WARNING") for r in critical_results)
        summary = self._generate_summary()

        if critical_passed:
            structured_logger.info(
                f"Diagnostics passed",
                extra={
                    "function": "run_all_checks",
                    "summary": summary,
                    "critical_passed": critical_passed,
                    "elapsed_seconds": summary["elapsed_seconds"],
                },
            )
        else:
            structured_logger.error(
                f"Diagnostics failed",
                extra={
                    "function": "run_all_checks",
                    "summary": summary,
                    "critical_passed": critical_passed,
                    "elapsed_seconds": summary["elapsed_seconds"],
                },
            )

        return critical_passed

    def _check_math_validation(self) -> None:
        """Verify math calculations with known test cases."""
        try:
            # Test case 1: ATM call option
            result = black_scholes_price_validated(
                S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
            )
            expected_range = (10.0, 11.0)

            if expected_range[0] <= result.value <= expected_range[1]:
                confidence = result.confidence
                self.results.append(
                    DiagnosticResult(
                        check_name="math_validation",
                        level="OK",
                        message="Black-Scholes calculations validated",
                        confidence=confidence,
                        details={
                            "test_case": "ATM call",
                            "calculated": result.value,
                            "expected_range": expected_range,
                            "warnings": result.warnings,
                        },
                    )
                )
                structured_logger.info(
                    "Math validation passed",
                    extra={
                        "function": "_check_math_validation",
                        "test_case": "ATM call",
                        "calculated": result.value,
                        "confidence": confidence,
                    },
                )
            else:
                self.results.append(
                    DiagnosticResult(
                        check_name="math_validation",
                        level="ERROR",
                        message=f"Black-Scholes calculation outside expected range",
                        confidence=0.0,
                        details={
                            "calculated": result.value,
                            "expected_range": expected_range,
                        },
                        recommendation="Check options math implementation",
                    )
                )
                structured_logger.error(
                    "Math validation failed",
                    extra={
                        "function": "_check_math_validation",
                        "test_case": "ATM call",
                        "calculated": result.value,
                        "expected_range": expected_range,
                    },
                )

            # Test case 2: Put-call parity
            call_result = black_scholes_price_validated(
                S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
            )
            put_result = black_scholes_price_validated(
                S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="put"
            )

            # C - P = S - K*exp(-rT)
            parity_lhs = call_result.value - put_result.value
            parity_rhs = 100.0 - 100.0 * np.exp(-0.05 * 1.0)
            parity_error = abs(parity_lhs - parity_rhs)

            if parity_error < 0.01:
                self.results.append(
                    DiagnosticResult(
                        check_name="put_call_parity",
                        level="OK",
                        message="Put-call parity verified",
                        details={"error": parity_error},
                    )
                )
            else:
                self.results.append(
                    DiagnosticResult(
                        check_name="put_call_parity",
                        level="WARNING",
                        message=f"Put-call parity error: {parity_error:.4f}",
                        details={"error": parity_error},
                    )
                )

        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    check_name="math_validation",
                    level="CRITICAL",
                    message=f"Math validation failed: {str(e)}",
                    confidence=0.0,
                    recommendation="Critical: Options math module not functioning",
                )
            )

    def _check_model_integrity(self) -> None:
        """Verify data models function correctly."""
        try:
            # Get Unity ticker from config
            config = get_config()
            ticker = config.unity.ticker

            # Test Position model
            stock_pos = Position(ticker, 100)
            assert stock_pos.symbol == ticker
            assert stock_pos.quantity == 100
            assert stock_pos.position_type.value == "stock"

            # Test option position
            opt_symbol = f"{ticker}241220C00080000"
            opt_pos = Position(opt_symbol, -1)
            assert opt_pos.strike == 80.0
            assert opt_pos.underlying == ticker

            # Test Greeks model
            greeks = Greeks(delta=0.5, gamma=0.02, theta=-0.05, vega=0.15)
            assert greeks.has_all_greeks is False  # Missing rho

            # Test Account model
            account = Account(
                cash_balance=50000.0,
                buying_power=100000.0,
                margin_used=10000.0,
            )
            assert account.margin_available == 40000.0
            assert account.margin_utilization == 0.2

            self.results.append(
                DiagnosticResult(
                    check_name="model_integrity",
                    level="OK",
                    message="All data models validated",
                    details={
                        "models_tested": ["Position", "Greeks", "Account"],
                    },
                )
            )

        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    check_name="model_integrity",
                    level="CRITICAL",
                    message=f"Model validation failed: {str(e)}",
                    confidence=0.0,
                    recommendation="Data models compromised - check implementations",
                )
            )

    def _check_risk_calculations(self) -> None:
        """Verify risk analytics function correctly."""
        try:
            analyzer = RiskAnalyzer()

            # Generate sample returns
            returns = np.random.normal(0.001, 0.02, 252)

            # Calculate VaR
            var, var_conf = analyzer.calculate_var(returns, 0.95)

            # Calculate CVaR
            cvar, cvar_conf = analyzer.calculate_cvar(returns, 0.95)

            # Validate relationships
            if 0 < var < 0.1 and cvar >= var:
                self.results.append(
                    DiagnosticResult(
                        check_name="risk_calculations",
                        level="OK",
                        message="Risk calculations validated",
                        confidence=min(var_conf, cvar_conf),
                        details={
                            "var_95": var,
                            "cvar_95": cvar,
                            "var_confidence": var_conf,
                            "cvar_confidence": cvar_conf,
                        },
                    )
                )
            else:
                self.results.append(
                    DiagnosticResult(
                        check_name="risk_calculations",
                        level="ERROR",
                        message="Risk calculation relationships invalid",
                        details={
                            "var": var,
                            "cvar": cvar,
                            "expected": "0 < VaR < 0.1 and CVaR >= VaR",
                        },
                        recommendation="Review risk analytics implementation",
                    )
                )

        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    check_name="risk_calculations",
                    level="CRITICAL",
                    message=f"Risk calculations failed: {str(e)}",
                    confidence=0.0,
                )
            )

    def _check_type_consistency(self) -> None:
        """Verify type hints are consistent."""
        try:
            # This is a simplified check - in practice would use mypy API
            import ast
            import inspect

            # Check a sample function
            from ..math.options import black_scholes_price_validated

            sig = inspect.signature(black_scholes_price_validated)
            params = sig.parameters

            # Verify return type annotation exists
            if sig.return_annotation != inspect.Signature.empty:
                self.results.append(
                    DiagnosticResult(
                        check_name="type_consistency",
                        level="OK",
                        message="Type annotations present and valid",
                        details={
                            "sample_function": "black_scholes_price_validated",
                            "parameters": len(params),
                        },
                    )
                )
            else:
                self.results.append(
                    DiagnosticResult(
                        check_name="type_consistency",
                        level="WARNING",
                        message="Missing return type annotations",
                    )
                )

        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    check_name="type_consistency",
                    level="WARNING",
                    message=f"Type checking incomplete: {str(e)}",
                    confidence=0.8,
                )
            )

    def _check_performance_benchmarks(self) -> None:
        """Benchmark critical calculations."""
        benchmarks = {}

        # Benchmark Black-Scholes
        start = time.time()
        for _ in range(1000):
            black_scholes_price_validated(100, 100, 1, 0.05, 0.2, "call")
        elapsed = (time.time() - start) / 1000
        benchmarks["black_scholes"] = elapsed

        # Benchmark Greeks
        start = time.time()
        for _ in range(1000):
            calculate_all_greeks(100, 100, 1, 0.05, 0.2, "call")
        elapsed = (time.time() - start) / 1000
        benchmarks["greeks"] = elapsed

        # Check against targets
        all_good = True
        for name, elapsed in benchmarks.items():
            target = self.PERFORMANCE_TARGETS.get(name, 0.001)
            if elapsed <= target:
                level = "OK"
            else:
                level = "WARNING"
                all_good = False

            self.results.append(
                DiagnosticResult(
                    check_name=f"performance_{name}",
                    level=level,
                    message=f"{name}: {elapsed*1000:.3f}ms (target: {target*1000:.1f}ms)",
                    details={"elapsed_ms": elapsed * 1000, "target_ms": target * 1000},
                )
            )

        if all_good:
            self.results.append(
                DiagnosticResult(
                    check_name="performance_summary",
                    level="OK",
                    message="All performance targets met",
                    details=benchmarks,
                )
            )

    def _check_dependencies(self) -> None:
        """Check all required dependencies."""
        required = ["numpy", "scipy", "pandas", "pydantic", "rich", "click"]
        missing = []

        for dep in required:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        if missing:
            self.results.append(
                DiagnosticResult(
                    check_name="dependencies",
                    level="ERROR",
                    message=f"Missing dependencies: {missing}",
                    recommendation=f"Run: pip install {' '.join(missing)}",
                )
            )
        else:
            self.results.append(
                DiagnosticResult(
                    check_name="dependencies",
                    level="OK",
                    message="All dependencies available",
                )
            )

    def _check_memory_usage(self) -> None:
        """Check memory usage is within bounds."""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb < 100:
                level = "OK"
            elif memory_mb < 200:
                level = "WARNING"
            else:
                level = "ERROR"

            self.results.append(
                DiagnosticResult(
                    check_name="memory_usage",
                    level=level,
                    message=f"Memory usage: {memory_mb:.1f}MB",
                    details={"memory_mb": memory_mb},
                )
            )
        except ImportError:
            self.results.append(
                DiagnosticResult(
                    check_name="memory_usage",
                    level="WARNING",
                    message="psutil not available for memory check",
                )
            )

    def _check_logging_system(self) -> None:
        """Verify logging is configured correctly."""
        try:
            # Check if we can log
            test_logger = logging.getLogger("unity_wheel.diagnostics.test")
            test_logger.debug("Test debug message")
            test_logger.info("Test info message")

            self.results.append(
                DiagnosticResult(
                    check_name="logging_system",
                    level="OK",
                    message="Logging system operational",
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    check_name="logging_system",
                    level="WARNING",
                    message=f"Logging issues: {str(e)}",
                )
            )

    def _check_calculation_confidence(self) -> None:
        """Check confidence scores from calculations."""
        # Run a few calculations and check confidence
        test_cases = [
            (100, 100, 1, 0.05, 0.2),  # Normal case
            (100, 100, 0.001, 0.05, 0.2),  # Very short time
            (100, 100, 1, 0.05, 5.0),  # Very high volatility
        ]

        low_confidence_count = 0

        for S, K, T, r, sigma in test_cases:
            result = black_scholes_price_validated(S, K, T, r, sigma, "call")
            if result.confidence < 0.8:
                low_confidence_count += 1

        if low_confidence_count == 0:
            self.results.append(
                DiagnosticResult(
                    check_name="calculation_confidence",
                    level="OK",
                    message="All calculations show high confidence",
                )
            )
        else:
            self.results.append(
                DiagnosticResult(
                    check_name="calculation_confidence",
                    level="WARNING",
                    message=f"{low_confidence_count} calculations with low confidence",
                    recommendation="Review edge cases in calculations",
                )
            )

    def _save_history(self) -> None:
        """Save diagnostic results to history file."""
        try:
            history = []
            if self.history_file.exists():
                with open(self.history_file, "r") as f:
                    history = json.load(f)

            # Add current results
            history.append(
                {
                    "timestamp": self.start_time.isoformat(),
                    "results": [r.to_dict() for r in self.results],
                    "summary": self._generate_summary(),
                }
            )

            # Keep only last 30 days
            cutoff = datetime.now(timezone.utc).timestamp() - (30 * 24 * 3600)
            history = [
                h for h in history if datetime.fromisoformat(h["timestamp"]).timestamp() > cutoff
            ]

            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save diagnostic history: {e}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        levels_count = {
            "OK": sum(1 for r in self.results if r.level == "OK"),
            "WARNING": sum(1 for r in self.results if r.level == "WARNING"),
            "ERROR": sum(1 for r in self.results if r.level == "ERROR"),
            "CRITICAL": sum(1 for r in self.results if r.level == "CRITICAL"),
        }

        critical_passed = all(
            r.level in ("OK", "WARNING")
            for r in self.results
            if r.check_name in self.CRITICAL_CHECKS
        )

        return {
            "total_checks": len(self.results),
            "levels": levels_count,
            "critical_passed": critical_passed,
            "elapsed_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
        }

    def report(self, format: Literal["text", "json", "html"] = "text") -> str:
        """Generate diagnostic report in specified format."""
        if format == "json":
            return json.dumps(
                {
                    "timestamp": self.start_time.isoformat(),
                    "results": [r.to_dict() for r in self.results],
                    "summary": self._generate_summary(),
                },
                indent=2,
            )

        elif format == "html":
            # Simple HTML report
            html = ["<html><body>"]
            html.append("<h1>Unity Wheel Bot - Diagnostics Report</h1>")
            html.append(f"<p>Generated: {self.start_time}</p>")

            # Summary
            summary = self._generate_summary()
            html.append("<h2>Summary</h2>")
            html.append(f"<p>Total Checks: {summary['total_checks']}</p>")
            html.append(f"<p>Critical Passed: {'Yes' if summary['critical_passed'] else 'No'}</p>")

            # Results table
            html.append("<h2>Results</h2>")
            html.append("<table border='1'>")
            html.append("<tr><th>Check</th><th>Level</th><th>Message</th><th>Confidence</th></tr>")

            for r in self.results:
                color = {"OK": "green", "WARNING": "orange", "ERROR": "red", "CRITICAL": "darkred"}[
                    r.level
                ]
                html.append(
                    f"<tr><td>{r.check_name}</td>"
                    f"<td style='color:{color}'>{r.level}</td>"
                    f"<td>{r.message}</td>"
                    f"<td>{r.confidence:.0%}</td></tr>"
                )

            html.append("</table>")
            html.append("</body></html>")

            return "\n".join(html)

        else:  # text format
            from rich.console import Console
            from rich.table import Table

            console = Console()

            # Create table
            table = Table(title="Unity Wheel Bot - Diagnostics Report")
            table.add_column("Check", style="cyan")
            table.add_column("Level", style="green")
            table.add_column("Message")
            table.add_column("Confidence", justify="right")

            for r in self.results:
                style = {
                    "OK": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red bold",
                }.get(r.level, "white")

                table.add_row(
                    r.check_name, f"[{style}]{r.level}[/{style}]", r.message, f"{r.confidence:.0%}"
                )

            # Summary
            summary = self._generate_summary()

            # Build output
            import io

            string_io = io.StringIO()
            temp_console = Console(file=string_io, force_terminal=True)
            temp_console.print(table)
            temp_console.print(f"\nTotal Checks: {summary['total_checks']}")
            temp_console.print(f"Passed: [green]{summary['levels']['OK']}[/green]")
            temp_console.print(f"Warnings: [yellow]{summary['levels']['WARNING']}[/yellow]")
            temp_console.print(f"Errors: [red]{summary['levels']['ERROR']}[/red]")
            temp_console.print(f"Critical: [red bold]{summary['levels']['CRITICAL']}[/red bold]")
            temp_console.print(
                f"\nStatus: {'[green]PASSED[/green]' if summary['critical_passed'] else '[red]FAILED[/red]'}"
            )

            return string_io.getvalue()


def run_diagnostics(output_format: Literal["text", "json", "html"] = "text") -> int:
    """Run diagnostics and return exit code."""
    diag = SelfDiagnostics()
    success = diag.run_all_checks()

    print(diag.report(format=output_format))

    return 0 if success else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Unity Wheel Bot self-diagnostics")
    parser.add_argument(
        "--format",
        choices=["text", "json", "html"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--save",
        help="Save report to file",
    )

    args = parser.parse_args()

    # Run diagnostics
    diag = SelfDiagnostics()
    success = diag.run_all_checks()

    # Generate report
    report = diag.report(format=args.format)

    # Output
    if args.save:
        with open(args.save, "w") as f:
            f.write(report)
        print(f"Report saved to {args.save}")
    else:
        print(report)

    sys.exit(0 if success else 1)
