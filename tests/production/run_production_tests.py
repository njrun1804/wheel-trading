#!/usr/bin/env python3
"""Production test runner for comprehensive deployment validation."""

import asyncio
import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()


class ProductionTestRunner:
    """Comprehensive production test execution and reporting."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    async def run_production_test_suite(
        self, 
        test_categories: List[str] = None,
        verbose: bool = False,
        save_report: bool = True,
        fail_fast: bool = False
    ) -> Dict[str, Any]:
        """Run comprehensive production test suite."""
        
        if test_categories is None:
            test_categories = [
                'end_to_end_workflows',
                'stress_testing', 
                'real_world_scenarios',
                'production_readiness'
            ]
        
        console.print("\n[bold blue]🚀 PRODUCTION DEPLOYMENT VALIDATION[/bold blue]")
        console.print("=" * 80)
        
        self.start_time = time.time()
        
        # Test configuration
        test_config = {
            'end_to_end_workflows': {
                'file': 'tests/production/test_end_to_end_workflows.py',
                'description': 'Complete trading workflow validation',
                'critical': True
            },
            'stress_testing': {
                'file': 'tests/production/test_stress_testing.py',
                'description': 'Concurrent usage and system limits',
                'critical': True
            },
            'real_world_scenarios': {
                'file': 'tests/production/test_real_world_scenarios.py',
                'description': 'Market scenarios and usage patterns',
                'critical': True
            },
            'production_readiness': {
                'file': 'tests/production/test_production_readiness.py',
                'description': 'Deployment readiness assessment',
                'critical': True
            }
        }
        
        total_categories = len(test_categories)
        passed_categories = 0
        failed_categories = []
        
        # Run each test category
        for i, category in enumerate(test_categories, 1):
            if category not in test_config:
                console.print(f"[red]❌ Unknown test category: {category}[/red]")
                continue
                
            config = test_config[category]
            
            console.print(f"\n[bold cyan]📋 Category {i}/{total_categories}: {category.replace('_', ' ').title()}[/bold cyan]")
            console.print(f"[dim]{config['description']}[/dim]")
            
            # Run tests for this category
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Running {category} tests...", total=None)
                
                try:
                    # Build pytest command
                    cmd_args = [
                        config['file'],
                        '-v' if verbose else '-q',
                        '--tb=short',
                        f'--junitxml=test_results_{category}.xml',
                        '--asyncio-mode=auto'
                    ]
                    
                    if fail_fast:
                        cmd_args.append('-x')
                    
                    # Run pytest
                    exit_code = pytest.main(cmd_args)
                    
                    if exit_code == 0:
                        self.results[category] = {
                            'status': 'PASSED',
                            'exit_code': exit_code,
                            'critical': config['critical']
                        }
                        passed_categories += 1
                        console.print(f"[green]✅ {category.replace('_', ' ').title()}: PASSED[/green]")
                    else:
                        self.results[category] = {
                            'status': 'FAILED',
                            'exit_code': exit_code,
                            'critical': config['critical']
                        }
                        failed_categories.append(category)
                        console.print(f"[red]❌ {category.replace('_', ' ').title()}: FAILED[/red]")
                        
                        if fail_fast and config['critical']:
                            console.print("[red]⚠️  Failing fast due to critical test failure[/red]")
                            break
                            
                except Exception as e:
                    self.results[category] = {
                        'status': 'ERROR',
                        'error': str(e),
                        'critical': config['critical']
                    }
                    failed_categories.append(category)
                    console.print(f"[red]💥 {category.replace('_', ' ').title()}: ERROR - {e}[/red]")
                    
                    if fail_fast and config['critical']:
                        break
        
        self.end_time = time.time()
        
        # Generate summary report
        summary = self._generate_summary_report(
            total_categories, passed_categories, failed_categories
        )
        
        # Display results
        self._display_results_table()
        self._display_summary(summary)
        
        # Save report if requested
        if save_report:
            report_path = self._save_detailed_report(summary)
            console.print(f"\n[blue]📄 Detailed report saved to: {report_path}[/blue]")
        
        return summary
    
    def _generate_summary_report(
        self, 
        total_categories: int, 
        passed_categories: int, 
        failed_categories: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        
        critical_failures = [
            cat for cat in failed_categories 
            if self.results.get(cat, {}).get('critical', False)
        ]
        
        deployment_ready = len(critical_failures) == 0
        success_rate = passed_categories / total_categories if total_categories > 0 else 0
        
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'deployment_ready': deployment_ready,
            'overall_success_rate': success_rate,
            'total_categories': total_categories,
            'passed_categories': passed_categories,
            'failed_categories': len(failed_categories),
            'critical_failures': critical_failures,
            'test_duration_seconds': duration,
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations(failed_categories)
        }
    
    def _generate_recommendations(self, failed_categories: List[str]) -> List[str]:
        """Generate recommendations based on test failures."""
        recommendations = []
        
        if 'end_to_end_workflows' in failed_categories:
            recommendations.extend([
                "Review core trading workflow implementation",
                "Validate data pipeline integration",
                "Check error handling in critical paths"
            ])
        
        if 'stress_testing' in failed_categories:
            recommendations.extend([
                "Optimize for concurrent user load",
                "Review database connection pooling",
                "Implement rate limiting and throttling"
            ])
        
        if 'real_world_scenarios' in failed_categories:
            recommendations.extend([
                "Test with realistic market data",
                "Validate market hours handling",
                "Review volatility scenario responses"
            ])
        
        if 'production_readiness' in failed_categories:
            recommendations.extend([
                "Complete security configuration",
                "Set up monitoring and alerting",
                "Validate backup and recovery procedures"
            ])
        
        return recommendations
    
    def _display_results_table(self):
        """Display results in a formatted table."""
        table = Table(title="Production Test Results")
        
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Critical", justify="center")
        table.add_column("Details", style="dim")
        
        for category, result in self.results.items():
            status = result['status']
            status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "💥"
            status_text = f"{status_icon} {status}"
            
            critical_icon = "🔴" if result.get('critical') else "🟡"
            
            details = ""
            if status == "FAILED":
                details = f"Exit code: {result.get('exit_code', 'N/A')}"
            elif status == "ERROR":
                details = result.get('error', '')[:50] + "..."
            
            table.add_row(
                category.replace('_', ' ').title(),
                status_text,
                critical_icon,
                details
            )
        
        console.print("\n")
        console.print(table)
    
    def _display_summary(self, summary: Dict[str, Any]):
        """Display summary results."""
        
        # Overall status
        if summary['deployment_ready']:
            status_panel = Panel(
                "[bold green]✅ SYSTEM READY FOR PRODUCTION DEPLOYMENT[/bold green]\n\n"
                f"Success Rate: {summary['overall_success_rate']:.1%}\n"
                f"Test Duration: {summary['test_duration_seconds']:.1f}s",
                title="Deployment Status",
                border_style="green"
            )
        else:
            critical_failures_text = "\n".join(f"• {cat}" for cat in summary['critical_failures'])
            status_panel = Panel(
                "[bold red]❌ SYSTEM NOT READY FOR PRODUCTION[/bold red]\n\n"
                f"Critical Failures:\n{critical_failures_text}\n\n"
                f"Success Rate: {summary['overall_success_rate']:.1%}\n"
                f"Test Duration: {summary['test_duration_seconds']:.1f}s",
                title="Deployment Status",
                border_style="red"
            )
        
        console.print("\n")
        console.print(status_panel)
        
        # Recommendations
        if summary['recommendations']:
            recommendations_text = "\n".join(f"• {rec}" for rec in summary['recommendations'][:10])
            rec_panel = Panel(
                recommendations_text,
                title="Recommendations",
                border_style="yellow"
            )
            console.print("\n")
            console.print(rec_panel)
    
    def _save_detailed_report(self, summary: Dict[str, Any]) -> Path:
        """Save detailed report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"production_validation_report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return report_path


async def main():
    """Main entry point for production test runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive production deployment validation tests"
    )
    parser.add_argument(
        '--categories',
        nargs='*',
        choices=['end_to_end_workflows', 'stress_testing', 'real_world_scenarios', 'production_readiness'],
        help='Test categories to run (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose test output'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip saving detailed report'
    )
    parser.add_argument(
        '--fail-fast', '-x',
        action='store_true',
        help='Stop on first critical failure'
    )
    
    args = parser.parse_args()
    
    runner = ProductionTestRunner()
    
    try:
        summary = await runner.run_production_test_suite(
            test_categories=args.categories,
            verbose=args.verbose,
            save_report=not args.no_report,
            fail_fast=args.fail_fast
        )
        
        # Exit with appropriate code
        if summary['deployment_ready']:
            console.print("\n[green]🎉 All production validation tests passed![/green]")
            sys.exit(0)
        else:
            console.print("\n[red]⚠️  Production validation failed. See recommendations above.[/red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Test execution interrupted by user[/yellow]")
        sys.exit(2)
    except Exception as e:
        console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())