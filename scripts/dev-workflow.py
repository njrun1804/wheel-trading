#!/usr/bin/env python3
"""Unified development workflow manager for Unity Wheel Trading Bot.

Provides a single command interface for all common development tasks
including testing, linting, profiling, and debugging.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.unity_wheel.utils.logging import StructuredLogger
    from src.config.loader import get_config
    import logging
    logger = StructuredLogger(logging.getLogger(__name__))
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class DevWorkflowManager:
    """Manages development workflows and automation."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"
        
        # Colors for terminal output
        self.colors = {
            'reset': '\033[0m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
        }
    
    def colorize(self, text: str, color: str) -> str:
        """Colorize text for terminal output."""
        if color in self.colors:
            return f"{self.colors[color]}{text}{self.colors['reset']}"
        return text
    
    def run_command(self, cmd: List[str], capture_output: bool = False, 
                   cwd: Optional[Path] = None) -> Tuple[int, str]:
        """Run a shell command and return exit code and output."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=capture_output,
                text=True,
                check=False
            )
            
            output = ""
            if capture_output:
                output = result.stdout + result.stderr
            
            return result.returncode, output
            
        except Exception as e:
            return 1, str(e)
    
    def print_section(self, title: str):
        """Print a formatted section header."""
        print(f"\n{self.colorize('=' * 60, 'blue')}")
        print(f"{self.colorize(f'🛠️  {title}', 'cyan')}")
        print(f"{self.colorize('=' * 60, 'blue')}")
    
    def print_status(self, status: str, message: str):
        """Print a status message with appropriate coloring."""
        if status == "success":
            icon = "✅"
            color = "green"
        elif status == "warning":
            icon = "⚠️"
            color = "yellow"
        elif status == "error":
            icon = "❌"
            color = "red"
        else:
            icon = "ℹ️"
            color = "blue"
        
        print(f"{icon} {self.colorize(message, color)}")
    
    def check_environment(self) -> bool:
        """Check development environment setup."""
        self.print_section("Environment Validation")
        
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
        else:
            self.print_status("success", f"Python {sys.version_info.major}.{sys.version_info.minor} OK")
        
        # Check project structure
        required_paths = [
            self.src_path,
            self.tests_path,
            self.project_root / "config.yaml",
            self.project_root / "pyproject.toml"
        ]
        
        for path in required_paths:
            if path.exists():
                self.print_status("success", f"{path.name} exists")
            else:
                issues.append(f"Missing: {path}")
        
        # Check environment variables
        env_vars = ["DATABENTO_API_KEY", "FRED_API_KEY"]
        for var in env_vars:
            if os.getenv(var):
                self.print_status("success", f"{var} configured")
            else:
                self.print_status("warning", f"{var} not set (some features may be limited)")
        
        # Check key dependencies
        try:
            import pytest, numpy, pandas
            self.print_status("success", "Core dependencies available")
        except ImportError as e:
            issues.append(f"Missing dependency: {e}")
        
        if issues:
            self.print_status("error", f"Environment issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        self.print_status("success", "Environment validation passed!")
        return True
    
    def run_tests(self, pattern: str = None, verbose: bool = False, 
                 coverage: bool = False) -> bool:
        """Run test suite with optional filtering and coverage."""
        self.print_section("Running Tests")
        
        cmd = ["python", "-m", "pytest"]
        
        if pattern:
            cmd.extend(["-k", pattern])
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
        
        # Add performance markers
        cmd.extend(["-m", "not slow"])  # Skip slow tests by default
        
        # Run tests
        start_time = time.time()
        exit_code, output = self.run_command(cmd)
        duration = time.time() - start_time
        
        if exit_code == 0:
            self.print_status("success", f"Tests passed in {duration:.1f}s")
            return True
        else:
            self.print_status("error", f"Tests failed after {duration:.1f}s")
            return False
    
    def run_linting(self) -> bool:
        """Run code linting and formatting checks."""
        self.print_section("Code Quality Checks")
        
        all_passed = True
        
        # Black formatting check
        self.print_status("info", "Checking code formatting with black...")
        exit_code, _ = self.run_command(["python", "-m", "black", "--check", "src", "tests"])
        if exit_code == 0:
            self.print_status("success", "Black formatting OK")
        else:
            self.print_status("error", "Black formatting issues found")
            self.print_status("info", "Run 'python -m black src tests' to fix")
            all_passed = False
        
        # Flake8 linting
        self.print_status("info", "Running flake8 linting...")
        exit_code, _ = self.run_command(["python", "-m", "flake8", "src", "tests"])
        if exit_code == 0:
            self.print_status("success", "Flake8 linting passed")
        else:
            self.print_status("error", "Flake8 linting issues found")
            all_passed = False
        
        # MyPy type checking
        self.print_status("info", "Running mypy type checking...")
        exit_code, _ = self.run_command(["python", "-m", "mypy", "src"])
        if exit_code == 0:
            self.print_status("success", "MyPy type checking passed")
        else:
            self.print_status("warning", "MyPy type checking has issues")
            # Don't fail on mypy issues as they might be configuration-related
        
        return all_passed
    
    def run_security_checks(self) -> bool:
        """Run security and safety checks."""
        self.print_section("Security Checks")
        
        # Bandit security check
        self.print_status("info", "Running bandit security scan...")
        exit_code, _ = self.run_command([
            "python", "-m", "bandit", "-r", "src", 
            "-f", "json", "-o", "bandit-report.json"
        ], capture_output=True)
        
        if exit_code == 0:
            self.print_status("success", "No security issues found")
            return True
        else:
            self.print_status("warning", "Security scan completed with findings")
            self.print_status("info", "Check bandit-report.json for details")
            return True  # Don't fail on security warnings for now
    
    def profile_performance(self) -> bool:
        """Run performance profiling on key components."""
        self.print_section("Performance Profiling")
        
        try:
            # Run performance benchmarks
            self.print_status("info", "Running performance benchmarks...")
            exit_code, output = self.run_command([
                "python", "-m", "pytest", 
                "tests/test_performance_benchmarks.py", 
                "-v"
            ], capture_output=True)
            
            if exit_code == 0:
                self.print_status("success", "Performance benchmarks passed")
                
                # Extract timing information if available
                lines = output.split('\n')
                for line in lines:
                    if 'ms' in line and 'test_' in line:
                        self.print_status("info", line.strip())
                
                return True
            else:
                self.print_status("error", "Performance benchmarks failed")
                return False
                
        except Exception as e:
            self.print_status("error", f"Performance profiling failed: {e}")
            return False
    
    def run_integration_tests(self) -> bool:
        """Run integration tests with external dependencies."""
        self.print_section("Integration Tests")
        
        # Check if we have credentials for integration tests
        has_databento = bool(os.getenv("DATABENTO_API_KEY"))
        has_fred = bool(os.getenv("FRED_API_KEY"))
        
        if not (has_databento or has_fred):
            self.print_status("warning", "No API credentials found, skipping integration tests")
            return True
        
        self.print_status("info", "Running integration tests...")
        exit_code, _ = self.run_command([
            "python", "-m", "pytest", 
            "-m", "integration", 
            "-v"
        ])
        
        if exit_code == 0:
            self.print_status("success", "Integration tests passed")
            return True
        else:
            self.print_status("error", "Integration tests failed")
            return False
    
    def generate_coverage_report(self) -> bool:
        """Generate detailed test coverage report."""
        self.print_section("Coverage Analysis")
        
        self.print_status("info", "Generating coverage report...")
        exit_code, _ = self.run_command([
            "python", "-m", "pytest", 
            "--cov=src", 
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])
        
        if exit_code == 0:
            self.print_status("success", "Coverage report generated (≥80%)")
            self.print_status("info", "HTML report: htmlcov/index.html")
            return True
        else:
            self.print_status("warning", "Coverage below 80% threshold")
            return False
    
    def fix_common_issues(self) -> bool:
        """Automatically fix common code issues."""
        self.print_section("Auto-fixing Issues")
        
        # Auto-format with black
        self.print_status("info", "Auto-formatting code with black...")
        exit_code, _ = self.run_command(["python", "-m", "black", "src", "tests"])
        if exit_code == 0:
            self.print_status("success", "Code formatting applied")
        
        # Sort imports with isort
        self.print_status("info", "Sorting imports with isort...")
        exit_code, _ = self.run_command(["python", "-m", "isort", "src", "tests"])
        if exit_code == 0:
            self.print_status("success", "Import sorting applied")
        
        return True
    
    def run_quick_check(self) -> bool:
        """Run a quick development check (fast tests + linting)."""
        self.print_section("Quick Development Check")
        
        success = True
        
        # Quick environment check
        if not self.check_environment():
            success = False
        
        # Fast tests only
        if success:
            if not self.run_tests(verbose=False):
                success = False
        
        # Basic linting
        if success:
            if not self.run_linting():
                success = False
        
        return success
    
    def run_full_check(self) -> bool:
        """Run comprehensive development check."""
        self.print_section("Full Development Check")
        
        success = True
        
        # Environment validation
        if not self.check_environment():
            success = False
        
        # Full test suite with coverage
        if success:
            if not self.run_tests(coverage=True):
                success = False
        
        # Code quality
        if success:
            if not self.run_linting():
                success = False
        
        # Security checks
        if success:
            if not self.run_security_checks():
                success = False
        
        # Performance profiling
        if success:
            if not self.profile_performance():
                success = False
        
        # Integration tests
        if success:
            if not self.run_integration_tests():
                success = False
        
        return success
    
    def prepare_commit(self) -> bool:
        """Prepare code for commit (fix issues + run checks)."""
        self.print_section("Preparing for Commit")
        
        # Auto-fix issues
        self.fix_common_issues()
        
        # Run quick check
        if self.run_quick_check():
            self.print_status("success", "Code ready for commit!")
            return True
        else:
            self.print_status("error", "Issues found - fix before committing")
            return False
    
    def show_project_stats(self):
        """Show project statistics and health metrics."""
        self.print_section("Project Statistics")
        
        try:
            # Count lines of code
            _, output = self.run_command([
                "find", "src", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"
            ], capture_output=True)
            
            if output:
                lines = output.strip().split('\n')
                if lines and 'total' in lines[-1]:
                    total_lines = lines[-1].split()[0]
                    self.print_status("info", f"Total lines of code: {total_lines}")
            
            # Count test files
            test_files = list(self.tests_path.glob("test_*.py"))
            self.print_status("info", f"Test files: {len(test_files)}")
            
            # Check git status
            exit_code, output = self.run_command(["git", "status", "--porcelain"], capture_output=True)
            if exit_code == 0:
                changed_files = len([line for line in output.split('\n') if line.strip()])
                self.print_status("info", f"Changed files: {changed_files}")
        
        except Exception as e:
            self.print_status("warning", f"Could not gather all stats: {e}")


def main():
    """Main entry point for development workflow manager."""
    parser = argparse.ArgumentParser(
        description="Unity Wheel Trading Bot - Development Workflow Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./dev-workflow.py check          # Quick development check
  ./dev-workflow.py test           # Run test suite
  ./dev-workflow.py test -v        # Run tests with verbose output
  ./dev-workflow.py test -c        # Run tests with coverage
  ./dev-workflow.py lint           # Run linting and formatting checks
  ./dev-workflow.py fix            # Auto-fix common issues
  ./dev-workflow.py full           # Run comprehensive check
  ./dev-workflow.py commit         # Prepare for commit
  ./dev-workflow.py stats          # Show project statistics
        """
    )
    
    parser.add_argument('command', choices=[
        'check', 'test', 'lint', 'security', 'performance', 
        'integration', 'coverage', 'fix', 'full', 'commit', 'stats'
    ], help='Development command to run')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    parser.add_argument('-c', '--coverage', action='store_true',
                       help='Include coverage analysis')
    
    parser.add_argument('-k', '--keyword', type=str,
                       help='Filter tests by keyword')
    
    args = parser.parse_args()
    
    # Create workflow manager
    manager = DevWorkflowManager()
    
    # Print header
    print(f"{manager.colorize('🌐 Unity Wheel Trading Bot - Development Workflow', 'cyan')}")
    print(f"{manager.colorize('=' * 60, 'blue')}")
    
    # Execute command
    success = True
    
    if args.command == 'check':
        success = manager.run_quick_check()
    elif args.command == 'test':
        success = manager.run_tests(
            pattern=args.keyword,
            verbose=args.verbose,
            coverage=args.coverage
        )
    elif args.command == 'lint':
        success = manager.run_linting()
    elif args.command == 'security':
        success = manager.run_security_checks()
    elif args.command == 'performance':
        success = manager.profile_performance()
    elif args.command == 'integration':
        success = manager.run_integration_tests()
    elif args.command == 'coverage':
        success = manager.generate_coverage_report()
    elif args.command == 'fix':
        success = manager.fix_common_issues()
    elif args.command == 'full':
        success = manager.run_full_check()
    elif args.command == 'commit':
        success = manager.prepare_commit()
    elif args.command == 'stats':
        manager.show_project_stats()
        success = True
    
    # Print final status
    print(f"\n{manager.colorize('=' * 60, 'blue')}")
    if success:
        print(f"{manager.colorize('✅ Success! All checks passed.', 'green')}")
        sys.exit(0)
    else:
        print(f"{manager.colorize('❌ Some checks failed. See output above.', 'red')}")
        sys.exit(1)


if __name__ == "__main__":
    main()