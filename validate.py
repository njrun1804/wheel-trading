#!/usr/bin/env python3
"""
Unified validation script for wheel trading database.

Usage:
    python validate.py --daily          # Quick daily health check
    python validate.py --comprehensive  # Full validation with detailed report
    python validate.py --greeks         # Investigate Greek calculation issues
    python validate.py --quick          # Same as --daily
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name, description):
    """Run a validation script and handle the output."""
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"❌ Error: {script_name} not found")
        return False

    print(f"Running {description}...")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)], capture_output=False, text=True
        )

        print("=" * 60)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(
                f"⚠️  {description} completed with warnings (code: {result.returncode})"
            )
            return False

    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate wheel trading database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python validate.py --daily          # Quick morning health check
    python validate.py --comprehensive  # Full evening validation
    python validate.py --greeks         # Investigate Greek issues
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--daily", "--quick", action="store_true", help="Run quick daily health check"
    )
    group.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive validation with detailed report",
    )
    group.add_argument(
        "--greeks", action="store_true", help="Investigate Greek calculation issues"
    )

    args = parser.parse_args()

    print("WHEEL TRADING DATABASE VALIDATION")
    print("=" * 60)

    success = True

    if args.daily:
        success = run_script("daily_health_check.py", "Daily Health Check")

    elif args.comprehensive:
        success = run_script(
            "validate_database_comprehensive.py", "Comprehensive Validation"
        )
        print("\n📊 Detailed report saved to: database_validation_report.json")
        print("📋 Summary available in: database_health_summary.md")

    elif args.greeks:
        success = run_script(
            "investigate_greek_issues.py", "Greek Calculation Investigation"
        )

    print("=" * 60)

    if success:
        print("🎉 Validation completed successfully!")
        return 0
    else:
        print("⚠️  Validation completed with issues - review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
