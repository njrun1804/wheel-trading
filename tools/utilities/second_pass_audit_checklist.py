#!/usr/bin/env python3
"""
Second-Pass Audit Checklist Implementation
Tracks completion of all audit items before production.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import duckdb
import yaml

from unity_wheel.config.unified_config import get_config
config = get_config()



class AuditChecklist:
    def __init__(self):
        self.checks = {
            "data_integrity": {
                "negative_dte_fix": False,
                "extreme_strikes_fix": False,
                "liquidity_filters": False,
                "data_scrub_rerun": False,
            },
            "statistical_validation": {
                "walk_forward_nested": False,
                "fdr_applied": False,
                "scenario_shocks": False,
                "early_assignment_model": False,
            },
            "operational_plumbing": {
                "automated_monitoring": False,
                "position_tracking": False,
                "alert_channels": False,
                "policy_engine": False,
                "paper_trading": False,
            },
            "market_context": {
                "insider_selling_flag": False,
                "vector_migration_flag": False,
                "event_override": False,
            },
        }
        self.results = {}

    def check_data_integrity(self):
        """Verify all data integrity fixes are applied."""

        print("\n1️⃣ DATA INTEGRITY CHECKS")
        print("-" * 50)

        db_path = Path("data/unified_wheel_trading.duckdb")
        if not db_path.exists():
            print("❌ Database not found")
            return

        conn = duckdb.connect(str(db_path), read_only=True)

        # Check for negative DTE
        negative_dte = conn.execute(
            """
            SELECT COUNT(*) as count
            FROM market_data_clean md
            JOIN options_metadata_clean om ON md.symbol = om.symbol
            WHERE md.data_type = 'option'
            AND DATEDIFF('day', md.date, om.expiration) < 0
        """
        ).fetchone()[0]

        self.checks["data_integrity"]["negative_dte_fix"] = negative_dte == 0
        print(
            f"   ✓ Negative DTE removed: {'✅' if negative_dte == 0 else f'❌ ({negative_dte} remain)'}"
        )

        # Check extreme strikes
        extreme_strikes = conn.execute(
            """
            WITH strike_check AS (
                SELECT
                    om.strike,
                    s.close,
                    ABS(om.strike - s.close) / s.close as distance
                FROM options_metadata_clean om
                JOIN market_data_clean s ON s.symbol = config.trading.symbol
                    AND s.data_type = 'stock'
                    AND s.date = (
                        SELECT MAX(date) FROM market_data_clean
                        WHERE symbol = config.trading.symbol AND data_type = 'stock'
                    )
                WHERE om.underlying = 'U'
            )
            SELECT COUNT(*) FROM strike_check WHERE distance > 3.0
        """
        ).fetchone()[0]

        self.checks["data_integrity"]["extreme_strikes_fix"] = extreme_strikes == 0
        print(
            f"   ✓ Extreme strikes removed: {'✅' if extreme_strikes == 0 else f'❌ ({extreme_strikes} remain)'}"
        )

        # Check if liquidity filters exist
        try:
            tradeable = conn.execute("SELECT COUNT(*) FROM tradeable_options").fetchone()[0]
            self.checks["data_integrity"]["liquidity_filters"] = tradeable > 0
            print(f"   ✓ Liquidity filters applied: ✅ ({tradeable:,} tradeable options)")
        except:
            self.checks["data_integrity"]["liquidity_filters"] = False
            print("   ✓ Liquidity filters applied: ❌ (table not found)")

        conn.close()

    def check_walk_forward_validation(self):
        """Verify nested walk-forward validation."""

        print("\n2️⃣ STATISTICAL VALIDATION CHECKS")
        print("-" * 50)

        # Check if validation results exist
        validation_files = {
            "walk_forward": Path("results/walk_forward_nested.json"),
            "fdr_results": Path("results/fdr_parameters.csv"),
            "scenario_shocks": Path("results/scenario_shocks.json"),
            "assignment_model": Path("results/early_assignment_analysis.csv"),
        }

        for check_name, file_path in validation_files.items():
            exists = file_path.exists()
            key = check_name.replace("_results", "").replace("_", "_")
            if key in self.checks["statistical_validation"]:
                self.checks["statistical_validation"][key] = exists
                print(f"   ✓ {check_name}: {'✅' if exists else '❌ (not found)'}")

    def check_operational_setup(self):
        """Verify operational plumbing is configured."""

        print("\n3️⃣ OPERATIONAL PLUMBING CHECKS")
        print("-" * 50)

        # Check monitoring script
        monitor_script = Path("src/unity_wheel/monitoring/scripts/daily_health_check.py")
        self.checks["operational_plumbing"]["automated_monitoring"] = monitor_script.exists()
        print(f"   ✓ Daily monitor script: {'✅' if monitor_script.exists() else '❌'}")

        # Check crontab
        try:
            cron_output = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
            has_monitor = "daily_health_check" in cron_output.stdout
            print(f"   ✓ Crontab configured: {'✅' if has_monitor else '❌'}")
        except:
            print("   ✓ Crontab configured: ❌ (could not check)")

        # Check position tracking
        positions_file = Path("my_positions.yaml")
        self.checks["operational_plumbing"]["position_tracking"] = positions_file.exists()
        print(f"   ✓ Position tracking: {'✅' if positions_file.exists() else '❌'}")

        # Check alert configuration
        env_file = Path(".env.monitoring")
        self.checks["operational_plumbing"]["alert_channels"] = env_file.exists()
        print(f"   ✓ Alert config: {'✅' if env_file.exists() else '❌'}")

        # Check policy engine
        self.check_policy_engine()

    def check_policy_engine(self):
        """Verify stop-trading rules are encoded."""

        # Check if advisor has volatility checks
        advisor_file = Path("src/unity_wheel/api/advisor.py")
        if advisor_file.exists():
            with open(advisor_file) as f:
                content = f.read()

            has_vol_check = "volatility > 1.20" in content or "vol > 1.20" in content
            self.checks["operational_plumbing"]["policy_engine"] = has_vol_check
            print(f"   ✓ Policy engine (vol>120% stop): {'✅' if has_vol_check else '❌'}")
        else:
            print("   ✓ Policy engine: ❌ (advisor not found)")

    def check_market_context(self):
        """Check Unity-specific event flags."""

        print("\n4️⃣ MARKET CONTEXT AWARENESS")
        print("-" * 50)

        # Check for event monitoring
        config_file = Path(os.getenv("WHEEL_CONFIG_PATH", "config/unified.yaml"))
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)

            # Look for Unity-specific settings
            has_event_config = False
            if config and "unity" in config:
                unity_config = config["unity"]
                if "monitor_insider_sales" in unity_config or "vector_migration" in unity_config:
                    has_event_config = True

            print(f"   ✓ Unity event monitoring: {'✅' if has_event_config else '❌'}")

        # Current Unity-specific risks
        print("\n   ⚠️  CURRENT UNITY RISKS:")
        print("   • Heavy insider selling (~15M shares YTD)")
        print("   • Vector ad-tech migration causing volatility")
        print("   • Recommend: Add 8-K filing monitor")

    def run_final_validation(self):
        """Run final go/no-go validation."""

        print("\n5️⃣ FINAL GO/NO-GO VALIDATION")
        print("-" * 50)

        # Re-run key metrics with clean data
        db_path = Path("data/unified_wheel_trading.duckdb")
        if db_path.exists():
            conn = duckdb.connect(str(db_path), read_only=True)

            # Check if clean tables exist
            try:
                # Get Q2 2025 performance
                q2_perf = conn.execute(
                    """
                    WITH q2_data AS (
                        SELECT
                            returns,
                            volatility_20d
                        FROM backtest_features_clean
                        WHERE symbol = config.trading.symbol
                        AND date >= '2025-04-01'
                    )
                    SELECT
                        AVG(returns) * 252 as annual_return,
                        STDDEV(returns) * SQRT(252) as annual_vol,
                        COUNT(*) as days
                    FROM q2_data
                """
                ).fetchone()

                if q2_perf and q2_perf[2] > 0:
                    annual_return, annual_vol, days = q2_perf
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

                    print("\n   📊 Q2 2025 Clean Data Performance:")
                    print(f"      Annualized Return: {annual_return:.1%}")
                    print(f"      Annualized Vol: {annual_vol:.1%}")
                    print(f"      Sharpe Ratio: {sharpe:.2f}")
                    print(f"      Days: {days}")

                    self.results["clean_sharpe"] = sharpe
                    self.results["clean_return"] = annual_return

                    if sharpe >= 1.2:
                        print("      ✅ Sharpe > 1.2 threshold")
                    else:
                        print("      ❌ Sharpe < 1.2 threshold")

            except Exception as e:
                print(f"   ❌ Could not calculate clean metrics: {e}")

            conn.close()

    def generate_report(self):
        """Generate final readiness report."""

        print("\n" + "=" * 60)
        print("📋 PRODUCTION READINESS REPORT")
        print("=" * 60)

        # Count completed items
        total_checks = 0
        completed_checks = 0

        for category, items in self.checks.items():
            for check, status in items.items():
                total_checks += 1
                if status:
                    completed_checks += 1

        completion_rate = completed_checks / total_checks * 100 if total_checks > 0 else 0

        print(f"\nOverall Completion: {completed_checks}/{total_checks} ({completion_rate:.0f}%)")

        # Category breakdown
        print("\nCategory Breakdown:")
        for category, items in self.checks.items():
            cat_total = len(items)
            cat_complete = sum(1 for status in items.values() if status)
            print(f"   {category}: {cat_complete}/{cat_total}")

        # Critical items
        print("\n🚨 CRITICAL ITEMS:")
        critical_items = [
            ("Data scrub complete", self.checks["data_integrity"]["negative_dte_fix"]),
            ("Monitoring automated", self.checks["operational_plumbing"]["automated_monitoring"]),
            ("Alert channels configured", self.checks["operational_plumbing"]["alert_channels"]),
            ("Policy engine active", self.checks["operational_plumbing"]["policy_engine"]),
        ]

        for item, status in critical_items:
            print(f"   {item}: {'✅' if status else '❌ MUST FIX'}")

        # Final recommendation
        print("\n📊 FINAL RECOMMENDATION:")

        if completion_rate >= 80 and all(status for _, status in critical_items):
            print("   ✅ READY FOR PAPER TRADING")
            print("   Run 2-week paper trading before live deployment")
        elif completion_rate >= 60:
            print("   ⚠️  CLOSE TO READY - Fix critical items first")
        else:
            print("   ❌ NOT READY - Significant work required")

        # Save report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "completion_rate": completion_rate,
            "checks": self.checks,
            "results": self.results,
            "recommendation": "READY" if completion_rate >= 80 else "NOT READY",
        }

        report_file = Path("audit_report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n✅ Report saved to: {report_file}")


def main():
    """Run complete second-pass audit."""

    print("SECOND-PASS AUDIT CHECKLIST")
    print("=" * 60)
    print("Verifying all critical items before production...\n")

    auditor = AuditChecklist()

    # Run all checks
    auditor.check_data_integrity()
    auditor.check_walk_forward_validation()
    auditor.check_operational_setup()
    auditor.check_market_context()
    auditor.run_final_validation()

    # Generate report
    auditor.generate_report()

    print("\n" + "=" * 60)
    print("✅ Audit complete - review report above")
    print("\nNEXT STEPS:")
    print("1. Fix any ❌ items")
    print("2. Re-run data integrity fixes if needed")
    print("3. Configure monitoring and alerts")
    print("4. Start 2-week paper trading")


if __name__ == "__main__":
    main()
