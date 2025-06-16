#!/usr/bin/env python3
"""
Run Claude's 12-step audit on Jarvis2 Strategic Architect files
"""

import json
from pathlib import Path

from meta_auditor import MetaAuditor


def run_jarvis2_audit():
    """Run comprehensive audit on Jarvis2 files"""

    print("🚀 Starting Claude's 12-Step Audit on Jarvis2 Strategic Architect")
    print("=" * 80)

    # Initialize auditor
    auditor = MetaAuditor("jarvis2_audit.db")

    # Files to audit
    jarvis2_files = [
        Path("jarvis2_core.py"),
        Path("jarvis2_mcts.py"),
        Path("jarvis2_complete.py"),
    ]

    # Run audit on each file
    all_results = {}
    for file_path in jarvis2_files:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        print(f"\n{'='*80}")
        print(f"📄 Auditing: {file_path.name}")
        print(f"{'='*80}")

        results = auditor.audit_file(file_path, generation=1)
        all_results[str(file_path)] = results

        # Print detailed results for each step
        for result in results:
            print(f"\n📋 Step: {result.step_name}")
            print(f"   Status: {result.status}")
            print(f"   Confidence: {result.confidence:.2f}")

            if result.findings:
                print(f"   Findings ({len(result.findings)}):")
                for finding in result.findings[:5]:  # Top 5 findings
                    print(f"     • {finding}")
                if len(result.findings) > 5:
                    print(f"     ... and {len(result.findings) - 5} more")

            if result.concerns:
                print("   Concerns:")
                for concern in result.concerns:
                    print(f"     ⚠️  {concern}")

    # Generate summary report
    print(f"\n{'='*80}")
    print("📊 AUDIT SUMMARY")
    print(f"{'='*80}")

    total_issues = 0
    critical_issues = []

    for file_name, results in all_results.items():
        file_issues = sum(1 for r in results if r.status == "NEEDS_WORK")
        total_issues += file_issues

        print(f"\n📄 {file_name}:")
        print(f"   Total checks: {len(results)}")
        print(f"   Issues found: {file_issues}")

        # Collect critical issues
        for result in results:
            if result.status == "NEEDS_WORK" and result.step_name in [
                "pattern_search_anti_stub",
                "shallow_implementation",
                "error_handling",
                "async_completeness",
            ]:
                critical_issues.append(
                    {
                        "file": file_name,
                        "step": result.step_name,
                        "concerns": result.concerns,
                    }
                )

    # Print critical issues
    if critical_issues:
        print("\n🚨 CRITICAL ISSUES REQUIRING ATTENTION:")
        for issue in critical_issues:
            print(f"\n   File: {issue['file']}")
            print(f"   Step: {issue['step']}")
            for concern in issue["concerns"]:
                print(f"     • {concern}")

    # Generate detailed report
    print("\n📋 DETAILED AUDIT REPORT:")
    print(auditor.get_audit_report(generation=1))

    # Save results to JSON
    output_file = "jarvis2_audit_results.json"
    with open(output_file, "w") as f:
        json_results = {}
        for file_name, results in all_results.items():
            json_results[file_name] = [
                {
                    "step_name": r.step_name,
                    "status": r.status,
                    "findings": r.findings,
                    "concerns": r.concerns,
                    "confidence": r.confidence,
                }
                for r in results
            ]
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_jarvis2_audit()
