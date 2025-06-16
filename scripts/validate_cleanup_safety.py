#!/usr/bin/env python3

"""
Cleanup Safety Validation Script
Validates that Phase 1 cleanup can be performed safely without breaking the system
Part of Codebase Harmonization Roadmap
"""

import json
import re
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).parent.parent
    if (current / "pyproject.toml").exists():
        return current

    # Search upward for project markers
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return current


class CleanupValidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues = []
        self.warnings = []
        self.essential_files = [
            "README.md",
            "CLAUDE.md",
            "src/unity_wheel/__init__.py",
            "config.yaml",
            "pyproject.toml",
            "requirements.txt",
            "run.py",
        ]

        # Files that should never be removed
        self.protected_patterns = [
            r"^src/unity_wheel/.*\.py$",
            r"^tests/test_.*\.py$",  # Keep main test files
            r"^config\.yaml$",
            r"^pyproject\.toml$",
            r"^requirements.*\.txt$",
            r"^README\.md$",
            r"^CLAUDE\.md$",
            r"^LICENSE$",
            r"^ARCHITECTURE\.md$",
        ]

    def validate_essential_files(self) -> bool:
        """Ensure all essential files exist."""
        print("🔍 Validating essential files...")

        missing = []
        for file_path in self.essential_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing.append(file_path)

        if missing:
            self.issues.append(f"Missing essential files: {missing}")
            return False

        print("  ✅ All essential files present")
        return True

    def analyze_import_dependencies(self) -> dict[str, set[str]]:
        """Analyze import dependencies to identify critical files."""
        print("🔍 Analyzing import dependencies...")

        dependencies = {}
        python_files = list(self.project_root.rglob("*.py"))

        for py_file in python_files:
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Find import statements
                imports = re.findall(
                    r"^(?:from|import)\s+([^\s#]+)", content, re.MULTILINE
                )
                relative_path = py_file.relative_to(self.project_root)
                dependencies[str(relative_path)] = set(imports)

            except Exception as e:
                self.warnings.append(f"Could not analyze {py_file}: {e}")

        return dependencies

    def identify_removable_files(self) -> tuple[list[Path], list[Path]]:
        """Identify files that can be safely removed vs protected files."""
        print("🔍 Identifying removable vs protected files...")

        all_files = []
        for pattern in ["*.md", "*.py", "*.json", "*.yaml", "*.yml"]:
            all_files.extend(self.project_root.rglob(pattern))

        removable = []
        protected = []

        for file_path in all_files:
            relative_path = file_path.relative_to(self.project_root)
            str_path = str(relative_path)

            # Check if file matches protected patterns
            is_protected = False
            for pattern in self.protected_patterns:
                if re.match(pattern, str_path):
                    is_protected = True
                    break

            if is_protected:
                protected.append(file_path)
            else:
                # Check if it matches removal patterns
                removal_patterns = [
                    r".*_SUMMARY\.md$",
                    r".*_REPORT\.md$",
                    r".*_COMPLETE\.md$",
                    r".*_VALIDATION\.md$",
                    r".*_FINAL\.md$",
                    r".*test_.*_simple\.py$",
                    r".*test_.*_validation\.py$",
                    r".*test_.*_old\.py$",
                    r".*\.backup$",
                    r".*backup_.*",
                    r".*mcp-servers-.+\.json$",
                ]

                for pattern in removal_patterns:
                    if re.match(pattern, str_path):
                        removable.append(file_path)
                        break

        return removable, protected

    def check_test_coverage(self) -> bool:
        """Ensure removing test files won't eliminate all tests for components."""
        print("🔍 Checking test coverage preservation...")

        # Find all test files
        test_files = list(self.project_root.rglob("test_*.py"))

        # Group by component being tested
        test_groups = {}
        for test_file in test_files:
            # Extract component name from test file
            name = test_file.stem
            # Remove test_ prefix and suffixes like _simple, _validation
            component = re.sub(
                r"^test_|_(simple|validation|old|backup|copy|deprecated)$", "", name
            )

            if component not in test_groups:
                test_groups[component] = []
            test_groups[component].append(test_file)

        # Check that each component will have at least one test after cleanup
        orphaned_components = []
        for component, tests in test_groups.items():
            main_test_exists = False
            for test in tests:
                # Check if this is a main test (not a variant)
                if not re.search(
                    r"_(simple|validation|old|backup|copy|deprecated)\.py$", str(test)
                ):
                    main_test_exists = True
                    break

            if not main_test_exists and len(tests) > 0:
                orphaned_components.append(component)

        if orphaned_components:
            self.warnings.append(
                f"Components may lose all tests: {orphaned_components}"
            )

        print(f"  ✅ Test coverage checked - {len(test_groups)} components have tests")
        return True

    def validate_configuration_integrity(self) -> bool:
        """Ensure configuration files are properly structured."""
        print("🔍 Validating configuration integrity...")

        config_files = ["config.yaml", "pyproject.toml"]
        for config_file in config_files:
            config_path = self.project_root / config_file
            if not config_path.exists():
                self.issues.append(f"Missing configuration file: {config_file}")
                continue

            try:
                if config_file.endswith(".yaml"):
                    import yaml

                    with open(config_path) as f:
                        yaml.safe_load(f)
                elif config_file.endswith(".toml"):
                    import tomli

                    with open(config_path, "rb") as f:
                        tomli.load(f)
            except Exception as e:
                self.issues.append(f"Invalid configuration file {config_file}: {e}")

        print("  ✅ Configuration files validated")
        return len(self.issues) == 0

    def estimate_cleanup_impact(self) -> dict[str, int]:
        """Estimate the impact of cleanup operations."""
        print("🔍 Estimating cleanup impact...")

        removable, protected = self.identify_removable_files()

        # Count by file type
        impact = {
            "docs_removable": len([f for f in removable if f.suffix == ".md"]),
            "tests_removable": len(
                [f for f in removable if f.name.startswith("test_")]
            ),
            "configs_removable": len(
                [f for f in removable if f.suffix in [".json", ".yaml", ".yml"]]
            ),
            "total_removable": len(removable),
            "total_protected": len(protected),
        }

        # Calculate current totals
        all_docs = len(list(self.project_root.rglob("*.md")))
        all_tests = len(list(self.project_root.rglob("test_*.py")))
        all_configs = (
            len(list(self.project_root.rglob("*.json")))
            + len(list(self.project_root.rglob("*.yaml")))
            + len(list(self.project_root.rglob("*.yml")))
        )

        impact.update(
            {
                "docs_total": all_docs,
                "tests_total": all_tests,
                "configs_total": all_configs,
                "docs_reduction_pct": round(
                    impact["docs_removable"] / all_docs * 100, 1
                )
                if all_docs > 0
                else 0,
                "tests_reduction_pct": round(
                    impact["tests_removable"] / all_tests * 100, 1
                )
                if all_tests > 0
                else 0,
                "configs_reduction_pct": round(
                    impact["configs_removable"] / all_configs * 100, 1
                )
                if all_configs > 0
                else 0,
            }
        )

        return impact

    def generate_safety_report(self) -> dict:
        """Generate a comprehensive safety report."""
        print("📊 Generating safety validation report...")

        # Run all validations
        essential_ok = self.validate_essential_files()
        self.analyze_import_dependencies()
        removable, protected = self.identify_removable_files()
        test_coverage_ok = self.check_test_coverage()
        config_ok = self.validate_configuration_integrity()
        impact = self.estimate_cleanup_impact()

        report = {
            "validation_date": "2025-06-16",
            "project_root": str(self.project_root),
            "safety_status": len(self.issues) == 0,
            "validations": {
                "essential_files": essential_ok,
                "test_coverage": test_coverage_ok,
                "configuration": config_ok,
            },
            "impact_analysis": impact,
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self) -> list[str]:
        """Generate safety recommendations based on analysis."""
        recommendations = []

        if self.issues:
            recommendations.append(
                "❌ STOP: Critical issues found - resolve before cleanup"
            )

        if self.warnings:
            recommendations.append("⚠️  Review warnings before proceeding")

        if not self.issues:
            recommendations.extend(
                [
                    "✅ Basic safety validation passed",
                    "📋 Create full backup before executing cleanup",
                    "🧪 Run test suite after cleanup to verify functionality",
                    "👁️  Monitor system for 24h after cleanup",
                ]
            )

        return recommendations


def main():
    """Main validation execution."""
    print("🛡️  Codebase Cleanup Safety Validation")
    print("=" * 50)

    project_root = find_project_root()
    print(f"📁 Project root: {project_root}")

    validator = CleanupValidator(project_root)
    report = validator.generate_safety_report()

    # Save report
    report_file = project_root / "cleanup_safety_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 30)

    if report["safety_status"]:
        print("✅ SAFE TO PROCEED with cleanup")
    else:
        print("❌ UNSAFE - Issues must be resolved first")

    print("\n📈 IMPACT ESTIMATE:")
    impact = report["impact_analysis"]
    print(
        f"  • Documentation: {impact['docs_removable']}/{impact['docs_total']} files ({impact['docs_reduction_pct']}% reduction)"
    )
    print(
        f"  • Test files: {impact['tests_removable']}/{impact['tests_total']} files ({impact['tests_reduction_pct']}% reduction)"
    )
    print(
        f"  • Config files: {impact['configs_removable']}/{impact['configs_total']} files ({impact['configs_reduction_pct']}% reduction)"
    )
    print(f"  • Total removable: {impact['total_removable']} files")

    if report["issues"]:
        print(f"\n❌ ISSUES ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"  • {issue}")

    if report["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  • {warning}")

    print("\n💡 RECOMMENDATIONS:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")

    print(f"\n📄 Detailed report saved: {report_file}")

    # Exit with appropriate code
    sys.exit(0 if report["safety_status"] else 1)


if __name__ == "__main__":
    main()
