#!/usr/bin/env python3
"""
Detect and resolve circular import dependencies between modules.
"""

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path


class CircularImportDetector:
    """Detect circular import dependencies in Python codebase."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.imports: dict[str, set[str]] = defaultdict(set)
        self.module_paths: dict[str, str] = {}

    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for imports."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Convert file path to module name
            rel_path = file_path.relative_to(self.root_path)
            module_name = self._path_to_module(rel_path)
            self.module_paths[module_name] = str(file_path)

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_module = alias.name
                        self.imports[module_name].add(imported_module)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:
                            imported_module = self._resolve_relative_import(
                                module_name, node.module, node.level
                            )
                        else:
                            imported_module = node.module

                        if imported_module:
                            self.imports[module_name].add(imported_module)

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    def _path_to_module(self, path: Path) -> str:
        """Convert file path to module name."""
        parts = list(path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        elif parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        return ".".join(parts)

    def _resolve_relative_import(
        self, current_module: str, imported_module: str | None, level: int
    ) -> str | None:
        """Resolve relative imports to absolute module names."""
        current_parts = current_module.split(".")

        # Go up 'level' directories
        if level > len(current_parts):
            return None

        base_parts = current_parts[:-level] if level > 0 else current_parts

        if imported_module:
            return ".".join(base_parts + imported_module.split("."))
        else:
            return ".".join(base_parts)

    def find_python_files(self) -> list[Path]:
        """Find all Python files in the codebase."""
        python_files = []
        exclude_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "venv",
            "env",
            "node_modules",
        }

        for file_path in self.root_path.rglob("*.py"):
            if not any(part in exclude_dirs for part in file_path.parts):
                python_files.append(file_path)

        return python_files

    def build_import_graph(self) -> None:
        """Build the import dependency graph."""
        python_files = self.find_python_files()
        print(f"Analyzing {len(python_files)} Python files for circular imports...")

        for file_path in python_files:
            self.analyze_file(file_path)

    def detect_cycles(self) -> list[list[str]]:
        """Detect cycles in the import graph using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.imports.get(node, set()):
                if neighbor in self.imports:  # Only consider modules we know about
                    if neighbor not in visited:
                        dfs(neighbor, path.copy())
                    elif neighbor in rec_stack:
                        # Found cycle
                        try:
                            cycle_start = path.index(neighbor)
                            cycle = path[cycle_start:] + [neighbor]
                            cycles.append(cycle)
                        except ValueError:
                            # neighbor not in path, add partial cycle
                            cycles.append(path + [neighbor])

            rec_stack.remove(node)

        # Run DFS from each unvisited node
        for node in self.imports:
            if node not in visited:
                dfs(node, [])

        return cycles

    def categorize_cycles(self, cycles: list[list[str]]) -> dict[str, list[list[str]]]:
        """Categorize cycles by the components involved."""
        categorized = {"einstein_bolt": [], "internal": [], "external": [], "other": []}

        for cycle in cycles:
            # Check if this involves Einstein and Bolt
            has_einstein = any("einstein" in module.lower() for module in cycle)
            has_bolt = any("bolt" in module.lower() for module in cycle)

            if has_einstein and has_bolt:
                categorized["einstein_bolt"].append(cycle)
            elif all(
                module.startswith("src.") or module.startswith("unity_wheel.")
                for module in cycle
            ):
                categorized["internal"].append(cycle)
            elif any(
                not (module.startswith("src.") or module.startswith("unity_wheel."))
                for module in cycle
            ):
                categorized["external"].append(cycle)
            else:
                categorized["other"].append(cycle)

        return categorized

    def suggest_fixes(self, cycles: list[list[str]]) -> list[dict[str, str]]:
        """Suggest fixes for circular import issues."""
        fixes = []

        for cycle in cycles:
            if len(cycle) == 2:
                # Simple A -> B -> A cycle
                module_a, module_b = cycle[0], cycle[1]
                fixes.append(
                    {
                        "type": "move_import",
                        "description": f"Move import from {module_a} to function level in {module_b}",
                        "cycle": " -> ".join(cycle),
                        "action": "Convert module-level import to function-level import",
                    }
                )
            else:
                # Complex cycle
                # Suggest breaking the cycle at the weakest link
                weakest_link = self._find_weakest_link(cycle)
                fixes.append(
                    {
                        "type": "break_cycle",
                        "description": f"Break cycle by moving import to function level at {weakest_link}",
                        "cycle": " -> ".join(cycle),
                        "action": f"Convert import at {weakest_link} to function-level or use TYPE_CHECKING",
                    }
                )

        return fixes

    def _find_weakest_link(self, cycle: list[str]) -> str:
        """Find the best place to break a cycle."""
        # Prefer breaking at test files, config files, or utility modules
        weak_patterns = ["test_", "config", "util", "_test", "setup"]

        for module in cycle:
            if any(pattern in module.lower() for pattern in weak_patterns):
                return module

        # Otherwise, return the last module in the cycle
        return cycle[-1]

    def generate_report(self) -> dict:
        """Generate a comprehensive report of circular imports."""
        cycles = self.detect_cycles()
        categorized = self.categorize_cycles(cycles)
        fixes = self.suggest_fixes(cycles)

        # Calculate statistics
        total_modules = len(self.imports)
        total_imports = sum(len(imports) for imports in self.imports.values())

        report = {
            "summary": {
                "total_modules_analyzed": total_modules,
                "total_imports": total_imports,
                "cycles_found": len(cycles),
                "einstein_bolt_cycles": len(categorized["einstein_bolt"]),
                "internal_cycles": len(categorized["internal"]),
                "external_cycles": len(categorized["external"]),
            },
            "cycles_by_category": categorized,
            "suggested_fixes": fixes,
            "detailed_cycles": [
                {
                    "cycle": cycle,
                    "length": len(cycle),
                    "modules_involved": cycle,
                    "files_involved": [
                        self.module_paths.get(mod, "Unknown") for mod in cycle
                    ],
                }
                for cycle in cycles
            ],
        }

        return report


def main():
    parser = argparse.ArgumentParser(description="Detect circular import dependencies")
    parser.add_argument("--root", "-r", default=".", help="Root directory to analyze")
    parser.add_argument(
        "--output", "-o", default="circular_imports_report.json", help="Output file"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix simple cycles"
    )
    args = parser.parse_args()

    detector = CircularImportDetector(args.root)
    detector.build_import_graph()

    report = detector.generate_report()

    # Print summary
    print("\n📊 Circular Import Analysis Results:")
    print(f"  Total modules analyzed: {report['summary']['total_modules_analyzed']}")
    print(f"  Total import statements: {report['summary']['total_imports']}")
    print(f"  Circular dependencies found: {report['summary']['cycles_found']}")
    print(f"  Einstein-Bolt cycles: {report['summary']['einstein_bolt_cycles']}")
    print(f"  Internal cycles: {report['summary']['internal_cycles']}")
    print(f"  External cycles: {report['summary']['external_cycles']}")

    if report["summary"]["cycles_found"] > 0:
        print("\n🔄 Detected Circular Dependencies:")
        for i, cycle_info in enumerate(report["detailed_cycles"][:10]):  # Show first 10
            cycle = cycle_info["cycle"]
            print(f"  {i+1}. {' -> '.join(cycle)} (length: {len(cycle)})")

        if len(report["detailed_cycles"]) > 10:
            print(f"  ... and {len(report['detailed_cycles']) - 10} more cycles")

        print("\n💡 Suggested Fixes:")
        for i, fix in enumerate(report["suggested_fixes"][:5]):  # Show first 5
            print(f"  {i+1}. {fix['description']}")
            print(f"     Cycle: {fix['cycle']}")
            print(f"     Action: {fix['action']}")
            print()

    # Save detailed report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"💾 Detailed report saved to {args.output}")

    if args.fix:
        print("\n🔧 Attempting to fix simple cycles...")
        # TODO: Implement automatic fixes
        print("   Automatic fixing not implemented yet - use suggested fixes above")

    return 0 if report["summary"]["cycles_found"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
