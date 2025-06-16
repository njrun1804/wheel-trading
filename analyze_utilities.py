#!/usr/bin/env python3
"""Deep Utility Function Analysis for Agent 6"""

import ast
from collections import defaultdict
from pathlib import Path

# Utility categories
UTILITY_CATEGORIES = {
    "memory": ["memory", "cache", "buffer", "pool", "alloc"],
    "logging": ["log", "logger", "debug", "trace", "monitor"],
    "validation": ["validate", "check", "verify", "assert", "ensure"],
    "io": ["load", "save", "read", "write", "parse", "dump"],
    "formatting": ["format", "encode", "decode", "serialize", "deserialize"],
    "conversion": ["convert", "transform", "cast", "normalize", "clean"],
    "error": ["error", "exception", "recover", "retry", "handle"],
    "config": ["config", "settings", "options", "params", "env"],
}


class UtilityAnalyzer:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.utilities = defaultdict(list)
        self.duplicates = defaultdict(set)
        self.import_graph = defaultdict(set)

    def categorize_function(self, func_name: str) -> str:
        """Categorize a function based on its name"""
        func_lower = func_name.lower()
        for category, keywords in UTILITY_CATEGORIES.items():
            if any(kw in func_lower for kw in keywords):
                return category
        return "other"

    def extract_functions(self, file_path: Path) -> list[tuple[str, str, int]]:
        """Extract function definitions from a Python file"""
        functions = []
        try:
            with open(file_path) as f:
                content = f.read()
                tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    category = self.categorize_function(node.name)
                    if category != "other":
                        functions.append((node.name, category, node.lineno))

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

        return functions

    def find_imports(self, file_path: Path) -> set[str]:
        """Find all imports in a file"""
        imports = set()
        try:
            with open(file_path) as f:
                content = f.read()
                tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)

        except Exception:
            pass

        return imports

    def analyze_codebase(self):
        """Analyze the entire codebase for utilities"""
        py_files = list(self.root_path.glob("**/*.py"))

        print(f"Analyzing {len(py_files)} Python files...")

        # First pass: collect all utilities
        for file_path in py_files:
            if any(
                skip in str(file_path)
                for skip in ["__pycache__", ".git", "node_modules", "venv"]
            ):
                continue

            functions = self.extract_functions(file_path)
            imports = self.find_imports(file_path)

            for func_name, category, line_no in functions:
                self.utilities[category].append(
                    {
                        "name": func_name,
                        "file": str(file_path.relative_to(self.root_path)),
                        "line": line_no,
                    }
                )

                # Track potential duplicates
                self.duplicates[func_name].add(
                    str(file_path.relative_to(self.root_path))
                )

            # Build import graph
            self.import_graph[str(file_path.relative_to(self.root_path))] = imports

    def find_true_duplicates(self) -> dict[str, list[str]]:
        """Find functions that are truly duplicated (same name, multiple files)"""
        true_duplicates = {}
        for func_name, files in self.duplicates.items():
            if len(files) > 1:
                true_duplicates[func_name] = sorted(list(files))
        return true_duplicates

    def generate_report(self):
        """Generate comprehensive utility analysis report"""
        print("\n" + "=" * 80)
        print("UTILITY FUNCTION ANALYSIS REPORT")
        print("=" * 80)

        # Category breakdown
        print("\n## UTILITY CATEGORIES")
        total_utils = 0
        for category, utils in sorted(self.utilities.items()):
            print(f"\n### {category.upper()} ({len(utils)} functions)")
            total_utils += len(utils)

            # Show top 5 functions per category
            for util in utils[:5]:
                print(f"  - {util['name']} ({util['file']}:{util['line']})")
            if len(utils) > 5:
                print(f"  ... and {len(utils) - 5} more")

        print(f"\nTOTAL UTILITY FUNCTIONS: {total_utils}")

        # Duplicates analysis
        true_duplicates = self.find_true_duplicates()
        print(f"\n## DUPLICATE FUNCTIONS ({len(true_duplicates)} found)")

        for func_name, files in sorted(true_duplicates.items())[:10]:
            print(f"\n{func_name} appears in {len(files)} files:")
            for file in files[:3]:
                print(f"  - {file}")
            if len(files) > 3:
                print(f"  ... and {len(files) - 3} more")

        # Import complexity
        print("\n## IMPORT COMPLEXITY")
        high_import_files = [
            (f, len(imports))
            for f, imports in self.import_graph.items()
            if len(imports) > 20
        ]
        high_import_files.sort(key=lambda x: x[1], reverse=True)

        print("\nFiles with high import counts (>20):")
        for file, count in high_import_files[:10]:
            print(f"  - {file}: {count} imports")

        # Quick wins for consolidation
        print("\n## QUICK CONSOLIDATION WINS")

        # Memory utilities
        memory_utils = self.utilities.get("memory", [])
        print(f"\n1. Memory Utilities ({len(memory_utils)} functions)")
        memory_files = set(u["file"] for u in memory_utils)
        print(f"   Spread across {len(memory_files)} files")
        print("   Recommended: Consolidate into src/unity_wheel/memory/")

        # Logging utilities
        logging_utils = self.utilities.get("logging", [])
        print(f"\n2. Logging Utilities ({len(logging_utils)} functions)")
        logging_files = set(u["file"] for u in logging_utils)
        print(f"   Spread across {len(logging_files)} files")
        print("   Recommended: Consolidate into src/unity_wheel/utils/logging.py")

        # Validation utilities
        validation_utils = self.utilities.get("validation", [])
        print(f"\n3. Validation Utilities ({len(validation_utils)} functions)")
        validation_files = set(u["file"] for u in validation_utils)
        print(f"   Spread across {len(validation_files)} files")
        print("   Recommended: Consolidate into src/unity_wheel/utils/validation.py")

        # Save detailed results
        self.save_detailed_results()

    def save_detailed_results(self):
        """Save detailed analysis results"""
        output_file = self.root_path / "utility_analysis_results.json"

        import json

        results = {
            "summary": {
                "total_utilities": sum(len(utils) for utils in self.utilities.values()),
                "categories": {
                    cat: len(utils) for cat, utils in self.utilities.items()
                },
                "duplicate_functions": len(self.find_true_duplicates()),
            },
            "utilities_by_category": self.utilities,
            "duplicates": self.find_true_duplicates(),
            "high_import_files": [
                (f, len(imports))
                for f, imports in self.import_graph.items()
                if len(imports) > 20
            ],
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    root = Path(
        "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
    )
    analyzer = UtilityAnalyzer(root)
    analyzer.analyze_codebase()
    analyzer.generate_report()
