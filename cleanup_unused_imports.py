#!/usr/bin/env python3
"""
Automated script to identify and remove unused imports across the entire codebase.
Focuses on high-impact areas: MLX/Metal, AsyncIO, and cross-module dependencies.
"""

import argparse
import ast
import json
import sys
from pathlib import Path


class ImportUsageAnalyzer(ast.NodeVisitor):
    """Analyzes Python files to find unused imports."""

    def __init__(self):
        self.imports = {}  # {alias: (module, name)}
        self.used_names = set()
        self.import_lines = []  # Track line numbers for removal

    def visit_Import(self, node):
        """Track regular imports: import module"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = (alias.name, None)
            self.import_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track from imports: from module import name"""
        if node.module:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.imports[name] = (node.module, alias.name)
                self.import_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node):
        """Track name usage"""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Track attribute access like module.function"""
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)


def analyze_file(file_path: Path) -> dict:
    """Analyze a single Python file for unused imports."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        analyzer = ImportUsageAnalyzer()
        analyzer.visit(tree)

        unused_imports = []
        for name, (module, original_name) in analyzer.imports.items():
            if name not in analyzer.used_names:
                unused_imports.append(
                    {
                        "name": name,
                        "module": module,
                        "original_name": original_name,
                        "type": "mlx"
                        if "mlx" in module.lower()
                        else "metal"
                        if "metal" in module.lower()
                        else "asyncio"
                        if "asyncio" in module.lower()
                        else "standard",
                    }
                )

        return {
            "file": str(file_path),
            "unused_imports": unused_imports,
            "total_imports": len(analyzer.imports),
            "unused_count": len(unused_imports),
            "import_lines": analyzer.import_lines,
        }

    except Exception as e:
        return {
            "file": str(file_path),
            "error": str(e),
            "unused_imports": [],
            "total_imports": 0,
            "unused_count": 0,
            "import_lines": [],
        }


def find_python_files(root_path: Path) -> list[Path]:
    """Find all Python files in the codebase."""
    python_files = []
    for file_path in root_path.rglob("*.py"):
        # Skip certain directories
        skip_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "venv",
            "env",
            "node_modules",
        }
        if not any(part in skip_dirs for part in file_path.parts):
            python_files.append(file_path)
    return python_files


def categorize_imports(analysis_results: list[dict]) -> dict:
    """Categorize unused imports by type for targeted cleanup."""
    categories = {
        "mlx": [],
        "metal": [],
        "asyncio": [],
        "standard": [],
        "high_impact": [],  # Files with many unused imports
    }

    for result in analysis_results:
        if result.get("error"):
            continue

        file_path = result["file"]
        unused_imports = result["unused_imports"]

        # High impact files (>5 unused imports)
        if result["unused_count"] > 5:
            categories["high_impact"].append(
                {
                    "file": file_path,
                    "count": result["unused_count"],
                    "imports": unused_imports,
                }
            )

        # Categorize by type
        for imp in unused_imports:
            categories[imp["type"]].append({"file": file_path, "import": imp})

    return categories


def generate_cleanup_commands(categories: dict) -> list[str]:
    """Generate commands to clean up imports."""
    commands = []

    # Priority order: MLX/Metal first (heavy dependencies)
    for category in ["mlx", "metal", "asyncio", "standard"]:
        if categories[category]:
            commands.append(f"# Clean up {category.upper()} imports")
            files_to_clean = set(item["file"] for item in categories[category])
            for file_path in files_to_clean:
                commands.append(f"# Process {file_path}")

    return commands


def remove_unused_imports_from_file(
    file_path: Path, unused_imports: list[dict]
) -> bool:
    """Remove unused imports from a specific file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Parse the file to get import statements
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        # Find import statements to remove
        imports_to_remove = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    if any(imp["name"] == name for imp in unused_imports):
                        imports_to_remove.add(node.lineno)

        # Remove lines (in reverse order to maintain line numbers)
        for line_num in sorted(imports_to_remove, reverse=True):
            if 1 <= line_num <= len(lines):
                lines.pop(line_num - 1)

        # Write back the cleaned content
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return True

    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Clean up unused imports")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze, do not remove imports",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="import_analysis.json",
        help="Output file for analysis results",
    )
    parser.add_argument("--root", "-r", default=".", help="Root directory to analyze")
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: Path {root_path} does not exist")
        return 1

    print("🔍 Analyzing Python files for unused imports...")
    python_files = find_python_files(root_path)
    print(f"Found {len(python_files)} Python files")

    # Analyze all files
    analysis_results = []
    for i, file_path in enumerate(python_files):
        if i % 50 == 0:
            print(f"  Analyzed {i}/{len(python_files)} files...")

        result = analyze_file(file_path)
        analysis_results.append(result)

    # Categorize results
    categories = categorize_imports(analysis_results)

    # Generate summary
    total_unused = sum(
        r["unused_count"] for r in analysis_results if not r.get("error")
    )
    total_files_with_unused = sum(1 for r in analysis_results if r["unused_count"] > 0)

    summary = {
        "total_files": len(python_files),
        "total_unused_imports": total_unused,
        "files_with_unused_imports": total_files_with_unused,
        "categories": {k: len(v) for k, v in categories.items()},
        "high_impact_files": len(categories["high_impact"]),
    }

    print("\n📊 Analysis Summary:")
    print(f"  Total files analyzed: {summary['total_files']}")
    print(f"  Total unused imports: {summary['total_unused_imports']}")
    print(f"  Files with unused imports: {summary['files_with_unused_imports']}")
    print(
        f"  MLX/Metal unused imports: {summary['categories']['mlx'] + summary['categories']['metal']}"
    )
    print(f"  AsyncIO unused imports: {summary['categories']['asyncio']}")
    print(f"  High-impact files (>5 unused): {summary['high_impact_files']}")

    # Save detailed results
    detailed_results = {
        "summary": summary,
        "categories": categories,
        "analysis_results": analysis_results,
    }

    with open(args.output, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\n💾 Detailed results saved to {args.output}")

    if not args.analyze_only:
        print("\n🧹 Starting cleanup process...")

        # Clean up high-impact files first
        cleaned_files = 0
        for high_impact in categories["high_impact"]:
            file_path = Path(high_impact["file"])
            if remove_unused_imports_from_file(file_path, high_impact["imports"]):
                cleaned_files += 1
                print(
                    f"  ✅ Cleaned {file_path} ({high_impact['count']} imports removed)"
                )

        print(f"\n✨ Cleanup complete! Cleaned {cleaned_files} files")
        print(f"   Removed approximately {total_unused} unused imports")

    return 0


if __name__ == "__main__":
    sys.exit(main())
