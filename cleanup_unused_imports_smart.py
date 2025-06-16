#!/usr/bin/env python3
"""
Smart unused import cleanup that handles edge cases like:
- Imports inside try/except blocks
- Imports used only for side effects
- Imports aliased but never used with alias
- Star imports
"""

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path


class SmartImportAnalyzer(ast.NodeVisitor):
    """Enhanced import analyzer that handles complex usage patterns."""

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_lines = source_code.split("\n")
        self.imports = {}  # {alias: ImportInfo}
        self.used_names = set()
        self.string_usage = set()  # Names used in strings (like in getattr calls)
        self.import_nodes = []  # Store actual import nodes

    def visit_Import(self, node):
        """Track regular imports: import module"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            import_info = ImportInfo(
                name=name,
                module=alias.name,
                original_name=None,
                line_no=node.lineno,
                import_type="import",
                in_try_except=self._in_try_except(node),
                node=node,
            )
            self.imports[name] = import_info
            self.import_nodes.append(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track from imports: from module import name"""
        if node.module:
            for alias in node.names:
                if alias.name == "*":
                    # Star import - mark as used to avoid removal
                    continue

                name = alias.asname if alias.asname else alias.name
                import_info = ImportInfo(
                    name=name,
                    module=node.module,
                    original_name=alias.name,
                    line_no=node.lineno,
                    import_type="from_import",
                    in_try_except=self._in_try_except(node),
                    node=node,
                )
                self.imports[name] = import_info
                self.import_nodes.append(node)
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

    def visit_Str(self, node):
        """Track string literals that might contain module names"""
        # Look for module names in strings (common in getattr, hasattr, etc.)
        for name in self.imports:
            if name in node.s:
                self.string_usage.add(name)
        self.generic_visit(node)

    def visit_Constant(self, node):
        """Track constant values (Python 3.8+)"""
        if isinstance(node.value, str):
            for name in self.imports:
                if name in node.value:
                    self.string_usage.add(name)
        self.generic_visit(node)

    def _in_try_except(self, node) -> bool:
        """Check if a node is inside a try/except block."""
        # Simple heuristic: check line numbers of try blocks
        line_no = node.lineno
        for line_idx in range(
            max(0, line_no - 10), min(len(self.source_lines), line_no + 5)
        ):
            line = self.source_lines[line_idx].strip()
            if line.startswith("try:") and line_idx < line_no:
                return True
        return False

    def analyze_unused_imports(self) -> list["ImportInfo"]:
        """Find truly unused imports with smart detection."""
        potentially_unused = []

        for name, import_info in self.imports.items():
            # Skip if directly used
            if name in self.used_names:
                continue

            # Skip if used in strings (side effects)
            if name in self.string_usage:
                continue

            # Skip certain patterns that are likely used
            if self._is_likely_used(import_info):
                continue

            potentially_unused.append(import_info)

        return potentially_unused

    def _is_likely_used(self, import_info: "ImportInfo") -> bool:
        """Check if import is likely used even if not detected."""
        # Type checking imports
        if import_info.module.startswith("typing"):
            return True

        # Common side-effect imports
        side_effect_modules = {
            "logging",
            "warnings",
            "sys",
            "os",
            "signal",
            "matplotlib.pyplot",
            "seaborn",
            "pandas",
        }
        if import_info.module in side_effect_modules:
            return True

        # If import is in a try/except, it might be for optional functionality
        if import_info.in_try_except:
            # Look for usage in the same try block or later
            return self._check_try_block_usage(import_info)

        return False

    def _check_try_block_usage(self, import_info: "ImportInfo") -> bool:
        """Check if import in try/except is used within that context."""
        line_no = import_info.line_no

        # Look for usage in the lines following the import
        for i in range(line_no, min(len(self.source_lines), line_no + 20)):
            line = self.source_lines[i]
            if import_info.name in line and "import" not in line:
                return True

        return False


class ImportInfo:
    """Information about an import statement."""

    def __init__(
        self,
        name: str,
        module: str,
        original_name: str | None,
        line_no: int,
        import_type: str,
        in_try_except: bool,
        node: ast.AST,
    ):
        self.name = name
        self.module = module
        self.original_name = original_name
        self.line_no = line_no
        self.import_type = import_type
        self.in_try_except = in_try_except
        self.node = node

        # Categorize by type for prioritized cleanup
        self.category = self._categorize()

    def _categorize(self) -> str:
        """Categorize import for prioritized cleanup."""
        module_lower = self.module.lower()

        if "mlx" in module_lower:
            return "mlx"
        elif "metal" in module_lower:
            return "metal"
        elif "asyncio" in module_lower or "async" in module_lower:
            return "asyncio"
        elif any(
            heavy in module_lower
            for heavy in ["torch", "tensorflow", "sklearn", "pandas", "numpy"]
        ):
            return "heavy"
        else:
            return "standard"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "module": self.module,
            "original_name": self.original_name,
            "line_no": self.line_no,
            "import_type": self.import_type,
            "in_try_except": self.in_try_except,
            "category": self.category,
        }


def analyze_file_smart(file_path: Path) -> dict:
    """Smart analysis of a single Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        analyzer = SmartImportAnalyzer(content)
        analyzer.visit(tree)

        unused_imports = analyzer.analyze_unused_imports()

        return {
            "file": str(file_path),
            "unused_imports": [imp.to_dict() for imp in unused_imports],
            "total_imports": len(analyzer.imports),
            "unused_count": len(unused_imports),
            "high_priority": len(
                [
                    imp
                    for imp in unused_imports
                    if imp.category in ["mlx", "metal", "heavy"]
                ]
            ),
            "categories": {
                cat: len([imp for imp in unused_imports if imp.category == cat])
                for cat in ["mlx", "metal", "asyncio", "heavy", "standard"]
            },
        }

    except Exception as e:
        return {
            "file": str(file_path),
            "error": str(e),
            "unused_imports": [],
            "total_imports": 0,
            "unused_count": 0,
            "high_priority": 0,
            "categories": {},
        }


def remove_unused_imports_smart(file_path: Path, unused_imports: list[dict]) -> bool:
    """Intelligently remove unused imports from a file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Group imports by line number for efficient removal
        lines_to_remove = set()

        for imp in unused_imports:
            line_no = imp["line_no"]
            if 1 <= line_no <= len(lines):
                # Check if this is a multi-import line (e.g., "from x import a, b, c")
                line_content = lines[line_no - 1].strip()

                if imp["import_type"] == "from_import" and "," in line_content:
                    # Handle multi-import lines
                    lines[line_no - 1] = remove_from_multi_import_line(
                        line_content, imp["name"]
                    )
                else:
                    # Single import - remove entire line
                    lines_to_remove.add(line_no - 1)

        # Remove lines (in reverse order to maintain indices)
        for line_idx in sorted(lines_to_remove, reverse=True):
            lines.pop(line_idx)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return True

    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return False


def remove_from_multi_import_line(line: str, name_to_remove: str) -> str:
    """Remove a specific import from a multi-import line."""
    # Handle lines like "from module import a, b, c"
    if "from " in line and " import " in line:
        parts = line.split(" import ", 1)
        if len(parts) == 2:
            prefix = parts[0] + " import "
            imports_part = parts[1]

            # Split imports and remove the target
            imports = [imp.strip() for imp in imports_part.split(",")]
            imports = [imp for imp in imports if imp != name_to_remove]

            if imports:
                return prefix + ", ".join(imports) + "\n"
            else:
                return ""  # Remove entire line if no imports left

    return line


def main():
    parser = argparse.ArgumentParser(description="Smart unused import cleanup")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze, do not remove imports",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="smart_import_analysis.json",
        help="Output file for analysis results",
    )
    parser.add_argument("--root", "-r", default=".", help="Root directory to analyze")
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        choices=["mlx", "metal", "asyncio", "heavy", "standard"],
        default=["mlx", "metal", "heavy"],
        help="Categories to process",
    )
    args = parser.parse_args()

    root_path = Path(args.root)
    print("🔍 Smart analysis of Python files for unused imports...")

    # Find Python files
    python_files = []
    for file_path in root_path.rglob("*.py"):
        skip_dirs = {".git", "__pycache__", ".pytest_cache", "venv", "env"}
        if not any(part in skip_dirs for part in file_path.parts):
            python_files.append(file_path)

    print(f"Found {len(python_files)} Python files")

    # Analyze files
    results = []
    high_priority_files = []

    for i, file_path in enumerate(python_files):
        if i % 100 == 0:
            print(f"  Analyzed {i}/{len(python_files)} files...")

        result = analyze_file_smart(file_path)
        results.append(result)

        if result["high_priority"] > 0:
            high_priority_files.append(result)

    # Summary
    total_unused = sum(r["unused_count"] for r in results if not r.get("error"))
    total_high_priority = sum(r["high_priority"] for r in results if not r.get("error"))

    category_totals = defaultdict(int)
    for result in results:
        if not result.get("error"):
            for cat, count in result["categories"].items():
                category_totals[cat] += count

    print("\n📊 Smart Analysis Results:")
    print(f"  Total unused imports: {total_unused}")
    print(f"  High priority (MLX/Metal/Heavy): {total_high_priority}")
    print(f"  MLX imports: {category_totals['mlx']}")
    print(f"  Metal imports: {category_totals['metal']}")
    print(f"  Heavy imports: {category_totals['heavy']}")
    print(f"  AsyncIO imports: {category_totals['asyncio']}")
    print(f"  High-priority files: {len(high_priority_files)}")

    # Save results
    analysis_data = {
        "summary": {
            "total_files": len(python_files),
            "total_unused_imports": total_unused,
            "high_priority_imports": total_high_priority,
            "category_totals": dict(category_totals),
            "high_priority_files": len(high_priority_files),
        },
        "high_priority_files": high_priority_files[:20],  # Top 20
        "all_results": results,
    }

    with open(args.output, "w") as f:
        json.dump(analysis_data, f, indent=2)

    print(f"💾 Results saved to {args.output}")

    if not args.analyze_only:
        print("\n🧹 Starting smart cleanup...")

        # Process files with high-priority unused imports
        cleaned_files = 0
        removed_imports = 0

        for result in high_priority_files:
            if result.get("error"):
                continue

            file_path = Path(result["file"])
            unused_imports = result["unused_imports"]

            # Filter by requested categories
            filtered_imports = [
                imp for imp in unused_imports if imp["category"] in args.categories
            ]

            if filtered_imports:
                if remove_unused_imports_smart(file_path, filtered_imports):
                    cleaned_files += 1
                    removed_imports += len(filtered_imports)
                    print(
                        f"  ✅ {file_path.name}: removed {len(filtered_imports)} imports"
                    )

        print("\n✨ Smart cleanup complete!")
        print(f"  Files cleaned: {cleaned_files}")
        print(f"  Imports removed: {removed_imports}")
        print(f"  Categories processed: {', '.join(args.categories)}")


if __name__ == "__main__":
    main()
