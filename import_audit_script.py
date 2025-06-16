#!/usr/bin/env python3
"""
Import Usage Audit for Einstein and Bolt directories
Analyzes all Python files to identify:
1. All import statements
2. Actual usage of imported modules/functions
3. Unused imports that can be removed
4. Circular import issues
5. MLX/Metal usage patterns
"""

import ast
import re
from collections import defaultdict
from pathlib import Path


class ImportUsageAnalyzer:
    def __init__(self):
        self.files_analyzed = 0
        self.total_imports = 0
        self.unused_imports = []
        self.mlx_usage = []
        self.metal_usage = []
        self.circular_imports = []
        self.import_map = defaultdict(list)  # module -> files that import it
        self.usage_map = defaultdict(list)  # module -> actual usage locations

    def analyze_file(self, file_path: Path) -> dict:
        """Analyze a single Python file for imports and usage."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            file_analysis = {
                "file": str(file_path),
                "imports": [],
                "usage": [],
                "unused": [],
                "mlx_related": [],
                "metal_related": [],
            }

            # Extract imports
            imports = self._extract_imports(tree)
            file_analysis["imports"] = imports

            # Check usage
            used_modules = self._find_usage(content, imports)
            file_analysis["usage"] = used_modules

            # Find unused
            unused = [imp for imp in imports if imp["name"] not in used_modules]
            file_analysis["unused"] = unused

            # Check for MLX/Metal usage
            mlx_items = [imp for imp in imports if "mlx" in imp["name"].lower()]
            metal_items = [imp for imp in imports if "metal" in imp["name"].lower()]

            if mlx_items or "mlx" in content.lower():
                file_analysis["mlx_related"] = mlx_items
                self.mlx_usage.append(
                    {
                        "file": str(file_path),
                        "imports": mlx_items,
                        "usage_lines": self._find_usage_lines(content, ["mlx", "MLX"]),
                    }
                )

            if metal_items or "metal" in content.lower():
                file_analysis["metal_related"] = metal_items
                self.metal_usage.append(
                    {
                        "file": str(file_path),
                        "imports": metal_items,
                        "usage_lines": self._find_usage_lines(
                            content, ["metal", "Metal", "METAL"]
                        ),
                    }
                )

            self.files_analyzed += 1
            self.total_imports += len(imports)
            self.unused_imports.extend(unused)

            return file_analysis

        except Exception as e:
            return {
                "file": str(file_path),
                "error": str(e),
                "imports": [],
                "usage": [],
                "unused": [],
            }

    def _extract_imports(self, tree: ast.AST) -> list[dict]:
        """Extract all import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        }
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": node.module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        }
                    )

        return imports

    def _find_usage(self, content: str, imports: list[dict]) -> set[str]:
        """Find actual usage of imported modules/functions in content."""
        used = set()

        for imp in imports:
            name_to_check = imp.get("alias") or imp["name"]

            # Skip the import line itself
            lines = content.split("\n")
            import_line = imp["line"] - 1  # Convert to 0-based index

            for i, line in enumerate(lines):
                if i == import_line:
                    continue

                # Check if the imported name is used
                if self._is_name_used_in_line(line, name_to_check):
                    used.add(name_to_check)
                    break

        return used

    def _is_name_used_in_line(self, line: str, name: str) -> bool:
        """Check if a name is actually used in a line of code."""
        # Skip comments and strings (basic check)
        if line.strip().startswith("#"):
            return False

        # Use word boundaries to avoid false positives
        pattern = r"\b" + re.escape(name) + r"\b"
        return bool(re.search(pattern, line))

    def _find_usage_lines(self, content: str, patterns: list[str]) -> list[dict]:
        """Find lines where specific patterns are used."""
        usage_lines = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern in patterns:
                if pattern.lower() in line.lower():
                    usage_lines.append(
                        {"line": i, "content": line.strip(), "pattern": pattern}
                    )

        return usage_lines

    def analyze_directory(self, directory: Path) -> dict:
        """Analyze all Python files in a directory."""
        results = []

        for py_file in directory.rglob("*.py"):
            if py_file.name.startswith("."):
                continue

            file_result = self.analyze_file(py_file)
            results.append(file_result)

            # Build import map for circular dependency detection
            for imp in file_result["imports"]:
                module_name = imp.get("module", imp["name"])
                self.import_map[module_name].append(str(py_file))

        return {
            "directory": str(directory),
            "files": results,
            "summary": {
                "files_analyzed": len(results),
                "total_imports": sum(len(f["imports"]) for f in results),
                "total_unused": sum(len(f["unused"]) for f in results),
                "mlx_files": len([f for f in results if f.get("mlx_related")]),
                "metal_files": len([f for f in results if f.get("metal_related")]),
            },
        }

    def detect_circular_imports(self, results: dict) -> list[dict]:
        """Detect potential circular import issues."""
        circular = []

        for file_result in results["files"]:
            file_path = file_result["file"]
            file_dir = Path(file_path).parent.name

            for imp in file_result["imports"]:
                module_name = imp.get("module", imp["name"])

                # Check if this is a relative import within the same directory
                if module_name.startswith(".") or module_name.startswith(file_dir):
                    # Check if the imported module also imports back
                    imported_files = self.import_map.get(module_name, [])
                    for imported_file in imported_files:
                        if file_path != imported_file:
                            # Check if imported file imports back to this file's module
                            this_module = Path(file_path).stem
                            if any(
                                this_module in imp_file for imp_file in self.import_map
                            ):
                                circular.append(
                                    {
                                        "file1": file_path,
                                        "file2": imported_file,
                                        "module": module_name,
                                    }
                                )

        return circular

    def generate_report(self, einstein_results: dict, bolt_results: dict) -> str:
        """Generate comprehensive import audit report."""
        report = []
        report.append("# DEPENDENCY USAGE AUDIT REPORT")
        report.append("=" * 50)
        report.append("")

        # Summary
        report.append("## SUMMARY")
        report.append(
            f"Einstein directory: {einstein_results['summary']['files_analyzed']} files, "
            f"{einstein_results['summary']['total_imports']} imports, "
            f"{einstein_results['summary']['total_unused']} unused"
        )
        report.append(
            f"Bolt directory: {bolt_results['summary']['files_analyzed']} files, "
            f"{bolt_results['summary']['total_imports']} imports, "
            f"{bolt_results['summary']['total_unused']} unused"
        )
        report.append("")

        # MLX/Metal Usage Analysis
        report.append("## MLX/METAL USAGE ANALYSIS")
        report.append("")

        if self.mlx_usage:
            report.append("### MLX Usage:")
            for usage in self.mlx_usage:
                report.append(f"File: {usage['file']}")
                if usage["imports"]:
                    report.append(
                        "  Imports: "
                        + ", ".join(imp["name"] for imp in usage["imports"])
                    )
                if usage["usage_lines"]:
                    report.append("  Usage lines:")
                    for line_info in usage["usage_lines"][:3]:  # Limit to first 3
                        report.append(
                            f"    Line {line_info['line']}: {line_info['content']}"
                        )
                report.append("")
        else:
            report.append("No MLX usage found")
            report.append("")

        if self.metal_usage:
            report.append("### Metal Usage:")
            for usage in self.metal_usage:
                report.append(f"File: {usage['file']}")
                if usage["imports"]:
                    report.append(
                        "  Imports: "
                        + ", ".join(imp["name"] for imp in usage["imports"])
                    )
                if usage["usage_lines"]:
                    report.append("  Usage lines:")
                    for line_info in usage["usage_lines"][:3]:  # Limit to first 3
                        report.append(
                            f"    Line {line_info['line']}: {line_info['content']}"
                        )
                report.append("")
        else:
            report.append("No Metal usage found")
            report.append("")

        # Unused Imports
        report.append("## UNUSED IMPORTS CLEANUP RECOMMENDATIONS")
        report.append("")

        for results in [einstein_results, bolt_results]:
            dir_name = Path(results["directory"]).name
            report.append(f"### {dir_name.upper()} Directory:")

            for file_result in results["files"]:
                if file_result.get("unused"):
                    report.append(f"File: {file_result['file']}")
                    for unused in file_result["unused"]:
                        line_num = unused["line"]
                        if unused["type"] == "import":
                            report.append(
                                f"  Line {line_num}: import {unused['name']} - REMOVE"
                            )
                        else:
                            module = unused.get("module", "")
                            report.append(
                                f"  Line {line_num}: from {module} import {unused['name']} - REMOVE"
                            )
                    report.append("")
            report.append("")

        # Cross-module Dependencies
        report.append("## CROSS-MODULE DEPENDENCIES")
        report.append("")

        # Check for Einstein->Bolt and Bolt->Einstein dependencies
        einstein_to_bolt = []
        bolt_to_einstein = []

        for file_result in einstein_results["files"]:
            for imp in file_result["imports"]:
                module_name = imp.get("module", imp["name"])
                if "bolt" in module_name.lower():
                    einstein_to_bolt.append(
                        {
                            "file": file_result["file"],
                            "import": module_name,
                            "line": imp["line"],
                        }
                    )

        for file_result in bolt_results["files"]:
            for imp in file_result["imports"]:
                module_name = imp.get("module", imp["name"])
                if "einstein" in module_name.lower():
                    bolt_to_einstein.append(
                        {
                            "file": file_result["file"],
                            "import": module_name,
                            "line": imp["line"],
                        }
                    )

        if einstein_to_bolt:
            report.append("### Einstein -> Bolt Dependencies:")
            for dep in einstein_to_bolt:
                report.append(f"  {dep['file']} line {dep['line']}: {dep['import']}")
            report.append("")

        if bolt_to_einstein:
            report.append("### Bolt -> Einstein Dependencies:")
            for dep in bolt_to_einstein:
                report.append(f"  {dep['file']} line {dep['line']}: {dep['import']}")
            report.append("")

        if not einstein_to_bolt and not bolt_to_einstein:
            report.append("✅ No cross-module dependencies found - good separation")
            report.append("")

        # Accelerated Tools Usage
        report.append("## ACCELERATED TOOLS USAGE")
        report.append("")

        accelerated_usage = []
        for results in [einstein_results, bolt_results]:
            for file_result in results["files"]:
                for imp in file_result["imports"]:
                    module_name = imp.get("module", imp["name"])
                    if "accelerated_tools" in module_name:
                        accelerated_usage.append(
                            {
                                "file": file_result["file"],
                                "import": module_name,
                                "name": imp["name"],
                                "line": imp["line"],
                            }
                        )

        if accelerated_usage:
            report.append("### Accelerated Tools Usage:")
            for usage in accelerated_usage:
                report.append(
                    f"  {usage['file']} line {usage['line']}: from {usage['import']} import {usage['name']}"
                )
            report.append("")
        else:
            report.append("No accelerated tools usage found")
            report.append("")

        return "\n".join(report)


def main():
    """Run the import audit analysis."""
    analyzer = ImportUsageAnalyzer()

    # Analyze Einstein directory
    einstein_dir = Path("einstein")
    if einstein_dir.exists():
        print("Analyzing Einstein directory...")
        einstein_results = analyzer.analyze_directory(einstein_dir)
    else:
        print("Einstein directory not found")
        einstein_results = {"directory": "einstein", "files": [], "summary": {}}

    # Analyze Bolt directory
    bolt_dir = Path("bolt")
    if bolt_dir.exists():
        print("Analyzing Bolt directory...")
        bolt_results = analyzer.analyze_directory(bolt_dir)
    else:
        print("Bolt directory not found")
        bolt_results = {"directory": "bolt", "files": [], "summary": {}}

    # Generate and print report
    report = analyzer.generate_report(einstein_results, bolt_results)
    print("\n" + report)

    # Save report to file
    with open("import_audit_report.txt", "w") as f:
        f.write(report)

    print("\n📝 Report saved to import_audit_report.txt")
    print(f"📊 Total files analyzed: {analyzer.files_analyzed}")
    print(f"📈 Total imports found: {analyzer.total_imports}")
    print(f"🗑️  Total unused imports: {len(analyzer.unused_imports)}")


if __name__ == "__main__":
    main()
