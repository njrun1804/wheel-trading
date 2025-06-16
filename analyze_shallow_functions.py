#!/usr/bin/env python3
"""
Analyze Einstein and Bolt directories for shallow function implementations.
"""

import ast
import os
from typing import Any


class ShallowFunctionAnalyzer(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.shallow_functions = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze function definitions for shallow implementations."""
        self.analyze_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Analyze async function definitions for shallow implementations."""
        self.analyze_function(node)
        self.generic_visit(node)

    def analyze_function(self, node: ast.FunctionDef) -> None:
        """Analyze a function for shallow implementation patterns."""
        # Get source lines
        source_lines = self.get_source_lines(node)

        # Count non-comment lines
        non_comment_lines = self.count_non_comment_lines(source_lines)

        # Check for simple return patterns
        is_constant_return = self.is_constant_return(node)
        is_pass_only = self.is_pass_only(node)
        is_raise_only = self.is_raise_only(node)
        is_simple_getter = self.is_simple_getter(node)

        # Determine if function is shallow
        is_shallow = non_comment_lines < 3 or is_constant_return or is_pass_only

        if is_shallow:
            justification = self.get_justification(
                node,
                non_comment_lines,
                is_constant_return,
                is_pass_only,
                is_raise_only,
                is_simple_getter,
            )

            self.shallow_functions.append(
                {
                    "function": node.name,
                    "line": node.lineno,
                    "non_comment_lines": non_comment_lines,
                    "is_constant_return": is_constant_return,
                    "is_pass_only": is_pass_only,
                    "is_raise_only": is_raise_only,
                    "is_simple_getter": is_simple_getter,
                    "justification": justification,
                    "source_lines": source_lines,
                }
            )

    def get_source_lines(self, node: ast.FunctionDef) -> list[str]:
        """Get source lines for a function."""
        try:
            with open(self.filename) as f:
                lines = f.readlines()

            # Get function body lines
            start_line = node.lineno - 1
            end_line = (
                node.end_lineno if hasattr(node, "end_lineno") else start_line + 10
            )

            return lines[start_line:end_line]
        except:
            return []

    def count_non_comment_lines(self, lines: list[str]) -> int:
        """Count non-comment, non-empty lines."""
        count = 0
        for line in lines:
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith('"""')
                and not stripped.startswith("'''")
            ):
                # Skip docstrings
                if not (stripped.startswith('"""') or stripped.startswith("'''")):
                    count += 1
        return count

    def is_constant_return(self, node: ast.FunctionDef) -> bool:
        """Check if function only returns a constant."""
        if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
            return_value = node.body[0].value
            if return_value is None:
                return True
            if isinstance(
                return_value, ast.Constant | ast.Num | ast.Str | ast.NameConstant
            ):
                return True
            if isinstance(return_value, ast.Name) and return_value.id in [
                "True",
                "False",
                "None",
            ]:
                return True
        return False

    def is_pass_only(self, node: ast.FunctionDef) -> bool:
        """Check if function only contains pass statement."""
        return len(node.body) == 1 and isinstance(node.body[0], ast.Pass)

    def is_raise_only(self, node: ast.FunctionDef) -> bool:
        """Check if function only raises an exception."""
        return len(node.body) == 1 and isinstance(node.body[0], ast.Raise)

    def is_simple_getter(self, node: ast.FunctionDef) -> bool:
        """Check if function is a simple getter (returns self.attribute)."""
        if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
            return_value = node.body[0].value
            if isinstance(return_value, ast.Attribute) and isinstance(
                return_value.value, ast.Name
            ):
                return return_value.value.id == "self"
        return False

    def get_justification(
        self,
        node: ast.FunctionDef,
        non_comment_lines: int,
        is_constant_return: bool,
        is_pass_only: bool,
        is_raise_only: bool,
        is_simple_getter: bool,
    ) -> str:
        """Determine justification for shallow implementation."""

        # Interface methods (abstract methods)
        if is_pass_only:
            return "JUSTIFIED - Abstract/interface method placeholder"

        if is_raise_only:
            return "JUSTIFIED - Method that raises NotImplementedError"

        # Simple getters
        if is_simple_getter:
            return "JUSTIFIED - Simple getter method"

        # Property methods
        if node.name.startswith("_") or any(
            d.id in ["property", "cached_property"]
            for d in node.decorator_list
            if isinstance(d, ast.Name)
        ):
            return "JUSTIFIED - Property or private method"

        # Constant returns might be justified
        if is_constant_return:
            if node.name in ["__str__", "__repr__", "__bool__", "__len__"]:
                return "JUSTIFIED - Magic method with constant return"
            else:
                return "NEEDS_WORK - Constant return may indicate incomplete implementation"

        # Very short functions
        if non_comment_lines < 3:
            return "NEEDS_WORK - Very short function may need more implementation"

        return "JUSTIFIED - Function appears complete"


def analyze_directory(directory: str) -> dict[str, list[dict[str, Any]]]:
    """Analyze all Python files in a directory."""
    results = {}

    for root, _dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath) as f:
                        content = f.read()

                    tree = ast.parse(content)
                    analyzer = ShallowFunctionAnalyzer(filepath)
                    analyzer.visit(tree)

                    if analyzer.shallow_functions:
                        results[filepath] = analyzer.shallow_functions

                except Exception as e:
                    print(f"Error analyzing {filepath}: {e}")

    return results


def main():
    """Main analysis function."""
    print("=== SHALLOW IMPLEMENTATION SCAN ===\n")

    # Analyze Einstein directory
    einstein_dir = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/einstein"
    bolt_dir = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt"

    print("🔍 Analyzing Einstein directory...")
    einstein_results = analyze_directory(einstein_dir)

    print("🔍 Analyzing Bolt directory...")
    bolt_results = analyze_directory(bolt_dir)

    # Combine results
    all_results = {**einstein_results, **bolt_results}

    # Report findings
    print("\n📊 ANALYSIS RESULTS")
    print(f"Files analyzed: {len(all_results)}")

    needs_work_count = 0
    justified_count = 0

    for filepath, functions in all_results.items():
        rel_path = filepath.replace(
            "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/",
            "",
        )

        print(f"\n📁 {rel_path}")
        print("-" * 80)

        for func in functions:
            status = "⚠️ " if "NEEDS_WORK" in func["justification"] else "✅ "

            print(f"{status}Function: {func['function']}")
            print(f"   Line: {func['line']}")
            print(f"   Non-comment lines: {func['non_comment_lines']}")
            print(f"   Justification: {func['justification']}")

            if func["is_constant_return"]:
                print("   ⚠️  Returns constant value")
            if func["is_pass_only"]:
                print("   📝 Contains only 'pass' statement")
            if func["is_raise_only"]:
                print("   🚫 Contains only 'raise' statement")
            if func["is_simple_getter"]:
                print("   📖 Simple getter method")

            print()

            if "NEEDS_WORK" in func["justification"]:
                needs_work_count += 1
            else:
                justified_count += 1

    print("\n📈 SUMMARY")
    print(f"Total shallow functions found: {needs_work_count + justified_count}")
    print(f"Functions needing work: {needs_work_count}")
    print(f"Justified shallow functions: {justified_count}")

    if needs_work_count > 0:
        print("\n🔧 RECOMMENDATIONS")
        print("Focus on functions marked with 'NEEDS_WORK' - these may require:")
        print("- Additional implementation logic")
        print("- Proper error handling")
        print("- Input validation")
        print("- Business logic completion")


if __name__ == "__main__":
    main()
