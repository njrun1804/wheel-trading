"""
Fallback Python analyzer implementation using ast module.

Provides real Python code analysis when accelerated tools are not available.
"""

import ast
import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PythonAnalyzerFallback:
    """Fallback Python analyzer using ast module and introspection."""

    def __init__(self):
        self.analysis_cache = {}
        self.module_cache = {}

    async def analyze_directory(self, directory_path: str) -> dict[str, Any]:
        """Analyze all Python files in a directory."""
        dir_path = Path(directory_path)
        python_files = list(dir_path.rglob("*.py"))

        # Analyze files in parallel
        tasks = [self._analyze_file(file_path) for file_path in python_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        total_lines = 0
        total_functions = 0
        total_classes = 0
        total_imports = 0
        files_analyzed = 0
        errors = 0

        for file_path, result in zip(python_files, results, strict=False):
            if isinstance(result, Exception):
                logger.debug(f"Failed to analyze {file_path}: {result}")
                errors += 1
                continue

            files_analyzed += 1
            total_lines += result.get("lines", 0)
            total_functions += result.get("functions", 0)
            total_classes += result.get("classes", 0)
            total_imports += result.get("imports", 0)

            # Cache individual file results
            self.analysis_cache[str(file_path)] = result

        return {
            "files_analyzed": files_analyzed,
            "errors": errors,
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_imports": total_imports,
            "average_lines_per_file": total_lines / files_analyzed
            if files_analyzed > 0
            else 0,
        }

    async def _analyze_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze a single Python file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))

            analyzer = ASTAnalyzer()
            analyzer.visit(tree)

            return {
                "file": str(file_path),
                "lines": len(content.splitlines()),
                "functions": len(analyzer.functions),
                "classes": len(analyzer.classes),
                "imports": len(analyzer.imports),
                "complexity": analyzer.complexity,
                "function_details": analyzer.functions,
                "class_details": analyzer.classes,
                "import_details": analyzer.imports,
                "docstring_coverage": analyzer.calculate_docstring_coverage(),
            }

        except Exception as e:
            logger.debug(f"Failed to analyze {file_path}: {e}")
            return {"error": str(e)}

    async def get_function_info(
        self, module_name: str, function_name: str
    ) -> dict[str, Any]:
        """Get detailed information about a function."""
        try:
            # Try to import and inspect the function
            module = __import__(module_name, fromlist=[function_name])
            func = getattr(module, function_name, None)

            if func is None:
                return {"error": f"Function {function_name} not found in {module_name}"}

            signature = inspect.signature(func)
            source_lines = inspect.getsourcelines(func)

            return {
                "name": function_name,
                "module": module_name,
                "signature": str(signature),
                "parameters": [
                    {
                        "name": param.name,
                        "kind": param.kind.name,
                        "default": str(param.default)
                        if param.default != inspect.Parameter.empty
                        else None,
                        "annotation": str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else None,
                    }
                    for param in signature.parameters.values()
                ],
                "return_annotation": str(signature.return_annotation)
                if signature.return_annotation != inspect.Signature.empty
                else None,
                "docstring": inspect.getdoc(func),
                "source_file": inspect.getfile(func),
                "line_number": source_lines[1],
                "source_lines": source_lines[0],
            }

        except Exception as e:
            logger.debug(
                f"Failed to get function info for {module_name}.{function_name}: {e}"
            )
            return {"error": str(e)}

    async def get_class_info(self, module_name: str, class_name: str) -> dict[str, Any]:
        """Get detailed information about a class."""
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name, None)

            if cls is None:
                return {"error": f"Class {class_name} not found in {module_name}"}

            methods = []
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                if not name.startswith("_"):  # Skip private methods
                    try:
                        sig = inspect.signature(method)
                        methods.append(
                            {
                                "name": name,
                                "signature": str(sig),
                                "docstring": inspect.getdoc(method),
                            }
                        )
                    except Exception:
                        pass

            return {
                "name": class_name,
                "module": module_name,
                "mro": [c.__name__ for c in cls.__mro__],
                "methods": methods,
                "docstring": inspect.getdoc(cls),
                "source_file": inspect.getfile(cls),
                "bases": [base.__name__ for base in cls.__bases__],
            }

        except Exception as e:
            logger.debug(
                f"Failed to get class info for {module_name}.{class_name}: {e}"
            )
            return {"error": str(e)}

    async def find_usages(
        self, symbol_name: str, directory_path: str = "."
    ) -> list[dict[str, Any]]:
        """Find usages of a symbol in the codebase."""
        usages = []
        dir_path = Path(directory_path)

        for file_path in dir_path.rglob("*.py"):
            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.splitlines()

                for line_num, line in enumerate(lines, 1):
                    if symbol_name in line:
                        # Simple heuristic to avoid false positives
                        if (
                            line.strip().startswith("#")
                            or f'"{symbol_name}"' in line  # Skip comments
                            or f"'{symbol_name}'" in line  # Skip string literals
                        ):
                            continue

                        usages.append(
                            {
                                "file": str(file_path),
                                "line": line_num,
                                "content": line.strip(),
                                "context": self._get_context(lines, line_num - 1),
                            }
                        )

            except Exception as e:
                logger.debug(f"Failed to search {file_path}: {e}")

        return usages

    def _get_context(self, lines: list[str], line_index: int) -> str:
        """Get context around a line."""
        start = max(0, line_index - 2)
        end = min(len(lines), line_index + 3)
        return "\n".join(lines[start:end])

    async def analyze_complexity(self, file_path: str) -> dict[str, Any]:
        """Analyze code complexity of a file."""
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            tree = ast.parse(content, filename=file_path)

            complexity_analyzer = ComplexityAnalyzer()
            complexity_analyzer.visit(tree)

            return {
                "file": file_path,
                "cyclomatic_complexity": complexity_analyzer.complexity,
                "function_complexities": complexity_analyzer.function_complexities,
                "max_complexity": max(
                    complexity_analyzer.function_complexities.values()
                )
                if complexity_analyzer.function_complexities
                else 0,
                "average_complexity": sum(
                    complexity_analyzer.function_complexities.values()
                )
                / len(complexity_analyzer.function_complexities)
                if complexity_analyzer.function_complexities
                else 0,
            }

        except Exception as e:
            logger.debug(f"Failed to analyze complexity of {file_path}: {e}")
            return {"error": str(e)}

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "backend": "ast_analyzer",
            "files_cached": len(self.analysis_cache),
            "modules_cached": len(self.module_cache),
            "introspection_available": True,
            "parallel_capable": True,
        }


class ASTAnalyzer(ast.NodeVisitor):
    """AST node visitor for code analysis."""

    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.complexity = 0

    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        self.functions.append(
            {
                "name": node.name,
                "line": node.lineno,
                "args": [arg.arg for arg in node.args.args],
                "decorators": [
                    ast.unparse(dec) if hasattr(ast, "unparse") else "unknown"
                    for dec in node.decorator_list
                ],
                "docstring": ast.get_docstring(node),
                "is_async": False,
            }
        )
        self.complexity += self._calculate_complexity(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions."""
        self.functions.append(
            {
                "name": node.name,
                "line": node.lineno,
                "args": [arg.arg for arg in node.args.args],
                "decorators": [
                    ast.unparse(dec) if hasattr(ast, "unparse") else "unknown"
                    for dec in node.decorator_list
                ],
                "docstring": ast.get_docstring(node),
                "is_async": True,
            }
        )
        self.complexity += self._calculate_complexity(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit class definitions."""
        self.classes.append(
            {
                "name": node.name,
                "line": node.lineno,
                "bases": [
                    ast.unparse(base) if hasattr(ast, "unparse") else "unknown"
                    for base in node.bases
                ],
                "decorators": [
                    ast.unparse(dec) if hasattr(ast, "unparse") else "unknown"
                    for dec in node.decorator_list
                ],
                "docstring": ast.get_docstring(node),
            }
        )
        self.generic_visit(node)

    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(
                {
                    "module": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                    "type": "import",
                }
            )

    def visit_ImportFrom(self, node):
        """Visit from...import statements."""
        for alias in node.names:
            self.imports.append(
                {
                    "module": node.module,
                    "name": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                    "type": "from_import",
                }
            )

    def _calculate_complexity(self, node):
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)) or isinstance(child, ast.ExceptHandler) or isinstance(child, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    def calculate_docstring_coverage(self) -> float:
        """Calculate percentage of functions and classes with docstrings."""
        total = len(self.functions) + len(self.classes)
        if total == 0:
            return 100.0

        documented = sum(1 for func in self.functions if func["docstring"]) + sum(
            1 for cls in self.classes if cls["docstring"]
        )

        return (documented / total) * 100


class ComplexityAnalyzer(ast.NodeVisitor):
    """Specialized AST visitor for complexity analysis."""

    def __init__(self):
        self.complexity = 0
        self.function_complexities = {}
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Visit function and calculate its complexity."""
        old_function = self.current_function
        self.current_function = node.name

        func_complexity = self._calculate_function_complexity(node)
        self.function_complexities[node.name] = func_complexity
        self.complexity += func_complexity

        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        """Visit async function and calculate its complexity."""
        self.visit_FunctionDef(node)  # Same logic

    def _calculate_function_complexity(self, node):
        """Calculate cyclomatic complexity for a specific function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(
                child,
                (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith),
            ) or isinstance(child, ast.ExceptHandler) or isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.ListComp):
                complexity += 1  # List comprehensions add complexity

        return complexity
