"""
Fallback code helper implementation using Python introspection.

Provides real code analysis and helper functions when accelerated tools are not available.
"""

import ast
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CodeHelperFallback:
    """Fallback code helper using Python introspection and AST parsing."""

    def __init__(self):
        self.function_cache = {}
        self.signature_cache = {}
        self.source_cache = {}

    async def get_function_signature(
        self, module_path: str, function_name: str
    ) -> dict[str, Any]:
        """Get function signature from module."""
        cache_key = f"{module_path}:{function_name}"

        if cache_key in self.signature_cache:
            return self.signature_cache[cache_key]

        try:
            # Try to get from loaded modules first
            for name, module in sys.modules.items():
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    if callable(func):
                        signature_info = await self._analyze_function(
                            func, function_name
                        )
                        self.signature_cache[cache_key] = signature_info
                        return signature_info

            # Try to parse from file
            if Path(module_path).exists():
                signature_info = await self._get_signature_from_file(
                    module_path, function_name
                )
                if signature_info:
                    self.signature_cache[cache_key] = signature_info
                    return signature_info

            return {"error": f"Function {function_name} not found in {module_path}"}

        except Exception as e:
            logger.debug(f"Failed to get function signature: {e}")
            return {"error": str(e)}

    async def _analyze_function(self, func, function_name: str) -> dict[str, Any]:
        """Analyze a function object."""
        try:
            sig = inspect.signature(func)

            parameters = []
            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "kind": param.kind.name,
                    "default": str(param.default)
                    if param.default != inspect.Parameter.empty
                    else None,
                    "annotation": str(param.annotation)
                    if param.annotation != inspect.Parameter.empty
                    else None,
                }
                parameters.append(param_info)

            # Get source info if available
            source_file = None
            source_lines = None
            line_number = None

            try:
                source_file = inspect.getfile(func)
                source_lines, line_number = inspect.getsourcelines(func)
            except (OSError, TypeError):
                pass

            return {
                "name": function_name,
                "signature": str(sig),
                "parameters": parameters,
                "return_annotation": str(sig.return_annotation)
                if sig.return_annotation != inspect.Signature.empty
                else None,
                "docstring": inspect.getdoc(func),
                "source_file": source_file,
                "line_number": line_number,
                "is_async": inspect.iscoroutinefunction(func),
                "is_method": inspect.ismethod(func),
                "source_available": source_lines is not None,
            }

        except Exception as e:
            logger.debug(f"Failed to analyze function {function_name}: {e}")
            return {"error": str(e)}

    async def _get_signature_from_file(
        self, file_path: str, function_name: str
    ) -> dict[str, Any] | None:
        """Get function signature by parsing the source file."""
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            tree = ast.parse(content, filename=file_path)

            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == function_name
                ):
                    return await self._analyze_ast_function(node, file_path)

            return None

        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return None

    async def _analyze_ast_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str
    ) -> dict[str, Any]:
        """Analyze function from AST node."""
        parameters = []

        # Process arguments
        args = node.args

        # Regular arguments
        for i, arg in enumerate(args.args):
            param_info = {
                "name": arg.arg,
                "kind": "POSITIONAL_OR_KEYWORD",
                "annotation": ast.unparse(arg.annotation)
                if arg.annotation and hasattr(ast, "unparse")
                else None,
                "default": None,
            }

            # Check for default values
            default_index = i - (len(args.args) - len(args.defaults))
            if default_index >= 0 and default_index < len(args.defaults):
                if hasattr(ast, "unparse"):
                    param_info["default"] = ast.unparse(args.defaults[default_index])
                else:
                    param_info["default"] = "default_value"

            parameters.append(param_info)

        # *args parameter
        if args.vararg:
            parameters.append(
                {
                    "name": args.vararg.arg,
                    "kind": "VAR_POSITIONAL",
                    "annotation": ast.unparse(args.vararg.annotation)
                    if args.vararg.annotation and hasattr(ast, "unparse")
                    else None,
                    "default": None,
                }
            )

        # **kwargs parameter
        if args.kwarg:
            parameters.append(
                {
                    "name": args.kwarg.arg,
                    "kind": "VAR_KEYWORD",
                    "annotation": ast.unparse(args.kwarg.annotation)
                    if args.kwarg.annotation and hasattr(ast, "unparse")
                    else None,
                    "default": None,
                }
            )

        # Keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            param_info = {
                "name": arg.arg,
                "kind": "KEYWORD_ONLY",
                "annotation": ast.unparse(arg.annotation)
                if arg.annotation and hasattr(ast, "unparse")
                else None,
                "default": None,
            }

            # Check for default values in kw_defaults
            if i < len(args.kw_defaults) and args.kw_defaults[i]:
                if hasattr(ast, "unparse"):
                    param_info["default"] = ast.unparse(args.kw_defaults[i])
                else:
                    param_info["default"] = "default_value"

            parameters.append(param_info)

        return {
            "name": node.name,
            "parameters": parameters,
            "return_annotation": ast.unparse(node.returns)
            if node.returns and hasattr(ast, "unparse")
            else None,
            "docstring": ast.get_docstring(node),
            "source_file": file_path,
            "line_number": node.lineno,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "decorators": [
                ast.unparse(dec) if hasattr(ast, "unparse") else "decorator"
                for dec in node.decorator_list
            ],
        }

    async def find_function_calls(
        self, directory_path: str, function_name: str
    ) -> list[dict[str, Any]]:
        """Find all calls to a specific function in the codebase."""
        calls = []
        dir_path = Path(directory_path)

        for file_path in dir_path.rglob("*.py"):
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content, filename=str(file_path))

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # Handle different call patterns
                        call_name = self._extract_call_name(node)
                        if function_name in call_name:
                            calls.append(
                                {
                                    "file": str(file_path),
                                    "line": node.lineno,
                                    "column": node.col_offset,
                                    "call_name": call_name,
                                    "args_count": len(node.args),
                                    "kwargs_count": len(node.keywords),
                                }
                            )

            except Exception as e:
                logger.debug(f"Failed to analyze {file_path}: {e}")

        return calls

    def _extract_call_name(self, call_node: ast.Call) -> str:
        """Extract the name from a function call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            # Handle method calls like obj.method()
            if hasattr(ast, "unparse"):
                return ast.unparse(call_node.func)
            else:
                return call_node.func.attr
        else:
            return "unknown_call"

    async def analyze_imports(self, file_path: str) -> dict[str, Any]:
        """Analyze imports in a Python file."""
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            tree = ast.parse(content, filename=file_path)

            imports = {
                "standard_library": [],
                "third_party": [],
                "local": [],
                "from_imports": {},
                "import_aliases": {},
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_type = self._classify_import(alias.name)
                        import_info = {
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        }
                        imports[module_type].append(import_info)

                        if alias.asname:
                            imports["import_aliases"][alias.asname] = alias.name

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_type = self._classify_import(node.module)
                        if node.module not in imports["from_imports"]:
                            imports["from_imports"][node.module] = []

                        for alias in node.names:
                            import_info = {
                                "name": alias.name,
                                "alias": alias.asname,
                                "line": node.lineno,
                            }
                            imports["from_imports"][node.module].append(import_info)

                            if alias.asname:
                                imports["import_aliases"][
                                    alias.asname
                                ] = f"{node.module}.{alias.name}"

            return {
                "file": file_path,
                "imports": imports,
                "total_imports": len(imports["standard_library"])
                + len(imports["third_party"])
                + len(imports["local"]),
                "from_imports_count": len(imports["from_imports"]),
            }

        except Exception as e:
            logger.debug(f"Failed to analyze imports in {file_path}: {e}")
            return {"error": str(e)}

    def _classify_import(self, module_name: str) -> str:
        """Classify import as standard library, third party, or local."""
        # Simple heuristic classification
        if module_name.startswith(".") or "." not in module_name:
            return "local"

        # Common standard library modules
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "datetime",
            "time",
            "math",
            "random",
            "itertools",
            "collections",
            "functools",
            "operator",
            "pathlib",
            "urllib",
            "http",
            "logging",
            "threading",
            "asyncio",
            "concurrent",
            "multiprocessing",
            "sqlite3",
            "csv",
            "xml",
            "html",
            "email",
            "unittest",
            "typing",
        }

        root_module = module_name.split(".")[0]
        if root_module in stdlib_modules:
            return "standard_library"
        else:
            return "third_party"

    async def get_class_hierarchy(
        self, module_path: str, class_name: str
    ) -> dict[str, Any]:
        """Get class hierarchy information."""
        try:
            # Try to import and inspect
            for name, module in sys.modules.items():
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    if inspect.isclass(cls):
                        return await self._analyze_class_hierarchy(cls, class_name)

            # Try to parse from file
            if Path(module_path).exists():
                return await self._get_class_hierarchy_from_file(
                    module_path, class_name
                )

            return {"error": f"Class {class_name} not found"}

        except Exception as e:
            logger.debug(f"Failed to get class hierarchy: {e}")
            return {"error": str(e)}

    async def _analyze_class_hierarchy(self, cls, class_name: str) -> dict[str, Any]:
        """Analyze class hierarchy from class object."""
        try:
            mro = [c.__name__ for c in cls.__mro__]
            bases = [base.__name__ for base in cls.__bases__]

            methods = []
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                if not name.startswith("__"):
                    methods.append(
                        {
                            "name": name,
                            "signature": str(inspect.signature(method)),
                            "docstring": inspect.getdoc(method),
                        }
                    )

            return {
                "name": class_name,
                "mro": mro,
                "bases": bases,
                "methods": methods,
                "docstring": inspect.getdoc(cls),
                "module": cls.__module__,
            }

        except Exception as e:
            logger.debug(f"Failed to analyze class hierarchy: {e}")
            return {"error": str(e)}

    async def _get_class_hierarchy_from_file(
        self, file_path: str, class_name: str
    ) -> dict[str, Any]:
        """Get class hierarchy by parsing source file."""
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            tree = ast.parse(content, filename=file_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    bases = []
                    if hasattr(ast, "unparse"):
                        bases = [ast.unparse(base) for base in node.bases]
                    else:
                        bases = ["base_class" for _ in node.bases]

                    methods = []
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.append(
                                {
                                    "name": item.name,
                                    "line": item.lineno,
                                    "is_async": isinstance(item, ast.AsyncFunctionDef),
                                    "docstring": ast.get_docstring(item),
                                }
                            )

                    return {
                        "name": class_name,
                        "bases": bases,
                        "methods": methods,
                        "docstring": ast.get_docstring(node),
                        "line": node.lineno,
                        "source_file": file_path,
                    }

            return {"error": f"Class {class_name} not found in {file_path}"}

        except Exception as e:
            logger.debug(f"Failed to parse class from {file_path}: {e}")
            return {"error": str(e)}

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "backend": "introspection_code_helper",
            "function_signatures_cached": len(self.signature_cache),
            "source_files_cached": len(self.source_cache),
            "introspection_available": True,
            "ast_parsing_available": True,
            "parallel_capable": True,
        }
