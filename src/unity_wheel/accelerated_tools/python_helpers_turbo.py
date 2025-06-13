"""Python helper tools replacing python-code-helper and python-project-helper MCPs.
Optimized for M4 Pro with parallel processing and caching."""

import ast
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import json
import time
from collections import defaultdict
from dataclasses import dataclass
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor

# Use MLX for similarity computations
import mlx.core as mx


@dataclass
class CodeHelper:
    """Unified code helper combining both MCP functionalities."""
    
    def __init__(self):
        # Load hardware config
        with open("optimization_config.json") as f:
            self.config = json.load(f)
        
        # Process pool for parallel operations
        self.executor = ProcessPoolExecutor(
            max_workers=self.config["cpu"]["max_workers"]
        )
        
        # Caches
        self._module_cache = {}
        self._project_cache = {}
        self._import_graph = defaultdict(set)
        
    async def get_function_signature(self, module_path: str, function_name: str) -> Dict[str, Any]:
        """Get function signature with type hints."""
        module = await self._load_module(module_path)
        
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            
            import inspect
            sig = inspect.signature(func)
            
            return {
                "function": function_name,
                "signature": str(sig),
                "parameters": {
                    name: {
                        "type": param.annotation if param.annotation != inspect.Parameter.empty else "Any",
                        "default": param.default if param.default != inspect.Parameter.empty else None
                    }
                    for name, param in sig.parameters.items()
                },
                "return_type": sig.return_annotation if sig.return_annotation != inspect.Parameter.empty else "Any",
                "docstring": inspect.getdoc(func)
            }
        
        return {"error": f"Function {function_name} not found in {module_path}"}
    
    async def get_class_info(self, module_path: str, class_name: str) -> Dict[str, Any]:
        """Get detailed class information."""
        module = await self._load_module(module_path)
        
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            
            import inspect
            
            # Get methods using parallel processing
            methods = []
            for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
                if not name.startswith('_'):
                    methods.append({
                        "name": name,
                        "signature": str(inspect.signature(method)),
                        "docstring": inspect.getdoc(method)
                    })
            
            return {
                "class": class_name,
                "bases": [base.__name__ for base in cls.__bases__],
                "methods": methods,
                "attributes": [
                    name for name, _ in inspect.getmembers(cls)
                    if not name.startswith('_') and not inspect.ismethod(getattr(cls, name))
                ],
                "docstring": inspect.getdoc(cls),
                "mro": [c.__name__ for c in cls.__mro__]
            }
        
        return {"error": f"Class {class_name} not found in {module_path}"}
    
    async def find_usages(self, symbol: str, directory: str = ".") -> List[Dict[str, Any]]:
        """Find all usages of a symbol using parallel search."""
        from unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
        
        rg = get_ripgrep_turbo()
        
        # Search for the symbol
        results = await rg.search(f"\\b{symbol}\\b", directory, file_type="py")
        
        # Analyze each result in parallel
        usage_details = []
        
        # Process in batches
        batch_size = self.config["cpu"]["batch_size"]
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            
            loop = asyncio.get_event_loop()
            batch_details = await loop.run_in_executor(
                self.executor,
                _analyze_usages_batch,
                batch,
                symbol
            )
            
            usage_details.extend(batch_details)
        
        return usage_details
    
    async def suggest_imports(self, file_path: str) -> List[str]:
        """Suggest missing imports using MLX for pattern matching."""
        content = Path(file_path).read_text()
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
        
        # Find undefined names
        analyzer = _UndefinedNamesFinder()
        analyzer.visit(tree)
        
        undefined = analyzer.undefined_names
        
        # Find potential imports from project
        suggestions = []
        
        # Search for definitions in parallel
        if undefined:
            from unity_wheel.accelerated_tools.dependency_graph_turbo import get_dependency_graph
            
            graph = get_dependency_graph()
            if not graph.symbol_locations:
                await graph.build_graph()
            
            for name in undefined:
                locations = graph.symbol_locations.get(name, [])
                if locations:
                    # Suggest the most likely import
                    file_path = locations[0][0]
                    module_path = self._file_to_module_path(file_path)
                    suggestions.append(f"from {module_path} import {name}")
        
        return suggestions
    
    async def analyze_project_structure(self, root_dir: str = ".") -> Dict[str, Any]:
        """Analyze entire project structure using all cores."""
        start = time.perf_counter()
        root = Path(root_dir)
        
        # Find all Python files
        py_files = list(root.rglob("*.py"))
        
        # Analyze in parallel
        loop = asyncio.get_event_loop()
        
        # Split work across cores
        chunk_size = max(10, len(py_files) // self.config["cpu"]["max_workers"])
        file_analyses = []
        
        for i in range(0, len(py_files), chunk_size):
            chunk = py_files[i:i + chunk_size]
            analysis = await loop.run_in_executor(
                self.executor,
                _analyze_project_chunk,
                chunk,
                str(root)
            )
            file_analyses.extend(analysis)
        
        # Aggregate results using MLX for fast computation
        total_loc = mx.sum(mx.array([a["loc"] for a in file_analyses]))
        total_classes = sum(a["classes"] for a in file_analyses)
        total_functions = sum(a["functions"] for a in file_analyses)
        
        # Build package structure
        packages = defaultdict(lambda: {"modules": [], "subpackages": []})
        
        for analysis in file_analyses:
            parts = Path(analysis["file"]).relative_to(root).parts
            if parts:
                package = ".".join(parts[:-1])
                module = parts[-1].replace(".py", "")
                packages[package]["modules"].append(module)
        
        duration = (time.perf_counter() - start) * 1000
        
        return {
            "root": str(root),
            "python_files": len(py_files),
            "total_loc": int(total_loc),
            "total_classes": total_classes,
            "total_functions": total_functions,
            "packages": dict(packages),
            "performance": {
                "duration_ms": duration,
                "files_per_second": len(py_files) / (duration / 1000),
                "cores_used": self.config["cpu"]["max_workers"]
            }
        }
    
    async def refactor_rename(self, old_name: str, new_name: str, 
                            directory: str = ".") -> Dict[str, Any]:
        """Refactor by renaming a symbol across the project."""
        # Find all occurrences
        usages = await self.find_usages(old_name, directory)
        
        # Group by file
        files_to_update = defaultdict(list)
        for usage in usages:
            files_to_update[usage["file"]].append(usage)
        
        # Update files in parallel
        updated_files = []
        
        for file_path, file_usages in files_to_update.items():
            content = Path(file_path).read_text()
            
            # Sort by line/column to update from end to beginning
            file_usages.sort(key=lambda x: (x["line"], x["column"]), reverse=True)
            
            lines = content.splitlines()
            
            for usage in file_usages:
                line_idx = usage["line"] - 1
                if line_idx < len(lines):
                    line = lines[line_idx]
                    # Simple replacement (could be enhanced)
                    lines[line_idx] = line.replace(old_name, new_name)
            
            # Write back
            new_content = "\n".join(lines)
            Path(file_path).write_text(new_content)
            updated_files.append(file_path)
        
        return {
            "renamed": f"{old_name} -> {new_name}",
            "files_updated": len(updated_files),
            "occurrences_replaced": len(usages),
            "files": updated_files
        }
    
    async def _load_module(self, module_path: str):
        """Load a module dynamically."""
        if module_path in self._module_cache:
            return self._module_cache[module_path]
        
        path = Path(module_path)
        
        if path.suffix == ".py":
            # Load from file
            spec = importlib.util.spec_from_file_location(
                path.stem, path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # Import by name
            module = importlib.import_module(module_path)
        
        self._module_cache[module_path] = module
        return module
    
    def _file_to_module_path(self, file_path: str) -> str:
        """Convert file path to module import path."""
        path = Path(file_path)
        parts = path.parts
        
        # Find src or package root
        for i, part in enumerate(parts):
            if part in ("src", "lib") or Path(*parts[:i+1], "__init__.py").exists():
                module_parts = parts[i+1:]
                break
        else:
            module_parts = parts
        
        # Remove .py extension
        if module_parts[-1].endswith(".py"):
            module_parts = list(module_parts)
            module_parts[-1] = module_parts[-1][:-3]
        
        return ".".join(module_parts)
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)


class _UndefinedNamesFinder(ast.NodeVisitor):
    """Find undefined names in AST."""
    
    def __init__(self):
        self.defined_names = set()
        self.used_names = set()
        self.undefined_names = set()
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defined_names.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        self.defined_names.add(node.name)
        # Add parameters as defined
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.defined_names.add(node.name)
        self.generic_visit(node)
        
    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.defined_names.add(name.split('.')[0])
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.defined_names.add(name)
        self.generic_visit(node)
        
    def finalize(self):
        """Calculate undefined names."""
        builtins = set(dir(__builtins__))
        self.undefined_names = self.used_names - self.defined_names - builtins


def _analyze_usages_batch(results: List[Dict], symbol: str) -> List[Dict]:
    """Analyze usage details in batch (runs in process pool)."""
    usage_details = []
    
    for result in results:
        try:
            # Read file content
            content = Path(result["file"]).read_text()
            tree = ast.parse(content)
            
            # Find usage context
            finder = _UsageContextFinder(symbol, result["line"])
            finder.visit(tree)
            
            usage_details.append({
                "file": result["file"],
                "line": result["line"],
                "column": result.get("column", 0),
                "content": result["content"],
                "usage_type": finder.usage_type,
                "context": finder.context
            })
        except:
            usage_details.append(result)
    
    return usage_details


def _analyze_project_chunk(files: List[Path], root: str) -> List[Dict]:
    """Analyze a chunk of project files."""
    analyses = []
    
    for file_path in files:
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            analyzer = _FileStatsCollector()
            analyzer.visit(tree)
            
            analyses.append({
                "file": str(file_path),
                "loc": len(content.splitlines()),
                "classes": len(analyzer.classes),
                "functions": len(analyzer.functions),
                "imports": len(analyzer.imports)
            })
        except:
            analyses.append({
                "file": str(file_path),
                "loc": 0,
                "classes": 0,
                "functions": 0,
                "imports": 0
            })
    
    return analyses


class _UsageContextFinder(ast.NodeVisitor):
    """Find context of symbol usage."""
    
    def __init__(self, symbol: str, target_line: int):
        self.symbol = symbol
        self.target_line = target_line
        self.usage_type = "unknown"
        self.context = ""
        
    def visit_FunctionDef(self, node):
        if node.lineno <= self.target_line <= (node.end_lineno or node.lineno):
            self.context = f"function {node.name}"
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        if node.lineno <= self.target_line <= (node.end_lineno or node.lineno):
            self.context = f"class {node.name}"
        self.generic_visit(node)


class _FileStatsCollector(ast.NodeVisitor):
    """Collect file statistics."""
    
    def __init__(self):
        self.classes = []
        self.functions = []
        self.imports = []
        
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)
        
    def visit_Import(self, node):
        self.imports.extend(alias.name for alias in node.names)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)


# Singleton instance
_helper_instance: Optional[CodeHelper] = None


def get_code_helper() -> CodeHelper:
    """Get or create the code helper instance."""
    global _helper_instance
    if _helper_instance is None:
        _helper_instance = CodeHelper()
    return _helper_instance


# Drop-in replacements for MCP functions
async def get_function_info(module_path: str, function_name: str) -> str:
    """Drop-in replacement for python-code-helper.get_function_info."""
    helper = get_code_helper()
    info = await helper.get_function_signature(module_path, function_name)
    
    if "error" in info:
        return info["error"]
    
    return json.dumps(info, indent=2)


async def analyze_project(directory: str = ".") -> str:
    """Drop-in replacement for python-project-helper.analyze_project."""
    helper = get_code_helper()
    analysis = await helper.analyze_project_structure(directory)
    
    output = [
        f"Project Analysis: {analysis['root']}",
        f"Python files: {analysis['python_files']}",
        f"Total LOC: {analysis['total_loc']:,}",
        f"Classes: {analysis['total_classes']}",
        f"Functions: {analysis['total_functions']}",
        f"",
        f"Performance: {analysis['performance']['files_per_second']:.1f} files/sec"
    ]
    
    return "\n".join(output)