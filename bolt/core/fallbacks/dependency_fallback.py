"""
Fallback dependency graph implementation using AST parsing.

Provides real Python dependency analysis when accelerated tools are not available.
"""

import ast
import asyncio
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DependencyGraphFallback:
    """Fallback dependency graph analyzer using AST parsing."""

    def __init__(self):
        self.dependency_cache = {}
        self.symbol_cache = {}
        self.file_cache = {}
        self.import_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)

    async def build_graph(self, root_path: str = ".") -> dict[str, Any]:
        """Build dependency graph for Python project."""
        root_path = Path(root_path).resolve()

        # Find all Python files
        python_files = await self._find_python_files(root_path)

        # Parse files in parallel
        tasks = [self._parse_file(file_path) for file_path in python_files]
        parse_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build import graph
        for file_path, result in zip(python_files, parse_results, strict=False):
            if isinstance(result, Exception):
                logger.debug(f"Failed to parse {file_path}: {result}")
                continue

            imports, exports = result
            rel_path = str(file_path.relative_to(root_path))

            self.file_cache[rel_path] = {
                "imports": imports,
                "exports": exports,
                "path": str(file_path),
            }

            # Build import relationships
            for imported_module in imports:
                self.import_graph[rel_path].add(imported_module)
                self.reverse_graph[imported_module].add(rel_path)

        return {
            "files": len(python_files),
            "imports": len(self.import_graph),
            "modules": len(self.reverse_graph),
        }

    async def _find_python_files(self, root_path: Path) -> list[Path]:
        """Find all Python files in the project."""
        python_files = []

        for root, dirs, files in os.walk(root_path):
            # Skip common non-source directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in {"__pycache__", "node_modules", "venv", "env", ".venv"}
            ]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

        return python_files

    async def _parse_file(self, file_path: Path) -> tuple[list[str], list[str]]:
        """Parse a Python file to extract imports and exports."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))

            imports = []
            exports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")

                elif isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    exports.append(node.name)

                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            exports.append(target.id)

            return imports, exports

        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return [], []

    async def find_symbol(self, symbol_name: str) -> list[dict[str, Any]]:
        """Find definitions of a symbol across the codebase."""
        if symbol_name in self.symbol_cache:
            return self.symbol_cache[symbol_name]

        results = []

        for file_path, file_info in self.file_cache.items():
            if symbol_name in file_info["exports"]:
                # Parse file again to get exact location
                try:
                    content = Path(file_info["path"]).read_text(encoding="utf-8")
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if hasattr(node, "name") and node.name == symbol_name:
                            results.append(
                                {
                                    "file": file_path,
                                    "line": getattr(node, "lineno", 0),
                                    "column": getattr(node, "col_offset", 0),
                                    "type": type(node).__name__,
                                    "symbol": symbol_name,
                                }
                            )

                except Exception as e:
                    logger.debug(f"Failed to find symbol location in {file_path}: {e}")

        self.symbol_cache[symbol_name] = results
        return results

    async def get_dependencies(self, file_path: str) -> list[str]:
        """Get direct dependencies of a file."""
        rel_path = self._normalize_path(file_path)
        return list(self.import_graph.get(rel_path, set()))

    async def get_dependents(self, file_path: str) -> list[str]:
        """Get files that depend on the given file."""
        rel_path = self._normalize_path(file_path)
        return list(self.reverse_graph.get(rel_path, set()))

    async def detect_cycles(self) -> list[list[str]]:
        """Detect circular dependencies in the graph."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> bool:
            """DFS to detect cycles."""
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.import_graph.get(node, set()):
                if dfs(neighbor, path.copy()):
                    return True

            rec_stack.remove(node)
            return False

        for node in self.import_graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    async def get_import_chain(self, from_file: str, to_file: str) -> list[str] | None:
        """Find import chain from one file to another."""
        from_path = self._normalize_path(from_file)
        to_path = self._normalize_path(to_file)

        if from_path not in self.import_graph:
            return None

        # BFS to find shortest path
        queue = deque([(from_path, [from_path])])
        visited = {from_path}

        while queue:
            current, path = queue.popleft()

            if current == to_path:
                return path

            for neighbor in self.import_graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def _normalize_path(self, file_path: str) -> str:
        """Normalize file path for consistent lookup."""
        return file_path.replace("\\", "/").lstrip("./")

    async def search_code_fuzzy(self, term: str) -> list[dict[str, Any]]:
        """Fuzzy search for code symbols and content."""
        results = []

        # Search in symbol names
        for file_path, file_info in self.file_cache.items():
            for symbol in file_info["exports"]:
                if term.lower() in symbol.lower():
                    symbol_locations = await self.find_symbol(symbol)
                    for location in symbol_locations:
                        results.append(
                            {
                                "file": location["file"],
                                "line": location["line"],
                                "column": location["column"],
                                "symbol": symbol,
                                "type": "symbol_match",
                                "score": self._calculate_fuzzy_score(term, symbol),
                            }
                        )

        # Search in imports
        for file_path, imports in self.import_graph.items():
            for import_name in imports:
                if term.lower() in import_name.lower():
                    results.append(
                        {
                            "file": file_path,
                            "line": 0,
                            "column": 0,
                            "symbol": import_name,
                            "type": "import_match",
                            "score": self._calculate_fuzzy_score(term, import_name),
                        }
                    )

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:100]  # Limit results

    def _calculate_fuzzy_score(self, term: str, candidate: str) -> float:
        """Calculate fuzzy match score."""
        term = term.lower()
        candidate = candidate.lower()

        if term == candidate:
            return 1.0
        elif term in candidate:
            return 0.8 - (len(candidate) - len(term)) * 0.01
        else:
            # Simple substring scoring
            common_chars = set(term) & set(candidate)
            return len(common_chars) / max(len(term), len(candidate))

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "backend": "ast_parser",
            "files_analyzed": len(self.file_cache),
            "symbols_cached": len(self.symbol_cache),
            "import_relationships": len(self.import_graph),
            "reverse_relationships": len(self.reverse_graph),
            "parallel_capable": True,
        }
