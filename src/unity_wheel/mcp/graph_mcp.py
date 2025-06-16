"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


Static-Analysis / Dependency-Graph MCP for structural code understanding.
Provides AST parsing, call graphs, import analysis, and data flow tracking.
"""

import ast
import asyncio
import hashlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb


@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable)."""

    name: str
    type: str  # 'function', 'class', 'variable', 'module'
    file_path: str
    line_number: int
    column: int
    parent: str | None = None
    docstring: str | None = None
    signature: str | None = None


@dataclass
class Edge:
    """Represents a relationship between symbols."""

    from_symbol: str
    to_symbol: str
    edge_type: str  # 'CALLS', 'IMPORTS', 'EXTENDS', 'USES', 'ASSIGNS'
    file_path: str
    line_number: int


@dataclass
class GraphNode:
    """Node in the code graph with metadata."""

    symbol: Symbol
    incoming: set[str] = field(default_factory=set)
    outgoing: set[str] = field(default_factory=set)
    metrics: dict[str, Any] = field(default_factory=dict)


class ASTAnalyzer(ast.NodeVisitor):
    """Extract symbols and relationships from Python AST."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.symbols: list[Symbol] = []
        self.edges: list[Edge] = []
        self.current_class = None
        self.current_function = None
        self.imports: dict[str, str] = {}  # alias -> full_name

    def visit_Import(self, node: ast.Import):
        """Track import statements."""
        for alias in node.names:
            import_name = alias.name
            as_name = alias.asname or alias.name
            self.imports[as_name] = import_name

            # Create import edge
            self.edges.append(
                Edge(
                    from_symbol=self.file_path,
                    to_symbol=import_name,
                    edge_type="IMPORTS",
                    file_path=self.file_path,
                    line_number=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from X import Y statements."""
        module = node.module or ""
        for alias in node.names:
            import_name = f"{module}.{alias.name}" if module else alias.name
            as_name = alias.asname or alias.name
            self.imports[as_name] = import_name

            self.edges.append(
                Edge(
                    from_symbol=self.file_path,
                    to_symbol=import_name,
                    edge_type="IMPORTS",
                    file_path=self.file_path,
                    line_number=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class definitions."""
        symbol = Symbol(
            name=node.name,
            type="class",
            file_path=self.file_path,
            line_number=node.lineno,
            column=node.col_offset,
            parent=self.current_class,
            docstring=ast.get_docstring(node),
        )
        self.symbols.append(symbol)

        # Track inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.edges.append(
                    Edge(
                        from_symbol=f"{self.file_path}:{node.name}",
                        to_symbol=base.id,
                        edge_type="EXTENDS",
                        file_path=self.file_path,
                        line_number=node.lineno,
                    )
                )

        # Visit class body
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function definitions."""
        # Build signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        signature = f"({', '.join(args)})"

        symbol = Symbol(
            name=node.name,
            type="function",
            file_path=self.file_path,
            line_number=node.lineno,
            column=node.col_offset,
            parent=self.current_class or self.current_function,
            docstring=ast.get_docstring(node),
            signature=signature,
        )
        self.symbols.append(symbol)

        # Visit function body
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Call(self, node: ast.Call):
        """Track function calls."""
        if isinstance(node.func, ast.Name):
            called_name = node.func.id

            # Build caller context
            if self.current_function:
                caller = f"{self.file_path}:{self.current_function}"
            elif self.current_class:
                caller = f"{self.file_path}:{self.current_class}"
            else:
                caller = self.file_path

            self.edges.append(
                Edge(
                    from_symbol=caller,
                    to_symbol=called_name,
                    edge_type="CALLS",
                    file_path=self.file_path,
                    line_number=node.lineno,
                )
            )

        elif isinstance(node.func, ast.Attribute):
            # Handle method calls like obj.method()
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr
                called_name = f"{obj_name}.{method_name}"

                if self.current_function:
                    caller = f"{self.file_path}:{self.current_function}"
                else:
                    caller = self.file_path

                self.edges.append(
                    Edge(
                        from_symbol=caller,
                        to_symbol=called_name,
                        edge_type="CALLS",
                        file_path=self.file_path,
                        line_number=node.lineno,
                    )
                )

        self.generic_visit(node)


class GraphMCP:
    """
    Static analysis and dependency graph MCP.
    Provides structural understanding of the codebase.
    """

    def __init__(self, project_root: str, config: dict[str, Any] | None = None):
        self.project_root = Path(project_root)
        self.config = config or {}

        # Configuration
        self.graph_backend = self.config.get("graph_backend", "duckdb")
        self.index_dir = Path(self.config.get("graph_index_dir", ".uc_graph"))
        self.index_dir.mkdir(exist_ok=True)

        # Database connection
        self.db_path = self.index_dir / "code_graph.duckdb"
        self.conn = None

        # In-memory graph for fast queries
        self.graph: dict[str, GraphNode] = {}
        self.file_hashes: dict[str, str] = {}

    def connect(self):
        """Establish database connection."""
        self.conn = duckdb.connect(str(self.db_path))
        self._setup_schema()

    def _setup_schema(self):
        """Create database schema for code graph."""
        # Symbols table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                type VARCHAR,
                file_path VARCHAR,
                line_number INTEGER,
                column_offset INTEGER,
                parent VARCHAR,
                docstring TEXT,
                signature VARCHAR,
                complexity INTEGER DEFAULT 0
            )
        """
        )

        # Edges table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                from_symbol VARCHAR,
                to_symbol VARCHAR,
                edge_type VARCHAR,
                file_path VARCHAR,
                line_number INTEGER,
                PRIMARY KEY (from_symbol, to_symbol, edge_type)
            )
        """
        )

        # File metadata
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_path VARCHAR PRIMARY KEY,
                hash VARCHAR,
                last_parsed TIMESTAMP,
                parse_time_ms INTEGER,
                symbol_count INTEGER,
                edge_count INTEGER
            )
        """
        )

        # Create indexes for fast queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_name ON symbols(name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_type ON symbols(type)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edge_from ON edges(from_symbol)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_to ON edges(to_symbol)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edge_type ON edges(edge_type)"
        )

    async def build_or_load(self, force_rebuild: bool = False):
        """Build or load the code graph."""
        self.connect()

        ok_flag = self.index_dir / "ok.flag"
        if not force_rebuild and ok_flag.exists():
            logger.info("Loading existing code graph...")
            self._load_graph()
            return {"status": "loaded", "nodes": len(self.graph)}

        logger.info("Building code graph...")
        start_time = time.time()

        # Clear existing data
        self.conn.execute("DELETE FROM symbols")
        self.conn.execute("DELETE FROM edges")
        self.conn.execute("DELETE FROM files")

        # Parse all Python files
        await self._parse_all_files()

        # Build graph relationships
        self._build_graph_tables()

        # Mark as complete
        ok_flag.touch()

        duration = time.time() - start_time
        return {
            "status": "built",
            "nodes": len(self.graph),
            "duration_seconds": duration,
        }

    async def _parse_all_files(self):
        """Parse all Python files in parallel."""
        py_files = list(self.project_root.rglob("*.py"))

        # Skip common non-source directories
        skip_dirs = {"venv", "env", ".git", "__pycache__", "build", "dist"}
        py_files = [
            f for f in py_files if not any(skip in f.parts for skip in skip_dirs)
        ]

        logger.info("Parsing {len(py_files)} Python files...")

        # Process in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(min(8, os.cpu_count()))
        tasks = []

        for file_path in py_files:
            task = asyncio.create_task(self._parse_file_async(file_path, semaphore))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        sum(1 for r in results if not isinstance(r, Exception))
        logger.info("Successfully parsed {success_count}/{len(py_files)} files")

    async def _parse_file_async(self, file_path: Path, semaphore: asyncio.Semaphore):
        """Parse a single file asynchronously."""
        async with semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._parse_file, file_path
            )

    def _parse_file(self, file_path: Path) -> tuple[list[Symbol], list[Edge]] | None:
        """Parse a single Python file and extract symbols/edges."""
        try:
            start_time = time.time()

            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Calculate hash
            file_hash = hashlib.md5(content.encode()).hexdigest()

            # Skip if unchanged
            relative_path = str(file_path.relative_to(self.project_root))
            if (
                relative_path in self.file_hashes
                and self.file_hashes[relative_path] == file_hash
            ):
                return None

            # Parse AST
            tree = ast.parse(content, filename=str(file_path))

            # Analyze
            analyzer = ASTAnalyzer(relative_path)
            analyzer.visit(tree)

            # Store in database
            parse_time = (time.time() - start_time) * 1000

            # Insert symbols
            for symbol in analyzer.symbols:
                symbol_id = f"{relative_path}:{symbol.name}"
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO symbols 
                    (id, name, type, file_path, line_number, column_offset, 
                     parent, docstring, signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol_id,
                        symbol.name,
                        symbol.type,
                        symbol.file_path,
                        symbol.line_number,
                        symbol.column,
                        symbol.parent,
                        symbol.docstring,
                        symbol.signature,
                    ),
                )

            # Insert edges
            for edge in analyzer.edges:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO edges 
                    (from_symbol, to_symbol, edge_type, file_path, line_number)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        edge.from_symbol,
                        edge.to_symbol,
                        edge.edge_type,
                        edge.file_path,
                        edge.line_number,
                    ),
                )

            # Update file metadata
            self.conn.execute(
                """
                INSERT OR REPLACE INTO files 
                (file_path, hash, last_parsed, parse_time_ms, symbol_count, edge_count)
                VALUES (?, ?, datetime('now'), ?, ?, ?)
            """,
                (
                    relative_path,
                    file_hash,
                    parse_time,
                    len(analyzer.symbols),
                    len(analyzer.edges),
                ),
            )

            self.file_hashes[relative_path] = file_hash

            return (analyzer.symbols, analyzer.edges)

        except (ValueError, KeyError, AttributeError):
            logger.info("Error parsing {file_path}: {e}")
            return None

    def _build_graph_tables(self):
        """Build in-memory graph from database."""
        logger.info("Building in-memory graph...")

        # Load all symbols
        symbols = self.conn.execute("SELECT * FROM symbols").fetchall()
        for row in symbols:
            symbol_id = row[0]
            symbol = Symbol(
                name=row[1],
                type=row[2],
                file_path=row[3],
                line_number=row[4],
                column=row[5],
                parent=row[6],
                docstring=row[7],
                signature=row[8],
            )
            self.graph[symbol_id] = GraphNode(symbol=symbol)

        # Load all edges
        edges = self.conn.execute("SELECT * FROM edges").fetchall()
        for from_sym, to_sym, _edge_type, _, _ in edges:
            if from_sym in self.graph:
                self.graph[from_sym].outgoing.add(to_sym)
            if to_sym in self.graph:
                self.graph[to_sym].incoming.add(from_sym)

        # Calculate metrics
        for node in self.graph.values():
            node.metrics["in_degree"] = len(node.incoming)
            node.metrics["out_degree"] = len(node.outgoing)
            node.metrics["total_degree"] = len(node.incoming) + len(node.outgoing)

        self.conn.commit()

    def _load_graph(self):
        """Load existing graph from database."""
        self._build_graph_tables()

    # Query API

    def search_code(self, term: str, max_results: int = 250) -> list[dict[str, Any]]:
        """
        Ultra-fast code search using symbol index.
        Returns in <5ms for structural queries.
        """
        sql = """
            SELECT DISTINCT s.file_path, s.line_number, s.name, s.type, s.signature
            FROM symbols s
            WHERE s.name LIKE ?
            ORDER BY s.type, s.name
            LIMIT ?
        """

        results = self.conn.execute(sql, [f"%{term}%", max_results]).fetchall()

        return [
            {
                "file_path": r[0],
                "line_number": r[1],
                "name": r[2],
                "type": r[3],
                "signature": r[4],
            }
            for r in results
        ]

    def find_callers(self, function_name: str) -> list[dict[str, Any]]:
        """Find all functions that call the given function."""
        sql = """
            SELECT DISTINCT e.from_symbol, e.file_path, e.line_number
            FROM edges e
            WHERE e.to_symbol LIKE ? AND e.edge_type = 'CALLS'
            ORDER BY e.file_path, e.line_number
        """

        results = self.conn.execute(sql, [f"%{function_name}%"]).fetchall()

        return [
            {"caller": r[0], "file_path": r[1], "line_number": r[2]} for r in results
        ]

    def find_callees(self, function_name: str) -> list[dict[str, Any]]:
        """Find all functions called by the given function."""
        sql = """
            SELECT DISTINCT e.to_symbol, e.file_path, e.line_number
            FROM edges e
            WHERE e.from_symbol LIKE ? AND e.edge_type = 'CALLS'
            ORDER BY e.to_symbol
        """

        results = self.conn.execute(sql, [f"%{function_name}%"]).fetchall()

        return [
            {"callee": r[0], "file_path": r[1], "line_number": r[2]} for r in results
        ]

    def get_call_chain(
        self, start_func: str, end_func: str, max_depth: int = 5
    ) -> list[list[str]]:
        """Find call chains between two functions."""
        # BFS to find paths
        from collections import deque

        queue = deque([(start_func, [start_func])])
        visited = set()
        paths = []

        while queue and len(paths) < 10:  # Limit results
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current == end_func:
                paths.append(path)
                continue

            if current in visited:
                continue

            visited.add(current)

            # Get callees
            callees = self.find_callees(current)
            for callee in callees:
                callee_name = callee["callee"]
                if callee_name not in path:  # Avoid cycles
                    queue.append((callee_name, path + [callee_name]))

        return paths

    def find_imports(self, module_name: str) -> list[dict[str, Any]]:
        """Find all files that import the given module."""
        sql = """
            SELECT DISTINCT e.from_symbol as file_path, e.line_number
            FROM edges e
            WHERE e.to_symbol = ? AND e.edge_type = 'IMPORTS'
            ORDER BY e.from_symbol
        """

        results = self.conn.execute(sql, [module_name]).fetchall()

        return [{"file_path": r[0], "line_number": r[1]} for r in results]

    def detect_cycles(self) -> list[list[str]]:
        """Detect import cycles in the codebase."""
        # Get all import edges
        import_edges = self.conn.execute(
            """
            SELECT from_symbol, to_symbol 
            FROM edges 
            WHERE edge_type = 'IMPORTS'
        """
        ).fetchall()

        # Build adjacency list
        graph = defaultdict(set)
        for from_sym, to_sym in import_edges:
            graph[from_sym].add(to_sym)

        # DFS to find cycles
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = rec_stack.index(node)
                cycle = rec_stack[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.append(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor, path + [neighbor])

            rec_stack.pop()

        # Check all nodes
        for node in graph:
            if node not in visited:
                dfs(node, [node])

        return cycles[:10]  # Limit results

    def get_complexity_metrics(self) -> dict[str, Any]:
        """Get overall codebase complexity metrics."""
        metrics = self.conn.execute(
            """
            SELECT 
                COUNT(DISTINCT file_path) as total_files,
                COUNT(*) as total_symbols,
                COUNT(CASE WHEN type = 'function' THEN 1 END) as total_functions,
                COUNT(CASE WHEN type = 'class' THEN 1 END) as total_classes,
                AVG(CASE WHEN type = 'function' THEN length(signature) END) as avg_function_params
            FROM symbols
        """
        ).fetchone()

        edge_metrics = self.conn.execute(
            """
            SELECT 
                COUNT(*) as total_edges,
                COUNT(CASE WHEN edge_type = 'CALLS' THEN 1 END) as call_edges,
                COUNT(CASE WHEN edge_type = 'IMPORTS' THEN 1 END) as import_edges,
                COUNT(CASE WHEN edge_type = 'EXTENDS' THEN 1 END) as inheritance_edges
            FROM edges
        """
        ).fetchone()

        return {
            "files": metrics[0],
            "symbols": metrics[1],
            "functions": metrics[2],
            "classes": metrics[3],
            "avg_function_params": round(metrics[4] or 0, 2),
            "total_edges": edge_metrics[0],
            "call_edges": edge_metrics[1],
            "import_edges": edge_metrics[2],
            "inheritance_edges": edge_metrics[3],
            "graph_density": edge_metrics[0] / max(1, metrics[1] * (metrics[1] - 1)),
        }

    def expand_concepts(self, tokens: list[str], radius: int = 2) -> dict[str, Any]:
        """
        Expand query tokens to related code concepts.
        Used by Sequential Thinking for better seeding.
        """
        concepts = {"symbols": [], "modules": [], "patterns": []}

        for token in tokens:
            # Find matching symbols
            symbols = self.search_code(token, max_results=10)
            concepts["symbols"].extend(symbols)

            # Find related modules
            if "." in token:
                module_parts = token.split(".")
                for i in range(len(module_parts)):
                    module = ".".join(module_parts[: i + 1])
                    concepts["modules"].append(module)

        # Deduplicate
        concepts["symbols"] = list({s["name"]: s for s in concepts["symbols"]}.values())
        concepts["modules"] = list(set(concepts["modules"]))

        return concepts

    async def incremental_update(self, changed_files: list[Path]):
        """Incrementally update graph for changed files."""
        logger.info("Incrementally updating {len(changed_files)} files...")

        for file_path in changed_files:
            relative_path = str(file_path.relative_to(self.project_root))

            # Remove old data
            self.conn.execute(
                "DELETE FROM symbols WHERE file_path = ?", [relative_path]
            )
            self.conn.execute("DELETE FROM edges WHERE file_path = ?", [relative_path])

            # Re-parse
            await self._parse_file_async(file_path, asyncio.Semaphore(1))

        # Rebuild affected parts of graph
        self._build_graph_tables()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Convenience functions for integration


async def build_code_graph(project_root: str, force: bool = False) -> dict[str, Any]:
    """Build or refresh the code graph."""
    graph = GraphMCP(project_root)
    result = await graph.build_or_load(force_rebuild=force)
    graph.close()
    return result


def quick_find_callers(project_root: str, function_name: str) -> list[dict[str, Any]]:
    """Quick function to find callers."""
    graph = GraphMCP(project_root)
    graph.connect()
    results = graph.find_callers(function_name)
    graph.close()
    return results
