"""Pre-indexing system for Jarvis - inspired by orchestrator but simplified."""

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from unity_wheel.accelerated_tools import get_duckdb_turbo, get_python_analyzer


@dataclass
class FileIndex:
    """Index entry for a file."""

    path: str
    content_hash: str
    size: int
    modified_time: float
    extension: str
    directory: str

    # Code metrics
    lines_of_code: int
    complexity_score: int
    num_functions: int
    num_classes: int
    imports: list[str]
    exports: list[str]

    # Search optimization
    keywords: list[str]
    doc_strings: list[str]

    # Trading-specific
    has_trading_patterns: bool
    trading_keywords: list[str]


class IndexManager:
    """Manages pre-computed indexes for instant code retrieval."""

    def __init__(
        self, workspace_root: str = ".", index_path: str = ".jarvis/index.duckdb"
    ):
        self.workspace_root = Path(workspace_root)
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize DuckDB
        self.db = get_duckdb_turbo(str(self.index_path))

        # Python analyzer for metrics
        self.analyzer = get_python_analyzer()

        # Trading patterns to detect
        self.trading_patterns = {
            "options": ["option", "strike", "expiry", "call", "put"],
            "greeks": ["delta", "gamma", "theta", "vega", "rho"],
            "strategy": ["wheel", "spread", "collar", "straddle"],
            "risk": ["var", "exposure", "margin", "drawdown"],
            "backtest": ["backtest", "historical", "simulation"],
            "execution": ["order", "fill", "trade", "position"],
        }

        # Initialize schema
        asyncio.create_task(self._init_schema())

    async def _init_schema(self):
        """Create index schema if not exists."""
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS file_index (
                path TEXT PRIMARY KEY,
                content_hash TEXT,
                size INTEGER,
                modified_time REAL,
                extension TEXT,
                directory TEXT,
                
                -- Code metrics
                lines_of_code INTEGER,
                complexity_score INTEGER,
                num_functions INTEGER,
                num_classes INTEGER,
                imports TEXT,  -- JSON array
                exports TEXT,  -- JSON array
                
                -- Search optimization
                keywords TEXT,  -- JSON array
                doc_strings TEXT,  -- JSON array
                
                -- Trading specific
                has_trading_patterns BOOLEAN,
                trading_keywords TEXT,  -- JSON array
                
                -- Indexing metadata
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Indexes for fast lookup
                INDEX idx_extension (extension),
                INDEX idx_directory (directory),
                INDEX idx_complexity (complexity_score),
                INDEX idx_trading (has_trading_patterns),
                INDEX idx_modified (modified_time DESC)
            )
        """
        )

        # Create full-text search index
        await self.db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS file_search
            USING FTS5(path, content, keywords, doc_strings)
        """
        )

        # Create code patterns table
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS code_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,  -- 'optimization', 'refactoring', 'bug_fix'
                description TEXT,
                code_before TEXT,
                code_after TEXT,
                performance_gain REAL,
                complexity_reduction REAL,
                times_applied INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                last_applied TIMESTAMP
            )
        """
        )

        # Create execution history for learning
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_history (
                execution_id TEXT PRIMARY KEY,
                query TEXT,
                strategy_chosen TEXT,
                files_affected INTEGER,
                duration_ms REAL,
                success BOOLEAN,
                performance_impact REAL,
                metadata TEXT,  -- JSON
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    async def build_index(self, force_rebuild: bool = False) -> dict[str, Any]:
        """Build or update the file index."""
        start_time = time.perf_counter()

        # Check if index needs rebuilding
        if not force_rebuild:
            last_indexed = await self._get_last_indexed_time()
            if last_indexed and (time.time() - last_indexed) < 86400:  # 24 hours
                return {"status": "up_to_date", "last_indexed": last_indexed}

        # Find all Python files
        py_files = list(self.workspace_root.rglob("*.py"))

        # Filter out common ignore patterns
        ignore_patterns = ["__pycache__", ".git", "venv", "env", ".pytest_cache"]
        py_files = [
            f for f in py_files if not any(p in str(f) for p in ignore_patterns)
        ]

        # Process files in batches
        batch_size = 100
        total_indexed = 0

        for i in range(0, len(py_files), batch_size):
            batch = py_files[i : i + batch_size]
            batch_indexes = await self._process_file_batch(batch)

            # Insert into database
            await self._insert_indexes(batch_indexes)
            total_indexed += len(batch_indexes)

            # Update search index
            await self._update_search_index(batch_indexes)

        duration = (time.perf_counter() - start_time) * 1000

        return {
            "status": "completed",
            "files_indexed": total_indexed,
            "duration_ms": duration,
            "files_per_second": total_indexed / (duration / 1000)
            if duration > 0
            else 0,
        }

    async def _process_file_batch(self, files: list[Path]) -> list[FileIndex]:
        """Process a batch of files into index entries."""
        indexes = []

        for file_path in files:
            try:
                # Read file content
                content = file_path.read_text(encoding="utf-8")

                # Basic metadata
                stat = file_path.stat()
                content_hash = hashlib.sha256(content.encode()).hexdigest()

                # Analyze with Python analyzer
                analysis = await self.analyzer.analyze_file(str(file_path))

                # Extract keywords and docstrings
                keywords = self._extract_keywords(content)
                doc_strings = self._extract_docstrings(content)

                # Check for trading patterns
                has_trading, trading_keywords = self._detect_trading_patterns(content)

                # Create index entry
                index = FileIndex(
                    path=str(file_path.relative_to(self.workspace_root)),
                    content_hash=content_hash,
                    size=stat.st_size,
                    modified_time=stat.st_mtime,
                    extension=file_path.suffix,
                    directory=str(file_path.parent.relative_to(self.workspace_root)),
                    lines_of_code=analysis.loc,
                    complexity_score=analysis.complexity,
                    num_functions=len(analysis.functions),
                    num_classes=len(analysis.classes),
                    imports=analysis.imports,
                    exports=[c["name"] for c in analysis.classes]
                    + [f["name"] for f in analysis.functions],
                    keywords=keywords,
                    doc_strings=doc_strings,
                    has_trading_patterns=has_trading,
                    trading_keywords=trading_keywords,
                )

                indexes.append(index)

            except Exception:
                # Skip files that can't be processed
                continue

        return indexes

    def _extract_keywords(self, content: str) -> list[str]:
        """Extract important keywords from code."""
        import re

        # Extract function and class names
        functions = re.findall(r"def\s+(\w+)", content)
        classes = re.findall(r"class\s+(\w+)", content)

        # Extract variable assignments (potential important names)
        variables = re.findall(r"(\w+)\s*=", content)

        # Combine and deduplicate
        keywords = list(set(functions + classes + variables))

        # Filter out common Python keywords
        python_keywords = {
            "self",
            "def",
            "class",
            "if",
            "else",
            "for",
            "while",
            "return",
            "import",
            "from",
        }
        keywords = [k for k in keywords if k not in python_keywords and len(k) > 2]

        return keywords[:50]  # Limit to top 50

    def _extract_docstrings(self, content: str) -> list[str]:
        """Extract docstrings from code."""
        import re

        # Find triple-quoted strings
        docstrings = re.findall(r'"""([^"]+)"""', content, re.MULTILINE)
        docstrings.extend(re.findall(r"'''([^']+)'''", content, re.MULTILINE))

        # Clean and limit
        docstrings = [d.strip() for d in docstrings if len(d.strip()) > 10]

        return docstrings[:20]  # Limit to top 20

    def _detect_trading_patterns(self, content: str) -> tuple[bool, list[str]]:
        """Detect trading-specific patterns in code."""
        content_lower = content.lower()
        found_keywords = []

        for _category, keywords in self.trading_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    found_keywords.append(keyword)

        has_trading = len(found_keywords) > 0
        return has_trading, list(set(found_keywords))

    async def _insert_indexes(self, indexes: list[FileIndex]):
        """Insert index entries into database."""
        if not indexes:
            return

        # Convert to records
        records = []
        for idx in indexes:
            record = asdict(idx)
            # Convert lists to JSON
            record["imports"] = json.dumps(record["imports"])
            record["exports"] = json.dumps(record["exports"])
            record["keywords"] = json.dumps(record["keywords"])
            record["doc_strings"] = json.dumps(record["doc_strings"])
            record["trading_keywords"] = json.dumps(record["trading_keywords"])
            records.append(record)

        # Bulk insert with UPSERT
        await self.db.insert_batch("file_index", records)

    async def _update_search_index(self, indexes: list[FileIndex]):
        """Update full-text search index."""
        # Implementation would update FTS5 table
        pass

    async def _get_last_indexed_time(self) -> float | None:
        """Get the last indexing timestamp."""
        result = await self.db.query_to_pandas(
            "SELECT MAX(indexed_at) as last_indexed FROM file_index"
        )
        if not result.empty and result["last_indexed"][0]:
            return result["last_indexed"][0].timestamp()
        return None

    async def search_by_pattern(
        self, pattern: str, pattern_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Search for files using pre-built index."""
        query = """
            SELECT path, complexity_score, has_trading_patterns
            FROM file_index
            WHERE path IN (
                SELECT path FROM file_search WHERE file_search MATCH ?
            )
        """

        if pattern_type == "trading":
            query += " AND has_trading_patterns = true"

        query += " ORDER BY complexity_score DESC LIMIT 50"

        result = await self.db.query_to_pandas(query, (pattern,))
        return result.to_dict("records")

    async def get_file_metrics(self, file_path: str) -> dict[str, Any] | None:
        """Get pre-computed metrics for a file."""
        result = await self.db.query_to_pandas(
            "SELECT * FROM file_index WHERE path = ?", (file_path,)
        )

        if not result.empty:
            record = result.iloc[0].to_dict()
            # Parse JSON fields
            for field in [
                "imports",
                "exports",
                "keywords",
                "doc_strings",
                "trading_keywords",
            ]:
                if field in record and record[field]:
                    record[field] = json.loads(record[field])
            return record

        return None

    async def record_execution(self, execution_data: dict[str, Any]):
        """Record execution history for learning."""
        await self.db.execute(
            """
            INSERT INTO execution_history 
            (execution_id, query, strategy_chosen, files_affected, 
             duration_ms, success, performance_impact, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                execution_data.get("execution_id"),
                execution_data.get("query"),
                execution_data.get("strategy_chosen"),
                execution_data.get("files_affected", 0),
                execution_data.get("duration_ms", 0),
                execution_data.get("success", True),
                execution_data.get("performance_impact", 0),
                json.dumps(execution_data.get("metadata", {})),
            ),
        )

    async def get_learned_patterns(
        self, pattern_type: str = None
    ) -> list[dict[str, Any]]:
        """Get successful code transformation patterns."""
        query = """
            SELECT * FROM code_patterns
            WHERE success_rate > 0.7
        """

        if pattern_type:
            query += f" AND pattern_type = '{pattern_type}'"

        query += " ORDER BY times_applied * success_rate DESC LIMIT 20"

        result = await self.db.query_to_pandas(query)
        return result.to_dict("records")
