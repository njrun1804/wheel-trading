"""Deep Indexing Manager for Jarvis 2.0.

Hybrid indexing system that combines:
- FAISS for vector embeddings (semantic search)
- SQLite FTS5 for full-text search
- NetworkX for dependency graphs  
- DuckDB for structured queries and analytics
- LMDB for ultra-fast key-value lookups
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sqlite3
import pickle
import ast

import numpy as np
import networkx as nx
import lmdb
import duckdb

logger = logging.getLogger(__name__)

# Try to import FAISS, fallback to simple numpy if not available
try:
    import faiss
    FAISS_AVAILABLE = True
    # Optimize for CPU on Apple Silicon
    import os
    faiss.omp_set_num_threads(os.cpu_count() // 2)  # Use half the cores for FAISS
except ImportError:
    logger.warning("FAISS not available, using numpy for embeddings")
    FAISS_AVAILABLE = False


class DeepIndexManager:
    """Manages all indexes for intelligent code search and analysis."""
    
    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Different index components
        self.vector_index = None  # FAISS index for embeddings
        self.fts_db = None  # SQLite FTS5 for text search
        self.graph_db = None  # NetworkX graphs
        self.struct_db = None  # DuckDB for structured data
        self.kv_db = None  # LMDB for fast lookups
        
        # Caches
        self._embedding_cache = {}
        self._pattern_cache = {}
        
        # Stats
        self.total_files_indexed = 0
        self.last_update_time = 0
    
    async def initialize(self):
        """Initialize all index components."""
        logger.info("Initializing Deep Index Manager...")
        
        # Initialize each component
        await asyncio.gather(
            self._init_vector_index(),
            self._init_fts_index(),
            self._init_graph_index(),
            self._init_struct_index(),
            self._init_kv_store()
        )
        
        # Skip initial indexing for faster startup
        # Can be triggered manually or in background
        if self.total_files_indexed == 0:
            logger.info("Skipping initial indexing for faster startup")
            # await self.build_indexes()
        
        logger.info(f"Index manager initialized with {self.total_files_indexed} files")
    
    async def _init_vector_index(self):
        """Initialize FAISS vector index for semantic search."""
        vector_path = self.index_path / "vectors"
        vector_path.mkdir(exist_ok=True)
        
        if FAISS_AVAILABLE:
            # Use FAISS with HNSW index for fast approximate search
            dimension = 768  # Standard embedding dimension
            
            index_file = vector_path / "code_embeddings.index"
            if index_file.exists():
                self.vector_index = faiss.read_index(str(index_file))
                logger.info(f"Loaded FAISS index with {self.vector_index.ntotal} vectors")
            else:
                # Create new HNSW index
                self.vector_index = faiss.IndexHNSWFlat(dimension, 32)
                self.vector_index.hnsw.efConstruction = 200
                logger.info("Created new FAISS HNSW index")
        else:
            # Fallback to numpy arrays
            self.vector_index = NumpyVectorIndex(vector_path)
            await self.vector_index.load()
    
    async def _init_fts_index(self):
        """Initialize SQLite FTS5 for full-text search."""
        fts_path = self.index_path / "fts"
        fts_path.mkdir(exist_ok=True)
        
        self.fts_db = sqlite3.connect(str(fts_path / "code_fts.db"))
        self.fts_db.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        
        # Create FTS5 table if not exists
        self.fts_db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS code_search USING fts5(
                file_path,
                content,
                language,
                symbols,  -- JSON array of symbols
                imports,  -- JSON array of imports
                tokenize='porter unicode61'
            )
        """)
        
        # Create metadata table
        self.fts_db.execute("""
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT,
                last_modified REAL,
                lines INTEGER,
                complexity_score REAL,
                ast_hash TEXT
            )
        """)
        
        self.fts_db.commit()
    
    async def _init_graph_index(self):
        """Initialize NetworkX for dependency graphs."""
        graph_path = self.index_path / "graphs"
        graph_path.mkdir(exist_ok=True)
        
        # Load or create dependency graph
        dep_graph_file = graph_path / "dependencies.gpickle"
        if dep_graph_file.exists():
            with open(dep_graph_file, 'rb') as f:
                self.dependency_graph = pickle.load(f)
        else:
            self.dependency_graph = nx.DiGraph()
        
        # Load or create call graph
        call_graph_file = graph_path / "calls.gpickle"
        if call_graph_file.exists():
            with open(call_graph_file, 'rb') as f:
                self.call_graph = pickle.load(f)
        else:
            self.call_graph = nx.DiGraph()
        
        logger.info(f"Loaded graphs: {len(self.dependency_graph)} deps, {len(self.call_graph)} calls")
    
    async def _init_struct_index(self):
        """Initialize DuckDB for structured queries."""
        struct_path = self.index_path / "structured"
        struct_path.mkdir(exist_ok=True)
        
        self.struct_db = duckdb.connect(str(struct_path / "code_analytics.duckdb"))
        
        # Create tables for code analytics
        self.struct_db.execute("""
            CREATE TABLE IF NOT EXISTS code_metrics (
                file_path VARCHAR PRIMARY KEY,
                language VARCHAR,
                total_lines INTEGER,
                code_lines INTEGER,
                comment_lines INTEGER,
                complexity INTEGER,
                num_functions INTEGER,
                num_classes INTEGER,
                avg_function_length DOUBLE,
                max_nesting_depth INTEGER,
                technical_debt_score DOUBLE
            )
        """)
        
        self.struct_db.execute("""
            CREATE TABLE IF NOT EXISTS function_signatures (
                id INTEGER PRIMARY KEY,
                file_path VARCHAR,
                function_name VARCHAR,
                line_number INTEGER,
                parameters TEXT,  -- JSON
                return_type VARCHAR,
                decorators TEXT,  -- JSON
                complexity INTEGER,
                is_async BOOLEAN
            )
        """)
        
        self.struct_db.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id VARCHAR PRIMARY KEY,
                pattern_type VARCHAR,  -- 'design', 'anti', 'performance'
                description TEXT,
                occurrences INTEGER,
                files TEXT  -- JSON array
            )
        """)
        
        # Create indexes for fast queries
        self.struct_db.execute("CREATE INDEX IF NOT EXISTS idx_complexity ON code_metrics(complexity)")
        self.struct_db.execute("CREATE INDEX IF NOT EXISTS idx_tech_debt ON code_metrics(technical_debt_score)")
    
    async def _init_kv_store(self):
        """Initialize LMDB for ultra-fast key-value lookups."""
        kv_path = self.index_path / "kv"
        kv_path.mkdir(exist_ok=True)
        
        # Open LMDB with smaller initial size for faster startup
        try:
            self.kv_env = lmdb.open(
                str(kv_path),
                map_size=100 * 1024 * 1024,  # 100MB initially (smaller for stability)
                max_dbs=10,
                sync=False,  # Async writes for speed
                writemap=False  # Disable writemap for stability
            )
        except Exception as e:
            logger.warning(f"LMDB writemap failed ({e}), falling back to standard mode")
            self.kv_env = lmdb.open(
                str(kv_path),
                map_size=100 * 1024 * 1024,  # 100MB
                max_dbs=10,
                sync=False
            )
        
        # Create named databases
        self.kv_embeddings = self.kv_env.open_db(b'embeddings')
        self.kv_ast_cache = self.kv_env.open_db(b'ast_cache')
        self.kv_symbols = self.kv_env.open_db(b'symbols')
        self.kv_patterns = self.kv_env.open_db(b'patterns')
    
    async def build_indexes(self):
        """Build all indexes from scratch."""
        logger.info("Building indexes from scratch...")
        start_time = time.time()
        
        # Find all Python files
        py_files = list(Path.cwd().rglob("*.py"))
        logger.info(f"Found {len(py_files)} Python files to index")
        
        # Process in batches for efficiency
        batch_size = 50
        for i in range(0, len(py_files), batch_size):
            batch = py_files[i:i + batch_size]
            await self._index_batch(batch)
            
            if i % 200 == 0 and i > 0:
                logger.info(f"Indexed {i}/{len(py_files)} files...")
        
        # Build derived indexes
        await self._build_pattern_index()
        await self._build_performance_index()
        
        # Save indexes
        await self._save_indexes()
        
        self.total_files_indexed = len(py_files)
        self.last_update_time = time.time()
        
        elapsed = time.time() - start_time
        logger.info(f"Indexing complete: {len(py_files)} files in {elapsed:.2f}s")
    
    async def _index_batch(self, files: List[Path]):
        """Index a batch of files."""
        for file_path in files:
            try:
                await self._index_file(file_path)
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
    
    async def _index_file(self, file_path: Path):
        """Index a single file across all systems."""
        # Read file content
        try:
            content = file_path.read_text(encoding='utf-8')
        except:
            return
        
        # Calculate file hash
        file_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except:
            return
        
        # Extract metadata
        metadata = self._extract_metadata(tree, content)
        
        # 1. Update FTS index
        self.fts_db.execute("""
            INSERT OR REPLACE INTO code_search 
            (file_path, content, language, symbols, imports)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(file_path),
            content,
            'python',
            json.dumps(metadata['symbols']),
            json.dumps(metadata['imports'])
        ))
        
        # 2. Update structured DB
        self.struct_db.execute("""
            INSERT OR REPLACE INTO code_metrics
            (file_path, language, total_lines, code_lines, comment_lines,
             complexity, num_functions, num_classes, avg_function_length,
             max_nesting_depth, technical_debt_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(file_path),
            'python',
            metadata['total_lines'],
            metadata['code_lines'],
            metadata['comment_lines'],
            metadata['complexity'],
            metadata['num_functions'],
            metadata['num_classes'],
            metadata['avg_function_length'],
            metadata['max_nesting_depth'],
            metadata['technical_debt_score']
        ))
        
        # 3. Generate and store embeddings
        embedding = await self._generate_embedding(content)
        with self.kv_env.begin(write=True) as txn:
            txn.put(
                f"emb:{file_path}".encode(),
                embedding.tobytes(),
                db=self.kv_embeddings
            )
        
        # 4. Update dependency graph
        for imp in metadata['imports']:
            self.dependency_graph.add_edge(str(file_path), imp)
        
        # 5. Cache AST
        with self.kv_env.begin(write=True) as txn:
            txn.put(
                f"ast:{file_path}".encode(),
                pickle.dumps(tree),
                db=self.kv_ast_cache
            )
    
    def _extract_metadata(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Extract metadata from AST."""
        visitor = CodeAnalyzer()
        visitor.visit(tree)
        
        lines = content.split('\n')
        total_lines = len(lines)
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = total_lines - comment_lines - sum(1 for line in lines if not line.strip())
        
        return {
            'symbols': visitor.symbols,
            'imports': visitor.imports,
            'total_lines': total_lines,
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'complexity': visitor.complexity,
            'num_functions': len(visitor.functions),
            'num_classes': len(visitor.classes),
            'avg_function_length': visitor.avg_function_length(),
            'max_nesting_depth': visitor.max_nesting_depth,
            'technical_debt_score': visitor.calculate_tech_debt()
        }
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        # Simple hash-based embedding for now
        # In production, would use a proper code embedding model
        
        # Create 768-dim embedding
        embedding = np.zeros(768)
        
        # Hash-based features
        words = text.split()
        for i, word in enumerate(words[:768]):
            hash_val = hash(word) % 768
            embedding[hash_val] += 1.0 / (i + 1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        return embedding.astype(np.float32)
    
    async def get_context(self, query: str) -> Dict[str, Any]:
        """Get relevant context for a query."""
        context = {
            'files': [],
            'symbols': [],
            'dependencies': [],
            'patterns': [],
            'metrics': {}
        }
        
        # 1. Semantic search
        query_embedding = await self._generate_embedding(query)
        similar_files = await self._vector_search(query_embedding, k=10)
        context['files'].extend(similar_files)
        
        # 2. Full-text search
        fts_results = self._fts_search(query, limit=10)
        for result in fts_results:
            if result not in context['files']:
                context['files'].append(result)
        
        # 3. Symbol search
        symbols = self._find_symbols(query)
        context['symbols'] = symbols
        
        # 4. Dependency analysis
        for file in context['files'][:5]:
            deps = list(self.dependency_graph.successors(file))
            context['dependencies'].extend(deps)
        
        # 5. Pattern detection
        patterns = await self._find_patterns(query)
        context['patterns'] = patterns
        
        # 6. Metrics
        context['metrics'] = self._get_aggregate_metrics(context['files'])
        
        return context
    
    async def _vector_search(self, query_embedding: np.ndarray, k: int = 10) -> List[str]:
        """Search for similar embeddings."""
        if FAISS_AVAILABLE and self.vector_index and self.vector_index.ntotal > 0:
            # FAISS search
            distances, indices = self.vector_index.search(
                query_embedding.reshape(1, -1), k
            )
            
            # Map indices to file paths
            # (In production, would maintain an index-to-path mapping)
            return [f"file_{i}" for i in indices[0] if i >= 0]
        else:
            # Fallback to brute force
            return []
    
    def _fts_search(self, query: str, limit: int = 10) -> List[str]:
        """Full-text search."""
        cursor = self.fts_db.execute("""
            SELECT file_path, rank
            FROM code_search
            WHERE code_search MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        
        return [row[0] for row in cursor.fetchall()]
    
    def _find_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Find symbols matching query."""
        symbols = []
        
        with self.kv_env.begin() as txn:
            cursor = txn.cursor(db=self.kv_symbols)
            for key, value in cursor:
                if query.lower() in key.decode().lower():
                    symbol_data = json.loads(value)
                    symbols.append(symbol_data)
                    if len(symbols) >= 20:
                        break
        
        return symbols
    
    async def _find_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Find code patterns related to query."""
        # Query DuckDB for patterns
        results = self.struct_db.execute("""
            SELECT pattern_id, pattern_type, description, occurrences
            FROM patterns
            WHERE description ILIKE ?
            ORDER BY occurrences DESC
            LIMIT 10
        """, (f"%{query}%",)).fetchall()
        
        return [
            {
                'id': r[0],
                'type': r[1],
                'description': r[2],
                'occurrences': r[3]
            }
            for r in results
        ]
    
    def _get_aggregate_metrics(self, files: List[str]) -> Dict[str, float]:
        """Get aggregate metrics for files."""
        if not files:
            return {}
        
        placeholders = ','.join('?' * len(files[:10]))  # Limit to 10 files
        result = self.struct_db.execute(f"""
            SELECT 
                AVG(complexity) as avg_complexity,
                AVG(technical_debt_score) as avg_debt,
                SUM(code_lines) as total_lines,
                AVG(avg_function_length) as avg_func_length
            FROM code_metrics
            WHERE file_path IN ({placeholders})
        """, files[:10]).fetchone()
        
        return {
            'avg_complexity': result[0] or 0,
            'avg_technical_debt': result[1] or 0,
            'total_code_lines': result[2] or 0,
            'avg_function_length': result[3] or 0
        }
    
    async def update_incremental(self):
        """Incrementally update indexes."""
        # Find modified files since last update
        modified_files = []
        
        for py_file in Path.cwd().rglob("*.py"):
            if py_file.stat().st_mtime > self.last_update_time:
                modified_files.append(py_file)
        
        if modified_files:
            logger.info(f"Updating {len(modified_files)} modified files")
            await self._index_batch(modified_files)
            await self._save_indexes()
            self.last_update_time = time.time()
    
    async def _build_pattern_index(self):
        """Build index of code patterns."""
        # Simple pattern detection for now
        patterns = [
            ('singleton', 'Singleton pattern', 0),
            ('factory', 'Factory pattern', 0),
            ('observer', 'Observer pattern', 0),
            ('nested_loop', 'Nested loops (performance)', 0),
            ('global_var', 'Global variables (anti-pattern)', 0)
        ]
        
        for pattern_id, description, _ in patterns:
            count = self.fts_db.execute("""
                SELECT COUNT(*) FROM code_search 
                WHERE content MATCH ?
            """, (pattern_id,)).fetchone()[0]
            
            self.struct_db.execute("""
                INSERT OR REPLACE INTO patterns 
                (pattern_id, pattern_type, description, occurrences, files)
                VALUES (?, ?, ?, ?, ?)
            """, (pattern_id, 'design', description, count, '[]'))
    
    async def _build_performance_index(self):
        """Build performance-related indexes."""
        # Identify performance hotspots
        hotspots = self.struct_db.execute("""
            SELECT file_path, complexity, technical_debt_score
            FROM code_metrics
            WHERE complexity > 10 OR technical_debt_score > 0.7
            ORDER BY complexity DESC
            LIMIT 20
        """).fetchall()
        
        logger.info(f"Found {len(hotspots)} performance hotspots")
    
    async def _save_indexes(self):
        """Save all indexes to disk."""
        # Save FAISS index
        if FAISS_AVAILABLE and self.vector_index:
            faiss.write_index(
                self.vector_index,
                str(self.index_path / "vectors" / "code_embeddings.index")
            )
        
        # Save graphs
        with open(self.index_path / "graphs" / "dependencies.gpickle", 'wb') as f:
            pickle.dump(self.dependency_graph, f)
        
        with open(self.index_path / "graphs" / "calls.gpickle", 'wb') as f:
            pickle.dump(self.call_graph, f)
        
        # Commit databases
        self.fts_db.commit()
        self.struct_db.commit()
        self.kv_env.sync()
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = {
            'total_files': self.total_files_indexed,
            'last_update': self.last_update_time,
            'vector_index_size': self.vector_index.ntotal if FAISS_AVAILABLE and self.vector_index else 0,
            'dependency_nodes': len(self.dependency_graph),
            'dependency_edges': self.dependency_graph.number_of_edges(),
        }
        
        # Get DB stats
        fts_count = self.fts_db.execute("SELECT COUNT(*) FROM code_search").fetchone()[0]
        metrics_count = self.struct_db.execute("SELECT COUNT(*) FROM code_metrics").fetchone()[0]
        
        stats['fts_documents'] = fts_count
        stats['metrics_files'] = metrics_count
        
        return stats


class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor to extract code metadata."""
    
    def __init__(self):
        self.symbols = []
        self.imports = []
        self.functions = []
        self.classes = []
        self.complexity = 0
        self.max_nesting_depth = 0
        self._current_depth = 0
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        self.functions.append({
            'name': node.name,
            'line': node.lineno,
            'args': [arg.arg for arg in node.args.args]
        })
        self.symbols.append(f"func:{node.name}")
        self._visit_block(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        self.classes.append({
            'name': node.name,
            'line': node.lineno
        })
        self.symbols.append(f"class:{node.name}")
        self._visit_block(node)
    
    def _visit_block(self, node):
        self._current_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self._current_depth)
        self.complexity += 1
        self.generic_visit(node)
        self._current_depth -= 1
    
    def visit_If(self, node):
        self.complexity += 1
        self._visit_block(node)
    
    def visit_For(self, node):
        self.complexity += 2
        self._visit_block(node)
    
    def visit_While(self, node):
        self.complexity += 2
        self._visit_block(node)
    
    def avg_function_length(self) -> float:
        if not self.functions:
            return 0
        # Simplified - would need actual line counts
        return 10.0  # Placeholder
    
    def calculate_tech_debt(self) -> float:
        """Simple technical debt score."""
        score = 0.0
        
        # High complexity
        if self.complexity > 20:
            score += 0.3
        
        # Deep nesting
        if self.max_nesting_depth > 5:
            score += 0.2
        
        # Too many functions
        if len(self.functions) > 50:
            score += 0.2
        
        # Large classes
        if any(len(cls.get('methods', [])) > 20 for cls in self.classes):
            score += 0.3
        
        return min(1.0, score)


class NumpyVectorIndex:
    """Simple numpy-based vector index as FAISS fallback."""
    
    def __init__(self, path: Path):
        self.path = path
        self.vectors = None
        self.metadata = []
    
    async def load(self):
        index_file = self.path / "vectors.npy"
        if index_file.exists():
            self.vectors = np.load(index_file)
            
            meta_file = self.path / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    self.metadata = json.load(f)
    
    def add(self, vector: np.ndarray, metadata: Dict[str, Any]):
        if self.vectors is None:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector])
        
        self.metadata.append(metadata)
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([[]]), np.array([[]])
        
        # Cosine similarity
        similarities = np.dot(self.vectors, query)
        top_k = np.argsort(similarities)[-k:][::-1]
        
        return similarities[top_k].reshape(1, -1), top_k.reshape(1, -1)
    
    @property
    def ntotal(self) -> int:
        return len(self.vectors) if self.vectors is not None else 0