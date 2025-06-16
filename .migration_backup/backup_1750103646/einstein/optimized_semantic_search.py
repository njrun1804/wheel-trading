"""
Optimized Semantic Search for Code Analysis Queries

Fixes core issues with Einstein's semantic search:
1. Poor understanding of programming concepts vs keywords
2. Embedding quality issues for code structures  
3. Search result ranking not optimized for code relevance
4. Semantic search returning 0 relevant results for coding queries
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.unity_wheel.mcp.code_specific_embeddings import CodeEmbeddingPipeline

logger = logging.getLogger(__name__)


@dataclass
class CodeSearchResult:
    """Enhanced search result with code-specific metadata."""

    content: str
    file_path: str
    line_number: int
    score: float
    result_type: str
    context: dict[str, Any]
    timestamp: float

    # Code-specific fields
    code_metadata: dict[str, Any] | None = None
    semantic_score: float | None = None
    relevance_score: float | None = None
    code_concepts: list[str] | None = None


class OptimizedSemanticSearch:
    """
    Optimized semantic search specifically designed for coding analysis queries.

    Key improvements:
    1. Code-aware embedding generation and matching
    2. Programming concept understanding (async, inheritance, etc.)
    3. Context-aware result ranking for code relevance
    4. Multi-stage search with fallbacks
    5. Query expansion for coding terms
    """

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.embedding_pipeline = CodeEmbeddingPipeline()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize code index
        self.code_index = {}  # file_path -> embeddings
        self.file_metadata = {}  # file_path -> metadata
        self.semantic_cache = {}  # query_hash -> results

        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_search_time_ms": 0,
            "semantic_matches": 0,
            "fallback_searches": 0,
        }

        # Code-specific query patterns
        self.coding_query_patterns = {
            "function_search": ["function", "def", "method", "procedure"],
            "class_search": ["class", "object", "inheritance", "extends"],
            "async_search": ["async", "await", "asynchronous", "concurrent"],
            "import_search": ["import", "module", "package", "dependency"],
            "error_search": ["error", "exception", "try", "catch", "except"],
            "pattern_search": ["pattern", "strategy", "factory", "singleton"],
            "data_search": ["data", "model", "dataframe", "schema"],
            "optimization_search": ["optimize", "performance", "speed", "efficient"],
        }

        # Initialize FAISS index if available
        self.faiss_index = None
        self.faiss_available = False
        self._init_faiss_index()

    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search."""
        try:
            import faiss

            # Create index for 1536-dimensional embeddings
            self.faiss_index = faiss.IndexFlatIP(
                1536
            )  # Inner product for cosine similarity
            self.faiss_available = True
            logger.info("✅ FAISS index initialized for semantic search")

        except ImportError:
            logger.info("ℹ️ FAISS not available, using linear search fallback")
            self.faiss_available = False

    async def search(
        self, query: str, max_results: int = 50, search_type: str = "auto"
    ) -> list[CodeSearchResult]:
        """
        Perform optimized semantic search for coding queries.

        Args:
            query: Search query (e.g., "async functions", "class inheritance")
            max_results: Maximum number of results to return
            search_type: Type of search ('auto', 'semantic', 'hybrid')

        Returns:
            List of CodeSearchResult objects ranked by relevance
        """
        start_time = time.time()
        self.search_stats["total_searches"] += 1

        # Check cache first
        query_hash = hash(f"{query}_{max_results}_{search_type}")
        if query_hash in self.semantic_cache:
            self.search_stats["cache_hits"] += 1
            logger.debug(f"Cache hit for query: {query}")
            return self.semantic_cache[query_hash]

        try:
            # Analyze query to determine search strategy
            query_analysis = await self._analyze_coding_query(query)

            # Determine search strategy
            if search_type == "auto":
                search_strategy = self._determine_search_strategy(query_analysis)
            else:
                search_strategy = search_type

            # Execute search based on strategy
            if search_strategy == "semantic":
                results = await self._semantic_search(
                    query, query_analysis, max_results
                )
            elif search_strategy == "hybrid":
                results = await self._hybrid_search(query, query_analysis, max_results)
            else:
                results = await self._pattern_search(query, query_analysis, max_results)

            # If semantic/hybrid search returns few results, also try pattern search
            if len(results) < 3 and search_strategy in ["semantic", "hybrid"]:
                pattern_results = await self._pattern_search(
                    query, query_analysis, max_results // 2
                )

                # Add unique pattern results
                existing_keys = {f"{r.file_path}:{r.line_number}" for r in results}
                for pattern_result in pattern_results:
                    key = f"{pattern_result.file_path}:{pattern_result.line_number}"
                    if key not in existing_keys:
                        results.append(pattern_result)

            # If still too few results, try text search fallback
            if len(results) < 2:
                from src.unity_wheel.accelerated_tools.ripgrep_turbo import (
                    get_ripgrep_turbo,
                )

                rg = get_ripgrep_turbo()

                try:
                    text_results = await rg.search(
                        query, str(self.project_root), max_results=10
                    )

                    for text_result in text_results:
                        result = CodeSearchResult(
                            content=text_result["content"],
                            file_path=text_result["file"],
                            line_number=text_result["line"],
                            score=0.6,  # Lower score for text matches
                            result_type="text_fallback",
                            context={"method": "ripgrep"},
                            timestamp=time.time(),
                        )
                        results.append(result)

                except Exception as e:
                    logger.warning(f"Text search fallback failed: {e}")

            # Post-process and rank results
            ranked_results = await self._rank_results(results, query, query_analysis)

            # Cache results
            final_results = ranked_results[:max_results]
            self.semantic_cache[query_hash] = final_results

            # Update stats
            search_time = (time.time() - start_time) * 1000
            self.search_stats["avg_search_time_ms"] = (
                self.search_stats["avg_search_time_ms"]
                * (self.search_stats["total_searches"] - 1)
                + search_time
            ) / self.search_stats["total_searches"]

            if results:
                self.search_stats["semantic_matches"] += 1

            logger.info(
                f"✅ Semantic search completed: {len(final_results)} results in {search_time:.1f}ms"
            )

            return final_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}", exc_info=True)
            # Return empty results rather than crash
            return []

    async def _analyze_coding_query(self, query: str) -> dict[str, Any]:
        """Analyze query to understand coding intent and concepts."""

        query_lower = query.lower()
        analysis = {
            "original_query": query,
            "query_type": "generic",
            "coding_concepts": [],
            "intent": "search",
            "complexity": "simple",
            "domain": "general",
            "expanded_terms": [],
        }

        # Detect query type
        for pattern_type, keywords in self.coding_query_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis["query_type"] = pattern_type
                analysis["coding_concepts"].extend(keywords)
                break

        # Detect intent
        intent_keywords = {
            "find": ["find", "locate", "search", "get", "show"],
            "analyze": ["analyze", "examine", "review", "check"],
            "list": ["list", "show all", "display", "enumerate"],
            "count": ["count", "how many", "number of"],
            "identify": ["identify", "detect", "discover"],
        }

        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis["intent"] = intent
                break

        # Detect complexity
        if len(query.split()) > 5 or any(
            term in query_lower for term in ["all", "complex", "advanced"]
        ):
            analysis["complexity"] = "complex"
        elif len(query.split()) > 3:
            analysis["complexity"] = "medium"

        # Detect domain
        domain_keywords = {
            "trading": [
                "trading",
                "wheel",
                "options",
                "delta",
                "gamma",
                "theta",
                "vega",
            ],
            "ml": ["model", "training", "prediction", "neural", "learning"],
            "data": ["data", "dataframe", "pandas", "analysis", "query"],
            "web": ["api", "request", "response", "client", "server"],
            "async": ["async", "await", "concurrent", "parallel", "thread"],
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis["domain"] = domain
                break

        # Expand query terms
        analysis["expanded_terms"] = await self._expand_query_terms(query, analysis)

        return analysis

    async def _expand_query_terms(
        self, query: str, analysis: dict[str, Any]
    ) -> list[str]:
        """Expand query terms with coding-specific synonyms and related terms."""

        expanded = []
        query_words = query.lower().split()

        # Code-specific expansions
        expansions = {
            "function": ["def", "method", "procedure", "routine", "callable"],
            "class": ["object", "type", "constructor", "instance"],
            "async": ["asynchronous", "await", "concurrent", "coroutine"],
            "import": ["module", "package", "library", "dependency"],
            "error": ["exception", "failure", "bug", "issue"],
            "test": ["testing", "unittest", "pytest", "validation"],
            "data": ["dataframe", "dataset", "table", "records"],
            "optimize": ["optimization", "performance", "speed", "efficient"],
            "pattern": ["design pattern", "architecture", "structure"],
        }

        for word in query_words:
            if word in expansions:
                expanded.extend(expansions[word])

        # Domain-specific expansions
        if analysis["domain"] == "trading":
            trading_expansions = {
                "wheel": ["wheel strategy", "covered call", "cash secured put"],
                "delta": ["hedge ratio", "price sensitivity"],
                "options": ["derivatives", "contracts", "strikes"],
            }
            for word in query_words:
                if word in trading_expansions:
                    expanded.extend(trading_expansions[word])

        return expanded

    def _determine_search_strategy(self, analysis: dict[str, Any]) -> str:
        """Determine optimal search strategy based on query analysis."""

        # Use semantic search for complex queries with coding concepts
        if (
            analysis["complexity"] in ["medium", "complex"]
            and analysis["coding_concepts"]
        ):
            return "semantic"

        # Use hybrid search for specific domains
        if analysis["domain"] in ["trading", "ml", "data"]:
            return "hybrid"

        # Use pattern search for simple structural queries
        if analysis["query_type"] in [
            "function_search",
            "class_search",
            "import_search",
        ]:
            return "pattern"

        # Default to semantic search
        return "semantic"

    async def _semantic_search(
        self, query: str, analysis: dict[str, Any], max_results: int
    ) -> list[CodeSearchResult]:
        """Perform semantic search using code-aware embeddings."""

        # Generate query embedding
        (
            query_embedding,
            query_metadata,
        ) = await self.embedding_pipeline.embed_code_query(query)

        if not self.code_index:
            # Need to build index first
            await self._build_code_index()

        # Search using FAISS if available, otherwise linear search
        if self.faiss_available and self.faiss_index.ntotal > 0:
            return await self._faiss_search(
                query_embedding, query, analysis, max_results
            )
        else:
            return await self._linear_search(
                query_embedding, query, analysis, max_results
            )

    async def _faiss_search(
        self,
        query_embedding: np.ndarray,
        query: str,
        analysis: dict[str, Any],
        max_results: int,
    ) -> list[CodeSearchResult]:
        """Search using FAISS index for fast similarity search."""

        try:
            # Normalize query embedding for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm

            # Search FAISS index
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype("float32"),
                min(
                    max_results * 2, self.faiss_index.ntotal
                ),  # Get more results for filtering
            )

            results = []
            file_paths = list(self.code_index.keys())

            for _i, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False)):
                if idx < len(file_paths):
                    file_path = file_paths[idx]
                    metadata = self.file_metadata.get(file_path, {})

                    # Create result
                    result = CodeSearchResult(
                        content=metadata.get("preview", ""),
                        file_path=file_path,
                        line_number=metadata.get("line_number", 1),
                        score=float(score),
                        result_type="semantic",
                        context=metadata.get("context", {}),
                        timestamp=time.time(),
                        code_metadata=metadata,
                        semantic_score=float(score),
                        code_concepts=analysis["coding_concepts"],
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return await self._linear_search(
                query_embedding, query, analysis, max_results
            )

    async def _linear_search(
        self,
        query_embedding: np.ndarray,
        query: str,
        analysis: dict[str, Any],
        max_results: int,
    ) -> list[CodeSearchResult]:
        """Linear search fallback when FAISS is not available."""

        results = []

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Search through all indexed files
        for file_path, embeddings in self.code_index.items():
            if isinstance(embeddings, list):
                # Multiple embeddings per file (chunks)
                for _i, embedding in enumerate(embeddings):
                    similarity = np.dot(query_embedding, embedding)

                    if similarity > 0.01:  # Lower threshold for relevance
                        metadata = self.file_metadata.get(file_path, {})

                        result = CodeSearchResult(
                            content=metadata.get("preview", ""),
                            file_path=file_path,
                            line_number=metadata.get("line_number", 1),
                            score=float(similarity),
                            result_type="semantic",
                            context=metadata.get("context", {}),
                            timestamp=time.time(),
                            code_metadata=metadata,
                            semantic_score=float(similarity),
                            code_concepts=analysis["coding_concepts"],
                        )
                        results.append(result)

            else:
                # Single embedding per file
                similarity = np.dot(query_embedding, embeddings)

                if similarity > 0.01:  # Lower threshold for relevance
                    metadata = self.file_metadata.get(file_path, {})

                    result = CodeSearchResult(
                        content=metadata.get("preview", ""),
                        file_path=file_path,
                        line_number=metadata.get("line_number", 1),
                        score=float(similarity),
                        result_type="semantic",
                        context=metadata.get("context", {}),
                        timestamp=time.time(),
                        code_metadata=metadata,
                        semantic_score=float(similarity),
                        code_concepts=analysis["coding_concepts"],
                    )
                    results.append(result)

        # Sort by similarity
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    async def _hybrid_search(
        self, query: str, analysis: dict[str, Any], max_results: int
    ) -> list[CodeSearchResult]:
        """Hybrid search combining semantic and pattern matching."""

        # Perform semantic search
        semantic_results = await self._semantic_search(
            query, analysis, max_results // 2
        )

        # Perform pattern search
        pattern_results = await self._pattern_search(query, analysis, max_results // 2)

        # Combine and deduplicate results
        combined_results = {}

        # Add semantic results with higher weight
        for result in semantic_results:
            key = f"{result.file_path}:{result.line_number}"
            if key not in combined_results:
                result.score *= 1.2  # Boost semantic scores
                combined_results[key] = result

        # Add pattern results
        for result in pattern_results:
            key = f"{result.file_path}:{result.line_number}"
            if key not in combined_results:
                combined_results[key] = result
            else:
                # Combine scores
                existing = combined_results[key]
                existing.score = max(existing.score, result.score * 0.8)

        return list(combined_results.values())

    async def _pattern_search(
        self, query: str, analysis: dict[str, Any], max_results: int
    ) -> list[CodeSearchResult]:
        """Pattern-based search for structural code queries."""

        results = []

        # Search for specific patterns based on query type
        if analysis["query_type"] == "function_search":
            results.extend(await self._search_functions(query, max_results))
        elif analysis["query_type"] == "class_search":
            results.extend(await self._search_classes(query, max_results))
        elif analysis["query_type"] == "async_search":
            results.extend(await self._search_async_patterns(query, max_results))
        elif analysis["query_type"] == "import_search":
            results.extend(await self._search_imports(query, max_results))
        else:
            # Generic pattern search
            results.extend(await self._search_generic_patterns(query, max_results))

        return results

    async def _search_functions(
        self, query: str, max_results: int
    ) -> list[CodeSearchResult]:
        """Search for function definitions."""

        results = []

        # Search for function patterns
        for file_path, metadata in self.file_metadata.items():
            functions = metadata.get("functions", [])

            for func_name in functions:
                if query.lower() in func_name.lower():
                    result = CodeSearchResult(
                        content=f"def {func_name}(...)",
                        file_path=file_path,
                        line_number=metadata.get("line_number", 1),
                        score=0.9,
                        result_type="function",
                        context={"function_name": func_name},
                        timestamp=time.time(),
                        code_metadata=metadata,
                    )
                    results.append(result)

        return results[:max_results]

    async def _search_classes(
        self, query: str, max_results: int
    ) -> list[CodeSearchResult]:
        """Search for class definitions."""

        results = []

        for file_path, metadata in self.file_metadata.items():
            classes = metadata.get("classes", [])

            for class_name in classes:
                if query.lower() in class_name.lower():
                    result = CodeSearchResult(
                        content=f"class {class_name}:",
                        file_path=file_path,
                        line_number=metadata.get("line_number", 1),
                        score=0.9,
                        result_type="class",
                        context={"class_name": class_name},
                        timestamp=time.time(),
                        code_metadata=metadata,
                    )
                    results.append(result)

        return results[:max_results]

    async def _search_async_patterns(
        self, query: str, max_results: int
    ) -> list[CodeSearchResult]:
        """Search for async/await patterns."""

        results = []

        for file_path, metadata in self.file_metadata.items():
            if "async" in metadata.get("patterns", []):
                result = CodeSearchResult(
                    content=metadata.get("preview", ""),
                    file_path=file_path,
                    line_number=metadata.get("line_number", 1),
                    score=0.8,
                    result_type="async_pattern",
                    context={"has_async": True},
                    timestamp=time.time(),
                    code_metadata=metadata,
                )
                results.append(result)

        return results[:max_results]

    async def _search_imports(
        self, query: str, max_results: int
    ) -> list[CodeSearchResult]:
        """Search for import statements."""

        results = []

        for file_path, metadata in self.file_metadata.items():
            imports = metadata.get("imports", [])

            for import_stmt in imports:
                if query.lower() in import_stmt.lower():
                    result = CodeSearchResult(
                        content=import_stmt,
                        file_path=file_path,
                        line_number=metadata.get("line_number", 1),
                        score=0.85,
                        result_type="import",
                        context={"import_statement": import_stmt},
                        timestamp=time.time(),
                        code_metadata=metadata,
                    )
                    results.append(result)

        return results[:max_results]

    async def _search_generic_patterns(
        self, query: str, max_results: int
    ) -> list[CodeSearchResult]:
        """Generic pattern search for fallback."""

        results = []
        query_words = query.lower().split()

        for file_path, metadata in self.file_metadata.items():
            match_score = 0.0
            matched_elements = []

            # Check file path for matches
            file_path_lower = file_path.lower()
            for word in query_words:
                if word in file_path_lower:
                    match_score += 0.3
                    matched_elements.append(f"path:{word}")

            # Check keywords in metadata
            keywords = metadata.get("keywords", [])
            for keyword in keywords:
                for word in query_words:
                    if word in keyword.lower():
                        match_score += 0.2
                        matched_elements.append(f"keyword:{keyword}")

            # Check functions
            functions = metadata.get("functions", [])
            for func_name in functions:
                for word in query_words:
                    if word in func_name.lower():
                        match_score += 0.4
                        matched_elements.append(f"function:{func_name}")

            # Check classes
            classes = metadata.get("classes", [])
            for class_name in classes:
                for word in query_words:
                    if word in class_name.lower():
                        match_score += 0.4
                        matched_elements.append(f"class:{class_name}")

            # Check imports
            imports = metadata.get("imports", [])
            for import_stmt in imports:
                for word in query_words:
                    if word in import_stmt.lower():
                        match_score += 0.3
                        matched_elements.append(f"import:{import_stmt}")

            # Create result if we have matches
            if match_score > 0:
                result = CodeSearchResult(
                    content=metadata.get("preview", ""),
                    file_path=file_path,
                    line_number=metadata.get("line_number", 1),
                    score=min(1.0, match_score),
                    result_type="generic",
                    context={
                        "matched_elements": matched_elements,
                        "match_score": match_score,
                    },
                    timestamp=time.time(),
                    code_metadata=metadata,
                )
                results.append(result)

        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    async def _rank_results(
        self, results: list[CodeSearchResult], query: str, analysis: dict[str, Any]
    ) -> list[CodeSearchResult]:
        """Rank results based on code-specific relevance scoring."""

        for result in results:
            # Calculate relevance score based on multiple factors
            relevance_score = result.score  # Base score from similarity

            # Boost score based on query type match
            if (
                analysis["query_type"] == "function_search"
                and result.result_type == "function"
                or analysis["query_type"] == "class_search"
                and result.result_type == "class"
            ):
                relevance_score *= 1.5
            elif (
                analysis["query_type"] == "async_search"
                and result.result_type == "async_pattern"
            ):
                relevance_score *= 1.4

            # Boost score based on domain match
            if analysis["domain"] != "general":
                domain_keywords = {
                    "trading": ["wheel", "options", "delta", "trading"],
                    "ml": ["model", "training", "neural"],
                    "data": ["dataframe", "pandas", "data"],
                }

                if analysis["domain"] in domain_keywords:
                    keywords = domain_keywords[analysis["domain"]]
                    if any(keyword in result.content.lower() for keyword in keywords):
                        relevance_score *= 1.3

            # Boost score based on file type
            if result.file_path.endswith(".py"):
                relevance_score *= 1.1

            # Penalty for test files if not specifically searching for tests
            if "test" not in query.lower() and "test" in result.file_path.lower():
                relevance_score *= 0.8

            result.relevance_score = relevance_score

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score or x.score, reverse=True)

        return results

    async def _build_code_index(self):
        """Build the code index by scanning and embedding all Python files."""

        logger.info("🔍 Building code index for semantic search...")
        start_time = time.time()

        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))

        # Process files in parallel
        tasks = []
        for file_path in python_files:
            if (
                file_path.is_file() and file_path.stat().st_size < 5 * 1024 * 1024
            ):  # Skip large files
                task = asyncio.create_task(self._index_file(file_path))
                tasks.append(task)

        # Wait for all files to be processed
        await asyncio.gather(*tasks, return_exceptions=True)

        # Build FAISS index if available
        if self.faiss_available:
            await self._build_faiss_index()

        build_time = time.time() - start_time
        logger.info(
            f"✅ Code index built: {len(self.code_index)} files in {build_time:.1f}s"
        )

    async def _index_file(self, file_path: Path):
        """Index a single file for semantic search."""

        try:
            # Read file content
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Generate embedding and metadata
            embedding, metadata = await self.embedding_pipeline.embed_code_content(
                content, str(file_path), {}
            )

            # Store in index
            relative_path = str(file_path.relative_to(self.project_root))
            self.code_index[relative_path] = embedding

            # Enhance metadata with preview
            lines = content.split("\n")
            preview_lines = [line for line in lines[:10] if line.strip()]
            metadata["preview"] = "\n".join(preview_lines[:5])

            self.file_metadata[relative_path] = metadata

        except Exception as e:
            logger.warning(f"Failed to index {file_path}: {e}")

    async def _build_faiss_index(self):
        """Build FAISS index from code embeddings."""

        if not self.faiss_available or not self.code_index:
            return

        # Collect all embeddings
        embeddings = []
        for embedding in self.code_index.values():
            embeddings.append(embedding)

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_array = embeddings_array / norms

        # Add to FAISS index
        self.faiss_index.add(embeddings_array)

        logger.info(f"✅ FAISS index built with {self.faiss_index.ntotal} embeddings")

    def get_search_stats(self) -> dict[str, Any]:
        """Get search performance statistics."""

        cache_hit_rate = (
            self.search_stats["cache_hits"]
            / max(self.search_stats["total_searches"], 1)
        ) * 100

        semantic_match_rate = (
            self.search_stats["semantic_matches"]
            / max(self.search_stats["total_searches"], 1)
        ) * 100

        return {
            "total_searches": self.search_stats["total_searches"],
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "avg_search_time_ms": f"{self.search_stats['avg_search_time_ms']:.1f}ms",
            "semantic_match_rate": f"{semantic_match_rate:.1f}%",
            "indexed_files": len(self.code_index),
            "faiss_available": self.faiss_available,
            "cache_size": len(self.semantic_cache),
        }

    async def warm_up_cache(self, common_queries: list[str]):
        """Warm up the search cache with common coding queries."""

        logger.info(f"🔥 Warming up cache with {len(common_queries)} common queries...")

        for query in common_queries:
            await self.search(query, max_results=10)

        logger.info("✅ Cache warm-up complete")

    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)


# Default common coding queries for cache warm-up
DEFAULT_CODING_QUERIES = [
    "async functions",
    "class inheritance",
    "error handling",
    "import statements",
    "data processing",
    "wheel strategy",
    "options calculation",
    "database queries",
    "API endpoints",
    "performance optimization",
    "test functions",
    "configuration setup",
    "logging statements",
    "exception handling",
    "async await patterns",
]
