#!/usr/bin/env python3
"""
Einstein Fallback Implementations - Working alternatives when Einstein fails.

This module provides robust fallback implementations for Einstein's semantic
search and analysis capabilities. These implementations work when the full
Einstein system fails to initialize or encounters errors.

Key Features:
- Lightweight semantic search using TF-IDF and keyword matching
- Fast file indexing without complex dependencies
- Code analysis using AST parsing
- Trading domain-aware search patterns
- Works without MLX, GPU acceleration, or complex ML models
"""

import ast
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SearchDocument:
    """Document representation for search indexing."""

    path: str
    content: str
    tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    size: int = 0
    modified_time: float = 0

    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()
        if not self.size:
            self.size = len(self.content)
        if not self.modified_time:
            try:
                self.modified_time = os.path.getmtime(self.path)
            except:
                self.modified_time = time.time()


@dataclass
class SearchResult:
    """Search result with relevance scoring."""

    document: SearchDocument
    score: float
    matches: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    line_numbers: list[int] = field(default_factory=list)


class TextTokenizer:
    """Simple but effective text tokenization for code and documentation."""

    def __init__(self):
        # Common programming language keywords and tokens
        self.code_keywords = {
            "def",
            "class",
            "import",
            "from",
            "return",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "try",
            "except",
            "finally",
            "with",
            "as",
            "async",
            "await",
            "lambda",
            "yield",
            "pass",
            "break",
            "continue",
            "global",
            "nonlocal",
            "assert",
            "del",
            "raise",
            "True",
            "False",
            "None",
        }

        # Trading and finance domain terms
        self.domain_keywords = {
            "option",
            "options",
            "strike",
            "expiration",
            "premium",
            "volatility",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
            "wheel",
            "strategy",
            "portfolio",
            "position",
            "risk",
            "profit",
            "loss",
            "margin",
            "underlying",
            "contract",
            "exercise",
            "assignment",
            "liquidity",
            "bid",
            "ask",
            "spread",
            "volume",
            "iv",
            "implied",
            "realized",
            "greek",
            "greeks",
            "pricing",
            "model",
            "black",
            "scholes",
            "binomial",
            "monte",
            "carlo",
            "simulation",
            "backtest",
            "paper",
            "trading",
            "trade",
            "buy",
            "sell",
            "long",
            "short",
            "call",
            "put",
        }

        # Stop words to filter out
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "under",
            "over",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
        }

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into meaningful terms."""
        # Convert to lowercase
        text = text.lower()

        # Extract different types of tokens
        tokens = []

        # 1. Code identifiers (camelCase, snake_case, etc.)
        code_pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
        code_tokens = re.findall(code_pattern, text)
        tokens.extend(code_tokens)

        # 2. Extract compound terms (split camelCase and snake_case)
        for token in code_tokens:
            # Split camelCase
            camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", token)
            tokens.extend([part.lower() for part in camel_parts if len(part) > 1])

            # Split snake_case
            if "_" in token:
                snake_parts = token.split("_")
                tokens.extend([part for part in snake_parts if len(part) > 1])

        # 3. Numbers and numeric patterns
        numeric_pattern = r"\b\d+\.?\d*\b"
        numbers = re.findall(numeric_pattern, text)
        tokens.extend(numbers)

        # 4. File extensions and paths
        file_pattern = r"\.[a-zA-Z]{2,4}\b"
        extensions = re.findall(file_pattern, text)
        tokens.extend([ext[1:] for ext in extensions])  # Remove the dot

        # Filter and clean tokens
        filtered_tokens = []
        for token in tokens:
            token = token.lower().strip()
            if (
                len(token) >= 2
                and token not in self.stop_words
                and not token.isspace()
                and re.match(r"^[a-zA-Z0-9_\.]+$", token)
            ):
                filtered_tokens.append(token)

        # Add domain-specific weight to important terms
        weighted_tokens = []
        for token in filtered_tokens:
            weighted_tokens.append(token)

            # Add extra weight to domain keywords
            if token in self.domain_keywords:
                weighted_tokens.extend([token] * 2)  # Triple weight
            elif token in self.code_keywords:
                weighted_tokens.append(token)  # Double weight

        return weighted_tokens


class TFIDFIndex:
    """TF-IDF based search index for code and documentation."""

    def __init__(self):
        self.documents: dict[str, SearchDocument] = {}
        self.tokenizer = TextTokenizer()
        self.term_frequencies: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.document_frequencies: dict[str, int] = defaultdict(int)
        self.total_documents = 0

    def add_document(self, doc: SearchDocument):
        """Add document to index."""
        # Tokenize content
        doc.tokens = self.tokenizer.tokenize(doc.content)

        # Update document storage
        self.documents[doc.path] = doc

        # Update term frequencies for this document
        term_counts = Counter(doc.tokens)
        self.term_frequencies[doc.path] = dict(term_counts)

        # Update document frequencies
        unique_terms = set(doc.tokens)
        for term in unique_terms:
            self.document_frequencies[term] += 1

        self.total_documents += 1

    def remove_document(self, path: str):
        """Remove document from index."""
        if path in self.documents:
            doc = self.documents[path]

            # Update document frequencies
            unique_terms = set(doc.tokens)
            for term in unique_terms:
                self.document_frequencies[term] -= 1
                if self.document_frequencies[term] <= 0:
                    del self.document_frequencies[term]

            # Remove from storage
            del self.documents[path]
            del self.term_frequencies[path]
            self.total_documents -= 1

    def search(self, query: str, max_results: int = 20) -> list[SearchResult]:
        """Search index using TF-IDF scoring."""
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Calculate scores for each document
        scores = {}
        for doc_path, doc in self.documents.items():
            score = self._calculate_tfidf_score(query_tokens, doc_path)
            if score > 0:
                scores[doc_path] = score

        # Sort by score and return top results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_path, score in sorted_results[:max_results]:
            doc = self.documents[doc_path]

            # Find matches and snippets
            matches = self._find_matches(query_tokens, doc)
            snippets = self._extract_snippets(query_tokens, doc)

            results.append(
                SearchResult(
                    document=doc, score=score, matches=matches, snippets=snippets
                )
            )

        return results

    def _calculate_tfidf_score(self, query_tokens: list[str], doc_path: str) -> float:
        """Calculate TF-IDF score for query against document."""
        if doc_path not in self.term_frequencies:
            return 0.0

        doc_tf = self.term_frequencies[doc_path]
        doc_length = sum(doc_tf.values())

        if doc_length == 0:
            return 0.0

        score = 0.0
        query_term_counts = Counter(query_tokens)

        for term, query_count in query_term_counts.items():
            if term in doc_tf:
                # Term frequency (normalized)
                tf = doc_tf[term] / doc_length

                # Inverse document frequency
                if term in self.document_frequencies:
                    df = self.document_frequencies[term]
                    idf = math.log(self.total_documents / (df + 1)) + 1
                else:
                    idf = 1

                # Query term weight
                query_weight = query_count / len(query_tokens)

                score += tf * idf * query_weight

        return score

    def _find_matches(self, query_tokens: list[str], doc: SearchDocument) -> list[str]:
        """Find matching terms in document."""
        matches = []
        doc_tokens_set = set(doc.tokens)

        for token in query_tokens:
            if token in doc_tokens_set:
                matches.append(token)

        return list(set(matches))  # Remove duplicates

    def _extract_snippets(
        self, query_tokens: list[str], doc: SearchDocument, snippet_length: int = 150
    ) -> list[str]:
        """Extract relevant snippets from document."""
        content = doc.content
        lines = content.split("\n")
        snippets = []

        query_pattern = "|".join(re.escape(token) for token in query_tokens)

        for i, line in enumerate(lines):
            if re.search(query_pattern, line, re.IGNORECASE):
                # Extract snippet around the match
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                snippet_lines = lines[start:end]

                snippet = "\n".join(snippet_lines)
                if len(snippet) > snippet_length:
                    snippet = snippet[:snippet_length] + "..."

                snippets.append(snippet)

                if len(snippets) >= 3:  # Limit snippets per document
                    break

        return snippets


class CodeAnalyzer:
    """Code analysis using AST parsing for Python files."""

    def __init__(self):
        self.analysis_cache = {}

    def analyze_file(self, file_path: str) -> dict[str, Any]:
        """Analyze Python file structure."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, filename=file_path)

            analysis = {
                "file_path": file_path,
                "classes": [],
                "functions": [],
                "imports": [],
                "constants": [],
                "complexity_score": 0,
                "lines_of_code": len(content.split("\n")),
                "docstrings": [],
            }

            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis["classes"].append(
                        {
                            "name": node.name,
                            "line": node.lineno,
                            "methods": [
                                n.name
                                for n in node.body
                                if isinstance(n, ast.FunctionDef)
                            ],
                            "docstring": ast.get_docstring(node),
                        }
                    )

                elif isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_complexity(node)
                    analysis["functions"].append(
                        {
                            "name": node.name,
                            "line": node.lineno,
                            "args": [arg.arg for arg in node.args.args],
                            "complexity": complexity,
                            "docstring": ast.get_docstring(node),
                        }
                    )
                    analysis["complexity_score"] += complexity

                elif isinstance(node, ast.Import | ast.ImportFrom):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            analysis["imports"].append(f"{module}.{alias.name}")

                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            analysis["constants"].append(target.id)

            # Extract docstrings
            docstring = ast.get_docstring(tree)
            if docstring:
                analysis["docstrings"].append(docstring)

            return analysis

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return {"file_path": file_path, "error": str(e), "complexity_score": 0}

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(
                child,
                ast.If
                | ast.While
                | ast.For
                | ast.With
                | ast.ExceptHandler
                | (ast.And | ast.Or),
            ):
                complexity += 1

        return complexity


class EinsteinFallback:
    """Fallback implementation for Einstein semantic search."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.index = TFIDFIndex()
        self.code_analyzer = CodeAnalyzer()
        self.indexed_files = set()
        self.last_index_update = 0
        self.index_cache_file = self.project_root / ".einstein_fallback_cache.json"

        # File patterns to index
        self.include_patterns = [
            "*.py",
            "*.md",
            "*.txt",
            "*.rst",
            "*.yaml",
            "*.yml",
            "*.json",
            "*.sql",
            "*.sh",
            "*.toml",
        ]

        # Directories to exclude
        self.exclude_dirs = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            ".tox",
            "build",
            "dist",
            ".eggs",
            "*.egg-info",
        }

    async def initialize(self):
        """Initialize the fallback search system."""
        try:
            # Load cached index if available
            if self.index_cache_file.exists():
                self._load_index_cache()

            # Build or update index
            await self._build_index()

            logger.info(
                f"Einstein fallback initialized with {len(self.indexed_files)} files"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Einstein fallback: {e}")

    async def search(
        self, query: str, max_results: int = 10, **kwargs
    ) -> list[dict[str, Any]]:
        """Fallback semantic search implementation."""
        try:
            # Update index if needed
            await self._update_index_if_needed()

            # Perform TF-IDF search
            results = self.index.search(query, max_results)

            # Convert to Einstein-compatible format
            einstein_results = []
            for result in results:
                einstein_results.append(
                    {
                        "file_path": result.document.path,
                        "relevance_score": result.score,
                        "matches": result.matches,
                        "snippets": result.snippets,
                        "metadata": result.document.metadata,
                        "search_type": "fallback_tfidf",
                    }
                )

            return einstein_results

        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    async def search_code(
        self, query: str, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Code-specific search with structure analysis."""
        # Regular search first
        results = await self.search(query, max_results * 2)

        # Filter and enhance code results
        code_results = []
        for result in results:
            if result["file_path"].endswith(".py"):
                # Add code analysis
                analysis = self.code_analyzer.analyze_file(result["file_path"])

                # Check if query matches code structures
                code_relevance = self._calculate_code_relevance(query, analysis)

                if code_relevance > 0:
                    result["code_analysis"] = analysis
                    result["code_relevance"] = code_relevance
                    result["relevance_score"] += code_relevance
                    code_results.append(result)

        # Re-sort by enhanced relevance
        code_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return code_results[:max_results]

    def _calculate_code_relevance(self, query: str, analysis: dict[str, Any]) -> float:
        """Calculate relevance based on code structure."""
        query_lower = query.lower()
        relevance = 0.0

        # Check class names
        for cls in analysis.get("classes", []):
            if query_lower in cls["name"].lower():
                relevance += 2.0

        # Check function names
        for func in analysis.get("functions", []):
            if query_lower in func["name"].lower():
                relevance += 1.5

        # Check imports
        for imp in analysis.get("imports", []):
            if query_lower in imp.lower():
                relevance += 1.0

        # Check constants
        for const in analysis.get("constants", []):
            if query_lower in const.lower():
                relevance += 1.0

        return relevance

    async def _build_index(self):
        """Build search index from project files."""
        start_time = time.time()

        for file_path in self._find_indexable_files():
            try:
                if file_path not in self.indexed_files:
                    await self._index_file(file_path)
                    self.indexed_files.add(file_path)
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")

        # Save cache
        self._save_index_cache()

        self.last_index_update = time.time()
        build_time = time.time() - start_time

        logger.info(
            f"Index built in {build_time:.2f}s, {len(self.indexed_files)} files indexed"
        )

    async def _index_file(self, file_path: Path):
        """Index a single file."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Create document
            doc = SearchDocument(
                path=str(file_path),
                content=content,
                metadata={
                    "file_type": file_path.suffix[1:]
                    if file_path.suffix
                    else "unknown",
                    "size": len(content),
                    "relative_path": str(file_path.relative_to(self.project_root)),
                },
            )

            # Add code analysis for Python files
            if file_path.suffix == ".py":
                analysis = self.code_analyzer.analyze_file(str(file_path))
                doc.metadata["code_analysis"] = analysis

            self.index.add_document(doc)

        except Exception as e:
            logger.warning(f"Failed to index file {file_path}: {e}")

    def _find_indexable_files(self) -> list[Path]:
        """Find files that should be indexed."""
        files = []

        for pattern in self.include_patterns:
            files.extend(self.project_root.rglob(pattern))

        # Filter out excluded directories and files
        filtered_files = []
        for file_path in files:
            # Skip if in excluded directory
            if any(excluded in file_path.parts for excluded in self.exclude_dirs):
                continue

            # Skip if file is too large (>1MB)
            try:
                if file_path.stat().st_size > 1024 * 1024:
                    continue
            except:
                continue

            # Skip binary files
            if self._is_binary_file(file_path):
                continue

            filtered_files.append(file_path)

        return filtered_files

    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary."""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                return b"\0" in chunk
        except:
            return True

    async def _update_index_if_needed(self):
        """Update index if files have changed."""
        # Check if enough time has passed
        if time.time() - self.last_index_update < 300:  # 5 minutes
            return

        # Check for new or modified files
        current_files = set(self._find_indexable_files())

        # Remove deleted files
        deleted_files = self.indexed_files - set(str(f) for f in current_files)
        for file_path in deleted_files:
            self.index.remove_document(file_path)
            self.indexed_files.discard(file_path)

        # Add new files
        new_files = current_files - set(Path(f) for f in self.indexed_files)
        for file_path in new_files:
            await self._index_file(file_path)
            self.indexed_files.add(str(file_path))

        self.last_index_update = time.time()

        if deleted_files or new_files:
            self._save_index_cache()
            logger.info(
                f"Index updated: +{len(new_files)} files, -{len(deleted_files)} files"
            )

    def _save_index_cache(self):
        """Save index cache to disk."""
        try:
            cache_data = {
                "indexed_files": list(self.indexed_files),
                "last_update": self.last_index_update,
                "total_documents": self.index.total_documents,
            }

            with open(self.index_cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save index cache: {e}")

    def _load_index_cache(self):
        """Load index cache from disk."""
        try:
            with open(self.index_cache_file) as f:
                cache_data = json.load(f)

            self.indexed_files = set(cache_data.get("indexed_files", []))
            self.last_index_update = cache_data.get("last_update", 0)

            logger.info(f"Loaded index cache with {len(self.indexed_files)} files")

        except Exception as e:
            logger.warning(f"Failed to load index cache: {e}")

    async def shutdown(self):
        """Shutdown fallback system."""
        self._save_index_cache()
        logger.info("Einstein fallback shutdown complete")


# Factory function for easy integration
def create_einstein_fallback(project_root: str = ".") -> EinsteinFallback:
    """Create Einstein fallback instance."""
    return EinsteinFallback(project_root)


# Testing CLI
if __name__ == "__main__":
    import sys

    async def main():
        if len(sys.argv) < 2:
            print("Einstein Fallback Search")
            print(
                "Usage: python bolt_einstein_fallbacks.py 'search query' [project_root]"
            )
            return

        query = sys.argv[1]
        project_root = sys.argv[2] if len(sys.argv) > 2 else "."

        print("🔍 Einstein Fallback Search")
        print("=" * 50)
        print(f"Query: {query}")
        print(f"Project: {project_root}")

        # Initialize fallback
        fallback = EinsteinFallback(project_root)
        await fallback.initialize()

        # Perform search
        start_time = time.time()
        results = await fallback.search(query, max_results=10)
        search_time = time.time() - start_time

        print(f"\nFound {len(results)} results in {search_time:.2f}s")
        print("-" * 50)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['file_path']}")
            print(f"   Score: {result['relevance_score']:.3f}")
            print(f"   Matches: {', '.join(result['matches'])}")

            if result["snippets"]:
                print(f"   Snippet: {result['snippets'][0][:100]}...")
            print()

        await fallback.shutdown()

    asyncio.run(main())
