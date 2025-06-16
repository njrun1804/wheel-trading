#!/usr/bin/env python3
"""
Intelligent Query Autocomplete and Suggestion System

Features:
1. Autocomplete based on previously successful queries
2. Intelligent suggestions based on current codebase content 
3. Query refinement suggestions when searches return few results
4. Context-aware suggestions based on recently viewed files
5. Integration with both Einstein and Bolt query patterns

Performance: Optimized for M4 Pro with hardware acceleration
"""

import asyncio
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Try to import accelerated tools
try:
    from src.unity_wheel.accelerated_tools.dependency_graph_turbo import (
        get_dependency_graph,
    )
    from src.unity_wheel.accelerated_tools.python_analysis_turbo import (
        get_python_analyzer,
    )
    from src.unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo

    HAS_ACCELERATED_TOOLS = True
except (ImportError, SyntaxError, Exception):
    HAS_ACCELERATED_TOOLS = False


@dataclass
class QuerySuggestion:
    """A query suggestion with metadata."""

    text: str
    confidence: float
    category: str  # 'historical', 'codebase', 'refinement', 'contextual'
    reasoning: str
    estimated_results: int = 0
    query_type: str = "search"  # 'search', 'analysis', 'optimization'


@dataclass
class QueryHistory:
    """Historical query with success metrics."""

    query: str
    timestamp: float
    system_used: str  # 'einstein' or 'bolt'
    success: bool
    result_count: int
    execution_time: float
    user_rating: int | None = None  # 1-5 star rating


@dataclass
class ContextualInfo:
    """Contextual information about recently viewed/modified files."""

    file_path: str
    last_accessed: float
    access_count: int
    file_type: str
    primary_classes: list[str]
    primary_functions: list[str]


class CodebaseAnalyzer:
    """Analyzes codebase to extract meaningful symbols for suggestions."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.symbols_cache = {}
        self.last_scan = 0
        self.scan_interval = 300  # 5 minutes

    async def get_symbols(self) -> dict[str, set[str]]:
        """Get all symbols from codebase (classes, functions, etc.)."""

        current_time = time.time()
        if current_time - self.last_scan < self.scan_interval and self.symbols_cache:
            return self.symbols_cache

        symbols = {
            "classes": set(),
            "functions": set(),
            "modules": set(),
            "variables": set(),
            "imports": set(),
            "keywords": set(),
        }

        if HAS_ACCELERATED_TOOLS:
            await self._extract_symbols_accelerated(symbols)
        else:
            await self._extract_symbols_basic(symbols)

        # Add common trading-specific terms
        symbols["keywords"].update(
            {
                "WheelStrategy",
                "options",
                "trading",
                "risk",
                "portfolio",
                "databento",
                "unity",
                "greeks",
                "delta",
                "gamma",
                "theta",
                "volatility",
                "backtest",
                "optimization",
                "performance",
            }
        )

        self.symbols_cache = symbols
        self.last_scan = current_time
        return symbols

    async def _extract_symbols_accelerated(self, symbols: dict[str, set[str]]) -> None:
        """Extract symbols using accelerated tools."""
        try:
            analyzer = get_python_analyzer()

            # Get all Python files
            python_files = list(self.project_root.rglob("*.py"))

            # Analyze files in parallel
            tasks = []
            for file_path in python_files[:100]:  # Limit to avoid overload
                if not any(
                    skip in str(file_path) for skip in ["__pycache__", ".git", "test"]
                ):
                    tasks.append(
                        self._analyze_file_accelerated(analyzer, file_path, symbols)
                    )

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            print(f"Warning: Accelerated symbol extraction failed: {e}")
            await self._extract_symbols_basic(symbols)

    async def _analyze_file_accelerated(
        self, analyzer, file_path: Path, symbols: dict[str, set[str]]
    ) -> None:
        """Analyze a single file for symbols."""
        try:
            analysis = await analyzer.analyze_file(str(file_path))

            if analysis and "symbols" in analysis:
                file_symbols = analysis["symbols"]
                symbols["classes"].update(file_symbols.get("classes", []))
                symbols["functions"].update(file_symbols.get("functions", []))
                symbols["imports"].update(file_symbols.get("imports", []))
                symbols["variables"].update(file_symbols.get("variables", []))

            # Add module name
            module_name = file_path.stem
            if module_name not in ["__init__", "__pycache__"]:
                symbols["modules"].add(module_name)

        except Exception:
            pass  # Skip files that can't be analyzed

    async def _extract_symbols_basic(self, symbols: dict[str, set[str]]) -> None:
        """Extract symbols using basic AST parsing."""

        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files[:50]:  # Limit for performance
            if any(skip in str(file_path) for skip in ["__pycache__", ".git"]):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Basic regex patterns for quick extraction
                symbols["classes"].update(re.findall(r"class\s+(\w+)", content))
                symbols["functions"].update(re.findall(r"def\s+(\w+)", content))
                symbols["imports"].update(
                    re.findall(r"from\s+[\w.]+\s+import\s+(\w+)", content)
                )
                symbols["imports"].update(re.findall(r"import\s+(\w+)", content))

                # Module name
                module_name = file_path.stem
                if module_name not in ["__init__", "__pycache__"]:
                    symbols["modules"].add(module_name)

            except Exception:
                continue


class QueryIntelligence:
    """Main intelligence system for query suggestions and autocomplete."""

    def __init__(self, project_root: Path = None, db_path: Path = None):
        self.project_root = project_root or Path.cwd()
        self.db_path = db_path or (self.project_root / ".query_intelligence.db")
        self.codebase_analyzer = CodebaseAnalyzer(self.project_root)
        self.context_files = {}  # file_path -> ContextualInfo
        self.init_db()

    def init_db(self) -> None:
        """Initialize SQLite database for query history."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    system_used TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    result_count INTEGER NOT NULL,
                    execution_time REAL NOT NULL,
                    user_rating INTEGER
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contextual_info (
                    file_path TEXT PRIMARY KEY,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    file_type TEXT NOT NULL,
                    primary_classes TEXT,  -- JSON
                    primary_functions TEXT -- JSON
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_timestamp 
                ON query_history(timestamp DESC)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_success 
                ON query_history(success, result_count DESC)
            """
            )

    async def get_suggestions(
        self, partial_query: str, max_suggestions: int = 10
    ) -> list[QuerySuggestion]:
        """Get intelligent suggestions for a partial query."""

        suggestions = []
        partial_lower = partial_query.lower().strip()

        if len(partial_lower) < 2:
            # Show popular recent queries for very short input
            suggestions.extend(await self._get_popular_queries(max_suggestions // 2))
        else:
            # Get suggestions from different sources
            tasks = [
                self._get_historical_suggestions(partial_lower, max_suggestions // 4),
                self._get_codebase_suggestions(partial_lower, max_suggestions // 4),
                self._get_contextual_suggestions(partial_lower, max_suggestions // 4),
                self._get_pattern_suggestions(partial_lower, max_suggestions // 4),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    suggestions.extend(result)

        # Deduplicate and sort by confidence
        seen = set()
        unique_suggestions = []

        for suggestion in suggestions:
            if suggestion.text not in seen:
                seen.add(suggestion.text)
                unique_suggestions.append(suggestion)

        # Sort by confidence and return top results
        unique_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return unique_suggestions[:max_suggestions]

    async def _get_popular_queries(self, limit: int) -> list[QuerySuggestion]:
        """Get popular recent queries."""

        suggestions = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT query, COUNT(*) as frequency, AVG(result_count) as avg_results
                    FROM query_history 
                    WHERE success = 1 AND timestamp > ? 
                    GROUP BY query 
                    ORDER BY frequency DESC, avg_results DESC
                    LIMIT ?
                """,
                    (time.time() - 86400 * 7, limit),
                )  # Last 7 days

                for query, freq, avg_results in cursor.fetchall():
                    suggestions.append(
                        QuerySuggestion(
                            text=query,
                            confidence=min(0.9, 0.5 + freq * 0.1),
                            category="historical",
                            reasoning=f"Popular query (used {freq} times)",
                            estimated_results=int(avg_results),
                            query_type=self._classify_query_type(query),
                        )
                    )
        except Exception:
            pass

        return suggestions

    async def _get_historical_suggestions(
        self, partial: str, limit: int
    ) -> list[QuerySuggestion]:
        """Get suggestions based on query history."""

        suggestions = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT query, result_count, execution_time, user_rating
                    FROM query_history 
                    WHERE query LIKE ? AND success = 1
                    ORDER BY 
                        CASE WHEN user_rating IS NOT NULL THEN user_rating ELSE 3 END DESC,
                        result_count DESC, 
                        execution_time ASC
                    LIMIT ?
                """,
                    (f"%{partial}%", limit * 2),
                )

                for query, result_count, exec_time, rating in cursor.fetchall():
                    # Calculate confidence based on rating, results, and speed
                    base_confidence = 0.6
                    if rating:
                        base_confidence += rating * 0.1
                    if result_count > 0:
                        base_confidence += min(0.2, result_count * 0.02)
                    if exec_time < 1.0:
                        base_confidence += 0.1

                    suggestions.append(
                        QuerySuggestion(
                            text=query,
                            confidence=min(0.95, base_confidence),
                            category="historical",
                            reasoning=f"Previously successful ({result_count} results)",
                            estimated_results=result_count,
                            query_type=self._classify_query_type(query),
                        )
                    )
        except Exception:
            pass

        return suggestions[:limit]

    async def _get_codebase_suggestions(
        self, partial: str, limit: int
    ) -> list[QuerySuggestion]:
        """Get suggestions based on codebase symbols."""

        suggestions = []

        try:
            symbols = await self.codebase_analyzer.get_symbols()

            # Search for matching symbols
            matches = []

            for symbol_type, symbol_set in symbols.items():
                for symbol in symbol_set:
                    if partial in symbol.lower():
                        similarity = self._calculate_similarity(partial, symbol.lower())
                        matches.append((symbol, symbol_type, similarity))

            # Sort by similarity and take top matches
            matches.sort(key=lambda x: x[2], reverse=True)

            for symbol, symbol_type, similarity in matches[:limit]:
                # Generate appropriate query based on symbol type
                if symbol_type == "classes":
                    query_text = f"find class {symbol}"
                elif symbol_type == "functions":
                    query_text = f"find function {symbol}"
                elif symbol_type == "modules":
                    query_text = f"show {symbol}.py"
                else:
                    query_text = f"search {symbol}"

                suggestions.append(
                    QuerySuggestion(
                        text=query_text,
                        confidence=0.7 + similarity * 0.2,
                        category="codebase",
                        reasoning=f"Found {symbol_type[:-1]} in codebase",
                        estimated_results=5,
                        query_type="search",
                    )
                )

        except Exception:
            pass

        return suggestions

    async def _get_contextual_suggestions(
        self, partial: str, limit: int
    ) -> list[QuerySuggestion]:
        """Get suggestions based on recently accessed files."""

        suggestions = []

        # Get recently accessed files
        recent_files = sorted(
            self.context_files.values(), key=lambda x: x.last_accessed, reverse=True
        )[:10]

        for context in recent_files:
            # Check if partial matches file path or symbols
            file_matches = partial in context.file_path.lower()
            symbol_matches = any(
                partial in symbol.lower()
                for symbol in context.primary_classes + context.primary_functions
            )

            if file_matches or symbol_matches:
                confidence = 0.6
                if file_matches:
                    confidence += 0.2
                if symbol_matches:
                    confidence += 0.2

                # Boost confidence for recently accessed files
                recency_boost = min(0.1, (time.time() - context.last_accessed) / 3600)
                confidence += recency_boost

                query_text = f"show {Path(context.file_path).name}"

                suggestions.append(
                    QuerySuggestion(
                        text=query_text,
                        confidence=min(0.9, confidence),
                        category="contextual",
                        reasoning="Recently accessed file",
                        estimated_results=1,
                        query_type="search",
                    )
                )

        return suggestions[:limit]

    async def _get_pattern_suggestions(
        self, partial: str, limit: int
    ) -> list[QuerySuggestion]:
        """Get suggestions based on common query patterns."""

        suggestions = []

        # Common query patterns for trading system
        patterns = {
            "find": ["WheelStrategy", "options", "risk", "portfolio", "databento"],
            "show": ["wheel.py", "options.py", "advisor.py", "trading.py"],
            "search": ["TODO", "FIXME", "Bug", "Performance", "Optimization"],
            "optimize": [
                "database queries",
                "memory usage",
                "performance",
                "risk calculations",
            ],
            "fix": [
                "memory leak",
                "performance issues",
                "database connection",
                "calculation errors",
            ],
            "analyze": [
                "bottlenecks",
                "performance",
                "risk exposure",
                "portfolio balance",
            ],
        }

        # Check if partial matches any pattern start
        for pattern_start, completions in patterns.items():
            if partial.startswith(pattern_start) or pattern_start.startswith(partial):
                for completion in completions:
                    query_text = f"{pattern_start} {completion}"

                    # Calculate confidence based on how well it matches
                    if partial.startswith(pattern_start):
                        confidence = 0.8
                    else:
                        confidence = 0.6

                    query_type = (
                        "search"
                        if pattern_start in ["find", "show", "search"]
                        else "analysis"
                    )

                    suggestions.append(
                        QuerySuggestion(
                            text=query_text,
                            confidence=confidence,
                            category="pattern",
                            reasoning=f"Common {query_type} pattern",
                            estimated_results=10,
                            query_type=query_type,
                        )
                    )

        return suggestions[:limit]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity score between two strings."""

        if text1 == text2:
            return 1.0

        if text1 in text2 or text2 in text1:
            return 0.8

        # Simple character overlap
        common_chars = set(text1) & set(text2)
        total_chars = set(text1) | set(text2)

        if not total_chars:
            return 0.0

        return len(common_chars) / len(total_chars)

    def _classify_query_type(self, query: str) -> str:
        """Classify query as search, analysis, or optimization."""

        query_lower = query.lower()

        # Analysis keywords
        analysis_keywords = {"analyze", "review", "audit", "assess", "check", "debug"}
        if any(word in query_lower for word in analysis_keywords):
            return "analysis"

        # Optimization keywords
        opt_keywords = {
            "optimize",
            "fix",
            "improve",
            "solve",
            "refactor",
            "performance",
        }
        if any(word in query_lower for word in opt_keywords):
            return "optimization"

        # Default to search
        return "search"

    async def record_query(
        self,
        query: str,
        system_used: str,
        success: bool,
        result_count: int,
        execution_time: float,
        user_rating: int | None = None,
    ) -> None:
        """Record a query execution for learning."""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO query_history 
                    (query, timestamp, system_used, success, result_count, execution_time, user_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        query,
                        time.time(),
                        system_used,
                        success,
                        result_count,
                        execution_time,
                        user_rating,
                    ),
                )
        except Exception as e:
            print(f"Warning: Failed to record query: {e}")

    def update_context(
        self, file_path: str, classes: list[str] = None, functions: list[str] = None
    ) -> None:
        """Update contextual information about accessed files."""

        current_time = time.time()

        if file_path in self.context_files:
            context = self.context_files[file_path]
            context.last_accessed = current_time
            context.access_count += 1
        else:
            context = ContextualInfo(
                file_path=file_path,
                last_accessed=current_time,
                access_count=1,
                file_type=Path(file_path).suffix,
                primary_classes=classes or [],
                primary_functions=functions or [],
            )
            self.context_files[file_path] = context

        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO contextual_info 
                    (file_path, last_accessed, access_count, file_type, primary_classes, primary_functions)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        file_path,
                        context.last_accessed,
                        context.access_count,
                        context.file_type,
                        json.dumps(context.primary_classes),
                        json.dumps(context.primary_functions),
                    ),
                )
        except Exception:
            pass

    async def get_refinement_suggestions(
        self, original_query: str, result_count: int
    ) -> list[QuerySuggestion]:
        """Get suggestions to refine a query that returned few results."""

        suggestions = []

        if result_count == 0:
            # No results - suggest broader searches
            words = original_query.lower().split()

            # Try individual words
            for word in words:
                if len(word) > 3:
                    suggestions.append(
                        QuerySuggestion(
                            text=word,
                            confidence=0.7,
                            category="refinement",
                            reasoning="Broader search with individual term",
                            estimated_results=20,
                            query_type="search",
                        )
                    )

            # Try partial matches
            if len(words) > 1:
                suggestions.append(
                    QuerySuggestion(
                        text=" ".join(words[:-1]),
                        confidence=0.6,
                        category="refinement",
                        reasoning="Remove last term to broaden search",
                        estimated_results=15,
                        query_type="search",
                    )
                )

        elif result_count < 5:
            # Few results - suggest related terms
            symbols = await self.codebase_analyzer.get_symbols()

            # Find similar symbols
            query_words = set(original_query.lower().split())

            for symbol_type, symbol_set in symbols.items():
                for symbol in symbol_set:
                    symbol_words = set(re.findall(r"\w+", symbol.lower()))
                    if query_words & symbol_words:  # Has common words
                        suggestions.append(
                            QuerySuggestion(
                                text=f"search {symbol}",
                                confidence=0.6,
                                category="refinement",
                                reasoning=f"Related {symbol_type[:-1]} found",
                                estimated_results=8,
                                query_type="search",
                            )
                        )

        return suggestions[:5]

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about query intelligence system."""

        stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "avg_execution_time": 0.0,
            "popular_terms": [],
            "symbols_cached": 0,
            "context_files": len(self.context_files),
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Basic stats
                cursor = conn.execute("SELECT COUNT(*) FROM query_history")
                stats["total_queries"] = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT COUNT(*) FROM query_history WHERE success = 1"
                )
                stats["successful_queries"] = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT AVG(execution_time) FROM query_history WHERE success = 1"
                )
                result = cursor.fetchone()[0]
                stats["avg_execution_time"] = result if result else 0.0

                # Popular terms
                cursor = conn.execute(
                    """
                    SELECT query, COUNT(*) as freq 
                    FROM query_history 
                    WHERE success = 1 
                    GROUP BY query 
                    ORDER BY freq DESC 
                    LIMIT 10
                """
                )
                stats["popular_terms"] = [
                    {"query": q, "frequency": f} for q, f in cursor.fetchall()
                ]

            # Symbol cache stats
            if self.codebase_analyzer.symbols_cache:
                stats["symbols_cached"] = sum(
                    len(symbols)
                    for symbols in self.codebase_analyzer.symbols_cache.values()
                )

        except Exception:
            pass

        return stats


# Singleton instance for easy access
_query_intelligence_instance = None


def get_query_intelligence(project_root: Path = None) -> QueryIntelligence:
    """Get singleton QueryIntelligence instance."""
    global _query_intelligence_instance

    if _query_intelligence_instance is None:
        _query_intelligence_instance = QueryIntelligence(project_root)

    return _query_intelligence_instance


# Example usage and testing
async def main():
    """Test the query intelligence system."""

    qi = get_query_intelligence()

    print("🧠 Testing Query Intelligence System")

    # Test suggestions
    test_queries = ["find", "wheel", "opt", "risk", "data"]

    for query in test_queries:
        print(f"\n🔍 Suggestions for '{query}':")
        suggestions = await qi.get_suggestions(query, max_suggestions=5)

        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion.text}")
            print(f"     Confidence: {suggestion.confidence:.1%}")
            print(f"     Category: {suggestion.category}")
            print(f"     Reasoning: {suggestion.reasoning}")

    # Test stats
    print("\n📊 System Stats:")
    stats = await qi.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
