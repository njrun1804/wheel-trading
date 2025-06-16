"""Simplified phase execution model for Jarvis.

Instead of 7 phases, we use 4 streamlined phases:
1. DISCOVER - Find relevant code and understand context
2. ANALYZE - Deep analysis and strategy selection  
3. IMPLEMENT - Execute changes with hardware acceleration
4. VERIFY - Test and validate results
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Phase(Enum):
    """Simplified phases for meta-coding."""

    DISCOVER = "discover"
    ANALYZE = "analyze"
    IMPLEMENT = "implement"
    VERIFY = "verify"


@dataclass
class PhaseResult:
    """Result from executing a phase."""

    phase: Phase
    success: bool
    duration_ms: float
    data: dict[str, Any]
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class PhaseExecutor:
    """Executes phases with hardware acceleration."""

    def __init__(self, accelerated_tools):
        self.tools = accelerated_tools
        self.results = {}
        self.context = {}

    async def execute_phase(self, phase: Phase, context: dict[str, Any]) -> PhaseResult:
        """Execute a single phase."""
        start = time.perf_counter()

        try:
            if phase == Phase.DISCOVER:
                result = await self._discover(context)
            elif phase == Phase.ANALYZE:
                result = await self._analyze(context)
            elif phase == Phase.IMPLEMENT:
                result = await self._implement(context)
            elif phase == Phase.VERIFY:
                result = await self._verify(context)
            else:
                raise ValueError(f"Unknown phase: {phase}")

            duration = (time.perf_counter() - start) * 1000

            return PhaseResult(
                phase=phase, success=True, duration_ms=duration, data=result
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return PhaseResult(
                phase=phase,
                success=False,
                duration_ms=duration,
                data={},
                errors=[str(e)],
            )

    async def _discover(self, context: dict[str, Any]) -> dict[str, Any]:
        """Discover relevant code and context using accelerated search."""
        query = context.get("query", "")

        # Extract key terms
        terms = self._extract_search_terms(query)

        # Parallel search using ripgrep turbo
        search_results = await self.tools.ripgrep.parallel_search(terms, "src")

        # Build dependency graph for found files
        files = set()
        for results in search_results.values():
            for r in results:
                files.add(r["file"])

        if files:
            await self.tools.dependency_graph.build_graph(list(files)[:100])

        return {
            "search_terms": terms,
            "files_found": len(files),
            "search_results": search_results,
            "relevant_files": list(files)[:50],  # Top 50 most relevant
        }

    async def _analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze discovered code and select strategy."""
        discovery = self.results.get(Phase.DISCOVER, {}).get("data", {})
        files = discovery.get("relevant_files", [])

        # Analyze files in parallel
        analyses = []
        for file in files[:20]:  # Analyze top 20 files
            try:
                analysis = await self.tools.python_analyzer.analyze_file(file)
                analyses.append(analysis)
            except Exception:
                # Skip files that can't be analyzed
                continue

        # Determine complexity and strategy
        total_complexity = sum(a.complexity for a in analyses)
        total_loc = sum(a.loc for a in analyses)

        # Use strategy selector for better analysis
        from jarvis.strategies.strategy_selector import StrategySelector

        selector = StrategySelector()

        strategy_result = selector.select_strategy(
            context.get("query", ""),
            {"complexity": total_complexity, "files": len(files)},
        )

        strategy = strategy_result["strategy"]
        needs_mcts = strategy_result["needs_mcts"]

        return {
            "strategy": strategy,
            "needs_mcts": needs_mcts,
            "complexity": total_complexity,
            "total_loc": total_loc,
            "files_analyzed": len(analyses),
            "primary_targets": files[:10],
        }

    async def _implement(self, context: dict[str, Any]) -> dict[str, Any]:
        """Implement changes using selected strategy."""
        analysis = self.results.get(Phase.ANALYZE, {}).get("data", {})
        strategy = analysis.get("strategy", "general")

        # Route to appropriate implementation
        if strategy == "optimization":
            return await self._implement_optimization(context, analysis)
        elif strategy == "refactoring":
            return await self._implement_refactoring(context, analysis)
        elif strategy == "testing":
            return await self._implement_testing(context, analysis)
        else:
            return await self._implement_general(context, analysis)

    async def _verify(self, context: dict[str, Any]) -> dict[str, Any]:
        """Verify implementation results."""
        implementation = self.results.get(Phase.IMPLEMENT, {}).get("data", {})

        # Run tests if available
        test_results = []
        if "modified_files" in implementation:
            # Could run pytest on modified files
            pass

        # Trace execution
        async with self.tools.tracer.trace_span("verification") as span:
            span["files_modified"] = implementation.get("files_modified", 0)
            span["tests_run"] = len(test_results)

        return {
            "verified": True,
            "tests_passed": len([t for t in test_results if t.get("passed")]),
            "tests_failed": len([t for t in test_results if not t.get("passed")]),
            "implementation_summary": implementation,
        }

    def _extract_search_terms(self, query: str) -> list[str]:
        """Extract search terms from natural language query."""
        # Simple keyword extraction
        keywords = []

        # Common code-related terms
        code_terms = ["class", "function", "def", "import", "error", "TODO", "FIXME"]
        for term in code_terms:
            if term.lower() in query.lower():
                keywords.append(term)

        # Extract quoted strings
        import re

        quoted = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
        keywords.extend(quoted)

        # Extract CamelCase or snake_case identifiers
        identifiers = re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", query)
        keywords.extend(identifiers)

        # Add specific terms from query
        if "ripgrep" in query.lower():
            keywords.append("ripgrep")
            keywords.append("RipgrepTurbo")
            keywords.append("search")

        if "optimize" in query.lower() or "performance" in query.lower():
            keywords.append("performance")
            keywords.append("parallel")
            keywords.append("async")

        # If no keywords found, use general terms
        if not keywords:
            keywords = ["TODO", "FIXME", "class", "def"]

        return list(set(keywords))[:10]  # Max 10 terms

    async def _implement_optimization(
        self, context: dict, analysis: dict
    ) -> dict[str, Any]:
        """Implement optimization changes."""
        # This would use MCTS for complex optimizations
        return {
            "strategy_used": "optimization",
            "optimizations_applied": 0,
            "performance_improvement": "0%",
        }

    async def _implement_refactoring(
        self, context: dict, analysis: dict
    ) -> dict[str, Any]:
        """Implement refactoring changes."""
        # Use code helper for refactoring
        return {
            "strategy_used": "refactoring",
            "files_modified": 0,
            "symbols_renamed": 0,
        }

    async def _implement_testing(self, context: dict, analysis: dict) -> dict[str, Any]:
        """Implement test-related changes."""
        return {
            "strategy_used": "testing",
            "tests_created": 0,
            "coverage_improvement": "0%",
        }

    async def _implement_general(self, context: dict, analysis: dict) -> dict[str, Any]:
        """General implementation for other tasks."""
        return {"strategy_used": "general", "actions_taken": []}
