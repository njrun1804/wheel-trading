"""Strategy selection for Jarvis - simplified from orchestrator."""

import re
from typing import Any


class StrategySelector:
    """Selects appropriate strategy based on query analysis."""

    # Strategy patterns
    PATTERNS = {
        "optimization": [
            r"optimi[zs]e",
            r"performance",
            r"speed up",
            r"faster",
            r"accelerat",
            r"improv\w+ (speed|performance)",
            r"reduce (time|latency)",
            r"bottleneck",
        ],
        "refactoring": [
            r"refactor",
            r"rename",
            r"restructure",
            r"reorgani[zs]e",
            r"clean up",
            r"simplify",
            r"extract (method|function|class)",
            r"move (to|from)",
        ],
        "testing": [
            r"test",
            r"verify",
            r"validate",
            r"check",
            r"coverage",
            r"unit test",
            r"integration test",
            r"add tests?",
        ],
        "debugging": [
            r"debug",
            r"fix",
            r"error",
            r"bug",
            r"issue",
            r"problem",
            r"crash",
            r"exception",
        ],
        "analysis": [
            r"analy[zs]e",
            r"understand",
            r"explain",
            r"document",
            r"what does",
            r"how does",
            r"find",
            r"search",
        ],
        "generation": [
            r"generat",
            r"creat",
            r"add",
            r"implement",
            r"build",
            r"write",
            r"develop",
            r"new",
        ],
    }

    def select_strategy(
        self, query: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Select the best strategy for the given query."""
        query_lower = query.lower()

        # Score each strategy
        scores = {}
        for strategy, patterns in self.PATTERNS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            scores[strategy] = score

        # Get the highest scoring strategy
        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]

        # If no patterns match, use general strategy
        if best_score == 0:
            best_strategy = "general"

        # Determine if MCTS is beneficial
        needs_mcts = self._needs_mcts(best_strategy, query_lower, context)

        # Determine parallelization strategy
        parallel_strategy = self._get_parallel_strategy(best_strategy)

        return {
            "strategy": best_strategy,
            "confidence": min(best_score / 3.0, 1.0),  # Normalize to 0-1
            "needs_mcts": needs_mcts,
            "parallel_strategy": parallel_strategy,
            "reasoning": self._get_reasoning(best_strategy, query_lower),
        }

    def _needs_mcts(
        self, strategy: str, query: str, context: dict | None = None
    ) -> bool:
        """Determine if MCTS would be beneficial."""
        # MCTS is useful for optimization and complex generation
        if strategy in ["optimization", "generation"]:
            return True

        # Check for complexity indicators
        complexity_indicators = [
            "complex",
            "difficult",
            "challenging",
            "optimize",
            "best",
            "maximum",
            "minimum",
            "optimal",
            "efficient",
        ]

        if any(indicator in query for indicator in complexity_indicators):
            return True

        # Check context for high complexity
        return bool(context and context.get("complexity", 0) > 100)

    def _get_parallel_strategy(self, strategy: str) -> str:
        """Determine the best parallelization approach."""
        if strategy in ["optimization", "analysis"]:
            return "cpu_intensive"  # Use all performance cores
        elif strategy in ["refactoring", "testing"]:
            return "io_intensive"  # Use thread pool for I/O
        elif strategy == "generation":
            return "gpu_accelerated"  # Use GPU for ML operations
        else:
            return "balanced"  # Mix of CPU and I/O

    def _get_reasoning(self, strategy: str, query: str) -> str:
        """Provide reasoning for strategy selection."""
        reasoning = {
            "optimization": "Focusing on performance improvements and speed optimization",
            "refactoring": "Restructuring code while maintaining functionality",
            "testing": "Ensuring code correctness through comprehensive testing",
            "debugging": "Identifying and fixing issues in the codebase",
            "analysis": "Understanding code structure and dependencies",
            "generation": "Creating new code or features",
            "general": "General code modification task",
        }

        return reasoning.get(strategy, "Processing general request")
