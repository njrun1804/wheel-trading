"""
Dynamic Token Optimization for Sonnet 4

Implements intelligent token utilization that scales with task complexity,
avoiding artificial padding while maximizing value per response.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ResponseComplexity(Enum):
    """Task complexity levels for dynamic token allocation."""

    SIMPLE = "simple"  # 1-2K tokens: Quick answers, confirmations
    MODERATE = "moderate"  # 5-8K tokens: Standard explanations, small code
    DETAILED = "detailed"  # 10-20K tokens: Comprehensive analysis, full implementations
    EXHAUSTIVE = (
        "exhaustive"  # 30-50K tokens: Multi-faceted deep dives, architecture docs
    )


@dataclass
class TokenBudget:
    """Dynamic token budget based on task requirements."""

    min_tokens: int
    target_tokens: int
    max_tokens: int
    complexity: ResponseComplexity

    @property
    def efficiency_ratio(self) -> float:
        """Calculate efficiency of token usage."""
        return self.target_tokens / 64000  # Ratio to max capacity


@dataclass
class TaskContext:
    """Context for intelligent token allocation."""

    query_length: int
    query_complexity: float  # 0-1 score
    domain_depth: float  # 0-1 score for technical depth needed
    code_generation: bool
    multi_step_reasoning: bool
    cross_component_analysis: bool
    documentation_needed: bool
    drift_compensation: float = 0.0  # Account for context growth

    def calculate_complexity_score(self) -> float:
        """Calculate overall complexity score for token allocation."""
        base_score = self.query_complexity * 0.3 + self.domain_depth * 0.3

        # Add modifiers for specific requirements
        if self.code_generation:
            base_score += 0.15
        if self.multi_step_reasoning:
            base_score += 0.15
        if self.cross_component_analysis:
            base_score += 0.1
        if self.documentation_needed:
            base_score += 0.1

        # Apply drift compensation (larger contexts need more tokens)
        drift_adjusted = base_score * (1 + self.drift_compensation)

        return min(1.0, drift_adjusted)


class DynamicTokenOptimizer:
    """
    Intelligent token allocation system that scales response depth
    with task complexity, avoiding waste while maximizing value.
    """

    def __init__(self):
        # Token allocation strategies
        self.complexity_budgets = {
            ResponseComplexity.SIMPLE: TokenBudget(
                1000, 2000, 3000, ResponseComplexity.SIMPLE
            ),
            ResponseComplexity.MODERATE: TokenBudget(
                4000, 6000, 8000, ResponseComplexity.MODERATE
            ),
            ResponseComplexity.DETAILED: TokenBudget(
                10000, 15000, 20000, ResponseComplexity.DETAILED
            ),
            ResponseComplexity.EXHAUSTIVE: TokenBudget(
                25000, 40000, 50000, ResponseComplexity.EXHAUSTIVE
            ),
        }

        # Drift tracking for large context management
        self.context_history: list[int] = []
        self.max_history = 10

        # Performance metrics
        self.allocation_history: list[tuple[float, TokenBudget]] = []

    def analyze_task(
        self, instruction: str, context: dict[str, Any] | None = None
    ) -> TaskContext:
        """Analyze task to determine complexity and requirements."""
        query_length = len(instruction)

        # Basic complexity heuristics
        complexity_indicators = {
            "optimize": 0.7,
            "analyze": 0.6,
            "implement": 0.8,
            "debug": 0.7,
            "explain": 0.4,
            "assess": 0.6,
            "design": 0.8,
            "refactor": 0.9,
        }

        # Calculate query complexity
        query_complexity = 0.3  # Base complexity
        for keyword, weight in complexity_indicators.items():
            if keyword in instruction.lower():
                query_complexity = max(query_complexity, weight)

        # Determine specific requirements
        code_generation = any(
            kw in instruction.lower()
            for kw in ["implement", "code", "generate", "create", "write"]
        )

        multi_step = any(
            kw in instruction.lower()
            for kw in ["step", "phase", "workflow", "process", "stages"]
        )

        cross_component = any(
            kw in instruction.lower()
            for kw in ["integration", "system", "architecture", "cross"]
        )

        documentation = any(
            kw in instruction.lower()
            for kw in ["document", "explain", "guide", "tutorial"]
        )

        # Calculate domain depth based on context
        domain_depth = 0.5  # Default moderate depth
        if context:
            if context.get("technical_level", "medium") == "expert":
                domain_depth = 0.8
            elif context.get("technical_level", "medium") == "beginner":
                domain_depth = 0.3

        # Calculate drift compensation
        drift = self._calculate_drift_compensation()

        return TaskContext(
            query_length=query_length,
            query_complexity=query_complexity,
            domain_depth=domain_depth,
            code_generation=code_generation,
            multi_step_reasoning=multi_step,
            cross_component_analysis=cross_component,
            documentation_needed=documentation,
            drift_compensation=drift,
        )

    def allocate_tokens(self, task_context: TaskContext) -> TokenBudget:
        """Allocate appropriate token budget based on task analysis."""
        complexity_score = task_context.calculate_complexity_score()

        # Map complexity score to response complexity level
        if complexity_score < 0.3:
            complexity = ResponseComplexity.SIMPLE
        elif complexity_score < 0.5:
            complexity = ResponseComplexity.MODERATE
        elif complexity_score < 0.7:
            complexity = ResponseComplexity.DETAILED
        else:
            complexity = ResponseComplexity.EXHAUSTIVE

        # Get base budget
        budget = self.complexity_budgets[complexity]

        # Apply drift adjustments if needed
        if task_context.drift_compensation > 0.1:
            # Scale up budget for context growth
            scale_factor = 1 + (task_context.drift_compensation * 0.5)
            adjusted_budget = TokenBudget(
                min_tokens=int(budget.min_tokens * scale_factor),
                target_tokens=int(budget.target_tokens * scale_factor),
                max_tokens=min(64000, int(budget.max_tokens * scale_factor)),
                complexity=complexity,
            )
            budget = adjusted_budget

        # Track allocation
        self.allocation_history.append((complexity_score, budget))

        logger.info(
            f"Token allocation: {budget.complexity.value} "
            f"({budget.target_tokens:,} tokens, "
            f"{budget.efficiency_ratio:.1%} capacity)"
        )

        return budget

    def _calculate_drift_compensation(self) -> float:
        """Calculate drift compensation based on context growth."""
        if len(self.context_history) < 2:
            return 0.0

        # Calculate average growth rate
        recent_sizes = self.context_history[-5:]
        if len(recent_sizes) < 2:
            return 0.0

        growth_rates = []
        for i in range(1, len(recent_sizes)):
            growth = (recent_sizes[i] - recent_sizes[i - 1]) / recent_sizes[i - 1]
            growth_rates.append(growth)

        avg_growth = np.mean(growth_rates)

        # Convert to compensation factor (0-0.5 range)
        compensation = min(0.5, max(0.0, avg_growth * 2))

        return compensation

    def update_context_size(self, size: int) -> None:
        """Update context size history for drift tracking."""
        self.context_history.append(size)
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)

    def get_response_guidelines(self, budget: TokenBudget) -> dict[str, Any]:
        """Get specific guidelines for response generation."""
        guidelines = {
            ResponseComplexity.SIMPLE: {
                "style": "concise and direct",
                "code_examples": "minimal snippets only",
                "explanations": "brief, focused on key points",
                "documentation": "none unless critical",
            },
            ResponseComplexity.MODERATE: {
                "style": "clear with good detail",
                "code_examples": "relevant examples with context",
                "explanations": "thorough but not exhaustive",
                "documentation": "inline comments and brief notes",
            },
            ResponseComplexity.DETAILED: {
                "style": "comprehensive and educational",
                "code_examples": "full implementations with tests",
                "explanations": "detailed with reasoning and alternatives",
                "documentation": "extensive inline docs and usage examples",
            },
            ResponseComplexity.EXHAUSTIVE: {
                "style": "authoritative and complete",
                "code_examples": "production-ready with error handling",
                "explanations": "multi-faceted with deep technical analysis",
                "documentation": "full technical documentation with tutorials",
            },
        }

        return {
            "budget": budget,
            "guidelines": guidelines[budget.complexity],
            "token_range": f"{budget.min_tokens:,}-{budget.max_tokens:,}",
            "target": f"{budget.target_tokens:,}",
            "efficiency": f"{budget.efficiency_ratio:.1%}",
        }

    def analyze_response_efficiency(self) -> dict[str, Any]:
        """Analyze token allocation efficiency over time."""
        if not self.allocation_history:
            return {"status": "no data"}

        # Calculate statistics
        complexities = [score for score, _ in self.allocation_history]
        budgets = [budget for _, budget in self.allocation_history]

        avg_complexity = np.mean(complexities)
        avg_tokens = np.mean([b.target_tokens for b in budgets])
        avg_efficiency = np.mean([b.efficiency_ratio for b in budgets])

        # Distribution by complexity level
        distribution = {}
        for _, budget in self.allocation_history:
            level = budget.complexity.value
            distribution[level] = distribution.get(level, 0) + 1

        return {
            "total_allocations": len(self.allocation_history),
            "average_complexity_score": round(avg_complexity, 3),
            "average_tokens": int(avg_tokens),
            "average_efficiency": f"{avg_efficiency:.1%}",
            "complexity_distribution": distribution,
            "drift_active": len(self.context_history) > 0,
        }


# Global instance for easy access
_token_optimizer: DynamicTokenOptimizer | None = None


def get_token_optimizer() -> DynamicTokenOptimizer:
    """Get global token optimizer instance."""
    global _token_optimizer
    if _token_optimizer is None:
        _token_optimizer = DynamicTokenOptimizer()
    return _token_optimizer


def analyze_and_allocate(
    instruction: str, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convenience function for token analysis and allocation."""
    optimizer = get_token_optimizer()
    task_context = optimizer.analyze_task(instruction, context)
    budget = optimizer.allocate_tokens(task_context)
    guidelines = optimizer.get_response_guidelines(budget)

    return {
        "task_analysis": {
            "complexity_score": task_context.calculate_complexity_score(),
            "code_generation": task_context.code_generation,
            "multi_step": task_context.multi_step_reasoning,
            "drift_compensation": task_context.drift_compensation,
        },
        "token_budget": {
            "complexity": budget.complexity.value,
            "target": budget.target_tokens,
            "range": f"{budget.min_tokens}-{budget.max_tokens}",
            "efficiency": f"{budget.efficiency_ratio:.1%}",
        },
        "guidelines": guidelines,
    }
