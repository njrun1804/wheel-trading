"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


Adaptive configuration system that dynamically tunes parameters based on query complexity.
Optimizes resource usage by right-sizing computation for each query type.
"""

import multiprocessing
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedComputeConfig:
    """Base configuration with sensible defaults."""

    # Sequential thinking
    sequential_thoughts: int = 60
    parallel_branches: int = 8
    use_monte_carlo: bool = False
    use_adversarial: bool = False

    # Memory MCP
    memory_search_depth: int = 35
    memory_max_nodes: int = 100
    memory_similarity_threshold: float = 0.7

    # Filesystem MCP
    filesystem_search_breadth: int = 250
    filesystem_max_file_size: int = 1_000_000  # 1MB
    filesystem_use_index: bool = True

    # PyREPL
    pyrepl_experiment_iterations: int = 5
    pyrepl_batch_size: int = 5
    pyrepl_timeout_seconds: int = 30

    # System-wide
    max_iterations: int = 10
    confidence_threshold: float = 0.9
    early_termination_enabled: bool = True
    cache_ttl_minutes: int = 5

    # Hardware awareness
    cpu_cores: int = field(default_factory=multiprocessing.cpu_count)


class AdaptiveConfig(UnifiedComputeConfig):
    """
    Adaptive configuration that tunes parameters based on query complexity.
    Reduces waste on simple queries while ensuring complex ones get full resources.
    """

    def __init__(self):
        super().__init__()
        self.complexity_profiles = self._init_profiles()
        self.current_complexity = "medium"

    def _init_profiles(self) -> dict[str, dict[str, Any]]:
        """Initialize complexity-based configuration profiles."""
        return {
            "simple": {
                # Minimal resources for quick lookups
                "sequential_thoughts": 20,
                "parallel_branches": 4,
                "use_monte_carlo": False,
                "use_adversarial": False,
                "memory_search_depth": 10,
                "memory_max_nodes": 25,
                "filesystem_search_breadth": 100,
                "pyrepl_experiment_iterations": 2,
                "pyrepl_batch_size": 3,
                "max_iterations": 5,
                "early_termination_enabled": True,
            },
            "medium": {
                # Balanced for most queries
                "sequential_thoughts": 60,
                "parallel_branches": min(8, self.cpu_cores),
                "use_monte_carlo": False,
                "use_adversarial": False,
                "memory_search_depth": 35,
                "memory_max_nodes": 100,
                "filesystem_search_breadth": 250,
                "pyrepl_experiment_iterations": 5,
                "pyrepl_batch_size": 5,
                "max_iterations": 10,
                "early_termination_enabled": True,
            },
            "complex": {
                # Full resources for challenging problems
                "sequential_thoughts": 100,
                "parallel_branches": min(12, self.cpu_cores),
                "use_monte_carlo": True,
                "use_adversarial": True,
                "memory_search_depth": 50,
                "memory_max_nodes": 200,
                "filesystem_search_breadth": 500,
                "pyrepl_experiment_iterations": 10,
                "pyrepl_batch_size": 8,
                "max_iterations": 15,
                "early_termination_enabled": True,
            },
            "maximum": {
                # Maximum compute for critical analysis
                "sequential_thoughts": 150,
                "parallel_branches": min(16, self.cpu_cores),
                "use_monte_carlo": True,
                "use_adversarial": True,
                "memory_search_depth": 100,
                "memory_max_nodes": 500,
                "filesystem_search_breadth": 1000,
                "pyrepl_experiment_iterations": 20,
                "pyrepl_batch_size": 10,
                "max_iterations": 20,
                "early_termination_enabled": False,  # Complete analysis
            },
        }

    def tune(self, complexity: str) -> None:
        """
        Dynamically adjust configuration based on query complexity.

        Args:
            complexity: One of 'simple', 'medium', 'complex', 'maximum'
        """
        if complexity not in self.complexity_profiles:
            complexity = "medium"

        self.current_complexity = complexity
        profile = self.complexity_profiles[complexity]

        # Update all parameters
        for key, value in profile.items():
            setattr(self, key, value)

        # Adjust for available hardware
        self._adjust_for_hardware()

        logger.info("Configuration tuned for '{complexity}' complexity")

    def _adjust_for_hardware(self) -> None:
        """Adjust parameters based on available hardware."""
        # M-series Macs have performance cores
        if self.cpu_cores >= 8:
            # Can handle more parallel work
            self.parallel_branches = min(self.parallel_branches + 2, self.cpu_cores)
        elif self.cpu_cores < 4:
            # Reduce parallelism on weaker hardware
            self.parallel_branches = max(2, self.parallel_branches - 2)
            self.pyrepl_batch_size = max(2, self.pyrepl_batch_size - 2)

    def auto_tune(
        self, query: str, file_count: int = 0, history: dict[str, Any] | None = None
    ) -> str:
        """
        Automatically determine complexity based on query characteristics.

        Args:
            query: The user query
            file_count: Estimated files to analyze
            history: Previous query performance data

        Returns:
            Complexity level: 'simple', 'medium', 'complex', or 'maximum'
        """
        query_lower = query.lower()
        query_length = len(query.split())

        # Complexity indicators
        simple_keywords = {"find", "where", "list", "show", "what", "get", "check"}
        medium_keywords = {"explain", "analyze", "compare", "implement", "update"}
        complex_keywords = {
            "refactor",
            "optimize",
            "debug",
            "trace",
            "redesign",
            "investigate",
            "comprehensive",
            "all",
            "entire",
        }
        maximum_keywords = {
            "maximum",
            "deepest",
            "exhaustive",
            "complete analysis",
            "full codebase",
            "everything",
        }

        # Check for maximum indicators first
        if any(keyword in query_lower for keyword in maximum_keywords):
            complexity = "maximum"
        # Check query characteristics
        elif query_length < 10 and any(kw in query_lower for kw in simple_keywords):
            complexity = "simple"
        elif any(kw in query_lower for kw in complex_keywords) or query_length > 30:
            complexity = "complex"
        elif any(kw in query_lower for kw in medium_keywords) or query_length > 15:
            complexity = "medium"
        else:
            complexity = "simple"

        # Adjust based on file count
        if file_count > 1000 and complexity in ["simple", "medium"]:
            complexity = "complex"
        elif file_count > 5000:
            complexity = "maximum"

        # Learn from history if available
        if history and "avg_iterations_needed" in history:
            avg_iterations = history["avg_iterations_needed"]
            if avg_iterations > 12 and complexity != "maximum":
                complexity = "complex"
            elif avg_iterations < 5 and complexity not in ["simple"]:
                complexity = "medium"

        self.tune(complexity)
        return complexity

    def get_profile_summary(self) -> dict[str, Any]:
        """Get current configuration summary."""
        return {
            "complexity": self.current_complexity,
            "sequential_thoughts": self.sequential_thoughts,
            "memory_depth": self.memory_search_depth,
            "filesystem_breadth": self.filesystem_search_breadth,
            "max_iterations": self.max_iterations,
            "parallelism": self.parallel_branches,
            "monte_carlo": self.use_monte_carlo,
            "early_termination": self.early_termination_enabled,
            "estimated_time": self._estimate_execution_time(),
        }

    def _estimate_execution_time(self) -> str:
        """Estimate execution time based on current settings."""
        # Simple heuristic based on empirical data
        base_times = {
            "simple": 5,  # 5 seconds
            "medium": 15,  # 15 seconds
            "complex": 45,  # 45 seconds
            "maximum": 120,  # 2 minutes
        }

        base = base_times.get(self.current_complexity, 30)

        # Adjust for parallelism
        parallel_factor = 1.0 - (min(self.parallel_branches, 8) * 0.05)

        estimated = base * parallel_factor

        if estimated < 60:
            return f"{estimated:.0f}s"
        else:
            return f"{estimated/60:.1f}m"

    def suggest_optimization(self, metrics: dict[str, Any]) -> str | None:
        """
        Suggest configuration optimizations based on execution metrics.

        Args:
            metrics: Execution metrics from last run

        Returns:
            Optimization suggestion or None
        """
        if not metrics:
            return None

        # Check if we can reduce complexity
        if (
            self.current_complexity != "simple"
            and metrics.get("confidence_at_iteration_3", 0) > 0.95
        ):
            return (
                "Query converged quickly. Consider using 'simple' complexity next time."
            )

        # Check if we need more resources
        if (
            self.current_complexity != "maximum"
            and metrics.get("final_confidence", 0) < 0.8
        ):
            return "Low confidence achieved. Consider using 'complex' or 'maximum' complexity."

        # Check cache performance
        cache_hit_rate = metrics.get("cache_hit_rate", 0)
        if cache_hit_rate < 0.5:
            return "Low cache hit rate. Consider increasing cache_ttl_minutes."

        return None


# Convenience functions
def create_adaptive_config() -> AdaptiveConfig:
    """Create a new adaptive configuration instance."""
    return AdaptiveConfig()


def get_optimal_config(query: str, file_count: int = 0) -> AdaptiveConfig:
    """Get an optimally tuned configuration for a query."""
    config = AdaptiveConfig()
    config.auto_tune(query, file_count)
    return config
