"""Solution data structures for Jarvis 2.0.

Defines the structure of code solutions and their associated metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SolutionMetrics:
    """Metrics for a generated solution."""
    generation_time_ms: float
    simulations_performed: int
    variants_generated: int
    confidence_score: float
    complexity_score: float
    gpu_utilization: float
    memory_used_mb: float
    
    # Additional metrics
    search_depth_reached: int = 0
    pruned_branches: int = 0
    cache_hits: int = 0
    neural_evaluations: int = 0
    
    def __str__(self) -> str:
        return (
            f"Time: {self.generation_time_ms:.0f}ms, "
            f"Simulations: {self.simulations_performed}, "
            f"Confidence: {self.confidence_score:.2f}, "
            f"GPU: {self.gpu_utilization:.1f}%"
        )


@dataclass
class CodeSolution:
    """A complete code solution with metadata."""
    query: str
    code: str
    explanation: str
    confidence: float
    alternatives: List[Dict[str, Any]]
    metrics: SolutionMetrics
    
    # Optional fields
    language: str = "python"
    dependencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Behavioral characteristics
    approach: str = "standard"
    optimization_focus: str = "balanced"
    patterns_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "code": self.code,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "language": self.language,
            "approach": self.approach,
            "optimization_focus": self.optimization_focus,
            "patterns_used": self.patterns_used,
            "dependencies": self.dependencies,
            "warnings": self.warnings,
            "alternatives": self.alternatives,
            "metrics": {
                "generation_time_ms": self.metrics.generation_time_ms,
                "simulations_performed": self.metrics.simulations_performed,
                "variants_generated": self.metrics.variants_generated,
                "confidence_score": self.metrics.confidence_score,
                "complexity_score": self.metrics.complexity_score,
                "gpu_utilization": self.metrics.gpu_utilization,
                "memory_used_mb": self.metrics.memory_used_mb,
            },
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CodeSolution:
        """Create from dictionary."""
        metrics_data = data.pop("metrics", {})
        metrics = SolutionMetrics(**metrics_data)
        
        # Handle timestamp
        timestamp_str = data.pop("timestamp", None)
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = datetime.now()
        
        return cls(
            metrics=metrics,
            timestamp=timestamp,
            **data
        )
    
    def format_display(self) -> str:
        """Format for display to user."""
        output = []
        
        # Main solution
        output.append(f"## Solution (Confidence: {self.confidence:.1%})")
        output.append("")
        output.append("```" + self.language)
        output.append(self.code)
        output.append("```")
        output.append("")
        
        # Explanation
        if self.explanation:
            output.append("### Explanation")
            output.append(self.explanation)
            output.append("")
        
        # Warnings
        if self.warnings:
            output.append("### ⚠️ Warnings")
            for warning in self.warnings:
                output.append(f"- {warning}")
            output.append("")
        
        # Dependencies
        if self.dependencies:
            output.append("### Dependencies")
            for dep in self.dependencies:
                output.append(f"- {dep}")
            output.append("")
        
        # Alternatives
        if self.alternatives:
            output.append("### Alternative Approaches")
            for i, alt in enumerate(self.alternatives, 1):
                output.append(f"**Option {i}** ({alt.get('differentiator', 'Alternative')})")
                output.append(f"Confidence: {alt.get('confidence', 0):.1%}")
                output.append("```" + self.language)
                output.append(alt.get('code', ''))
                output.append("```")
                output.append("")
        
        # Metrics
        output.append("### Generation Metrics")
        output.append(str(self.metrics))
        
        return "\n".join(output)


@dataclass
class SearchNode:
    """Node in the search tree."""
    code: str
    parent: Optional[SearchNode]
    children: List[SearchNode] = field(default_factory=list)
    
    # MCTS statistics
    visits: int = 0
    value_sum: float = 0.0
    prior_probability: float = 0.0
    
    # Node metadata
    action_taken: str = ""
    depth: int = 0
    
    def add_child(self, child_code: str, action: str, prior: float = 0.0) -> SearchNode:
        """Add a child node."""
        child = SearchNode(
            code=child_code,
            parent=self,
            action_taken=action,
            prior_probability=prior,
            depth=self.depth + 1
        )
        self.children.append(child)
        return child
    
    def update(self, value: float):
        """Update node statistics after simulation."""
        self.visits += 1
        self.value_sum += value
    
    @property
    def average_value(self) -> float:
        """Average value of this node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def get_path_to_root(self) -> List[SearchNode]:
        """Get path from this node to root."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def to_solution_tree(self) -> Dict[str, Any]:
        """Convert to solution tree format."""
        return {
            "code": self.code,
            "action": self.action_taken,
            "visits": self.visits,
            "value": self.average_value,
            "depth": self.depth,
            "children": [child.to_solution_tree() for child in self.children]
        }


@dataclass
class Experience:
    """A single experience for learning."""
    query: str
    context: Dict[str, Any]
    solution: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: float
    
    # Outcome tracking
    was_selected: bool = True
    user_feedback: Optional[float] = None
    execution_success: Optional[bool] = None
    
    def to_training_example(self) -> Dict[str, Any]:
        """Convert to training example."""
        return {
            "input": {
                "query": self.query,
                "context": self.context
            },
            "output": self.solution,
            "reward": self._calculate_reward(),
            "timestamp": self.timestamp
        }
    
    def _calculate_reward(self) -> float:
        """Calculate reward for reinforcement learning."""
        reward = 0.0
        
        # Base reward from metrics
        if "confidence_score" in self.metrics:
            reward += self.metrics["confidence_score"] * 0.3
        
        if "performance" in self.metrics:
            reward += self.metrics["performance"] * 0.2
        
        # User feedback is most important
        if self.user_feedback is not None:
            reward = 0.5 * reward + 0.5 * self.user_feedback
        
        # Execution success
        if self.execution_success is not None:
            reward *= (1.0 if self.execution_success else 0.5)
        
        return max(0.0, min(1.0, reward))