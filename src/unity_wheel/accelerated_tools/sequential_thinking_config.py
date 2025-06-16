"""
Configuration and wrapper for sequential thinking engine.
Provides both MCP compatibility and direct hardware-accelerated access.
"""

from dataclasses import asdict
from typing import Any

from .sequential_thinking_turbo import ThinkingStep, get_sequential_thinking


class SequentialThinkingEngine:
    """
    Unified interface for sequential thinking.
    Can use either MCP server or hardware-accelerated implementation.
    """

    def __init__(self, use_mcp: bool = False):
        self.use_mcp = use_mcp

        if not use_mcp:
            # Use our hardware-accelerated implementation
            self.engine = get_sequential_thinking()
        else:
            # MCP compatibility mode
            self.engine = None  # Would need MCP client setup

    async def think_through_problem(
        self,
        problem: str,
        constraints: list[str] | None = None,
        context: dict[str, Any] | None = None,
        max_steps: int = 100,
        strategy: str = "parallel_explore",
    ) -> dict[str, Any]:
        """
        Think through a problem step by step.

        Args:
            problem: The problem to solve
            constraints: Any constraints or requirements
            context: Additional context
            max_steps: Maximum thinking steps
            strategy: Thinking strategy to use

        Returns:
            Dictionary with solution and steps taken
        """
        if self.use_mcp:
            # MCP mode - would call MCP server
            raise NotImplementedError(
                "MCP mode not implemented - use hardware acceleration instead"
            )

        # Hardware-accelerated mode
        steps = await self.engine.think(
            goal=problem,
            constraints=constraints,
            initial_state=context or {},
            strategy=strategy,
            max_steps=max_steps,
        )

        return {
            "problem": problem,
            "steps": [self._step_to_dict(step) for step in steps],
            "solution": self._extract_solution(steps),
            "stats": self.engine.get_stats(),
        }

    def _step_to_dict(self, step: ThinkingStep) -> dict[str, Any]:
        """Convert ThinkingStep to dictionary."""
        return asdict(step)

    def _extract_solution(self, steps: list[ThinkingStep]) -> dict[str, Any]:
        """Extract the final solution from thinking steps."""
        if not steps:
            return {"status": "no_solution", "reason": "No steps taken"}

        final_step = steps[-1]

        return {
            "status": "solved" if final_step.confidence > 0.7 else "partial",
            "confidence": final_step.confidence,
            "final_action": final_step.action,
            "total_steps": len(steps),
            "result": final_step.result,
        }

    async def plan_implementation(
        self,
        feature: str,
        requirements: list[str],
        existing_code: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Plan the implementation of a feature.

        Specialized method for code planning tasks.
        """
        context = {
            "feature": feature,
            "existing_code": existing_code or {},
            "codebase_type": "python",
            "hardware_available": True,
        }

        result = await self.think_through_problem(
            problem=f"Plan implementation of {feature}",
            constraints=requirements
            + [
                "Generate specific implementation steps",
                "Consider existing code patterns",
                "Optimize for performance",
                "Use available hardware acceleration",
            ],
            context=context,
            strategy="depth_first",  # Better for implementation planning
        )

        # Extract implementation-specific details
        implementation_plan = {
            "feature": feature,
            "steps": [],
            "estimated_effort": "medium",
            "dependencies": [],
            "risks": [],
        }

        for step in result["steps"]:
            if "implement" in step["action"].lower():
                implementation_plan["steps"].append(
                    {
                        "action": step["action"],
                        "details": step["reasoning"],
                        "order": step["step_number"],
                    }
                )

        return implementation_plan

    async def analyze_code(
        self, code: str, objective: str = "optimization opportunities"
    ) -> dict[str, Any]:
        """
        Analyze code for specific objectives.
        """
        result = await self.think_through_problem(
            problem=f"Analyze code for {objective}",
            constraints=[
                "Identify specific improvements",
                "Consider performance implications",
                "Maintain correctness",
                "Leverage hardware capabilities",
            ],
            context={"code": code, "language": "python"},
            strategy="beam_search",
        )

        return {
            "objective": objective,
            "findings": result["solution"],
            "recommendations": [
                s for s in result["steps"] if "recommend" in s["action"].lower()
            ],
            "confidence": result["solution"]["confidence"],
        }

    def close(self):
        """Clean up resources."""
        if hasattr(self.engine, "close"):
            self.engine.close()


# Convenience functions for direct access
async def think_sequential(problem: str, **kwargs) -> dict[str, Any]:
    """Quick access to sequential thinking."""
    engine = SequentialThinkingEngine(use_mcp=False)
    try:
        return await engine.think_through_problem(problem, **kwargs)
    finally:
        engine.close()


async def plan_feature(
    feature: str, requirements: list[str], **kwargs
) -> dict[str, Any]:
    """Plan a feature implementation."""
    engine = SequentialThinkingEngine(use_mcp=False)
    try:
        return await engine.plan_implementation(feature, requirements, **kwargs)
    finally:
        engine.close()


# Export for compatibility
__all__ = ["SequentialThinkingEngine", "think_sequential", "plan_feature"]
