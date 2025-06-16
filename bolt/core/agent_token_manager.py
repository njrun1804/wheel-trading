#!/usr/bin/env python3
"""
Agent Token Manager Integration
Integrates with Claude Code's token limits and provides intelligent response management.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

from .dynamic_token_optimizer import (
    analyze_and_allocate,
    get_token_optimizer,
)
from .output_token_manager import (
    ResponseStrategy,
    TokenBudget,
    get_output_token_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentResponseContext:
    """Context for agent response generation."""

    agent_id: str
    task_type: str
    complexity_score: float
    is_final_response: bool = True
    allow_chunking: bool = False
    user_request: str | None = None


class AgentTokenManager:
    """
    Unified token management for Bolt agents.
    Handles both output limits and dynamic optimization.
    """

    def __init__(self):
        self.output_manager = get_output_token_manager()
        self.dynamic_optimizer = get_token_optimizer()

        # Get token limits from environment
        self.max_output_tokens = int(
            os.getenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        )
        self.thinking_budget = int(
            os.getenv("CLAUDE_CODE_THINKING_BUDGET_TOKENS", "50000")
        )

        # Update token budget based on environment
        self.output_manager.budget.max_tokens = self.max_output_tokens

        logger.info(
            f"Agent Token Manager initialized: "
            f"output_limit={self.max_output_tokens}, "
            f"thinking_budget={self.thinking_budget}"
        )

    def prepare_response(
        self, content: Any, context: AgentResponseContext
    ) -> dict[str, Any]:
        """
        Prepare agent response with intelligent token management.

        Args:
            content: The response content to optimize
            context: Context about the response being generated

        Returns:
            Optimized response within token limits
        """
        try:
            # First, get dynamic token allocation
            allocation = analyze_and_allocate(
                context.user_request or context.task_type,
                {
                    "technical_level": "expert",
                    "complexity_score": context.complexity_score,
                },
            )

            # Update token budget based on allocation
            dynamic_budget = allocation["token_budget"]
            target_tokens = dynamic_budget["target"]

            # Don't exceed system limits
            effective_limit = min(target_tokens, self.max_output_tokens)

            # Create response-specific budget
            response_budget = TokenBudget(
                max_tokens=effective_limit,
                reserved_for_metadata=300,
                reserved_for_summary=200,
                chunk_overlap=100,
            )

            # Update output manager budget
            original_budget = self.output_manager.budget
            self.output_manager.budget = response_budget

            try:
                # Determine optimization strategy
                strategy = self._get_optimization_strategy(context, effective_limit)

                # Optimize the response
                optimized_content = self.output_manager.optimize_response(
                    self._prepare_content(content, context), strategy
                )

                # Add metadata about token management
                if isinstance(optimized_content, dict):
                    optimized_content["_agent_context"] = {
                        "agent_id": context.agent_id,
                        "task_type": context.task_type,
                        "complexity_score": context.complexity_score,
                        "token_allocation": dynamic_budget,
                        "effective_limit": effective_limit,
                        "strategy_used": strategy.value,
                    }

                # Validate final response
                (
                    is_valid,
                    token_count,
                    status,
                ) = self.output_manager.validate_response_size(optimized_content)

                if not is_valid:
                    logger.warning(f"Response exceeds limits: {token_count} tokens")
                    # Force more aggressive optimization
                    optimized_content = self.output_manager.optimize_response(
                        optimized_content, ResponseStrategy.SUMMARIZE
                    )

                logger.info(
                    f"Response prepared: {token_count} tokens, "
                    f"strategy={strategy.value}, valid={is_valid}"
                )

                return optimized_content

            finally:
                # Restore original budget
                self.output_manager.budget = original_budget

        except Exception as e:
            logger.error(f"Error preparing response: {e}")
            # Fallback to simple truncation
            return self._emergency_truncate(content, context)

    def _get_optimization_strategy(
        self, context: AgentResponseContext, token_limit: int
    ) -> ResponseStrategy:
        """Determine the best optimization strategy."""

        # For final responses, prioritize quality
        if context.is_final_response:
            if token_limit >= 20000:
                return ResponseStrategy.PRIORITIZE
            elif token_limit >= 10000:
                return ResponseStrategy.SUMMARIZE
            else:
                return ResponseStrategy.TRUNCATE

        # For intermediate responses, allow chunking
        if context.allow_chunking:
            return ResponseStrategy.CHUNK

        # For complex analysis tasks, prioritize key information
        if context.complexity_score > 0.7:
            return ResponseStrategy.PRIORITIZE

        # Default to summarization
        return ResponseStrategy.SUMMARIZE

    def _prepare_content(
        self, content: Any, context: AgentResponseContext
    ) -> dict[str, Any]:
        """Prepare content for optimization."""
        if isinstance(content, dict):
            return content
        elif isinstance(content, str):
            return {
                "response": content,
                "agent_id": context.agent_id,
                "task_type": context.task_type,
            }
        else:
            return {
                "response": str(content),
                "agent_id": context.agent_id,
                "task_type": context.task_type,
                "raw_content": content,
            }

    def _emergency_truncate(
        self, content: Any, context: AgentResponseContext
    ) -> dict[str, Any]:
        """Emergency truncation when optimization fails."""
        try:
            text_content = str(content)

            # Simple truncation to 80% of limit
            max_chars = int(
                self.max_output_tokens * 3.8 * 0.8
            )  # Rough char-to-token conversion

            if len(text_content) > max_chars:
                truncated = (
                    text_content[:max_chars]
                    + "\n\n[Response truncated due to token limits]"
                )
            else:
                truncated = text_content

            return {
                "response": truncated,
                "agent_id": context.agent_id,
                "task_type": context.task_type,
                "_emergency_truncation": True,
                "_original_length": len(text_content),
                "_truncated_length": len(truncated),
            }

        except Exception as e:
            logger.error(f"Emergency truncation failed: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "agent_id": context.agent_id,
                "task_type": context.task_type,
                "_error": True,
            }

    def validate_response(self, response: Any) -> tuple[bool, int, str]:
        """Validate response meets token requirements."""
        return self.output_manager.validate_response_size(response)

    def get_token_stats(self) -> dict[str, Any]:
        """Get token management statistics."""
        return {
            "limits": {
                "max_output_tokens": self.max_output_tokens,
                "thinking_budget": self.thinking_budget,
            },
            "output_manager": self.output_manager.get_optimization_stats(),
            "dynamic_optimizer": self.dynamic_optimizer.analyze_response_efficiency(),
        }


# Global instance
_agent_token_manager: AgentTokenManager | None = None


def get_agent_token_manager() -> AgentTokenManager:
    """Get global agent token manager."""
    global _agent_token_manager
    if _agent_token_manager is None:
        _agent_token_manager = AgentTokenManager()
    return _agent_token_manager


def prepare_agent_response(
    content: Any,
    agent_id: str,
    task_type: str,
    complexity_score: float = 0.5,
    user_request: str | None = None,
) -> Any:
    """Convenience function to prepare agent response."""
    manager = get_agent_token_manager()
    context = AgentResponseContext(
        agent_id=agent_id,
        task_type=task_type,
        complexity_score=complexity_score,
        user_request=user_request,
    )
    return manager.prepare_response(content, context)


# Decorator for automatic response preparation
def token_managed_response(
    agent_id: str, task_type: str, complexity_score: float = 0.5
):
    """Decorator to automatically manage response tokens."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return prepare_agent_response(result, agent_id, task_type, complexity_score)

        return wrapper

    return decorator
