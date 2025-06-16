#!/usr/bin/env python3
"""
Output Token Limit Manager for Bolt Agents

Ensures agents never exceed 8192 output token maximum by intelligent
truncation, summarization, and chunked responses.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ResponseStrategy(Enum):
    """Strategies for handling large outputs."""

    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    CHUNK = "chunk"
    PRIORITIZE = "prioritize"


@dataclass
class TokenBudget:
    """Token budget for agent responses."""

    max_tokens: int = 8192
    reserved_for_metadata: int = 500
    reserved_for_summary: int = 200
    chunk_overlap: int = 100

    @property
    def available_tokens(self) -> int:
        return self.max_tokens - self.reserved_for_metadata


class OutputTokenManager:
    """Production-ready output token management."""

    def __init__(self, budget: TokenBudget | None = None):
        self.budget = budget or TokenBudget()
        self.token_patterns = {
            "code_block": r"```[\s\S]*?```",
            "inline_code": r"`[^`]+`",
            "headers": r"^#{1,6}\s+.*$",
            "lists": r"^[\s]*[-*+]\s+.*$",
            "numbers": r"^\s*\d+\.\s+.*$",
        }

        logger.info(
            f"Initialized OutputTokenManager with {self.budget.max_tokens} token limit"
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
        if not text:
            return 0

        # Basic tokenization estimate
        # More accurate would use tiktoken, but this is lightweight
        words = len(text.split())
        chars = len(text)

        # Heuristic: average of word-based and char-based estimates
        word_tokens = words * 1.3  # Average 1.3 tokens per word
        char_tokens = chars / 3.8  # Average 3.8 chars per token

        return int((word_tokens + char_tokens) / 2)

    def optimize_response(
        self,
        content: dict[str, Any],
        strategy: ResponseStrategy = ResponseStrategy.PRIORITIZE,
    ) -> dict[str, Any]:
        """Optimize response content to fit token budget."""

        # Convert to text for token estimation
        text_content = self._extract_text_content(content)
        estimated_tokens = self.estimate_tokens(text_content)

        if estimated_tokens <= self.budget.available_tokens:
            return content  # No optimization needed

        logger.info(
            f"Optimizing response: {estimated_tokens} tokens -> {self.budget.available_tokens} target"
        )

        if strategy == ResponseStrategy.TRUNCATE:
            return self._truncate_content(content, estimated_tokens)
        elif strategy == ResponseStrategy.SUMMARIZE:
            return self._summarize_content(content, estimated_tokens)
        elif strategy == ResponseStrategy.CHUNK:
            return self._chunk_content(content, estimated_tokens)
        else:  # PRIORITIZE
            return self._prioritize_content(content, estimated_tokens)

    def _extract_text_content(self, content: dict[str, Any]) -> str:
        """Extract text content for token estimation."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            text_parts = []
            for key, value in content.items():
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, list | dict):
                    text_parts.append(f"{key}: {str(value)}")
            return "\n".join(text_parts)
        else:
            return str(content)

    def _truncate_content(
        self, content: dict[str, Any], estimated_tokens: int
    ) -> dict[str, Any]:
        """Truncate content to fit budget."""
        target_ratio = self.budget.available_tokens / estimated_tokens

        if isinstance(content, dict):
            optimized = {}
            for key, value in content.items():
                if isinstance(value, str):
                    target_length = int(len(value) * target_ratio)
                    optimized[key] = value[:target_length]
                    if len(value) > target_length:
                        optimized[key] += "... [truncated]"
                elif isinstance(value, list):
                    target_count = max(1, int(len(value) * target_ratio))
                    optimized[key] = value[:target_count]
                    if len(value) > target_count:
                        optimized[key].append("... [truncated]")
                else:
                    optimized[key] = value

            # Add truncation notice
            optimized["_token_management"] = {
                "strategy": "truncated",
                "original_tokens": estimated_tokens,
                "target_tokens": self.budget.available_tokens,
            }

            return optimized
        else:
            # Simple string truncation
            text = str(content)
            target_length = int(len(text) * target_ratio)
            return text[:target_length] + "... [truncated]"

    def _summarize_content(
        self, content: dict[str, Any], estimated_tokens: int
    ) -> dict[str, Any]:
        """Summarize content intelligently."""
        if isinstance(content, dict):
            summary = {
                "_summary": "Content summarized due to token limits",
                "_token_management": {
                    "strategy": "summarized",
                    "original_tokens": estimated_tokens,
                    "target_tokens": self.budget.available_tokens,
                },
            }

            # Keep high-priority keys
            priority_keys = [
                "success",
                "error",
                "status",
                "result",
                "findings",
                "recommendations",
            ]

            for key in priority_keys:
                if key in content:
                    if isinstance(content[key], str) and len(content[key]) > 500:
                        # Summarize long strings
                        summary[key] = self._summarize_string(content[key])
                    else:
                        summary[key] = content[key]

            # Add counts for other data
            for key, value in content.items():
                if key not in priority_keys and key not in summary:
                    if isinstance(value, list):
                        summary[f"{key}_count"] = len(value)
                        if value:  # Include sample
                            summary[f"{key}_sample"] = value[0]
                    elif isinstance(value, dict):
                        summary[f"{key}_keys"] = list(value.keys())[:5]

            return summary
        else:
            return self._summarize_string(str(content))

    def _summarize_string(self, text: str, max_length: int = 200) -> str:
        """Summarize a long string."""
        if len(text) <= max_length:
            return text

        # Try to find sentence boundaries
        sentences = re.split(r"[.!?]+", text)

        if len(sentences) > 1:
            # Take first and last sentences
            summary = sentences[0].strip()
            if len(summary) < max_length - 50:
                summary += f" ... {sentences[-1].strip()}"
            return summary[:max_length]
        else:
            # Simple truncation with ellipsis
            return text[: max_length - 3] + "..."

    def _chunk_content(
        self, content: dict[str, Any], estimated_tokens: int
    ) -> dict[str, Any]:
        """Chunk content into multiple responses."""
        chunks_needed = (estimated_tokens // self.budget.available_tokens) + 1

        if isinstance(content, dict):
            # Split dict into chunks
            items = list(content.items())
            chunk_size = len(items) // chunks_needed

            first_chunk = dict(items[:chunk_size])
            first_chunk["_token_management"] = {
                "strategy": "chunked",
                "chunk": 1,
                "total_chunks": chunks_needed,
                "original_tokens": estimated_tokens,
                "remaining_keys": [k for k, _ in items[chunk_size:]],
            }

            return first_chunk
        else:
            # Split text into chunks
            text = str(content)
            chunk_size = len(text) // chunks_needed

            first_chunk = text[:chunk_size]
            return {
                "content": first_chunk,
                "_token_management": {
                    "strategy": "chunked",
                    "chunk": 1,
                    "total_chunks": chunks_needed,
                    "original_tokens": estimated_tokens,
                },
            }

    def _prioritize_content(
        self, content: dict[str, Any], estimated_tokens: int
    ) -> dict[str, Any]:
        """Prioritize most important content."""
        if isinstance(content, dict):
            # Priority levels for different types of content
            priority_map = {
                # Critical information
                "success": 1,
                "error": 1,
                "errors": 1,
                "status": 1,
                "result": 1,
                "results": 1,
                # Important findings
                "findings": 2,
                "recommendations": 2,
                "summary": 2,
                "analysis": 2,
                "metrics": 2,
                # Supporting data
                "performance": 3,
                "stats": 3,
                "details": 3,
                "metadata": 3,
                "configuration": 3,
                # Debug/verbose information
                "debug": 4,
                "trace": 4,
                "logs": 4,
                "raw_data": 4,
            }

            # Sort items by priority
            items = list(content.items())
            items.sort(key=lambda x: priority_map.get(x[0].lower(), 3))

            # Build optimized response within budget
            optimized = {}
            current_tokens = 0
            included_items = 0

            for key, value in items:
                item_text = f"{key}: {str(value)}"
                item_tokens = self.estimate_tokens(item_text)

                if current_tokens + item_tokens <= self.budget.available_tokens:
                    optimized[key] = value
                    current_tokens += item_tokens
                    included_items += 1
                else:
                    # Try to include a summary/sample
                    if isinstance(value, list) and value:
                        sample_text = f"{key}_sample: {str(value[0])}"
                        sample_tokens = self.estimate_tokens(sample_text)
                        if (
                            current_tokens + sample_tokens
                            <= self.budget.available_tokens
                        ):
                            optimized[f"{key}_sample"] = value[0]
                            optimized[f"{key}_count"] = len(value)
                            current_tokens += sample_tokens
                    break

            # Add management info
            optimized["_token_management"] = {
                "strategy": "prioritized",
                "original_tokens": estimated_tokens,
                "optimized_tokens": current_tokens,
                "included_items": included_items,
                "total_items": len(items),
                "excluded_items": len(items) - included_items,
            }

            return optimized
        else:
            # For non-dict content, use summarization
            return self._summarize_content({"content": content}, estimated_tokens)

    def validate_response_size(self, response: Any) -> tuple[bool, int, str]:
        """Validate response is within token limits."""
        text_content = self._extract_text_content(response)
        token_count = self.estimate_tokens(text_content)

        is_valid = token_count <= self.budget.max_tokens
        status = "valid" if is_valid else "exceeds_limit"

        return is_valid, token_count, status

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get token optimization statistics."""
        return {
            "token_budget": {
                "max_tokens": self.budget.max_tokens,
                "available_tokens": self.budget.available_tokens,
                "reserved_metadata": self.budget.reserved_for_metadata,
                "reserved_summary": self.budget.reserved_for_summary,
            },
            "supported_strategies": [s.value for s in ResponseStrategy],
            "token_estimation": "heuristic_based",
        }


# Global token manager
_token_manager: OutputTokenManager | None = None


def get_output_token_manager() -> OutputTokenManager:
    """Get global output token manager."""
    global _token_manager
    if _token_manager is None:
        _token_manager = OutputTokenManager()
    return _token_manager


def optimize_agent_response(
    response: Any, strategy: ResponseStrategy = ResponseStrategy.PRIORITIZE
) -> Any:
    """Optimize agent response for token limits."""
    manager = get_output_token_manager()
    return manager.optimize_response(response, strategy)


def validate_agent_response(response: Any) -> tuple[bool, int, str]:
    """Validate agent response token count."""
    manager = get_output_token_manager()
    return manager.validate_response_size(response)


# Decorator for automatic response optimization
def token_optimized(strategy: ResponseStrategy = ResponseStrategy.PRIORITIZE):
    """Decorator to automatically optimize function responses."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return optimize_agent_response(result, strategy)

        return wrapper

    return decorator
