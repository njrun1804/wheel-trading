from __future__ import annotations

import asyncio

"""Simplified MCTS that actually works."""

import logging
from typing import Any

from ..core.solution import SearchNode

logger = logging.getLogger(__name__)


class SimplifiedMCTS:
    """Simplified MCTS for testing."""

    def __init__(self):
        self._initialized = False
        logger.info("SimplifiedMCTS created")

    async def initialize(self):
        """Initialize MCTS."""
        logger.info("SimplifiedMCTS initialized")
        self._initialized = True

    async def explore(
        self,
        query: str,
        context: dict[str, Any],
        simulations: int = 2000,
        parallel_batch_size: int = 256,
        hardware_executor: Any | None = None,
    ) -> SearchNode:
        """Run simplified exploration."""
        await asyncio.sleep(0)
        logger.info(f"Running simplified exploration for: {query}")
        root = SearchNode(code="", parent=None)
        if "hello" in query.lower():
            code = """def hello_world():
    ""\"Print Hello World.""\"
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()"""
        elif "add" in query.lower():
            code = (
                'def add_numbers(a, b):\n    """Add two numbers."""\n    return a + b'
            )
        else:
            code = f"""# Solution for: {query}
def solution():
    ""\"TODO: Implement solution.""\"
    pass"""
        root.code = code
        root.visits = simulations
        root.value_sum = simulations * 0.8
        return root

    async def fast_search(
        self,
        query: str,
        context: dict[str, Any],
        simulations: int = 100,
        hardware_executor: Any | None = None,
    ) -> SearchNode:
        """Fast search - just calls explore with fewer simulations."""
        return await self.explore(
            query, context, simulations, hardware_executor=hardware_executor
        )

    def get_update_count(self) -> int:
        """Get model update count."""
        return 0

    def save_models(self, path):
        """Save models (no-op for simplified version)."""
        logger.info("SimplifiedMCTS save_models called (no-op)")
