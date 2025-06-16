#!/usr/bin/env python3
"""
Create a simple, reliable text search that works when ripgrep fails.
"""

import re
from pathlib import Path
from typing import Any


class SimpleTextSearch:
    """Simple text search that works without external dependencies."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    async def search(self, query: str, max_results: int = 100) -> list[dict[str, Any]]:
        """Search for text in Python files."""
        results = []

        # Get all Python files
        python_files = list(self.project_root.rglob("*.py"))

        # Create regex pattern for case-insensitive search
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        except re.error:
            # If query has regex chars, fall back to literal search
            pattern = re.compile(query, re.IGNORECASE)

        # Search through files
        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    if pattern.search(line):
                        results.append(
                            {
                                "content": line.strip(),
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": line_num,
                                "column": 0,
                            }
                        )

                        if len(results) >= max_results:
                            return results

            except Exception:
                # Skip files that can't be read
                continue

        return results


# Test the simple search
async def test_simple_search():
    """Test the simple text search."""
    project_root = Path(__file__).parent
    search = SimpleTextSearch(project_root)

    test_queries = ["WheelStrategy", "options", "class", "def ", "import"]

    for query in test_queries:
        print(f"\nSearching for '{query}'...")
        results = await search.search(query, max_results=5)
        print(f"Found {len(results)} results:")

        for i, result in enumerate(results[:3], 1):
            print(
                f"  {i}. {result['file']}:{result['line']} - {result['content'][:80]}..."
            )


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_simple_search())
