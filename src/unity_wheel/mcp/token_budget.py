"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


Dynamic Token Budget Allocation for Claude Code CLI
Adaptively manages token budgets based on file complexity and MCP capabilities
"""

import ast
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import tiktoken


@dataclass
class FileComplexity:
    """Represents complexity metrics for a file"""

    path: str
    lines_of_code: int
    cyclomatic_complexity: int
    import_depth: int
    class_count: int
    function_count: int
    token_estimate: int
    complexity_score: float


class TokenBudgetAllocator:
    """Manages dynamic token budget allocation across files and MCPs"""

    def __init__(self, max_context_tokens: int = 100000):
        self.max_context_tokens = max_context_tokens
        self.safe_limit = int(max_context_tokens * 0.8)  # 80% safety margin
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.complexity_cache: dict[str, FileComplexity] = {}
        self.usage_history: list[dict] = []
        self.mcp_capabilities = self._load_mcp_capabilities()

    def _load_mcp_capabilities(self) -> dict[str, dict]:
        """Load MCP capabilities and token efficiency"""
        return {
            "filesystem": {
                "max_file_size": 1_000_000,
                "token_efficiency": 1.0,
                "specialization": ["general"],
            },
            "dependency-graph": {
                "max_file_size": 100_000,
                "token_efficiency": 0.3,  # Only extracts structure
                "specialization": ["imports", "structure"],
            },
            "ripgrep": {
                "max_file_size": 10_000_000,
                "token_efficiency": 0.1,  # Only returns matches
                "specialization": ["search"],
            },
            "python_analysis": {
                "max_file_size": 500_000,
                "token_efficiency": 0.5,
                "specialization": ["analysis", "execution"],
            },
        }

    def calculate_file_complexity(self, file_path: str) -> FileComplexity:
        """Calculate complexity metrics for a file"""
        if file_path in self.complexity_cache:
            cached = self.complexity_cache[file_path]
            # Check if file has been modified
            if os.path.getmtime(file_path) <= cached.token_estimate:
                return cached

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()

            # Token estimation
            token_count = len(self.encoding.encode(content))

            # Initialize metrics
            complexity = FileComplexity(
                path=file_path,
                lines_of_code=len(lines),
                cyclomatic_complexity=1,
                import_depth=0,
                class_count=0,
                function_count=0,
                token_estimate=token_count,
                complexity_score=0.0,
            )

            # Parse AST for Python files
            if file_path.endswith(".py"):
                try:
                    tree = ast.parse(content)
                    complexity.class_count = len(
                        [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                    )
                    complexity.function_count = len(
                        [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    )
                    complexity.import_depth = self._calculate_import_depth(tree)
                    complexity.cyclomatic_complexity = (
                        self._calculate_cyclomatic_complexity(tree)
                    )
                except:
                    pass  # If parsing fails, use basic metrics

            # Calculate composite complexity score (0-1)
            complexity.complexity_score = min(
                1.0,
                (
                    (complexity.lines_of_code / 1000) * 0.2
                    + (complexity.cyclomatic_complexity / 50) * 0.3
                    + (complexity.import_depth / 10) * 0.1
                    + (complexity.class_count / 20) * 0.2
                    + (complexity.function_count / 50) * 0.2
                ),
            )

            self.complexity_cache[file_path] = complexity
            return complexity

        except (ValueError, KeyError, AttributeError):
            # Return basic complexity for unreadable files
            return FileComplexity(
                path=file_path,
                lines_of_code=0,
                cyclomatic_complexity=1,
                import_depth=0,
                class_count=0,
                function_count=0,
                token_estimate=0,
                complexity_score=0.0,
            )

    def _calculate_import_depth(self, tree: ast.AST) -> int:
        """Calculate maximum import depth"""
        imports = [
            n for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom)
        ]
        if not imports:
            return 0

        max_depth = 0
        for imp in imports:
            if isinstance(imp, ast.ImportFrom) and imp.module:
                depth = len(imp.module.split("."))
                max_depth = max(max_depth, depth)
        return max_depth

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Simple cyclomatic complexity calculation"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, ast.If | ast.While | ast.For | ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def allocate_budget(
        self, files: list[str], task_type: str = "general"
    ) -> dict[str, dict]:
        """Allocate token budget across files based on complexity and task"""
        allocations = {}
        total_tokens = 0

        # Calculate complexity for all files
        complexities = []
        for file in files:
            complexity = self.calculate_file_complexity(file)
            complexities.append(complexity)
            total_tokens += complexity.token_estimate

        # Sort by complexity score
        complexities.sort(key=lambda x: x.complexity_score, reverse=True)

        # Allocate tokens based on complexity and available budget
        if total_tokens <= self.safe_limit:
            # All files fit, use full content
            for comp in complexities:
                allocations[comp.path] = {
                    "tokens": comp.token_estimate,
                    "strategy": "full",
                    "mcp": self._select_mcp(comp, task_type),
                }
        else:
            # Need to be selective
            remaining_budget = self.safe_limit

            # First pass: Include high-complexity files
            for comp in complexities:
                if (
                    comp.complexity_score > 0.7
                    and comp.token_estimate < remaining_budget * 0.3
                ):
                    allocations[comp.path] = {
                        "tokens": comp.token_estimate,
                        "strategy": "full",
                        "mcp": self._select_mcp(comp, task_type),
                    }
                    remaining_budget -= comp.token_estimate

            # Second pass: Use specialized MCPs for remaining files
            for comp in complexities:
                if comp.path not in allocations and remaining_budget > 1000:
                    if task_type == "search":
                        # Use ripgrep for search tasks
                        allocations[comp.path] = {
                            "tokens": min(500, remaining_budget),
                            "strategy": "search",
                            "mcp": "ripgrep",
                        }
                        remaining_budget -= 500
                    elif task_type == "structure":
                        # Use dependency graph for structure analysis
                        allocations[comp.path] = {
                            "tokens": min(1000, remaining_budget),
                            "strategy": "structure",
                            "mcp": "dependency-graph",
                        }
                        remaining_budget -= 1000
                    else:
                        # Chunk large files
                        chunk_size = min(comp.token_estimate // 3, remaining_budget)
                        if chunk_size > 500:
                            allocations[comp.path] = {
                                "tokens": chunk_size,
                                "strategy": "chunked",
                                "mcp": "filesystem",
                            }
                            remaining_budget -= chunk_size

        # Record usage for learning
        self.usage_history.append(
            {
                "timestamp": time.time(),
                "task_type": task_type,
                "total_files": len(files),
                "allocated_files": len(allocations),
                "total_tokens": sum(a["tokens"] for a in allocations.values()),
                "strategies": defaultdict(int),
            }
        )

        for alloc in allocations.values():
            self.usage_history[-1]["strategies"][alloc["strategy"]] += 1

        return allocations

    def _select_mcp(self, complexity: FileComplexity, task_type: str) -> str:
        """Select optimal MCP based on file complexity and task"""
        file_size = complexity.token_estimate * 4  # Rough byte estimate

        # Filter MCPs by file size capability
        suitable_mcps = []
        for mcp, caps in self.mcp_capabilities.items():
            if (
                file_size <= caps["max_file_size"]
                and task_type in caps["specialization"]
            ):
                suitable_mcps.append((mcp, caps["token_efficiency"]))

        # Sort by efficiency
        suitable_mcps.sort(key=lambda x: x[1])

        return suitable_mcps[0][0] if suitable_mcps else "filesystem"

    def get_usage_report(self) -> dict:
        """Generate usage report for optimization"""
        if not self.usage_history:
            return {}

        recent = self.usage_history[-100:]  # Last 100 operations

        return {
            "total_operations": len(recent),
            "avg_files_per_operation": sum(h["total_files"] for h in recent)
            / len(recent),
            "avg_tokens_used": sum(h["total_tokens"] for h in recent) / len(recent),
            "strategy_distribution": self._aggregate_strategies(recent),
            "efficiency_rate": sum(
                h["allocated_files"] / h["total_files"] for h in recent
            )
            / len(recent),
        }

    def _aggregate_strategies(self, history: list[dict]) -> dict[str, float]:
        """Aggregate strategy usage"""
        totals = defaultdict(int)
        for h in history:
            for strategy, count in h["strategies"].items():
                totals[strategy] += count

        total_count = sum(totals.values())
        return {s: c / total_count for s, c in totals.items()} if total_count else {}

    def optimize_from_history(self):
        """Learn from usage history to improve allocations"""
        report = self.get_usage_report()

        if report and report["efficiency_rate"] < 0.8:
            # Adjust token safety margin if we're excluding too many files
            self.safe_limit = min(
                int(self.max_context_tokens * 0.9), int(self.safe_limit * 1.1)
            )

        # Future enhancements:
        # - Track which files are most useful for different tasks
        # - Learn optimal chunk sizes dynamically
        # - Improve MCP selection based on historical success rates


# Example usage
if __name__ == "__main__":
    allocator = TokenBudgetAllocator()

    # Test files
    test_files = [
        "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/src/unity_wheel/strategy/wheel.py",
        "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/src/unity_wheel/math/options.py",
        "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/run.py",
    ]

    # Allocate budget for different task types
    for task in ["general", "search", "structure"]:
        allocations = allocator.allocate_budget(test_files, task)
        logger.info("\nTask: {task}")
        for _file, _alloc in allocations.items():
            logger.info(
                "  {Path(file).name}: {alloc['tokens']} tokens via {alloc['mcp']} ({alloc['strategy']})"
            )

    # Show usage report
    logger.info("\nUsage Report: {json.dumps(allocator.get_usage_report(), indent=2)}")
