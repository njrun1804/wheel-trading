"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


Batch PyREPL executor for efficient code testing.
Executes multiple code snippets in a single PyREPL call for better performance.
"""

import asyncio
import textwrap
from dataclasses import dataclass
from typing import Any


@dataclass
class CodeSnippet:
    """Individual code snippet for testing."""

    id: str
    code: str
    description: str
    expected_output_type: str | None = None
    timeout_seconds: int = 30


@dataclass
class BatchResult:
    """Result from batch execution."""

    snippet_id: str
    success: bool
    output: Any
    error: str | None
    execution_time: float


class BatchPyREPLExecutor:
    """
    Efficient batch executor for PyREPL code testing.
    Reduces overhead by executing multiple snippets in one call.
    """

    def __init__(self):
        self.execution_history = []
        self.batch_size_limits = {"simple": 3, "medium": 5, "complex": 8, "maximum": 10}

    async def execute_batch(
        self, snippets: list[CodeSnippet], complexity: str = "medium"
    ) -> list[BatchResult]:
        """
        Execute multiple code snippets in optimized batches.

        Args:
            snippets: List of code snippets to execute
            complexity: Query complexity level for batch sizing

        Returns:
            List of execution results
        """
        batch_size = self.batch_size_limits.get(complexity, 5)
        all_results = []

        # Process in batches
        for i in range(0, len(snippets), batch_size):
            batch = snippets[i : i + batch_size]
            batch_results = await self._execute_single_batch(batch)
            all_results.extend(batch_results)

        return all_results

    async def _execute_single_batch(
        self, snippets: list[CodeSnippet]
    ) -> list[BatchResult]:
        """Execute a single batch of snippets."""
        # Create batch execution code
        batch_code = self._create_batch_code(snippets)

        # Simulate PyREPL execution (would call actual MCP in production)
        results = await self._call_pyrepl_mcp(batch_code)

        # Parse results
        return self._parse_batch_results(results, snippets)

    def _create_batch_code(self, snippets: list[CodeSnippet]) -> str:
        """
        Create optimized batch execution code.
        Each snippet is wrapped in error handling and timing.
        """
        batch_parts = [
            "import time",
            "import traceback",
            "import json",
            "",
            "results = {}",
            "",
        ]

        for snippet in snippets:
            # Wrap each snippet for isolated execution
            wrapped = f"""
# Execute snippet: {snippet.id}
try:
    start_time = time.time()
    
    # User code
{textwrap.indent(snippet.code, '    ')}
    
    # Capture result
    if 'result' in locals():
        output = result
    else:
        output = "Execution completed"
    
    execution_time = time.time() - start_time
    
    results['{snippet.id}'] = {{
        'success': True,
        'output': output,
        'error': None,
        'execution_time': execution_time
    }}
    
except (ValueError, KeyError, AttributeError) as e:
    results['{snippet.id}'] = {{
        'success': False,
        'output': None,
        'error': str(e) + '\\n' + traceback.format_exc(),
        'execution_time': time.time() - start_time if 'start_time' in locals() else 0
    }}
    
# Clear namespace for next snippet
for var in list(locals()):
    if var not in ['results', 'time', 'traceback', 'json']:
        del locals()[var]
"""
            batch_parts.append(wrapped)

        # Add final result export
        batch_parts.extend(
            ["", "# Export results", "print(json.dumps(results, indent=2))"]
        )

        return "\n".join(batch_parts)

    async def _call_pyrepl_mcp(self, code: str) -> dict[str, Any]:
        """
        Call PyREPL MCP server with batch code.
        This is a placeholder - would integrate with actual MCP.
        """
        # Simulate execution delay
        await asyncio.sleep(0.1)

        # Simulate results
        # In production, this would call the actual PyREPL MCP
        return {
            "snippet_1": {
                "success": True,
                "output": 42,
                "error": None,
                "execution_time": 0.05,
            }
        }

    def _parse_batch_results(
        self, raw_results: dict[str, Any], snippets: list[CodeSnippet]
    ) -> list[BatchResult]:
        """Parse raw results into BatchResult objects."""
        results = []

        for snippet in snippets:
            if snippet.id in raw_results:
                result_data = raw_results[snippet.id]
                result = BatchResult(
                    snippet_id=snippet.id,
                    success=result_data.get("success", False),
                    output=result_data.get("output"),
                    error=result_data.get("error"),
                    execution_time=result_data.get("execution_time", 0),
                )
            else:
                # Missing result
                result = BatchResult(
                    snippet_id=snippet.id,
                    success=False,
                    output=None,
                    error="No result returned",
                    execution_time=0,
                )

            results.append(result)

        return results

    def create_test_variations(
        self, base_code: str, variations: list[dict[str, Any]]
    ) -> list[CodeSnippet]:
        """
        Create multiple test variations from a base code template.
        Useful for testing edge cases and parameter sweeps.
        """
        snippets = []

        for i, variation in enumerate(variations):
            # Replace placeholders in base code
            varied_code = base_code
            for key, value in variation.items():
                placeholder = f"{{{key}}}"
                varied_code = varied_code.replace(placeholder, str(value))

            snippet = CodeSnippet(
                id=f"variation_{i}",
                code=varied_code,
                description=f"Test variation {i}: {variation}",
            )
            snippets.append(snippet)

        return snippets

    async def test_hypothesis(
        self, hypothesis: str, test_cases: list[tuple[str, Any]]
    ) -> dict[str, Any]:
        """
        Test a hypothesis with multiple test cases.

        Args:
            hypothesis: Description of what we're testing
            test_cases: List of (code, expected_result) tuples

        Returns:
            Test results and conclusion
        """
        logger.info("\nTesting hypothesis: {hypothesis}")

        # Create snippets from test cases
        snippets = []
        for i, (code, expected) in enumerate(test_cases):
            snippet = CodeSnippet(
                id=f"test_{i}",
                code=code,
                description=f"Test case {i}",
                expected_output_type=type(expected).__name__ if expected else None,
            )
            snippets.append(snippet)

        # Execute batch
        results = await self.execute_batch(snippets)

        # Analyze results
        passed = 0
        failed = 0

        for i, result in enumerate(results):
            expected = test_cases[i][1]
            if result.success and result.output == expected:
                passed += 1
            else:
                failed += 1

        conclusion = (
            "CONFIRMED" if failed == 0 else "REJECTED" if passed == 0 else "PARTIAL"
        )

        return {
            "hypothesis": hypothesis,
            "conclusion": conclusion,
            "passed": passed,
            "failed": failed,
            "total": len(test_cases),
            "details": [
                {
                    "test": test_cases[i][0],
                    "expected": test_cases[i][1],
                    "actual": r.output,
                    "success": r.success,
                    "error": r.error,
                }
                for i, r in enumerate(results)
            ],
        }

    def optimize_batch_order(self, snippets: list[CodeSnippet]) -> list[CodeSnippet]:
        """
        Optimize execution order for better performance.
        Quick tests first, complex ones last.
        """

        # Estimate complexity by code length and operations
        def estimate_complexity(snippet: CodeSnippet) -> int:
            code = snippet.code
            complexity = len(code)

            # Add weight for expensive operations
            if "for" in code or "while" in code:
                complexity *= 2
            if "import" in code:
                complexity *= 1.5
            if "sleep" in code:
                complexity *= 3

            return complexity

        # Sort by estimated complexity
        return sorted(snippets, key=estimate_complexity)


# Integration with unified compute
class PyREPLIntegration:
    """Integration layer for unified compute system."""

    def __init__(self):
        self.executor = BatchPyREPLExecutor()

    async def validate_implementation(
        self, implementation: str, test_inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Validate an implementation against test inputs.
        """
        # Create test snippets
        snippets = []
        for i, inputs in enumerate(test_inputs):
            test_code = f"""
# Setup inputs
{chr(10).join(f'{k} = {repr(v)}' for k, v in inputs.items())}

# Implementation
{implementation}

# Capture result
result = output if 'output' in locals() else None
"""
            snippet = CodeSnippet(
                id=f"validation_{i}", code=test_code, description=f"Validation test {i}"
            )
            snippets.append(snippet)

        # Execute tests
        results = await self.executor.execute_batch(snippets)

        # Summarize
        success_rate = sum(1 for r in results if r.success) / len(results)

        return {
            "valid": success_rate >= 0.95,
            "success_rate": success_rate,
            "results": results,
        }

    async def explore_parameter_space(
        self, function_template: str, parameter_ranges: dict[str, list[Any]]
    ) -> dict[str, Any]:
        """
        Explore parameter space for optimization.
        """
        # Generate parameter combinations
        import itertools

        param_names = list(parameter_ranges.keys())
        param_values = [parameter_ranges[name] for name in param_names]

        combinations = list(itertools.product(*param_values))[:50]  # Limit to 50

        # Create test variations
        variations = [
            dict(zip(param_names, combo, strict=False)) for combo in combinations
        ]

        snippets = self.executor.create_test_variations(function_template, variations)

        # Execute
        results = await self.executor.execute_batch(
            self.executor.optimize_batch_order(snippets), complexity="complex"
        )

        # Find optimal parameters
        best_result = max(
            (r for r in results if r.success),
            key=lambda r: r.output if isinstance(r.output, int | float) else 0,
            default=None,
        )

        if best_result:
            best_idx = int(best_result.snippet_id.split("_")[1])
            best_params = variations[best_idx]
        else:
            best_params = None

        return {
            "optimal_parameters": best_params,
            "tested_combinations": len(combinations),
            "successful_tests": sum(1 for r in results if r.success),
            "best_output": best_result.output if best_result else None,
        }


# Convenience functions
def create_batch_executor() -> BatchPyREPLExecutor:
    """Create optimized batch executor."""
    return BatchPyREPLExecutor()


async def quick_test(code_snippets: list[str]) -> list[BatchResult]:
    """Quick test multiple code snippets."""
    executor = BatchPyREPLExecutor()

    snippets = [
        CodeSnippet(id=f"snippet_{i}", code=code, description=f"Quick test {i}")
        for i, code in enumerate(code_snippets)
    ]

    return await executor.execute_batch(snippets)
